"""E023 single-loop, finite closed-loop HTTP measurement client (no model imports).

Public API: ``await run_group(base_url, jobs, concurrency, *, qualification=None,
request_timeout_s=300, group_timeout_s=3600, cleanup_timeout_s=5, transport=None,
output_dir=None, phase='formal', headers=None)``. No listener is started here.

Input jobs are frozen JSON objects with request_id (unique string), fixture_id
(string), path (relative /v1/... path), and body (unchanged JSON POST object).
body must request stream=true, include_usage=true, n=1, a positive max_tokens or
max_completion_tokens, and normal EOS. No prompt, tokenizer, or 19/32 assumption
is built in. Formal N must equal max(64, 8*C); warm/admission_probe/qualification
may explicitly use fewer jobs. The first min(C,N) workers share a start barrier;
each DONE/error immediately refills from the fixed queue. No retry or replacement
is made, including after early EOS, 429, or an individual deadline.

Result schema E023-http-group-v1:
  requests: one row per planned job, in input order, including unsubmitted tails;
    identity/body, start/first_output/first_answer/first_reasoning/DONE/terminal
    perf_counter timestamps, HTTP status, raw_sse_chunks (base64 decoded HTTP
    body bytes with receipt times), sse_events (exact data field plus time),
    answer/reasoning events, usage_events/final usage, finish, response IDs,
    benchmark_* native fields, error/validation_errors, and qualified count.
  summary: attempted/submitted/DONE/success counts, full group and attempt windows,
    p50 median and p95 nearest-rank E2E/TTFT, actual length distribution, prompt
    counts, early-EOS/cap ratios, outstanding trace, and aggregate tokens/window.
  cleanup: separate bounded cleanup clock, errors, and any still-pending tasks.
  qualification: supplied receipt verbatim; this client does NOT grant engine,
    route, model, or scheduling correctness qualification.
    formal_point_eligible additionally excludes warm/probe/qualification phases.

Optional qualification contract (otherwise usage is retained but untrusted):
  status='qualified', receipt_sha256=<64 hex>, concurrency=C, normal_eos=true,
  fixture_contracts={fixture_id: {'path': '/v1/...', 'body_sha256':
    canonical_body_sha256(full_frozen_body)}},
  request_fixture_map={request_id: fixture_id} (exact complete queue mapping),
  usage_semantics=COUNT_SEMANTICS,
  usage_count_rule='identity' (usage already excludes terminal/rejected tokens),
  raw_ids_count_rule='none' | 'all_ids' | 'native_committed_ids',
  terminal_token_ids=[...] (required for native_committed_ids),
  expected_prompt_tokens_by_fixture={...} (optional),
  expected_raw_ids_by_request={...} (optional, only an explicitly qualified
    schedule's oracle; never implicitly reuse a C1 answer at another C).
native_committed_ids checks actual full IDs = committed IDs + declared terminal
EOS, or identical arrays with no EOS. It never infers token IDs from text. A new
external usage convention needs separate qualification and explicit adaptation;
this client never guesses an EOS subtraction from finish_reason='stop'.
Full body hashes include model, prompt/messages, cap, sampling/template options.
Matching hashes bind the HTTP request, not the server's actual loaded weights or
source: the independent controller must verify those before issuing the receipt.
Custom stop/minimum-output/ignore_eos fields are outside this normal-EOS protocol
and are rejected even if a caller supplies a matching hash.

Connection reuse is disabled for all arms; limit=2*C accommodates a completed
response's asynchronous close without imposing a hidden refill barrier. At most
C requests are POST-to-terminal outstanding. Timers include client pool wait.
Raw collection is in memory during measurement; output_dir must be new and stores
group.json after cleanup, including partial failures/cancellation. Hard process
kill is outside this persistence guarantee. External cancellation raises
GroupCancelled with its .result AFTER saving partial rows. Request/group limits
are absolute wall deadlines, not HTTP read-idle timeouts. Cleanup is separately
bounded; cancellation-resistant transports are reported incomplete, not claimed
closed. Controllers must treat any incomplete cleanup as fatal and reap servers.
"""

import asyncio
import base64
import codecs
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import time

import httpx


COUNT_SEMANTICS = 'committed_target_output_excluding_terminal_including_reasoning'
PHASES = {'formal', 'warm', 'admission_probe', 'qualification'}
UNSUPPORTED_EOS_FIELDS = {'stop', 'stop_token_ids', 'stop_strings', 'min_tokens',
                          'min_new_tokens', 'min_length', 'ignore_eos',
                          'forced_eos_token_id', 'include_stop_str_in_output'}


class GroupCancelled(asyncio.CancelledError):
    """Controller cancellation, carrying every completed and partial request."""
    def __init__(self, result):
        super().__init__('HTTP group cancelled; partial results retained')
        self.result = result


class StreamProtocolError(ValueError):
    pass


def _error(exc):
    return {'type': type(exc).__name__, 'message': str(exc)}


def _integer(value):
    return type(value) is int and value >= 0


def _ids(value):
    return isinstance(value, list) and all(_integer(x) for x in value)


def canonical_body_sha256(body):
    """SHA256 of UTF-8 sorted compact JSON; no defaults or prompt normalization.

    The controller uses this same function when freezing each fixture contract.
    JSON values only, finite numbers, and string object keys are required.
    """
    def validate(value):
        if isinstance(value, dict):
            if not all(isinstance(key, str) for key in value):
                raise ValueError('JSON object keys must be strings')
            for item in value.values():
                validate(item)
        elif isinstance(value, list):
            for item in value:
                validate(item)
        elif value is not None and type(value) not in (str, bool, int, float):
            raise ValueError('only JSON values allowed in frozen body')
    if not isinstance(body, dict):
        raise ValueError('frozen body must be an object')
    validate(body)
    raw = json.dumps(body, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                     allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def _distribution(values):
    if not values:
        return {'n': 0, 'min': None, 'p50': None, 'p95': None, 'max': None, 'sum': 0}
    ordered = sorted(values)
    return {'n': len(values), 'min': ordered[0], 'p50': statistics.median(ordered),
            'p95': ordered[math.ceil(.95 * len(ordered)) - 1], 'max': ordered[-1],
            'sum': sum(values)}


def _validate_jobs(jobs, concurrency, phase):
    if type(concurrency) is not int or concurrency < 1 or phase not in PHASES:
        raise ValueError('positive integer concurrency and an explicit supported phase required')
    jobs = copy.deepcopy(list(jobs))
    if not jobs or (phase == 'formal' and len(jobs) != max(64, 8 * concurrency)):
        raise ValueError('formal N must equal max(64,8*C); other phases need at least one job')
    seen = set()
    for job in jobs:
        for key in ('request_id', 'fixture_id', 'path'):
            if not isinstance(job.get(key), str) or not job[key]:
                raise ValueError(f'job {key} must be a nonempty string')
        if job['request_id'] in seen:
            raise ValueError('duplicate request_id')
        if any(ord(c) < 32 or ord(c) > 126 for c in job['request_id']):
            raise ValueError('request_id must be printable ASCII for X-Request-ID')
        seen.add(job['request_id'])
        if not job['path'].startswith('/') or job['path'].startswith('//'):
            raise ValueError('job path must be an origin-relative path')
        body = job.get('body')
        if not isinstance(body, dict) or body.get('stream') is not True:
            raise ValueError('body must request stream=true')
        if (body.get('stream_options') or {}).get('include_usage') is not True:
            raise ValueError('body must explicitly request include_usage=true')
        if body.get('n', 1) != 1:
            raise ValueError('single-choice normal EOS requests required')
        overrides = sorted(UNSUPPORTED_EOS_FIELDS & set(body))
        if overrides:
            raise ValueError('unqualified stop/minimum-output/EOS override fields: ' + ', '.join(overrides))
        caps = [body[k] for k in ('max_tokens', 'max_completion_tokens') if k in body]
        if not caps or any(type(x) is not int or x < 1 for x in caps) or len(set(caps)) != 1:
            raise ValueError('one consistent positive output cap required')
        canonical_body_sha256(body)
        json.dumps(job, allow_nan=False)
    return jobs


def _qualification_errors(qualification, jobs, concurrency):
    if qualification is None:
        return ['usage/raw-ID count semantics have not been independently qualified']
    checks = {
        'status': qualification.get('status') == 'qualified',
        'receipt_sha256': bool(re.fullmatch('[0-9a-fA-F]{64}', str(qualification.get('receipt_sha256', '')))),
        'concurrency': qualification.get('concurrency') == concurrency,
        'normal_eos': qualification.get('normal_eos') is True,
        'request_fixture_map': qualification.get('request_fixture_map') == {j['request_id']: j['fixture_id'] for j in jobs},
        'usage_semantics': qualification.get('usage_semantics') == COUNT_SEMANTICS,
        'usage_count_rule': qualification.get('usage_count_rule') == 'identity',
        'raw_ids_count_rule': qualification.get('raw_ids_count_rule') in {'none', 'all_ids', 'native_committed_ids'},
    }
    contracts = qualification.get('fixture_contracts')
    checks['fixture_contracts'] = isinstance(contracts, dict) and all(
        isinstance(contracts.get(j['fixture_id']), dict)
        and contracts[j['fixture_id']].get('path') == j['path']
        and contracts[j['fixture_id']].get('body_sha256') == canonical_body_sha256(j['body'])
        for j in jobs)
    if qualification.get('raw_ids_count_rule') == 'native_committed_ids':
        checks['terminal_token_ids'] = bool(qualification.get('terminal_token_ids')) and _ids(qualification['terminal_token_ids'])
    return ['invalid qualification field: ' + k for k, okay in checks.items() if not okay]


def _new_row(job):
    return {**job, 'status': 'not_started', 'http_status': None, 'done': False,
            'start_perf_counter_s': None, 'first_output_perf_counter_s': None,
            'first_answer_perf_counter_s': None, 'first_reasoning_perf_counter_s': None,
            'done_perf_counter_s': None, 'terminal_perf_counter_s': None,
            'raw_sse_chunks': [], 'raw_body_bytes': 0, 'raw_body_truncated': False,
            'sse_events': [], 'answer_events': [], 'reasoning_events': [],
            'usage_events': [], 'usage': None, 'finish_reason': None,
            'response_ids': [], 'native_fields': {}, 'error': None,
            'validation_errors': [], 'qualified_output_tokens': None,
            'e2e_s': None, 'ttft_s': None, 'reported_prompt_tokens': None,
            'reported_completion_tokens': None}


def _consume_event(row, payload, received):
    row['sse_events'].append({'client_perf_counter_s': received, 'data': payload})
    if payload == '[DONE]':
        row['done'] = True
        row['done_perf_counter_s'] = row['terminal_perf_counter_s'] = received
        return
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise StreamProtocolError('SSE JSON event must be an object')
    if 'error' in data:
        raise StreamProtocolError('server SSE error: ' + json.dumps(data['error']))
    if data.get('id') is not None:
        if not isinstance(data['id'], str):
            raise StreamProtocolError('response id must be a string')
        if data['id'] not in row['response_ids']:
            row['response_ids'].append(data['id'])
    if data.get('usage') is not None:
        row['usage_events'].append({'client_perf_counter_s': received, 'usage': data['usage']})
        row['usage'] = data['usage']
    for key, value in data.items():
        if key.startswith('benchmark_'):
            if key in row['native_fields'] and row['native_fields'][key] != value:
                raise StreamProtocolError('native final field changed: ' + key)
            row['native_fields'][key] = value
    choices = data.get('choices', [])
    if not isinstance(choices, list) or len(choices) > 1:
        raise StreamProtocolError('only one completion choice supported')
    for choice in choices:
        if choice.get('index', 0) != 0:
            raise StreamProtocolError('unexpected choice index')
        finish = choice.get('finish_reason')
        if finish is not None:
            if row['finish_reason'] is not None and row['finish_reason'] != finish:
                raise StreamProtocolError('finish reason changed')
            row['finish_reason'] = finish
        delta = choice.get('delta') or {}
        answer = choice.get('text', delta.get('content'))
        reasoning = delta.get('reasoning_content', delta.get('reasoning'))
        if delta.get('reasoning_content') is not None and delta.get('reasoning') is not None:
            if delta['reasoning_content'] != delta['reasoning']:
                raise StreamProtocolError('conflicting reasoning aliases')
        for kind, piece in (('answer', answer), ('reasoning', reasoning)):
            if piece is None or piece == '':
                continue
            if not isinstance(piece, str):
                raise StreamProtocolError('non-string output requires a new qualified adapter')
            row[kind + '_events'].append({'client_perf_counter_s': received, 'text': piece})
            if row['first_' + kind + '_perf_counter_s'] is None:
                row['first_' + kind + '_perf_counter_s'] = received
            if row['first_output_perf_counter_s'] is None:
                row['first_output_perf_counter_s'] = received


class _SSE:
    """Incremental UTF-8 and SSE framing; chunks are retained separately verbatim."""
    def __init__(self, row):
        self.row = row
        self.decoder = codecs.getincrementaldecoder('utf-8')('strict')
        self.buffer = ''
        self.data = []

    def feed(self, chunk, received):
        self.buffer += self.decoder.decode(chunk)
        while '\n' in self.buffer and not self.row['done']:
            line, self.buffer = self.buffer.split('\n', 1)
            line = line.removesuffix('\r')
            if line == '':
                if self.data:
                    payload, self.data = '\n'.join(self.data), []
                    _consume_event(self.row, payload, received)
            elif line.startswith('data:'):
                value = line[5:]
                self.data.append(value[1:] if value.startswith(' ') else value)


async def _request(client, base_url, row, timeout_s, closers, max_raw_bytes):
    response = None
    row['status'] = 'running'
    row['start_perf_counter_s'] = time.perf_counter()
    row['absolute_deadline_perf_counter_s'] = row['start_perf_counter_s'] + timeout_s
    try:
        # asyncio's monotonic loop clock may have a different epoch from perf_counter.
        async with asyncio.timeout_at(asyncio.get_running_loop().time() + timeout_s):
            request = client.build_request('POST', base_url + row['path'], json=row['body'],
                                           headers={'X-Request-ID': row['request_id']})
            response = await client.send(request, stream=True)
            row['http_status'] = response.status_code
            row['response_headers'] = dict(response.headers)
            row['headers_perf_counter_s'] = time.perf_counter()
            parser = _SSE(row)
            async for chunk in response.aiter_bytes():
                received = time.perf_counter()
                if len(row['raw_sse_chunks']) >= 65536:
                    row['raw_body_truncated'] = True
                    raise StreamProtocolError('65536 raw chunk bound exceeded; partial body retained')
                remaining = max_raw_bytes - row['raw_body_bytes']
                retained = chunk[:remaining]
                row['raw_sse_chunks'].append({'client_perf_counter_s': received,
                    'body_bytes_base64': base64.b64encode(retained).decode('ascii')})
                row['raw_body_bytes'] += len(retained)
                if len(retained) != len(chunk):
                    row['raw_body_truncated'] = True
                    raise StreamProtocolError('raw body limit exceeded; partial body retained')
                if response.status_code == 200:
                    parser.feed(chunk, received)
                    if row['done']:
                        break
            if response.status_code != 200:
                raise StreamProtocolError(f'HTTP {response.status_code}; body retained, no retry')
            if not row['done']:
                raise StreamProtocolError('SSE ended without DONE')
            row['status'] = 'done'
    except asyncio.CancelledError as exc:
        row['status'], row['error'] = 'cancelled', _error(exc)
        raise
    except Exception as exc:
        row['status'], row['error'] = 'error', _error(exc)
    finally:
        if row['terminal_perf_counter_s'] is None:
            row['terminal_perf_counter_s'] = time.perf_counter()
        if response is not None:
            # Closing is separately bounded after timers; never delay refill for close.
            closers.append(asyncio.create_task(response.aclose(), name='close:' + row['request_id']))


def _finalize_row(row, qualification, qualification_errors):
    row['answer'] = ''.join(x['text'] for x in row['answer_events'])
    row['reasoning'] = ''.join(x['text'] for x in row['reasoning_events'])
    start, done, first = (row[k] for k in ('start_perf_counter_s', 'done_perf_counter_s', 'first_output_perf_counter_s'))
    row['e2e_s'] = done - start if done is not None else None
    row['ttft_s'] = first - start if first is not None else None
    errors = row['validation_errors']
    if not row['done'] or row['error'] is not None:
        errors.append('request did not complete successfully')
    if not isinstance(row['finish_reason'], str) or row['finish_reason'] not in {'stop', 'length'}:
        errors.append('missing or unsupported normal-EOS/cap finish reason')
    if len(row['response_ids']) > 1:
        errors.append('multiple response IDs within one request')
    usage = row['usage'] if isinstance(row['usage'], dict) else {}
    n = usage.get('completion_tokens')
    p = usage.get('prompt_tokens')
    row['reported_completion_tokens'] = n
    row['reported_prompt_tokens'] = p
    if not _integer(n) or not _integer(p):
        errors.append('missing or invalid usage token counts')
    total = usage.get('total_tokens')
    if total is not None and (not _integer(total) or ((_integer(n) and _integer(p)) and total != n + p)):
        errors.append('usage total does not equal prompt plus completion')
    details = usage.get('prompt_tokens_details')
    if details is not None and not isinstance(details, dict):
        errors.append('invalid prompt_tokens_details object')
    cached = details.get('cached_tokens') if isinstance(details, dict) else None
    row['reported_cached_prompt_tokens'] = cached
    if cached is not None and (not _integer(cached) or cached != 0):
        errors.append('nonzero or invalid cached prompt count; uncached contract required')
    if qualification_errors:
        row['count_qualification_errors'] = list(qualification_errors)
    else:
        q = qualification
        cap = row['body'].get('max_tokens', row['body'].get('max_completion_tokens'))
        if _integer(n) and (n > cap or (row['finish_reason'] == 'length' and n != cap)):
            errors.append('qualified committed count conflicts with output cap/finish')
        expected = q.get('expected_prompt_tokens_by_fixture', {}).get(row['fixture_id'])
        if expected is not None and p != expected:
            errors.append('prompt usage differs from qualified fixture count')
        native = row['native_fields']
        raw = native.get('benchmark_output_ids')
        rule = q['raw_ids_count_rule']
        if rule == 'all_ids' and (not _ids(raw) or len(raw) != n):
            errors.append('actual raw-ID count differs from qualified usage')
        if rule == 'native_committed_ids':
            committed = native.get('benchmark_committed_output_ids')
            eos = native.get('benchmark_terminal_eos_id')
            terminal_ids = set(q['terminal_token_ids'])
            if not _ids(raw) or not _ids(committed):
                errors.append('native raw/committed IDs absent or invalid')
            else:
                expected_raw = committed + ([eos] if eos is not None else [])
                if raw != expected_raw or len(committed) != n or any(x in terminal_ids for x in committed):
                    errors.append('native full/committed IDs or usage disagree')
                if not 1 <= len(raw) <= cap:
                    errors.append('native full raw IDs exceed cap or are empty')
                if eos is not None and (not _integer(eos) or eos not in terminal_ids or row['finish_reason'] != 'stop'):
                    errors.append('native terminal EOS contradicts qualified terminal set/finish')
                if eos is None and row['finish_reason'] != 'length':
                    errors.append('native stop has no declared terminal EOS')
            if 'benchmark_prompt_ids' in native:
                prompt_ids = native['benchmark_prompt_ids']
                if not _ids(prompt_ids) or len(prompt_ids) != p:
                    errors.append('native prompt IDs disagree with prompt usage')
        expected_raw = q.get('expected_raw_ids_by_request', {}).get(row['request_id'])
        if expected_raw is not None and raw != expected_raw:
            errors.append('actual raw IDs differ from explicitly supplied per-request oracle')
        row['count_qualification_errors'] = []
        if not errors:
            row['qualified_output_tokens'] = n
    row['transport_valid'] = row['done'] and row['error'] is None and not errors


def _summarize(rows, qualification_errors, group_error, cleanup, attempt_end):
    submitted = [r for r in rows if r['start_perf_counter_s'] is not None]
    done = [r for r in rows if r['done']]
    # A reused SSE response ID is evidence of possible cross-request mixing.
    owners = {}
    for row in submitted:
        for response_id in row['response_ids']:
            if response_id in owners:
                for affected in (row, owners[response_id]):
                    affected['validation_errors'].append('response ID reused across requests')
                    affected['qualified_output_tokens'] = None
                    affected['transport_valid'] = False
            owners[response_id] = row
    valid_rows = [r for r in rows if r['transport_valid']]
    counted = [r for r in rows if r['qualified_output_tokens'] is not None]
    complete = len(valid_rows) == len(rows) and group_error is None and cleanup['complete']
    eligible = complete and not qualification_errors and len(counted) == len(rows)
    first = min((r['start_perf_counter_s'] for r in submitted), default=None)
    window = max(r['done_perf_counter_s'] for r in done) - first if len(done) == len(rows) else None
    attempt_window = attempt_end - first if first is not None else None
    events = []
    for row in submitted:
        events.append((row['start_perf_counter_s'], 1, row['request_id']))
        if row['terminal_perf_counter_s'] is not None:
            events.append((row['terminal_perf_counter_s'], -1, row['request_id']))
    trace, outstanding, peak, time_at_peak = [], 0, 0, 0.
    for stamp, delta, rid in sorted(events):
        outstanding += delta
        peak = max(peak, outstanding)
        trace.append({'perf_counter_s': stamp, 'delta': delta, 'outstanding': outstanding, 'request_id': rid})
    for left, right in zip(trace, trace[1:]):
        if left['outstanding'] == peak:
            time_at_peak += right['perf_counter_s'] - left['perf_counter_s']
    lengths = [r['qualified_output_tokens'] for r in counted]
    main_counts = lengths if eligible else []
    early = sum(r['finish_reason'] == 'stop' and r['qualified_output_tokens'] < r['body'].get('max_tokens', r['body'].get('max_completion_tokens')) for r in counted)
    capped = sum(r['qualified_output_tokens'] == r['body'].get('max_tokens', r['body'].get('max_completion_tokens')) for r in counted)
    return {'transport_complete': complete, 'eligible_under_supplied_qualification': eligible,
            'planned_requests': len(rows), 'submitted_requests': len(submitted),
            'done_requests': len(done), 'successful_requests': len(valid_rows),
            'count_qualified_requests': len(counted), 'group_window_s': window,
            'full_attempt_window_s': attempt_window,
            'aggregate_output_tok_s': sum(main_counts) / window if eligible and window and window > 0 else None,
            'e2e_s': _distribution([r['e2e_s'] for r in rows]) if eligible else _distribution([]),
            'ttft_s': _distribution([r['ttft_s'] for r in rows if r['ttft_s'] is not None]) if eligible else _distribution([]),
            'actual_output_tokens': _distribution(main_counts),
            'prompt_tokens_total': sum(r['reported_prompt_tokens'] for r in rows) if eligible else None,
            'early_eos_ratio': early / len(rows) if eligible else None,
            'cap_ratio': capped / len(rows) if eligible else None,
            'diagnostic_successful_output_tokens': sum(lengths) if not qualification_errors else None,
            'diagnostic_successful_tokens_per_full_attempt_s': sum(lengths) / attempt_window if not qualification_errors and attempt_window and attempt_window > 0 else None,
            'diagnostic_completed_e2e_s': _distribution([r['e2e_s'] for r in done]),
            'max_http_outstanding': peak, 'time_at_peak_outstanding_s': time_at_peak,
            'outstanding_trace': trace, 'p95_algorithm': 'nearest-rank: sorted[ceil(.95*N)-1]',
            'scope': 'HTTP concurrency is not evidence of GPU batch size; no token-level ITL inferred'}


async def _cleanup(workers, closers, client, budget_s):
    started = time.perf_counter()
    loop = asyncio.get_running_loop()
    deadline = loop.time() + budget_s
    all_tasks = list(workers)
    errors = []
    cancelled_again = False

    async def wait_bounded(tasks):
        nonlocal cancelled_again
        pending = {t for t in tasks if not t.done()}
        while pending and loop.time() < deadline:
            try:
                _, pending = await asyncio.wait(pending, timeout=max(0, deadline - loop.time()))
            except asyncio.CancelledError:
                cancelled_again = True
                for task in pending:
                    task.cancel()
        return pending

    for task in workers:
        if not task.done():
            task.cancel()
    await wait_bounded(workers)
    all_tasks.extend(closers)  # Worker finally blocks have now added their response closers.
    all_tasks.append(asyncio.create_task(client.aclose(), name='close:client'))
    pending = await wait_bounded(all_tasks)
    for task in pending:
        task.cancel()
    # Give cooperative cancellation one turn without extending the wall budget.
    try:
        await asyncio.sleep(0)
    except asyncio.CancelledError:
        cancelled_again = True
    for task in all_tasks:
        if task.done() and not task.cancelled():
            exc = task.exception()
            if exc is not None:
                errors.append({'task': task.get_name(), **_error(exc)})
    unresolved = [t.get_name() for t in all_tasks if not t.done()]
    return {'start_perf_counter_s': started, 'end_perf_counter_s': time.perf_counter(),
            'budget_s': budget_s, 'complete': not pending and not errors,
            'deadline_exceeded': bool(pending), 'pending_tasks': unresolved,
            'errors': errors, 'cancelled_again': cancelled_again}


async def run_group(base_url, jobs, concurrency, *, qualification=None,
                    request_timeout_s=300, group_timeout_s=3600, cleanup_timeout_s=5,
                    transport=None, output_dir=None, phase='formal', headers=None,
                    max_raw_bytes_per_request=8 << 20):
    """Run fixed jobs once; returns invalid results on deadlines, raises on cancel."""
    jobs = _validate_jobs(jobs, concurrency, phase)
    for value in (request_timeout_s, group_timeout_s, cleanup_timeout_s):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('all absolute deadlines/cleanup budgets must be finite and positive')
    if type(max_raw_bytes_per_request) is not int or max_raw_bytes_per_request < 1:
        raise ValueError('positive raw byte bound required')
    url = httpx.URL(base_url)
    if url.scheme not in {'http', 'https'} or not url.host or url.query or url.fragment or url.path not in {'', '/'}:
        raise ValueError('base_url must be an HTTP origin without a path/query/fragment')
    base_url = str(url).rstrip('/')
    qualification = copy.deepcopy(qualification)
    qerrors = _qualification_errors(qualification, jobs, concurrency)
    out = Path(output_dir) if output_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=False)
    limits = httpx.Limits(max_connections=2 * concurrency, max_keepalive_connections=0)
    # Construction and all client configuration occur before the group clock.
    client = httpx.AsyncClient(transport=transport, limits=limits, timeout=None,
                              trust_env=False, follow_redirects=False, headers=headers)
    rows = [_new_row(job) for job in jobs]
    queue = asyncio.Queue()
    for row in rows:
        queue.put_nowait(row)
    barrier, ready = asyncio.Event(), asyncio.Queue()
    stop, closers = asyncio.Event(), []

    async def worker(index):
        ready.put_nowait(index)
        await barrier.wait()
        while not stop.is_set():
            try:
                row = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            await _request(client, base_url, row, request_timeout_s, closers, max_raw_bytes_per_request)
            # A zero-time scheduler yield preserves initial barrier fairness even for
            # a MockTransport with an entirely synchronous, already-buffered response.
            await asyncio.sleep(0)

    workers = [asyncio.create_task(worker(i), name=f'worker:{i}') for i in range(min(concurrency, len(jobs)))]
    group_error = None
    was_cancelled = False
    released = None
    try:
        for _ in workers:
            await ready.get()
        released = time.perf_counter()
        barrier.set()
        _, pending = await asyncio.wait(workers, timeout=group_timeout_s)
        if pending:
            group_error = {'type': 'GroupDeadline', 'message': 'absolute group wall deadline reached'}
        else:
            for task in workers:
                if not task.cancelled() and task.exception() is not None:
                    group_error = {'type': 'WorkerFailure', **_error(task.exception())}
                    break
    except asyncio.CancelledError as exc:
        was_cancelled = True
        group_error = _error(exc)
    finally:
        attempt_end = time.perf_counter()
        stop.set()
        cleanup = await _cleanup(workers, closers, client, cleanup_timeout_s)
    if cleanup['cancelled_again']:
        group_error = group_error or {'type': 'CancelledError', 'message': 'controller cancelled during cleanup'}
    attempt_end = max([attempt_end] + [r['terminal_perf_counter_s'] for r in rows if r['terminal_perf_counter_s'] is not None])
    for row in rows:
        try:
            _finalize_row(row, qualification, qerrors)
        except Exception as exc:
            # Malformed server metadata must not discard other requests or partial SSE.
            row['finalization_error'] = _error(exc)
            row['validation_errors'].append('metadata finalization failed; raw evidence retained')
            row['transport_valid'] = False
            row['qualified_output_tokens'] = None
    result = {'schema': 'E023-http-group-v1', 'phase': phase, 'concurrency': concurrency,
              'base_url': base_url, 'qualification': qualification, 'qualification_errors': qerrors,
              'request_timeout_s': request_timeout_s, 'group_timeout_s': group_timeout_s,
              'barrier_release_perf_counter_s': released,
              'group_deadline_perf_counter_s': released + group_timeout_s if released is not None else None,
              'group_error': group_error, 'cleanup': cleanup, 'requests': rows,
              'connection_policy': {'max_connections': 2 * concurrency, 'max_keepalive_connections': 0,
                                    'retries': 0, 'follow_redirects': False},
              'raw_bytes_semantics': 'HTTP response body after content decoding, before SSE/UTF-8 parsing'}
    result['summary'] = _summarize(rows, qerrors, group_error, cleanup, attempt_end)
    result['summary']['formal_point_eligible'] = phase == 'formal' and result['summary']['eligible_under_supplied_qualification']
    if out is not None:
        temporary = out / 'group.json.tmp'
        temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        temporary.replace(out / 'group.json')
    if was_cancelled or cleanup['cancelled_again']:
        raise GroupCancelled(result)
    return result
