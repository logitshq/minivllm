"""Lithos /api/chat adapter for the frozen E023 finite HTTP client.

Same run_group API and scheduler; no server, socket listener, model, or tokenizer
is created here. A private source-pinned base module receives protocol hooks;
the public/frozen client module is never modified. Original NDJSON/body bytes,
counts and receive timestamps remain original evidence, never synthetic SSE.

Only the uncached-fixed-D-v1 contract is accepted for qualified counts. Its
independent receipt must bind the real model/route, messages, concurrency and
raw-ID EOS semantics. The controller must additionally check this run's complete
zero-hit Generation logs; this client cannot establish that from NDJSON.

DONE releases a worker immediately. A separate task drains to EOF and closes its
response. All such tasks must finish before closing the HTTP client, within one
cleanup deadline. Bad/missing tails invalidate the whole point. No child/server
is owned here: unresolved cleanup requires the external owner to reap it.
"""
import asyncio
import base64
import codecs
import copy
import hashlib
import json
from pathlib import Path
import time
import types


BASE_PATH = Path(__file__).with_name('client.py')
BASE_SHA256 = 'ef64a114f4f7d90ec5044d6bdcf058ca05f98dea445d6639db706a4fc75217dc'
PROTOCOL = 'lithos-ollama-ndjson-v1'
CACHE_CONTRACT = 'uncached-fixed-D-v1'
COUNT_RULE = 'lithos_eval_count_includes_one_terminal_eos_on_stop'
SCHEMA = 'E023-lithos-ndjson-group-v1'


def _private_base():
    raw = BASE_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != BASE_SHA256:
        raise ValueError('frozen HTTP client source changed; review adapter before use')
    module = types.ModuleType(__name__ + '._private_frozen_client')
    module.__file__ = str(BASE_PATH)
    exec(compile(raw, str(BASE_PATH), 'exec'), module.__dict__)
    return module


_BASE = _private_base()
_base_validate = _BASE._validate_jobs
_base_qualification = _BASE._qualification_errors
_base_new_row = _BASE._new_row
_base_summary = _BASE._summarize
canonical_body_sha256 = _BASE.canonical_body_sha256
COUNT_SEMANTICS = _BASE.COUNT_SEMANTICS
GroupCancelled = _BASE.GroupCancelled
StreamProtocolError = _BASE.StreamProtocolError
_integer = _BASE._integer
_error = _BASE._error


def _cap_view(row):
    """Private validation/statistics view; never sent or stored as original body."""
    return {**row, 'body': {'stream': True, 'stream_options': {'include_usage': True},
                           'max_tokens': row['body']['options']['num_predict']}}


def _validate_jobs(jobs, concurrency, phase):
    jobs = copy.deepcopy(list(jobs))
    for job in jobs:
        body = job.get('body')
        if job.get('path') != '/api/chat' or not isinstance(body, dict):
            raise ValueError('only original /api/chat JSON bodies are supported')
        if set(body) != {'model', 'messages', 'stream', 'options'} or body['stream'] is not True:
            raise ValueError('fixed model/messages/stream/options body required; no lifecycle/EOS overrides')
        if not isinstance(body['model'], str) or not body['model']:
            raise ValueError('explicit served model name required')
        messages = body['messages']
        if (not isinstance(messages, list) or len(messages) != 1
                or not isinstance(messages[0], dict) or set(messages[0]) != {'role', 'content'}
                or messages[0]['role'] != 'user' or not isinstance(messages[0]['content'], str)):
            raise ValueError('uncached D contract supports only one user string message')
        options = body['options']
        if not isinstance(options, dict) or set(options) != {'num_predict', 'temperature', 'top_p', 'top_k', 'seed'}:
            raise ValueError('only fixed greedy options and num_predict are supported')
        if (type(options['num_predict']) is not int or options['num_predict'] < 1
                or type(options['temperature']) not in (int, float) or options['temperature'] != 0
                or type(options['top_p']) not in (int, float) or options['top_p'] != 1
                or type(options['top_k']) is not int or options['top_k'] != 0
                or type(options['seed']) is not int or options['seed'] != 0):
            raise ValueError('positive num_predict and fixed greedy sampling required')
        canonical_body_sha256(body)
        json.dumps(job, allow_nan=False)
    _base_validate([_cap_view(job) for job in jobs], concurrency, phase)
    return jobs


def _qualification_errors(q, jobs, concurrency):
    if not isinstance(q, dict):
        return ['independent Lithos raw-ID/route/count qualification absent']
    checks = {
        'concurrency_type': type(q.get('concurrency')) is int,
        'protocol': q.get('protocol') == PROTOCOL,
        'cache_contract': q.get('cache_contract') == CACHE_CONTRACT,
        'usage_count_rule': q.get('usage_count_rule') == COUNT_RULE,
        'raw_ids_count_rule': q.get('raw_ids_count_rule') == 'none',
        'terminal_token_ids': q.get('terminal_token_ids') == [248046, 248044]
                              and all(type(x) is int for x in q.get('terminal_token_ids', [])),
        'no_post_terminal_tokens': q.get('no_post_terminal_tokens') is True,
        'expected_prompt_tokens_by_fixture': isinstance(q.get('expected_prompt_tokens_by_fixture'), dict)
            and all(_integer(q['expected_prompt_tokens_by_fixture'].get(j['fixture_id']))
                    and q['expected_prompt_tokens_by_fixture'][j['fixture_id']] > 0 for j in jobs),
        'no_wire_raw_id_oracle': not q.get('expected_raw_ids_by_request'),
    }
    # The frozen validator still binds the ORIGINAL body/path/fixture/request/C.
    normalized_rule = {**q, 'usage_count_rule': 'identity', 'raw_ids_count_rule': 'none'}
    return _base_qualification(normalized_rule, jobs, concurrency) + [
        'invalid qualification field: ' + name for name, okay in checks.items() if not okay]


def _new_row(job):
    row = _base_new_row(job)
    del row['raw_sse_chunks'], row['sse_events']
    row.update(protocol=PROTOCOL, body_sha256=canonical_body_sha256(job['body']),
               output_cap=job['body']['options']['num_predict'], raw_ndjson_chunks=[],
               ndjson_events=[], terminal_record=None, terminal_count=0,
               stream_eof=False, eof_perf_counter_s=None, reported_eval_count=None,
               count_normalization=None, drain_error=None)
    return row


def _json_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise StreamProtocolError('duplicate JSON key: ' + key)
        obj[key] = value
    return obj


def _bad_constant(value):
    raise StreamProtocolError('nonfinite JSON constant: ' + value)


class _NDJSON:
    def __init__(self, row):
        self.row, self.buffer = row, ''
        self.decoder = codecs.getincrementaldecoder('utf-8')('strict')

    def feed(self, chunk, received):
        self.buffer += self.decoder.decode(chunk)
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            line = line.removesuffix('\r')
            if self.row['done']:
                raise StreamProtocolError('data after terminal done record')
            self.row['ndjson_events'].append({'client_perf_counter_s': received, 'line': line})
            data = json.loads(line, object_pairs_hook=_json_object, parse_constant=_bad_constant)
            self.consume(data, received)
        if self.row['done'] and self.buffer:
            raise StreamProtocolError('unterminated data after terminal done record')

    def consume(self, data, received):
        row = self.row
        if not isinstance(data, dict) or 'error' in data:
            raise StreamProtocolError('non-object or error NDJSON record')
        if type(data.get('done')) is not bool or data.get('model') != row['body']['model']:
            raise StreamProtocolError('invalid done flag or unexpected served model')
        message = data.get('message')
        if (not isinstance(message, dict) or set(message) != {'role', 'content'}
                or message['role'] != 'assistant' or not isinstance(message['content'], str)):
            raise StreamProtocolError('unqualified message/reasoning/tool shape')
        text = message['content']
        if data['done']:
            row['terminal_count'] += 1
            row['terminal_record'] = copy.deepcopy(data)
            row['finish_reason'] = data.get('done_reason')
            row['usage'] = {k: v for k, v in data.items()
                            if k.endswith('_count') or k.endswith('_duration')}
            row['usage_events'].append({'client_perf_counter_s': received,
                                        'usage': copy.deepcopy(row['usage'])})
            if text:
                raise StreamProtocolError('official terminal message must be empty; no implicit text dedup')
            row['done'] = True
            row['done_perf_counter_s'] = row['terminal_perf_counter_s'] = received
        elif text:
            row['answer_events'].append({'client_perf_counter_s': received, 'text': text})
            if row['first_output_perf_counter_s'] is None:
                row['first_output_perf_counter_s'] = row['first_answer_perf_counter_s'] = received

    def eof(self, received):
        self.buffer += self.decoder.decode(b'', final=True)
        if self.buffer or not self.row['done'] or self.row['terminal_count'] != 1:
            raise StreamProtocolError('missing/truncated terminal NDJSON record')
        self.row['stream_eof'] = True
        self.row['eof_perf_counter_s'] = received


def _retain(row, chunk, received, max_raw_bytes):
    if len(row['raw_ndjson_chunks']) >= 65536:
        row['raw_body_truncated'] = True
        raise StreamProtocolError('65536 raw chunk bound exceeded; partial body retained')
    remaining = max_raw_bytes - row['raw_body_bytes']
    retained = chunk[:remaining]
    row['raw_ndjson_chunks'].append({'client_perf_counter_s': received,
        'body_bytes_base64': base64.b64encode(retained).decode('ascii')})
    row['raw_body_bytes'] += len(retained)
    if len(retained) != len(chunk):
        row['raw_body_truncated'] = True
        raise StreamProtocolError('raw byte bound exceeded; partial body retained')


async def _drain_close(response, iterator, parser, row, deadline, max_raw_bytes):
    try:
        async with asyncio.timeout_at(deadline):
            async for chunk in iterator:
                received = time.perf_counter()
                _retain(row, chunk, received, max_raw_bytes)
                parser.feed(chunk, received)
            parser.eof(time.perf_counter())
    except BaseException as exc:
        row['drain_error'] = _error(exc)
        row['status'], row['error'] = 'error', row['error'] or _error(exc)
        raise
    finally:
        try:
            await response.aclose()
        except BaseException as exc:
            # EOF can already be valid. Closing is still required, including
            # cancellation raised by the transport itself rather than our timer.
            row['drain_error'] = row['drain_error'] or _error(exc)
            row['status'], row['error'] = 'error', row['error'] or _error(exc)
            raise


async def _request(client, base_url, row, timeout_s, closers, max_raw_bytes):
    response, handed_off = None, False
    row['status'] = 'running'
    row['start_perf_counter_s'] = time.perf_counter()
    row['absolute_deadline_perf_counter_s'] = row['start_perf_counter_s'] + timeout_s
    deadline = asyncio.get_running_loop().time() + timeout_s
    try:
        async with asyncio.timeout_at(deadline):
            request = client.build_request('POST', base_url + row['path'], json=row['body'],
                                           headers={'X-Request-ID': row['request_id']})
            # Actual serialization is retained as well as the canonical body hash.
            row['request_body_bytes_base64'] = base64.b64encode(request.content).decode('ascii')
            row['request_body_bytes_sha256'] = hashlib.sha256(request.content).hexdigest()
            response = await client.send(request, stream=True)
            row['http_status'], row['response_headers'] = response.status_code, dict(response.headers)
            row['headers_perf_counter_s'] = time.perf_counter()
            parser, iterator = _NDJSON(row), response.aiter_bytes()
            async for chunk in iterator:
                received = time.perf_counter()
                _retain(row, chunk, received, max_raw_bytes)
                if response.status_code == 200:
                    parser.feed(chunk, received)
                    if row['done']:
                        closers.append(asyncio.create_task(_drain_close(response, iterator, parser, row,
                            deadline, max_raw_bytes), name='drain-close:' + row['request_id']))
                        handed_off = True
                        row['status'] = 'done'
                        return  # Keep the original DONE→immediate-refill scheduler.
            if response.status_code != 200:
                raise StreamProtocolError(f'HTTP {response.status_code}; body retained, no retry')
            parser.eof(time.perf_counter())
    except asyncio.CancelledError as exc:
        row['status'], row['error'] = 'cancelled', _error(exc)
        raise
    except Exception as exc:
        row['status'], row['error'] = 'error', _error(exc)
    finally:
        if row['terminal_perf_counter_s'] is None:
            row['terminal_perf_counter_s'] = time.perf_counter()
        if response is not None and not handed_off:
            closers.append(asyncio.create_task(response.aclose(), name='close:' + row['request_id']))


def _finalize_row(row, qualification, qerrors):
    row['answer'] = ''.join(x['text'] for x in row['answer_events'])
    row['reasoning'] = ''
    start, done, first = (row[k] for k in ('start_perf_counter_s', 'done_perf_counter_s', 'first_output_perf_counter_s'))
    row['e2e_s'] = done - start if done is not None else None
    row['ttft_s'] = first - start if first is not None else None
    errors = row['validation_errors']
    if not row['done'] or row['terminal_count'] != 1 or not row['stream_eof'] or row['error']:
        errors.append('request/terminal/tail did not complete successfully')
    finish = row['finish_reason']
    if finish not in {'stop', 'length'}:
        errors.append('normal EOS or output-cap finish required')
    raw = row['usage'] if isinstance(row['usage'], dict) else {}
    n, p = raw.get('eval_count'), raw.get('prompt_eval_count')
    row['reported_eval_count'] = row['reported_completion_tokens'] = n
    row['reported_prompt_tokens'] = p
    if not _integer(n) or not _integer(p) or n < 1 or p < 1:
        errors.append('missing/invalid raw token counts')
    # Lithos does not emit cached_prompt_tokens over this wire. Do not invent 0.
    row['reported_cached_prompt_tokens'] = None
    row['cache_evidence_scope'] = 'external complete Generation logs still required'
    if _integer(n) and (n > row['output_cap'] or (finish == 'length' and n != row['output_cap'])):
        errors.append('reported generated count conflicts with cap/finish')
    row['count_qualification_errors'] = list(qerrors)
    if not qerrors:
        expected = qualification['expected_prompt_tokens_by_fixture'][row['fixture_id']]
        if p != expected:
            errors.append('prompt count differs from independently qualified fixture')
        if _integer(n) and finish in {'stop', 'length'}:
            committed = n - (1 if finish == 'stop' else 0)
            row['count_normalization'] = {'rule': COUNT_RULE, 'reported_eval_count': n,
                'terminal_eos_subtracted': 1 if finish == 'stop' else 0,
                'committed_output_tokens': committed,
                'receipt_sha256': qualification['receipt_sha256']}
            if not errors:
                row['qualified_output_tokens'] = committed
    row['transport_valid'] = not errors


def _summarize(rows, qerrors, group_error, cleanup, attempt_end):
    result = _base_summary([_cap_view(row) for row in rows], qerrors, group_error, cleanup, attempt_end)
    result['scope'] = 'HTTP concurrency includes official single-generation FIFO queueing; no GPU batching claim'
    result['cache_evidence_scope'] = 'conditional on external per-run zero-hit log verification'
    return result


async def _cleanup(workers, closers, client, budget_s):
    started = time.perf_counter()
    deadline = asyncio.get_running_loop().time() + budget_s
    errors, all_tasks = [], list(workers)
    cancelled_again = False

    async def bounded(tasks):
        nonlocal cancelled_again
        pending = {task for task in tasks if not task.done()}
        while pending and asyncio.get_running_loop().time() < deadline:
            try:
                _, pending = await asyncio.wait(pending,
                    timeout=max(0, deadline - asyncio.get_running_loop().time()))
            except asyncio.CancelledError:
                cancelled_again = True
                for task in pending:
                    task.cancel()
        return pending

    for task in workers:
        if not task.done():
            task.cancel()
    pending = await bounded(workers)
    all_tasks.extend(closers)  # No racing client close while a response is read.
    pending |= await bounded(closers)
    timed_out = bool(pending)
    for task in pending:
        task.cancel()
    try:
        await asyncio.sleep(0)
    except asyncio.CancelledError:
        cancelled_again = True
    remaining = [task for task in all_tasks if not task.done()]
    client_close_started = not remaining
    if client_close_started:
        closer = asyncio.create_task(client.aclose(), name='close:client')
        all_tasks.append(closer)
        late = await bounded([closer])
        timed_out |= bool(late)
        for task in late:
            task.cancel()
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            cancelled_again = True
    else:
        errors.append({'task': 'close:client', 'type': 'BlockedByActiveResponse',
                       'message': 'client close not raced against unresolved response tasks'})
    for task in all_tasks:
        if task.cancelled() and task not in workers:
            errors.append({'task': task.get_name(), 'type': 'CancelledError',
                           'message': 'independent response/client close task cancelled'})
        elif task.done() and not task.cancelled() and task.exception() is not None:
            errors.append({'task': task.get_name(), **_error(task.exception())})
    unresolved = [task.get_name() for task in all_tasks if not task.done()]
    return {'start_perf_counter_s': started, 'end_perf_counter_s': time.perf_counter(),
            'budget_s': budget_s, 'complete': not timed_out and not errors and not unresolved,
            'deadline_exceeded': timed_out, 'pending_tasks': unresolved, 'errors': errors,
            'cancelled_again': cancelled_again, 'client_close_started': client_close_started,
            'order': 'workers, response-drain-and-close, client-close; one absolute budget'}


# These hooks affect only our unregistered private module, never client.py or
# another importer. The actual closed-loop scheduling code remains byte-pinned.
for _name in ('_validate_jobs', '_qualification_errors', '_new_row', '_request',
              '_finalize_row', '_summarize', '_cleanup'):
    setattr(_BASE, _name, globals()[_name])


async def run_group(base_url, jobs, concurrency, *, qualification=None,
                    request_timeout_s=300, group_timeout_s=3600, cleanup_timeout_s=5,
                    transport=None, output_dir=None, phase='formal', headers=None,
                    max_raw_bytes_per_request=8 << 20):
    """Original finite queue, protocol-specific evidence; no retry or model work."""
    jobs = _validate_jobs(jobs, concurrency, phase)
    out = Path(output_dir) if output_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=False)
    cancelled = False
    try:
        result = await _BASE.run_group(base_url, jobs, concurrency, qualification=qualification,
            request_timeout_s=request_timeout_s, group_timeout_s=group_timeout_s,
            cleanup_timeout_s=cleanup_timeout_s, transport=transport, output_dir=None,
            phase=phase, headers=headers, max_raw_bytes_per_request=max_raw_bytes_per_request)
    except GroupCancelled as exc:
        result, cancelled = exc.result, True
    result.update(schema=SCHEMA, protocol=PROTOCOL, cache_contract=CACHE_CONTRACT,
                  frozen_scheduler_sha256=BASE_SHA256,
                  raw_bytes_semantics='HTTP response body after content decoding, before NDJSON/UTF-8 parsing',
                  terminal_clock='receipt of bytes completing the unique newline-terminated done:true record')
    if out is not None:
        temporary = out / 'group.json.tmp'
        temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        temporary.replace(out / 'group.json')
    if cancelled:
        raise GroupCancelled(result)
    return result
