"""Offline oMLX D512 output-tap/raw-SSE join; stdlib, no runtime imports.

join(group, observer_lines, jobs, fixtures_by_id, decode_text, expected_probe_sha256)
raises on incomplete, extra or contradictory evidence. decode_text uses the
externally pinned tokenizer with skip_special_tokens=False. This establishes
only observed target-only route, normal-EOS/count/text/request binding, never
full-state numerical equivalence, model ownership, performance or quality parity.
"""
import base64
from collections import Counter
import hashlib
import json
import math
import re


def need(ok, message):
    if not ok:
        raise ValueError(message)


def parse(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'duplicate JSON key'); result[key] = value
        return result
    def bad(_):
        raise ValueError('nonfinite JSON value')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)


def ids(value):
    return isinstance(value, list) and all(type(x) is int and 0 <= x < 248320 for x in value)


def integer(value):
    return type(value) is int and value >= 0


def wire(row):
    """Strict original official framing, including the separate usage frame."""
    chunks = row['raw_sse_chunks']; need(0 < len(chunks) <= 65536, 'raw chunk bound')
    parts, size, previous = [], 0, None
    for chunk in chunks:
        stamp = chunk['client_perf_counter_s']
        need(type(stamp) in (float, int) and math.isfinite(stamp) and
             (previous is None or stamp >= previous), 'receive clock order')
        previous = stamp
        raw = base64.b64decode(chunk['body_bytes_base64'], validate=True)
        size += len(raw); need(size <= 8 << 20, 'raw body bound'); parts.append(raw)
    raw = b''.join(parts)
    need(row['raw_body_truncated'] is False and row['raw_body_bytes'] == size, 'raw body count/truncation')
    frames = raw.decode('utf-8', errors='strict').split('\n\n')
    need(frames[-1] == '' and frames[-2] == 'data: [DONE]', 'missing final DONE/retained tail')
    response_id, text, finish, usage = None, [], None, None
    for frame in frames[:-2]:
        need(frame.startswith('data: ') and '\n' not in frame, 'original one-line SSE required')
        event = parse(frame[6:])
        need(isinstance(event, dict) and 'error' not in event and
             re.fullmatch(r'cmpl-[0-9a-f]{8}', event.get('id', '')), 'official response ID/error')
        need(response_id in (None, event['id']), 'wire response ID changed'); response_id = event['id']
        need(event.get('object') == 'text_completion' and event.get('model') == 'Qwen3.8-27B-nvfp4', 'official object/model')
        choices = event.get('choices')
        if choices == []:
            need(finish is not None and usage is None, 'usage before terminal/duplicate usage')
            usage = event['usage']
        else:
            need(finish is None and usage is None and isinstance(choices, list) and len(choices) == 1, 'output after terminal/choice count')
            choice = choices[0]
            need(set(choice) == {'index', 'text', 'finish_reason'} and type(choice['index']) is int and
                 choice['index'] == 0 and type(choice['text']) is str, 'official choice shape')
            text.append(choice['text']); finish = choice['finish_reason']
            need(finish in (None, 'stop', 'length'), 'unsupported finish reason')
    need(response_id and finish and isinstance(usage, dict), 'complete terminal and usage required')
    need(type(row['start_perf_counter_s']) in (float, int) and math.isfinite(row['start_perf_counter_s']) and
         row['start_perf_counter_s'] <= chunks[0]['client_perf_counter_s'] and
         row['done_perf_counter_s'] == chunks[-1]['client_perf_counter_s'], 'POST/DONE binding')
    return dict(response_id=response_id, text=''.join(text), finish_reason=finish, usage=usage,
                bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def join(group, observer_lines, jobs, fixtures_by_id, decode_text, expected_probe_sha256):
    from .serve_probe import PINS, ROOT  # stdlib-only constants, no installed imports
    C = group['concurrency']; count = 8 if C == 1 else 16
    need(type(C) is int and C in (1, 8) and len(jobs) == count and len(fixtures_by_id) == 8, 'fixed C1/C8 D workload')
    need(group['schema'] == 'E023-http-group-v1' and group['phase'] == 'qualification' and
         group['qualification'] is None and group['group_error'] is None and
         group['summary']['formal_point_eligible'] is False, 'unqualified qualification group only')
    cleanup = group['cleanup']
    need(cleanup['complete'] is True and cleanup['deadline_exceeded'] is False and cleanup['errors'] == [] and
         cleanup['pending_tasks'] == [] and cleanup['cancelled_again'] is False and
         cleanup.get('adapter') == 'native-close-cancellation-v1', 'strict versioned client cleanup')
    need(len(group['requests']) == count and len({j['request_id'] for j in jobs}) == count, 'complete unique jobs')
    seen, bindings, streams, states, histogram = {}, {}, {}, {}, Counter()
    opened = closed = False; digest = hashlib.sha256(); lines = 0
    for raw in observer_lines:
        if isinstance(raw, str):
            raw = raw.encode()
        digest.update(raw); lines += 1
        need(lines <= 2 * count * 512 + count * 3 + 2 and raw.endswith(b'\n') and not closed, 'observer bound/truncation/tail')
        event = parse(raw); kind = event['event']
        if not opened:
            need(kind == 'open' and event['schema'] == 'E023-omlx-D512-observer-v1' and event['concurrency'] == C and
                 event['sources'] == PINS and event['source_root'] == str(ROOT) and
                 event['probe_sha256'] == expected_probe_sha256 and event['performance_qualified'] is False, 'pinned observer header')
            opened = True; continue
        if kind == 'response':
            rid = event['response_id']; need(rid not in seen, 'duplicate response ID'); seen[rid] = event
        elif kind == 'binding':
            rid = event['request_id']; need(rid not in bindings, 'duplicate engine request ID'); bindings[rid] = event
            expected = dict(max_tokens=512, temperature=0, top_p=1, top_k=0, min_p=0,
                repetition_penalty=1, presence_penalty=0, frequency_penalty=0, xtc_probability=0,
                stop=[], stop_token_ids=[], thinking_budget=None)
            need(event['sampling'] == expected, 'effective greedy/no-override sampling')
        elif kind == 'batch':
            expected_route = dict(max_num_seqs=C, vlm_mtp_drafter=False, vlm_mtp_active=False,
                block_drafter=False, embedded_mtp=False, specprefill=False, turboquant=False)
            need(event['route'] == expected_route, 'actual target-only route/config')
            tokens, outputs = event['tokens'], event['outputs']
            rids = [x['request_id'] for x in tokens]
            need(len(rids) == len(set(rids)) <= C and [x['request_id'] for x in outputs] == rids, 'actual response batch/outputs')
            histogram[len(rids)] += 1
            for token, output in zip(tokens, outputs):
                rid = token['request_id']; state = states.setdefault(rid, dict(raw_ids=[], finished=False,
                    prompt_ids=token['prompt_ids'], scheduler_id=event['scheduler_id']))
                need(not state['finished'] and state['prompt_ids'] == token['prompt_ids'] and
                     state['scheduler_id'] == event['scheduler_id'] and token['output_parser'] is False,
                     'request route/prompt changed or output after finish')
                need(ids([token['token']]) and ids(token['prompt_ids']) and token['prompt_ids'], 'actual original IDs')
                state['raw_ids'].append(token['token'])
                need(len(state['raw_ids']) <= 512 and output['error'] is None and
                     type(output['finished']) is bool and output['cached_tokens'] == 0 and
                     output['prompt_tokens'] == len(state['prompt_ids']), 'output/error/cache/count')
                need(output['finish_reason'] == token['finish_reason'] and
                     output['finished'] == (token['finish_reason'] is not None), 'raw versus committed finish')
                committed = state['raw_ids'][:-1] if token['finish_reason'] == 'stop' else state['raw_ids']
                need(type(output['completion_tokens']) is int and output['completion_tokens'] == len(committed), 'completion count includes excluded EOS')
                if output['finished']:
                    need(output['committed_ids'] == committed, 'actual committed output sequence')
                    state.update(finished=True, finish_reason=output['finish_reason'], committed_ids=list(committed))
                else:
                    need(output['committed_ids'] is None, 'unexpected full terminal output')
        elif kind == 'wire':
            rid = event['response_id']; need(rid not in streams and event['complete'] is True, 'incomplete/duplicate server stream')
            streams[rid] = event
        elif kind == 'close':
            need(event['restored'] is True and event['official_returned_normally'] is True, 'observer scope did not close'); closed = True
        else:
            raise ValueError('unknown observer event')
    need(opened and closed and len(seen) == len(bindings) == len(streams) == len(states) == count, 'complete observed request set')
    by_response = {v['response_id']: k for k, v in bindings.items()}
    need(len(by_response) == count and set(by_response) == set(seen) == set(streams) and set(bindings) == set(states), 'response/engine UUID bijection')
    completed, used = [], set()
    for job, row in zip(jobs, group['requests']):
        need({k: row[k] for k in job} == job and set(job) == {'request_id', 'fixture_id', 'path', 'body'}, 'raw row/job/body binding')
        fixture = fixtures_by_id[job['fixture_id']]; body = job['body']
        expected = dict(model='Qwen3.8-27B-nvfp4', prompt=fixture['prompt_text'], temperature=0,
            top_p=1, repetition_penalty=1, max_tokens=512, n=1, stream=True, stream_options={'include_usage': True})
        need(job['path'] == '/v1/completions' and body == expected and fixture['max_output_tokens'] == 512 and
             fixture['normal_eos'] is True, 'frozen D512 body/normal EOS')
        need(row['status'] == 'done' and row['http_status'] == 200 and row['error'] is None and row['done'] is True and
             row['validation_errors'] == [] and row['qualified_output_tokens'] is None, 'incomplete/prequalified HTTP row')
        actual = wire(row); rid = actual['response_id']
        need(rid in by_response and rid not in used, 'wire UUID missing/reused'); used.add(rid)
        state, response = states[by_response[rid]], seen[rid]
        need(response['body'] == body and response['prompt'] == fixture['prompt_text'] and
             response['prompt_ids'] == state['prompt_ids'] == fixture['prompt_ids'] and
             bindings[by_response[rid]]['scheduler_id'] == state['scheduler_id'], 'full prompt/actual request mapping')
        need(streams[rid]['sha256'] == actual['sha256'] and streams[rid]['bytes'] == actual['bytes'], 'server/client original wire bytes differ')
        raw_ids, committed, eos = state['raw_ids'], state.get('committed_ids'), fixture['eos_token_ids']
        need(state['finished'] and raw_ids and not any(x in eos for x in raw_ids[:-1]), 'normal EOS/no post-EOS output')
        stop = raw_ids[-1] in eos
        need(stop or len(raw_ids) == 512, 'early non-EOS output')
        need(state['finish_reason'] == actual['finish_reason'] == ('stop' if stop else 'length') and
             committed == (raw_ids[:-1] if stop else raw_ids), 'normal EOS/cap/committed IDs')
        usage = actual['usage']
        need(all(integer(usage.get(k)) for k in ('prompt_tokens','completion_tokens','total_tokens')) and
             integer(usage['prompt_tokens_details']['cached_tokens']) and
             usage['prompt_tokens'] == len(fixture['prompt_ids']) and usage['completion_tokens'] == len(committed) and
             usage['total_tokens'] == len(fixture['prompt_ids']) + len(committed) and
             usage['prompt_tokens_details']['cached_tokens'] == 0, 'actual wire count/cache semantics')
        need(actual['text'] == decode_text(committed), 'wire text differs from actual committed-ID decode')
        completed.append(dict(client_request_id=job['request_id'], fixture_id=job['fixture_id'], response_id=rid,
            engine_request_id=by_response[rid], prompt_ids=state['prompt_ids'], raw_ids=raw_ids,
            committed_ids=committed, finish_reason=actual['finish_reason']))
    need(C == 1 or max(histogram, default=0) >= 2, 'C8 did not exercise actual multirow response batch')
    return dict(schema='E023-omlx-D512-observed-join-v1', status='PASS', concurrency=C,
        protocol_route_count_qualified=True, full_state_numerical_qualified=False, performance_qualified=False,
        source_model_owner_qualified=False, observer_sha256=digest.hexdigest(), observer_lines=lines,
        requests=completed, batch_histogram={str(k): v for k, v in histogram.items()},
        raw_output_count=sum(len(x['raw_ids']) for x in completed),
        committed_output_count=sum(len(x['committed_ids']) for x in completed),
        observed_terminal_eos=sum(x['finish_reason'] == 'stop' for x in completed),
        count_rule='identity: official completion_tokens excludes terminal EOS',
        scope='Only this target-only captured workload; no full-state, cross-framework quality or speed claim.')
