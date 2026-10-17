"""Offline Lithos D512 raw NDJSON/Generation join; stdlib, no model imports.

join(group, observer_lines, jobs, fixtures_by_id, decode_text,
     expected_probe_sha256, *, expected_official_args, expected_pack_dir)
requires completed, observer-enabled 8/C1 or 16/C8 requests. decode_text uses the
externally pinned tokenizer, skip_special_tokens=True and cleanup=False, exactly
as official visible_text. No response re-tokenization or cross-engine oracle.

generation_log_window(raw, expected_count, require_setup_zero=...) validates an
owner-delimited complete official log window. It proves all entries in that
window satisfy the gates; official logs have no request IDs. The owner must bind
file identity/offsets at quiet phase boundaries, reject rotation/truncation and
unrelated traffic. This function alone cannot establish those boundaries.
"""
import ast
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


def same(a, b, label):
    need(json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False), label)


def ids(value):
    return isinstance(value, list) and all(type(x) is int and 0 <= x < 248320 for x in value)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def wire(row):
    """Parse original retained chunks through unique done and EOF, not cached row fields."""
    chunks = row.get('raw_ndjson_chunks'); need(isinstance(chunks, list) and 0 < len(chunks) <= 65536, 'raw chunks')
    buffer, text, terminal_line, terminal, done_time = b'', [], None, None, None
    size, previous = 0, None
    for chunk in chunks:
        stamp = chunk['client_perf_counter_s']
        need(finite(stamp) and (previous is None or stamp >= previous), 'receive clock order')
        previous = stamp
        raw = base64.b64decode(chunk['body_bytes_base64'], validate=True)
        size += len(raw); need(size <= 8 << 20 and (terminal is None or not raw), 'raw bound or data after DONE')
        buffer += raw
        while b'\n' in buffer:
            need(terminal is None, 'extra line after DONE')
            line, buffer = buffer.split(b'\n', 1)
            data = parse(line.decode('utf-8', errors='strict'))
            need(isinstance(data, dict) and 'error' not in data and type(data.get('done')) is bool, 'invalid NDJSON')
            need(data.get('model') == row['body']['model'] and isinstance(data.get('created_at'), str), 'wire model/time')
            message = data.get('message')
            need(isinstance(message, dict) and set(message) == {'role', 'content'} and
                 message['role'] == 'assistant' and type(message['content']) is str, 'unqualified tool/message')
            if data['done']:
                need(message['content'] == '', 'official final text must be empty')
                terminal, terminal_line, done_time = data, line.decode('utf-8')+'\n', stamp
            else:
                text.append(message['content'])
        need(terminal is None or not buffer, 'tail bytes after DONE')
    need(terminal is not None and not buffer, 'missing/truncated DONE')
    need(row.get('stream_eof') is True and row.get('drain_error') is None and row.get('error') is None and
         row.get('status') == 'done' and row.get('http_status') == 200, 'unsuccessful transport/EOF/drain')
    need(row.get('raw_body_truncated') is False and type(row.get('raw_body_bytes')) is int and
         row['raw_body_bytes'] == size, 'raw byte receipt')
    need(finite(row.get('start_perf_counter_s')) and row['start_perf_counter_s'] <= chunks[0]['client_perf_counter_s']
         and row.get('done_perf_counter_s') == done_time and finite(row.get('eof_perf_counter_s')) and
         row['eof_perf_counter_s'] >= done_time, 'POST/DONE/EOF clocks')
    return dict(terminal=terminal, terminal_line=terminal_line, text=''.join(text), bytes=size)


def check_route(route, pack):
    need(route['backend'] == 'm5_ultra_80c' and route['device']['name'] == 'Apple M5 Ultra'
         and route['device']['gpu_cores'] == 80, 'actual M5 Ultra route')
    need(route['pack_dir'] == pack and type(route['capacity']) is int and route['capacity'] == 2048,
         'actual existing pack/capacity')
    need(route['has_drafter'] is False and route['sampling'] is None and route['seed'] == 0
         and type(route['seed']) is int and route['decoder_kernel_config'] is None,
         'actual target-only greedy route without a substituted recipe')
    need(route['accelerator'] == 'on' and route['prefill_chunk_size'] == 128 and
         type(route['prefill_chunk_size']) is int and route['effective_prefill_exact'] is False,
         'original 128-row exact behavior; Session exact flag only applies above 128')
    need(route['prefix_cache'] is True and route['eos'] == [248046, 248044] and
         route['config_eos'] == route['eos'] and ids(route['eos']), 'actual cache/EOS identity')
    programs = route['programs']
    need(programs and all(p['backend'] == 'm5_ultra_80c' and type(p['capacity']) is int and p['capacity'] == 2048
             for p in programs), 'executed official prefill/decode programs')


def join(group, observer_lines, jobs, fixtures_by_id, decode_text, expected_probe_sha256, *, expected_official_args, expected_pack_dir):
    from .serve_probe import PINS, ROOT, SCHEMA
    C = group['concurrency']; count = 8 if C == 1 else 16
    need(type(C) is int and C in (1, 8) and len(jobs) == count and len(fixtures_by_id) == 8, 'fixed D8 C1/C8')
    need(group['schema'] == 'E023-lithos-ndjson-group-v1' and group['phase'] == 'qualification' and
         group['qualification'] is None and group['group_error'] is None and
         group['summary']['formal_point_eligible'] is False, 'unqualified qualification group only')
    cleanup = group['cleanup']
    need(cleanup['complete'] is True and cleanup['deadline_exceeded'] is False and cleanup['errors'] == [] and
         cleanup['pending_tasks'] == [] and cleanup['cancelled_again'] is False and
         cleanup['client_close_started'] is True, 'all worker/response/client cleanup required')
    need(len(group['requests']) == count and len({j['request_id'] for j in jobs}) == count, 'complete unique jobs')
    started, returned, bodies, terminals = {}, {}, {}, {}
    opened = closed = False; active = None; digest = hashlib.sha256(); line_count = 0
    for raw in observer_lines:
        raw = raw.encode() if isinstance(raw, str) else raw
        line_count += 1; digest.update(raw)
        need(raw.endswith(b'\n') and not closed and line_count <= 4*count+2, 'observer truncation/tail/bound')
        event = parse(raw); kind = event['event']
        if not opened:
            need(kind == 'open' and event['schema'] == SCHEMA and event['concurrency'] == C and
                 event['source_root'] == str(ROOT) and event['sources'] == PINS and
                 event['probe_sha256'] == expected_probe_sha256 and event['qualification_only'] is True,
                 'pinned observer open')
            same(event['official_args'], expected_official_args, 'exact registered official CLI'); opened = True
        elif kind == 'generation_start':
            index = event['index']
            need(type(index) is int and index == len(started)+1 and active is None and event['streaming'] is True,
                 'single-generation actual route/order')
            need(ids(event['prompt_ids']) and event['prompt_ids'] and type(event['cap']) is int and event['cap'] == 512,
                 'actual full prompt/cap')
            started[index] = event; active = index
        elif kind == 'generation_return':
            index = event['index']; need(type(index) is int and active == index and index not in returned, 'generation return ownership')
            check_route(event['route'], expected_pack_dir)
            gen = event['generation']
            need(ids(gen['tokens']) and 1 <= len(gen['tokens']) <= 512 and
                 type(gen['cached_prompt_tokens']) is int and gen['cached_prompt_tokens'] == 0 and
                 finite(gen['setup_ms']), 'actual returned IDs/zero cache/setup evidence')
            need(all(gen.get(k) is None for k in ('accepted','committed','verify_len','confidences')),
                 'target-only Generation has no speculative acceptance arrays')
            returned[index] = event; active = None
        elif kind == 'body':
            index, wid = event['index'], event['wire_id']
            need(type(index) is int and index in returned and index not in bodies and
                 isinstance(wid, str) and re.fullmatch('[0-9a-f]{32}', wid) and
                 wid not in {b['wire_id'] for b in bodies.values()}, 'unique worker/wire binding')
            gen, metrics = returned[index]['generation'], event['metrics']
            same(metrics['cached_prompt_tokens'], gen['cached_prompt_tokens'], 'official cache metric binding')
            same(metrics['setup_ms'], gen['setup_ms'], 'official setup metric binding')
            need(type(metrics['verify_tokens']) is int and metrics['verify_tokens'] == 1, 'official non-spec route')
            bodies[index] = event
        elif kind == 'wire':
            wid, line = event['wire_id'], event['line']
            need(wid in {b['wire_id'] for b in bodies.values()} and wid not in terminals and
                 isinstance(line, str) and line.endswith('\n') and line.count('\n') == 1,
                 'unique official final NDJSON')
            body = next(b['body'] for b in bodies.values() if b['wire_id'] == wid)
            same(parse(line), {**body, 'message': {'role':'assistant','content':''}}, 'official final body/string binding')
            need(body.get('done') is True and set(body['message']) == {'role','content'}, 'no tools/body mutation')
            terminals[wid] = line
        elif kind == 'sealed':
            need(event['status'] == 'COMPLETE' and event['source_pins_rechecked'] is True and active is None,
                 'observer did not finish/recheck sources')
            same({k:event[k] for k in ('started','returned','bodies','terminals','active')},
                 dict(started=count, returned=count, bodies=count, terminals=count, active=0), 'all observer calls closed')
            closed = True
        else:
            raise ValueError('observer error or unsupported event: '+str(kind))
    need(closed and len(started) == len(returned) == len(bodies) == len(terminals) == count,
         'all real generations and wires required')
    need(len(set(terminals.values())) == count, 'ambiguous identical terminal lines; never infer order')
    by_line = {line:wid for wid,line in terminals.items()}; matched = set(); rows = []; fixture_counts = Counter(); intervals = []
    for job, row in zip(jobs, group['requests']):
        need(set(job) == {'request_id','fixture_id','path','body'}, 'fixed job shape')
        same({k:row.get(k) for k in job}, job, 'job/row body binding')
        f = fixtures_by_id[job['fixture_id']]; body = job['body']
        need(f['fixture_id'] == job['fixture_id'] and f['normal_eos'] is True and f['max_output_tokens'] == 512
             and f['eos_token_ids'] == [248046,248044], 'fixed normal EOS fixture')
        same(body, dict(model=body['model'], messages=[dict(role='user',content=f['prompt_text'])], stream=True,
                       options=dict(num_predict=512,temperature=0,top_p=1,top_k=0,seed=0)), 'fixed original request body')
        need(job['path'] == '/api/chat', 'official FIFO NDJSON endpoint')
        payload = base64.b64decode(row['request_body_bytes_base64'], validate=True)
        need(hashlib.sha256(payload).hexdigest() == row['request_body_bytes_sha256'], 'sent request raw SHA')
        same(parse(payload), body, 'sent original body')
        w = wire(row); wid = by_line.get(w['terminal_line'])
        start, done = row['start_perf_counter_s'], row['done_perf_counter_s']
        need(finite(start) and finite(done) and start < done, 'positive finite POST-to-DONE interval')
        intervals.extend(((start, 1), (done, -1)))
        need(wid is not None and wid not in matched, 'raw final-line bijection failed'); matched.add(wid)
        b = next(b for b in bodies.values() if b['wire_id'] == wid); index = b['index']
        same(started[index]['prompt_ids'], f['prompt_ids'], 'actual full prompt differs from frozen fixture')
        raw_ids = returned[index]['generation']['tokens']; eos = [i for i,t in enumerate(raw_ids) if t in f['eos_token_ids']]
        need(not eos or eos == [len(raw_ids)-1], 'EOS is not unique terminal; post-EOS tokens')
        finish = 'stop' if eos else 'length'; committed = raw_ids[:-1] if eos else raw_ids
        need(bool(eos) or len(raw_ids) == 512, 'short output without normal EOS')
        final = w['terminal']
        need(final['done_reason'] == finish and b['body']['done_reason'] == finish and
             type(final['eval_count']) is int and final['eval_count'] == len(raw_ids) and
             type(final['prompt_eval_count']) is int and final['prompt_eval_count'] == len(f['prompt_ids']),
             'official final finish/raw usage counts')
        need(set(final) == {'model','created_at','message','done','done_reason','total_duration','load_duration',
                           'prompt_eval_count','prompt_eval_duration','eval_count','eval_duration'}, 'official final keys')
        need(all(type(v) is int and v >= 0 for k,v in final.items() if k.endswith('_duration')), 'duration units')
        metrics = b['metrics']
        need(all(finite(metrics[k]) for k in ('setup_ms','load_ms','prefill_wall_ms','decode_wall_ms')), 'finite official timings')
        same({k:final[k] for k in ('load_duration','prompt_eval_duration','eval_duration')},
             dict(load_duration=int((metrics['load_ms']+metrics['setup_ms'])*1e6),
                  prompt_eval_duration=int(metrics['prefill_wall_ms']*1e6),
                  eval_duration=int(metrics['decode_wall_ms']*1e6)), 'official metric-to-wire units')
        same(w['text'], b['body']['message']['content'], 'wire delta text/full body')
        same(w['text'], decode_text(committed), 'original IDs decoded text (no re-tokenization)')
        fixture_counts[job['fixture_id']] += 1
        rows.append(dict(request_id=job['request_id'], fixture_id=job['fixture_id'], wire_id=wid, generation_index=index,
            prompt_ids=f['prompt_ids'], raw_ids=raw_ids, committed_ids=committed, finish_reason=finish,
            cached_prompt_tokens=0, setup_ms=returned[index]['generation']['setup_ms'],
            terminal_line_sha256=hashlib.sha256(w['terminal_line'].encode()).hexdigest(), raw_body_bytes=w['bytes']))
    need(fixture_counts == Counter({fid:count//8 for fid in fixtures_by_id}), 'each fixed fixture coverage')
    # Half-open [POST, DONE) intervals: at a tied clock, retire before admitting.
    active = peak = 0
    for _, delta in sorted(intervals):
        active += delta
        need(active >= 0, 'HTTP completion without an outstanding request')
        peak = max(peak, active)
    need(active == 0 and peak == C, 'actual HTTP outstanding peak differs from declared concurrency')
    return dict(schema='E023-lithos-D512-join-v1', status='PASS', qualified=True, concurrency=C, requests=rows,
        max_http_outstanding=peak, http_concurrency_scope='Retained client POST-to-DONE intervals; not GPU batch size',
        observed_eos=sum(r['finish_reason']=='stop' for r in rows), observer_sha256=digest.hexdigest(),
        raw_output_count=sum(len(r['raw_ids']) for r in rows),
        committed_output_count=sum(len(r['committed_ids']) for r in rows),
        probe_sha256=expected_probe_sha256, source_pins=PINS, official_args=expected_official_args,
        scope='observed request/route/cache/raw-ID/count/text qualification only; official FIFO, not GPU batch C',
        usage_count_rule='lithos_eval_count_includes_one_terminal_eos_on_stop',
        protocol='lithos-ollama-ndjson-v1', cache_contract='uncached-fixed-D-v1', normal_eos=True,
        terminal_token_ids=[248046,248044], no_post_terminal_tokens=True,
        expected_prompt_tokens_by_fixture={fid:len(f['prompt_ids']) for fid,f in fixtures_by_id.items()},
        fixture_contracts={j['fixture_id']:dict(path=j['path'],body_sha256=hashlib.sha256(json.dumps(j['body'],
            sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()) for j in jobs},
        raw_ids_scope='actual returned Generation.tokens, not untruncated internal token-ring contents',
        full_state_qualified=False, performance_qualified=False, cross_framework_bit_equality=False)


def generation_log_window(raw, expected_count, *, require_setup_zero, expected_logger='__main__'):
    """No process/filesystem access; caller supplies a complete immutable phase byte window."""
    need(isinstance(raw, bytes) and raw.endswith(b'\n') and type(expected_count) is int and expected_count > 0
         and type(require_setup_zero) is bool, 'complete explicit log window/count')
    need(expected_logger in ('__main__','monolith.serve'), 'explicit original CLI or imported probe logger')
    prefix = f'INFO:{expected_logger}:Generation: '; rows = []
    for line in raw.decode('utf-8', errors='strict').splitlines():
        if 'Generation:' not in line:
            continue
        need(line.startswith(prefix), 'unknown/malformed Generation log prefix')
        expression = ast.parse(line[len(prefix):], mode='eval').body
        need(isinstance(expression, ast.Dict) and all(isinstance(k, ast.Constant) and type(k.value) is str
             for k in expression.keys) and len({k.value for k in expression.keys}) == len(expression.keys), 'unique log dict keys')
        metrics = ast.literal_eval(expression)
        need(type(metrics.get('cached_prompt_tokens')) is int and metrics['cached_prompt_tokens'] == 0,
             'actual official log cache miss required')
        need(finite(metrics.get('setup_ms')) and (not require_setup_zero or metrics['setup_ms'] == 0),
             'actual formal setup must be zero')
        need(type(metrics.get('verify_tokens')) is int and metrics['verify_tokens'] == 1, 'official target-only log route')
        rows.append(metrics)
    need(len(rows) == expected_count, 'missing/extra official Generation logs')
    return dict(status='PASS', count=len(rows), log_window_sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw),
        all_cached_prompt_tokens_zero=True, all_setup_ms_zero=all(r['setup_ms']==0 for r in rows),
        require_setup_zero=require_setup_zero, per_wire_request_id_bound=False,
        scope='all Generation entries in externally bounded quiet window; owner must prove offsets/file identity/no unrelated requests')
