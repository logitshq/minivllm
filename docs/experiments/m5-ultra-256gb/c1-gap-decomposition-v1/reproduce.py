"""Read-only C1 HTTP mean-gap decomposition from six already-audited raw groups.

Stdlib only; no model/runtime imports, service actions, device work or raw copies.
The two immutable PASS audits fix the selected C1 groups. This is a decomposition
of 192-request means per engine, not the chart's process medians or p50/p95. The
post-first-output interval is not GPU time, and SSE emission granularity alone
cannot explain or causally attribute the gap.
"""
import argparse
import base64
import codecs
import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback

AUDITS = {
    'native-on': {
        'archive': 'docs/experiments/m5-ultra-256gb/E024-route-http-formal-v2-reparse-v1',
        'raw_archive': 'docs/experiments/m5-ultra-256gb/E024-route-http-formal-v2-reparse-v1',
        'owner': '.artifacts/m5-ultra-256gb/route-http-formal-v2-owner',
        'audit_sha256': '16c285f78e46797fea7b747925caae6b1934b7a975e0ef6583bc3fbf42989a37',
        'audit_schema': 'E024-route-http-formal-independent-audit-v2',
        'source': 'bench/m5_ultra_256gb/route_http_formal/audit_formal_v2.py',
        'source_sha256': 'b2bc55b87043d66d79154fd32dc7579cc7c6fdb457844a1b1d3df932ff41224e',
        'member_prefix': 'owner/',
    },
    'omlx': {
        'archive': 'docs/experiments/m5-ultra-256gb/E023-omlx-d512-formal-v2-audit-supplement',
        'raw_archive': 'docs/experiments/m5-ultra-256gb/E023-omlx-d512-formal-v2',
        'owner': '.artifacts/m5-ultra-256gb/omlx-d512-formal-v2-owner',
        'audit_sha256': 'b8582928c375524f8a0286360d606adbac834494a73910a6d85dcd9ee212c591',
        'audit_schema': 'E023-omlx-D512-formal-independent-audit-v2',
        'source': 'bench/m5_ultra_256gb/omlx_d512_formal_retry/audit_formal_v2.py',
        'source_sha256': None,  # The fixed audit above supplies its exact actual source SHA.
        'member_prefix': 'raw/owner/',
    },
}
CLIENT = 'bench/m5_ultra_256gb/pareto_realistic/client.py'
CLIENT_SHA = 'ef64a114f4f7d90ec5044d6bdcf058ca05f98dea445d6639db706a4fc75217dc'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def parse(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            need(key not in out, 'duplicate JSON field: ' + key)
            out[key] = value
        return out
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON: ' + value)))


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def read_source(path, sources, repo):
    raw = path.read_bytes()
    sources[str(path.relative_to(repo))] = dict(path=str(path), sha256=digest(raw), bytes=len(raw))
    return raw


def write_new(path, value, compact=False):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False,
                  **({'separators': (',', ':')} if compact else {'indent': 2}))
        stream.write('\n')


def events_from_raw(row):
    """Independently reconstruct SSE events and their final-chunk receive clocks."""
    decoder = codecs.getincrementaldecoder('utf-8')('strict')
    buffer, data, events = '', [], []
    done, size, previous = False, 0, None
    for chunk in row['raw_sse_chunks']:
        need(not done, 'raw chunk follows DONE')
        stamp = chunk['client_perf_counter_s']
        need(finite(stamp) and (previous is None or stamp >= previous), 'raw chunk clocks')
        previous = stamp
        raw = base64.b64decode(chunk['body_bytes_base64'], validate=True)
        size += len(raw)
        buffer += decoder.decode(raw)
        while '\n' in buffer and not done:
            line, buffer = buffer.split('\n', 1)
            line = line.removesuffix('\r')
            if line == '':
                if data:
                    payload, data = '\n'.join(data), []
                    events.append(dict(client_perf_counter_s=stamp, data=payload))
                    done = payload == '[DONE]'
            elif line.startswith('data:'):
                data.append(line[6:] if line.startswith('data: ') else line[5:])
            else:
                need(line.startswith(':'), 'unexpected SSE field in fixed completion protocol')
    need(done and not data and not buffer and not decoder.decode(b'', final=True), 'complete raw SSE framing')
    need(size == row['raw_body_bytes'] and events == row['sse_events'], 'raw SSE bytes/events mismatch')
    need(sum(e['data'] == '[DONE]' for e in events) == 1, 'exactly one DONE')
    return events


def request_record(row, engine, phase):
    need(row['status'] == 'done' and row['http_status'] == 200 and row['done'] is True
         and row['raw_body_truncated'] is False and row['error'] is None
         and row['validation_errors'] == [], 'successful complete retained request')
    need(row['qualified_output_tokens'] == 512 and row['finish_reason'] == 'length', 'actual fixed D512 cap')
    events = events_from_raw(row)
    answer, reasoning, first = [], [], None
    for event in events:
        if event['data'] == '[DONE]':
            continue
        value = parse(event['data'])
        for choice in value.get('choices', []):
            delta = choice.get('delta') or {}
            if delta.get('reasoning_content') is not None and delta.get('reasoning') is not None:
                need(delta['reasoning_content'] == delta['reasoning'], 'conflicting reasoning aliases')
            for kind, piece in (('answer', choice.get('text', delta.get('content'))),
                                ('reasoning', delta.get('reasoning_content', delta.get('reasoning')))):
                if piece is None or piece == '':
                    continue
                need(type(piece) is str, 'output fragment must be text')
                (answer if kind == 'answer' else reasoning).append(
                    dict(client_perf_counter_s=event['client_perf_counter_s'], text=piece))
                if first is None:
                    first = event['client_perf_counter_s']
    need(answer == row['answer_events'] and reasoning == row['reasoning_events'], 'reparsed output events')
    need(first == row['first_output_perf_counter_s'] == row['first_answer_perf_counter_s']
         and row['first_reasoning_perf_counter_s'] is None and not reasoning,
         'actual sample has text-channel first output and no separate reasoning events')
    start, end = row['start_perf_counter_s'], events[-1]['client_perf_counter_s']
    need(all(finite(t) for t in (start, first, end)) and start < first < end
         and end == row['done_perf_counter_s'], 'request clock order and actual DONE')
    e2e, pre, post = end-start, first-start, end-first
    need(e2e == row['e2e_s'] and pre == row['ttft_s'], 'saved timing differs from raw receive clocks')
    return dict(engine=engine, phase=phase, request_id=row['request_id'], fixture_id=row['fixture_id'],
        start_perf_counter_s=start, first_output_perf_counter_s=first, done_perf_counter_s=end,
        e2e_s=e2e, pre_first_output_s=pre, post_first_output_s=post,
        nonempty_text_sse_events=len(answer), nonempty_reasoning_sse_events=len(reasoning),
        raw_sse_chunks=len(row['raw_sse_chunks']), raw_body_bytes=row['raw_body_bytes'],
        committed_output_tokens=row['qualified_output_tokens'], finish_reason=row['finish_reason'])


def average(rows):
    need(rows, 'nonempty request population')
    keys = ('e2e_s', 'pre_first_output_s', 'post_first_output_s', 'nonempty_text_sse_events',
            'nonempty_reasoning_sse_events', 'raw_sse_chunks', 'raw_body_bytes')
    return dict(requests=len(rows), means={k: math.fsum(r[k] for r in rows)/len(rows) for k in keys})


def diagnose(repo, sources):
    client_raw = read_source(repo/CLIENT, sources, repo)
    need(digest(client_raw) == CLIENT_SHA, 'shared frozen client source changed')
    requests, bindings, per_process = [], {}, []
    for engine, config in AUDITS.items():
        archive, raw_archive, owner = (repo/config[k] for k in ('archive', 'raw_archive', 'owner'))
        audit_raw = read_source(archive/'independent-formal.json', sources, repo)
        need(digest(audit_raw) == config['audit_sha256'], 'fixed independent PASS audit changed')
        audit = parse(audit_raw)
        need(audit['schema'] == config['audit_schema'] and audit['status'] == 'PASS'
             and audit['performance_qualified'] is True, 'actual audited formal eligibility')
        auditor_raw = read_source(repo/config['source'], sources, repo)
        need(digest(auditor_raw) == audit['audit_source_sha256']
             and (config['source_sha256'] is None or digest(auditor_raw) == config['source_sha256']),
             'actual independent auditor source identity')
        raw_manifest_raw = read_source(raw_archive/'manifest.json', sources, repo)
        raw_manifest = parse(raw_manifest_raw)['files']
        archive_manifest = parse(read_source(archive/'manifest.json', sources, repo))['files']
        need(any(row['sha256'] == digest(audit_raw) and row['bytes'] == len(audit_raw)
                 for row in archive_manifest.values()), 'audit retained in verified archive manifest')
        for directory in (archive, raw_archive):
            verified = parse(read_source(directory/'archive-verification.json', sources, repo))
            need(verified['status'] == 'PASS', 'existing archive verification PASS')
        need(any(name.endswith('pareto_realistic/client.py') and row['sha256'] == CLIENT_SHA
                 for name, row in raw_manifest.items()), 'same archived client clock semantics')
        if engine == 'omlx':
            need(archive_manifest['original-archive/manifest.json']['sha256'] == digest(raw_manifest_raw),
                 'oMLX supplement binds original raw archive manifest')
        groups = []
        for repeat in (1, 2, 3):
            phase = f'r{repeat}-c1' + ('-on' if engine == 'native-on' else '')
            relative = f'phases/{phase}/trial/formal-group.json'
            raw = read_source(owner/relative, sources, repo)
            pin = dict(sha256=digest(raw), bytes=len(raw))
            stored = raw_manifest[config['member_prefix']+relative]
            need({k: stored[k] for k in pin} == pin, 'raw group bytes differ from retained archive')
            if engine == 'native-on':
                need(audit['raw_file_pins'][relative] == pin, 'native raw_file_pins mismatch')
            else:
                point = next(p for p in audit['points'] if p['phase'] == phase)
                need(point['formal_raw_group'] == dict(path=str(Path(audit['owner_directory'])/relative), sha256=pin['sha256']),
                     'oMLX audited raw-group reference mismatch')
            group = parse(raw)
            need(group['phase'] == 'formal' and group['concurrency'] == 1 and len(group['requests']) == 64,
                 'fixed three C1 formal groups of 64')
            rows = [request_record(row, engine, phase) for row in group['requests']]
            requests.extend(rows);per_process.append(dict(engine=engine,phase=phase,**average(rows)))
            groups.append(dict(phase=phase,path=str(owner/relative),archive_member=config['member_prefix']+relative,**pin))
        bindings[engine] = dict(audit_path=str(archive/'independent-formal.json'),audit_sha256=config['audit_sha256'],
            audit_source_sha256=digest(auditor_raw),raw_archive_manifest_sha256=digest(raw_manifest_raw),groups=groups)
    need(len(requests) == 384 and len({(r['engine'],r['request_id']) for r in requests}) == 384, '384 distinct actual requests')
    population = {name:average([r for r in requests if r['engine'] == name]) for name in AUDITS}
    need(all(row['requests'] == 192 for row in population.values()), '192 requests per engine')
    difference = {k: population['native-on']['means'][k]-population['omlx']['means'][k]
                  for k in ('e2e_s','pre_first_output_s','post_first_output_s')}
    need(difference['e2e_s'] > 0, 'positive observed native-minus-oMLX total gap required for share')
    difference['post_first_share_of_e2e_gap'] = difference['post_first_output_s']/difference['e2e_s']
    difference['pre_first_share_of_e2e_gap'] = difference['pre_first_output_s']/difference['e2e_s']
    return requests, dict(population=population,per_process=per_process,
        native_minus_omlx_mean_seconds=difference,bindings=bindings,
        first_output_definition='Client receipt of first nonempty choices.text/delta.content or reasoning_content/reasoning string. Whitespace counts; not first GPU-token timestamp.',
        actual_reasoning_scope='All384 first outputs are text-channel events; zero separate reasoning events does not establish absence of reasoning in generated text.',
        limitations=[
            'Request means over3x64 per engine, not the Pareto chart process-metric medians or pooled p50/p95.',
            'Post-first includes remaining model work, scheduling/launch/eval, detokenization/SSE encoding, transport/buffering/client receive until DONE; it is not pure GPU decode time.',
            'SSE event counts are emission granularity, not token counts, kernel counts or a causal attribution of the latency gap.',
            'Different closed cycles and arrival histories; this arithmetic decomposition is descriptive, not a controlled causal profiler.',
            'Scope is fixed D512 C1, all512 committed output tokens. No C8/C64/general-model claim.',
            'Original PASS audits/manifest bindings reused; no archive payload, full audit, model or runtime re-execution.'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv);repo=args.repo.resolve();out=args.output.resolve()
    need(repo.is_dir() and not out.exists(), 'existing repository and new exclusive output required')
    out.mkdir(parents=True)
    summary = dict(schema='C1-http-mean-gap-diagnostic-v1',status='INCOMPLETE',
        started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        argv=[sys.executable,*sys.argv],repo=str(repo),output=str(out),
        diagnostic_source=dict(path=str(Path(__file__).resolve()),sha256=digest(Path(__file__).read_bytes())),
        model_actions=False,service_actions=False,device_actions=False,sources={})
    try:
        requests,result=diagnose(repo,summary['sources'])
        request_file=out/'requests.json'
        write_new(request_file,dict(schema='C1-http-request-time-decomposition-v1',requests=requests),compact=True)
        summary.update(result,status='PASS',requests_artifact=dict(path=str(request_file),sha256=digest(request_file.read_bytes()),bytes=request_file.stat().st_size))
    except Exception as error:
        summary.update(status='FAIL',error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
    finally:
        summary['ended_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_new(out/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ('status','diagnostic_source','output')}))
    return int(summary['status']!='PASS')


if __name__=='__main__':raise SystemExit(main())
