"""One read per selected closed group; stdlib timestamp diagnostic only.

No backend/service imports, test execution, recursive scanning or payload scans.
The old oMLX r1 point happens to equal its three-process throughput median.
Current native r1 OFF/ON remain unaudited partial results of an active cycle.
"""
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path('/Users/ryanli/code/minivllm')
OUT = Path('/tmp/combined-v2-r1-c1-timestamps.json')
NOTE = Path('/tmp/combined-v2-r1-c1-timestamps.md')
AUDIT = ROOT/'docs/experiments/m5-ultra-256gb/E023-omlx-d512-formal-v2-audit-supplement/independent-formal.json'
SUMMARY = AUDIT.with_name('summary.json')
TRIALS = {
    'omlx_r1_c1': ROOT/'.artifacts/m5-ultra-256gb/omlx-d512-formal-v2-owner/phases/r1-c1/trial',
    'native_r1_c1_off': ROOT/'.artifacts/m5-ultra-256gb/gdn-graph-combined-formal-v2-owner/phases/r1-c1-off/trial',
    'native_r1_c1_on': ROOT/'.artifacts/m5-ultra-256gb/gdn-graph-combined-formal-v2-owner/phases/r1-c1-on/trial',
}


def read_once(path):
    before = path.stat(); raw = path.read_bytes(); after = path.stat()
    assert (before.st_size,before.st_mtime_ns) == (after.st_size,after.st_mtime_ns), str(path)+' changed'
    return json.loads(raw),dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),mtime_ns=after.st_mtime_ns)


def main():
    assert not OUT.exists() and not NOTE.exists(), 'preserve previous attempt'
    result = dict(status='INCOMPLETE',scope='Closed-group HTTP timestamp decomposition, not independent cycle qualification or GPU profiling',
        current_cycle_independently_audited=False,groups={},checks={},
        formulas=dict(ttft='first_output_perf_counter_s - start_perf_counter_s',
            post_first='done_perf_counter_s - first_output_perf_counter_s',e2e='done_perf_counter_s - start_perf_counter_s'),
        limitations=['first_output is first nonempty answer OR reasoning text received by the client, not necessarily the first model token',
            'Only means are additively decomposed. Each median is computed independently; median differences are not a decomposition.',
            'POST timer starts before request construction/send. done-first includes generation, client/server scheduling, delivery and terminal overhead.',
            'One old oMLX process and current r1 OFF/ON; not three-repeat estimates, contemporary inter-engine pairing, or a causal GPU profile.',
            'Current groups are complete locally but the active cycle lacks final independent resource/restoration/performance approval.'])
    try:
        audit,audit_ref = read_once(AUDIT); summary,summary_ref = read_once(SUMMARY)
        point = next(p for p in summary['points'] if p['phase'] == 'r1-c1')
        result['old_audit'] = dict(ref=audit_ref,summary_ref=summary_ref,status=audit['status'],
            performance_qualified=audit['performance_qualified'],original_formal_status=summary['original']['original_formal_status'],
            supplement_formal_status=summary['formal_status'],post_run_parser_correction=summary['post_run_parser_correction'],
            old_point_tok_s=point['aggregate_output_tok_s'],old_three_process_metric_medians=summary['process_metric_medians'])
        specs, orders = {}, {}
        for label,folder in TRIALS.items():
            group,ref = read_once(folder/'formal-group.json'); spec,sref = read_once(folder/'spec.json')
            specs[label] = spec; jobs = spec['jobs']['formal']; rows = group['requests']; records = []
            assert len(rows) == len(jobs) == 64 and group['concurrency'] == spec['concurrency'] == 1
            assert group['phase'] == 'formal' and group['group_error'] is None and not group['qualification_errors']
            assert group['cleanup']['complete'] is True and not group['cleanup']['errors'] and not group['cleanup']['pending_tasks']
            assert group['summary']['eligible_under_supplied_qualification'] is True
            fixtures = {f['fixture_id']:f for f in spec['fixtures']}
            for ordinal,(row,job) in enumerate(zip(rows,jobs)):
                assert all(row[k] == job[k] for k in ('request_id','fixture_id','path','body'))
                f = fixtures[row['fixture_id']]; body = row['body']
                assert f['normal_eos'] is True and f['max_output_tokens'] == body['max_tokens'] == 512
                assert body['temperature'] == 0 and body['stream'] is True
                assert body['prompt'] == (f['prompt_ids'] if isinstance(body['prompt'],list) else f['prompt_text'])
                assert row['status'] == 'done' and row['done'] is True and row['http_status'] == 200
                assert row['error'] is None and not row['validation_errors'] and not row['raw_body_truncated']
                assert row['qualified_output_tokens'] == row['usage']['completion_tokens'] == 512 and row['finish_reason'] == 'length'
                assert row['usage']['prompt_tokens'] == len(f['prompt_ids'])
                start,first,done = [row[k] for k in ('start_perf_counter_s','first_output_perf_counter_s','done_perf_counter_s')]
                assert all(type(x) in (int,float) and math.isfinite(x) for x in (start,first,done)) and start <= first <= done
                output_times = [e['client_perf_counter_s'] for kind in ('answer_events','reasoning_events') for e in row[kind] if e['text']]
                assert first == min(output_times)
                done_events = [e['client_perf_counter_s'] for e in row['sse_events'] if e['data'] == '[DONE]']
                assert done_events == [done]
                ttft,tail,e2e = first-start,done-first,done-start
                assert row['ttft_s'] == ttft and row['e2e_s'] == e2e and math.isclose(ttft+tail,e2e,rel_tol=0,abs_tol=1e-12)
                records.append(dict(ordinal=ordinal,request_id=row['request_id'],fixture_id=row['fixture_id'],
                    prompt_tokens=len(f['prompt_ids']),completion_tokens=512,start=start,first_output=first,done=done,
                    ttft_s=ttft,post_first_s=tail,e2e_s=e2e))
            assert all(records[i]['done'] <= records[i+1]['start'] for i in range(63))
            metrics = {name:dict(mean=statistics.mean(r[name] for r in records),median=statistics.median(r[name] for r in records))
                       for name in ('ttft_s','post_first_s','e2e_s')}
            window = max(r['done'] for r in records)-min(r['start'] for r in records)
            rate = 32768/window
            assert math.isclose(metrics['ttft_s']['mean']+metrics['post_first_s']['mean'],metrics['e2e_s']['mean'],rel_tol=0,abs_tol=1e-12)
            assert math.isclose(rate,group['summary']['aggregate_output_tok_s'],rel_tol=0,abs_tol=1e-12)
            if label == 'omlx_r1_c1':
                assert ref['sha256'] == point['formal_raw_group']['sha256'] and math.isclose(rate,50.78156750755799,rel_tol=0,abs_tol=1e-12)
            orders[label] = [r['fixture_id'] for r in records]
            result['groups'][label] = dict(group=ref,spec=sref,n=64,metrics=metrics,common_window_s=window,
                committed_tokens=32768,aggregate_output_tok_s=rate,requests=records,
                qualification_scope='previous independent audit supplement PASS' if label.startswith('omlx') else 'CURRENT PARTIAL RESULTS, NOT INDEPENDENTLY AUDITED')
        old = specs['omlx_r1_c1']; off = specs['native_r1_c1_off']; on = specs['native_r1_c1_on']
        assert old['fixtures'] == off['fixtures'] == on['fixtures'] and old['base_plan'] == off['base_plan'] == on['base_plan']
        assert old['model_directory'] == off['identity']['model_snapshot_path'] == on['identity']['model_snapshot_path']
        assert orders['omlx_r1_c1'] == orders['native_r1_c1_off'] == orders['native_r1_c1_on']
        assert off['environment'] == on['environment'] and off['transport'] == on['transport']
        assert off['identity']['configuration'] == on['identity']['configuration']
        for arm,spec in [('off',off),('on',on)]:
            identity = spec['identity']['combined_candidates']
            assert spec['arm'] == arm and identity['route_decode_graph_enabled'] is True and identity['gdn_prework_enabled'] is (arm == 'on')
        result['checks'] = dict(same_original_numeric_plan=old['base_plan'],same_model_snapshot=old['model_directory'],
            same_full_fixtures_and_order=True,normal_eos=True,actual_512_outputs_all_192_requests=True,
            native_same_environment_transport_configuration=True,native_route_graph='ON both arms',native_gdn_prework='OFF versus ON')
        result['mean_deltas_s'] = {}
        for label,reference in [('native_r1_c1_off','omlx_r1_c1'),('native_r1_c1_on','omlx_r1_c1'),('native_r1_c1_on','native_r1_c1_off')]:
            delta = {k:result['groups'][label]['metrics'][k]['mean']-result['groups'][reference]['metrics'][k]['mean']
                     for k in ('ttft_s','post_first_s','e2e_s')}
            assert math.isclose(delta['ttft_s']+delta['post_first_s'],delta['e2e_s'],rel_tol=0,abs_tol=1e-12)
            result['mean_deltas_s'][label+' minus '+reference] = delta
        result['status'] = 'COMPUTED_TIMESTAMP_DIAGNOSTIC_ONLY'
    except Exception as error:
        result['status'] = 'FAIL'; result['error'] = dict(type=type(error).__name__,message=str(error)); raise
    finally:
        with OUT.open('x') as stream: json.dump(result,stream,indent=2,allow_nan=False); stream.write('\n')
    lines = ['# C1 first-output / DONE timestamp diagnostic','',
        'Current native r1 OFF/ON are **unaudited partial results** of the active formal cycle. Previous oMLX r1 is bound to the later audit-supplement PASS; original FAIL remains historical evidence.',
        '', '| Process (n=64 each) | TTFT mean / median s | DONE−first mean / median s | E2E mean / median s | common-window tok/s |',
        '|---|---:|---:|---:|---:|']
    for name,g in result['groups'].items():
        cells = ['%.9f / %.9f'%(g['metrics'][k]['mean'],g['metrics'][k]['median']) for k in ('ttft_s','post_first_s','e2e_s')]
        lines.append('| '+name+' | '+' | '.join(cells)+' | %.9f |'%g['aggregate_output_tok_s'])
    lines += ['', 'Means add exactly (within float rounding); component medians are measured separately and must not be subtracted to claim an exact decomposition.',
        '', 'The same Qwen3.8-27B-nvfp4 snapshot, eight full original fixtures repeated eight times, temperature0 and normalEOS/cap512 were verified from specs and request bodies. All192 observed requests returned512 committed tokens. Native graph is ON in both arms; only GDN prework toggles.',
        '', 'First output means the first nonempty answer or reasoning text received. This HTTP interval is not GPU kernel time, pure prefill time or ITL. No new independent numerical/resource audit is claimed.', '', '## Sources']
    for name,g in result['groups'].items(): lines += ['', '- '+name+': `'+g['group']['path']+'`', '  SHA256 `'+g['group']['sha256']+'`; n=64.']
    lines += ['', '- Old qualifying supplement: `'+str(AUDIT)+'`', '- Reproducer: `/tmp/review-combined-v2-r1-c1-timestamps.py`', '- Full per-request timestamps/results: `'+str(OUT)+'`']
    with NOTE.open('x') as stream: stream.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=result['status'],groups={k:{'n':v['n'],'metrics':v['metrics'],'tok_s':v['aggregate_output_tok_s']} for k,v in result['groups'].items()},mean_deltas_s=result['mean_deltas_s']),indent=2))


if __name__ == '__main__': main()
