"""Finish the retained one-read extraction; never reopen original request groups.

The first script preserved all192 validated request records, then rejected two
different paths for the same registered plan. This script checks those small
plan bytes explicitly and recomputes all statistics from retained timestamps.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics

FIRST = Path('/tmp/combined-v2-r1-c1-timestamps.json')
OUT = Path('/tmp/combined-v2-r1-c1-timestamps-v2.json')
NOTE = Path('/tmp/combined-v2-r1-c1-timestamps-v2.md')
EXPECTED_PLAN = '21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb'


def small_read(path):
    path = Path(path); raw = path.read_bytes()
    return json.loads(raw), dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))


def main():
    assert not OUT.exists() and not NOTE.exists(), 'preserve first output'
    prior,prior_ref = small_read(FIRST)
    assert prior['status'] == 'FAIL' and prior['error']['type'] == 'AssertionError'
    result = copy.deepcopy(prior); result.pop('error')
    result['first_calculation_attempt'] = dict(result=prior_ref,
        script='/tmp/review-combined-v2-r1-c1-timestamps.py',
        reason='The same registered numeric plan was copied at two distinct paths; a path-equality assertion rejected after all three raw groups had been validated/extracted.')
    result['raw_group_rereads_in_this_step'] = 0
    specs = {}; plans = {}; plan_refs = {}
    for name,group in result['groups'].items():
        spec,spec_ref = small_read(group['spec']['path'])
        assert spec_ref['sha256'] == group['spec']['sha256'], 'spec changed after extraction'
        specs[name] = spec
        path = spec['base_plan']['path']
        if path not in plans: plans[path],plan_refs[path] = small_read(path)
        assert spec['base_plan']['sha256'] == plan_refs[path]['sha256'] == EXPECTED_PLAN
        records = group['requests']; assert group['n'] == len(records) == 64
        for row in records:
            assert row['completion_tokens'] == 512
            start,first,done = row['start'],row['first_output'],row['done']
            assert all(type(x) in (int,float) and math.isfinite(x) for x in (start,first,done)) and start <= first <= done
            row['ttft_s'],row['post_first_s'],row['e2e_s'] = first-start,done-first,done-start
        group['metrics'] = {k:dict(mean=statistics.mean(r[k] for r in records),median=statistics.median(r[k] for r in records))
                            for k in ('ttft_s','post_first_s','e2e_s')}
        m = group['metrics']; assert math.isclose(m['ttft_s']['mean']+m['post_first_s']['mean'],m['e2e_s']['mean'],rel_tol=0,abs_tol=1e-12)
        group['common_window_s'] = max(r['done'] for r in records)-min(r['start'] for r in records)
        group['aggregate_output_tok_s'] = 32768/group['common_window_s']
    assert len(plans) == 2 and list(plans.values())[0] == list(plans.values())[1]
    old,off,on = [specs[k] for k in ('omlx_r1_c1','native_r1_c1_off','native_r1_c1_on')]
    assert old['fixtures'] == off['fixtures'] == on['fixtures']
    assert old['model_directory'] == off['identity']['model_snapshot_path'] == on['identity']['model_snapshot_path']
    groups = result['groups']; orders = [[r['fixture_id'] for r in g['requests']] for g in groups.values()]
    assert orders[0] == orders[1] == orders[2]
    assert off['environment'] == on['environment'] and off['transport'] == on['transport']
    assert off['identity']['configuration'] == on['identity']['configuration']
    for arm,spec in [('off',off),('on',on)]:
        identity = spec['identity']['combined_candidates']
        assert spec['arm'] == arm and identity['route_decode_graph_enabled'] is True and identity['gdn_prework_enabled'] is (arm == 'on')
    result['checks'] = dict(same_numeric_plan_sha256=EXPECTED_PLAN,numeric_plan_copies=list(plan_refs.values()),
        same_model_snapshot=old['model_directory'],same_full_fixtures_and_order=True,normal_eos=True,
        actual_512_outputs_all_192_requests=True,native_same_environment_transport_configuration=True,
        native_route_graph='ON both arms',native_gdn_prework='OFF versus ON')
    result['mean_deltas_s'] = {}
    for label,reference in [('native_r1_c1_off','omlx_r1_c1'),('native_r1_c1_on','omlx_r1_c1'),('native_r1_c1_on','native_r1_c1_off')]:
        delta = {k:groups[label]['metrics'][k]['mean']-groups[reference]['metrics'][k]['mean'] for k in ('ttft_s','post_first_s','e2e_s')}
        assert math.isclose(delta['ttft_s']+delta['post_first_s'],delta['e2e_s'],rel_tol=0,abs_tol=1e-12)
        result['mean_deltas_s'][label+' minus '+reference] = delta
    result['status'] = 'COMPUTED_TIMESTAMP_DIAGNOSTIC_ONLY'
    result['calculation_script'] = str(Path(__file__))
    with OUT.open('x') as stream: json.dump(result,stream,indent=2,allow_nan=False); stream.write('\n')
    lines = ['# C1 HTTP timestamp decomposition','',
        'Current native r1 OFF/ON are **unaudited partial results**, each one completed process of the active formal cycle. Historical oMLX r1 matches the requested50.781567507558 tok/s point and its three-process throughput median; it is not a pooled three-process timestamp distribution.',
        '', '| Process (n=64 each) | TTFT mean / median s | DONE−first mean / median s | E2E mean / median s | common-window tok/s |',
        '|---|---:|---:|---:|---:|']
    for name,g in groups.items():
        cells = ['%.9f / %.9f'%(g['metrics'][k]['mean'],g['metrics'][k]['median']) for k in ('ttft_s','post_first_s','e2e_s')]
        lines.append('| '+name+' | '+' | '.join(cells)+' | %.9f |'%g['aggregate_output_tok_s'])
    lines += ['', 'Means add within floating-point rounding. Each component median is computed directly from64 request durations; no subtraction of medians is used as an exact decomposition.',
        '', 'Native ON minus OFF mean deltas: TTFT %.9f s; DONE−first %.9f s; E2E %.9f s.'%tuple(result['mean_deltas_s']['native_r1_c1_on minus native_r1_c1_off'][k] for k in ('ttft_s','post_first_s','e2e_s')),
        'Native ON minus historical oMLX: TTFT %.9f s; DONE−first %.9f s; E2E %.9f s.'%tuple(result['mean_deltas_s']['native_r1_c1_on minus omlx_r1_c1'][k] for k in ('ttft_s','post_first_s','e2e_s')),
        '', 'Same Qwen3.8-27B-nvfp4 snapshot and identical original eight fixtures, in identical order repeated eight times, normal EOS/cap512 and temperature0. All192 observed requests returned512 committed tokens. The two numeric-plan paths are copies with identical actual SHA21ba52e4…547fb. Native graph is ON in both arms; only GDN prework toggles.',
        '', 'First output means first nonempty answer or reasoning text received. The start timer precedes request construction/send. DONE−first includes model generation, scheduling, server/client transport and terminal overhead; it is not exclusive GPU time or pure TPOT. Historical oMLX was measured earlier, not interleaved with these native processes.',
        '', 'oMLX qualification lineage: original formal/resource audit FAIL retained, later versioned parser supplement independently PASS. The current native cycle has no completed independent final resource/restoration/performance audit yet.',
        '', '## Exact input groups']
    for name,g in groups.items(): lines += ['', '- '+name+': `'+g['group']['path']+'`', '  SHA256 `'+g['group']['sha256']+'`; n=64.']
    lines += ['', '## Calculation and retained attempt',
        '', '- First raw extractor: `/tmp/review-combined-v2-r1-c1-timestamps.py`.',
        '- Its retained first result `/tmp/combined-v2-r1-c1-timestamps.json` contains192 validated timestamp rows; final path-equality assertion failed on legitimate same-SHA plan copies.',
        '- Finish script: `'+str(Path(__file__))+'`; reads only the retained extract and small specs/plans, never the raw groups again.',
        '- Recomputed result and per-request rows: `'+str(OUT)+'`.',
        '- Historical qualifying receipt: `'+prior['old_audit']['ref']['path']+'`.']
    with NOTE.open('x') as stream: stream.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=result['status'],metrics={k:v['metrics'] for k,v in groups.items()},mean_deltas_s=result['mean_deltas_s']),indent=2))


if __name__ == '__main__': main()
