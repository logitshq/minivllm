"""Offline E045 raw-wire screening audit; no backend/service work, no Pareto claim.

Each C has its own E033 entry: combined graph+GDN at C1, combined graph-only
at C8 (historical 170.46064 tok/s, not E024's 170.619844). Inherited numerical
receipts bind the old configuration only. Exact returned IDs are a narrower
screen than full-vocabulary/state equality. Root audits resources/restoration.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
NATIVE = HERE.parent / 'pareto_realistic'
LABELS = ('c1-start50', 'c1-mb200', 'c1-mb16', 'c1-mb800', 'c1-end50',
          'c8-start50', 'c8-mb800', 'c8-mb16', 'c8-mb200', 'c8-end50')
BASE = {1: ('r1-c1-on.json', 'd0b993baf8374ae1e6e1219dfdc624009bbc426f356fe7c613a70c03b1a220b0'),
        8: ('r1-c8-off.json', '25e0f4b4d27c87ebf429adbb56cc63590e70cfa67be3b9623346720adbdce562')}
FIXED = {'http_join.py': 'a362c321c2329d7ab53ab6a1a5a96981bb95b44733739fdbb56dd3d38bef7b4e',
         'http_capture_v2.py': '536fe41d6c8e2bdffe83ba6f43fba87dd6265deb4c84f6204bc9c7220f8280e3'}


def need(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(1 << 20): h.update(chunk)
    return h.hexdigest()


def unique(items):
    result = {}
    for key, value in items:
        need(key not in result, 'duplicate JSON key: ' + key); result[key] = value
    return result


def read(path):
    def invalid(x): raise ValueError('nonfinite JSON: ' + x)
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique, parse_constant=invalid)


def same(a, b, message):
    need(json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False), message)


def ref(path): return dict(path=str(Path(path).resolve()), sha256=sha(path))
def finite(x): return type(x) in (int, float) and math.isfinite(x)


def load(name):
    path = NATIVE / name; need(sha(path) == FIXED[name], 'frozen raw parser changed')
    spec = importlib.util.spec_from_file_location('_e045_audit_' + path.stem, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def specification(spec, label):
    C = int(label[1]); mb = 50 if '50' in label else int(label.split('mb')[1])
    need(spec['schema'] == 'concurrency-config-screen-spec-v1' and spec['label'] == label
         and type(spec['mb_threshold']) is int and spec['mb_threshold'] == mb
         and spec['concurrency'] == C, 'trial label/configuration')
    base_ref = spec['base_spec']; path = Path(base_ref['path'])
    need(path.is_absolute() and path.name == BASE[C][0] and base_ref['sha256'] == BASE[C][1]
         and sha(path) == BASE[C][1], 'frozen actual E033 first-repeat specification')
    base = read(path); expected = copy.deepcopy(base)
    expected['environment']['MLX_MAX_MB_PER_BUFFER'] = str(mb)
    expected['deadlines']['resource_interval'] = 3
    expected['jobs']['formal'] = expected['jobs']['formal'][:8]
    metadata = {'schema', 'base_spec', 'label', 'mb_threshold', 'sources', 'numerical_qualified',
                'full_state_qualified', 'performance_qualified', 'qualification_scope', 'scope'}
    same({k:v for k,v in spec.items() if k not in metadata},
         {k:v for k,v in expected.items() if k not in metadata}, 'only registered changes to frozen E033 base')
    need(all(spec[k] is False for k in ('numerical_qualified', 'full_state_qualified', 'performance_qualified')),
         'old numeric receipt cannot qualify new environment')
    need({k:v for k,v in spec['environment'].items() if k.startswith('MLX_')} == {
         'MLX_MAX_OPS_PER_BUFFER':'50', 'MLX_MAX_MB_PER_BUFFER':str(mb),
         'MLX_BFS_MAX_WIDTH':'20', 'MLX_METAL_FAST_SYNCH':'0'}, 'exact one-axis MLX environment')
    for source, digest in spec['sources'].items(): need(sha(source) == digest, 'source changed: ' + source)
    need(str(HERE/'trial.py') in spec['sources'], 'actual trial source pinned')
    return C, mb


def group_check(group, spec, phase, parser, decode):
    """Independent eight-request validation and metric arithmetic from raw SSE."""
    jobs = spec['jobs'][phase]; fixtures = {f['fixture_id']: f for f in spec['fixtures']}
    need(len(jobs) == len(group['requests']) == len(fixtures) == 8, 'exact eight requests/fixtures')
    need(group['schema'] == 'E023-http-group-v1' and group['phase'] == ('warm' if phase == 'warm' else 'admission_probe')
         and group['concurrency'] == spec['concurrency'] and group['group_error'] is None
         and group['qualification_errors'] == [], 'complete group and concurrency')
    need(group['base_url'] == f"http://127.0.0.1:{spec['transport']['port']}"
         and group['request_timeout_s'] == spec['deadlines']['request']
         and group['group_timeout_s'] == spec['deadlines'][phase], 'endpoint/deadlines')
    expected_contract = dict(status='qualified', receipt_sha256=spec['qualification']['audit']['sha256'],
        concurrency=spec['concurrency'], normal_eos=True,
        fixture_contracts={j['fixture_id']:dict(path=j['path'],body_sha256=hashlib.sha256(
            json.dumps(j['body'],sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()) for j in jobs},
        request_fixture_map={j['request_id']:j['fixture_id'] for j in jobs},
        usage_semantics='committed_target_output_excluding_terminal_including_reasoning',
        usage_count_rule='identity',raw_ids_count_rule='native_committed_ids',
        terminal_token_ids=spec['identity']['eos_ids'],
        expected_prompt_tokens_by_fixture={f['fixture_id']:len(f['prompt_ids']) for f in spec['fixtures']})
    same(group['qualification'],expected_contract,'original wire/count contract only')
    same(group['connection_policy'],dict(max_connections=2*spec['concurrency'],max_keepalive_connections=0,
         retries=0,follow_redirects=False),'unchanged connection policy')
    close = group['cleanup']
    need(close['complete'] is True and close['deadline_exceeded'] is False and close['errors'] == []
         and close['pending_tasks'] == [] and close['cancelled_again'] is False
         and close['adapter'] == 'native-close-cancellation-v1'
         and close['budget_s'] == spec['deadlines']['close']
         and close['client_close_started'] is True
         and close['frozen_client_sha256']=='ef64a114f4f7d90ec5044d6bdcf058ca05f98dea445d6639db706a4fc75217dc', 'client close')
    starts=[]; ends=[]; latencies=[]; events=[]; outputs={}; uuids=[]; total=0
    eos = set(spec['identity']['eos_ids'])
    for job, row in zip(jobs, group['requests']):
        same({k:row.get(k) for k in job}, job, 'original request/path/body')
        need(row['status'] == 'done' and row['http_status'] == 200 and row['error'] is None
             and row['validation_errors'] == row['count_qualification_errors'] == [], 'successful wire request')
        rid, terminal, text = parser.raw_completion(row)
        fixture = fixtures[job['fixture_id']]; prompt = fixture['prompt_ids']
        same(terminal['benchmark_prompt_ids'], prompt, 'complete prompt IDs')
        raw = terminal['benchmark_output_ids']; committed = terminal['benchmark_committed_output_ids']
        need(parser.ids(raw) and 1 <= len(raw) <= fixture['max_output_tokens'] == 512
             and parser.ids(committed) and not any(i in eos for i in raw[:-1]), 'raw IDs/cap/normal EOS')
        stopped = raw[-1] in eos
        need(stopped or len(raw) == 512, 'early non-EOS termination')
        same(committed, raw[:-1] if stopped else raw, 'committed complete IDs')
        same(terminal['benchmark_terminal_eos_id'], raw[-1] if stopped else None, 'terminal EOS ID')
        need(terminal['choices'][0]['finish_reason'] == ('stop' if stopped else 'length'), 'finish reason')
        same(terminal['usage'], dict(prompt_tokens=len(prompt), completion_tokens=len(committed),
             total_tokens=len(prompt)+len(committed), prompt_tokens_details={'cached_tokens':0}), 'zero-cache usage')
        need(decode(committed) == text == row['answer'] and row['reasoning'] == ''
             and type(row['qualified_output_tokens']) is int
             and row['qualified_output_tokens'] == len(committed), 'decoded wire/count')
        start, end = row['start_perf_counter_s'], row['done_perf_counter_s']
        need(finite(start) and finite(end) and end > start and row['e2e_s'] == end-start
             and row['absolute_deadline_perf_counter_s'] == start+group['request_timeout_s']
             and end <= row['absolute_deadline_perf_counter_s'], 'request clocks')
        fid = job['fixture_id']; need(fid not in outputs and rid not in uuids, 'unique fixture/response IDs')
        outputs[fid] = raw; uuids.append(rid); total += len(committed)
        starts.append(start); ends.append(end); latencies.append(end-start)
        events.extend(((start,1), (end,-1)))
    active=peak=0
    for _, delta in sorted(events):
        active += delta; need(active >= 0, 'request lifetime'); peak = max(peak, active)
    need(active == 0 and peak == spec['concurrency'], 'actual HTTP concurrency')
    barrier, deadline = group['barrier_release_perf_counter_s'], group['group_deadline_perf_counter_s']
    need(finite(barrier) and finite(deadline) and barrier <= min(starts) and max(ends) <= deadline
         and deadline == barrier+group['group_timeout_s'], 'group clock bounds')
    need(finite(close['start_perf_counter_s']) and finite(close['end_perf_counter_s'])
         and max(ends) <= close['start_perf_counter_s'] <= close['end_perf_counter_s'], 'natural group drain')
    window = max(ends)-min(starts)
    point = dict(concurrency=spec['concurrency'], repeat=spec['repeat'], requests=8,
        committed_output_tokens=total, common_window_s=window, aggregate_output_tok_s=total/window,
        e2e_p50_s=statistics.median(latencies), e2e_p95_s=sorted(latencies)[math.ceil(.95*8)-1])
    summary = group['summary']
    need(summary['transport_complete'] is True and summary['eligible_under_supplied_qualification'] is True
         and summary['formal_point_eligible'] is False, 'wire/count eligibility only')
    same([summary['group_window_s'], summary['aggregate_output_tok_s'], summary['e2e_s']['p50'], summary['e2e_s']['p95']],
         [window, total/window, point['e2e_p50_s'], point['e2e_p95_s']], 'independent raw metrics')
    return dict(point=point, outputs=outputs, uuids=uuids, first_post=min(starts),
                close_end=close['end_perf_counter_s'], max_http_outstanding=peak)


def compare_tokens(reference, observed):
    need(set(reference) == set(observed), 'same fixture set for complete ID comparison')
    mismatches=[]
    for fid, expected in reference.items():
        actual = observed[fid]
        if expected != actual:
            i = next((i for i,(a,b) in enumerate(zip(expected,actual)) if a != b), min(len(expected),len(actual)))
            mismatches.append(dict(fixture_id=fid, expected_length=len(expected), actual_length=len(actual),
                first_index=i, expected_id=expected[i] if i<len(expected) else None,
                actual_id=actual[i] if i<len(actual) else None))
    return mismatches


def rank(points, tokens):
    result={}
    for C in (1,8):
        names = [x for x in LABELS if x.startswith(f'c{C}-')]; anchors = [points[names[0]],points[names[-1]]]
        drift = {k:max(a[k] for a in anchors)/min(a[k] for a in anchors)-1
                 for k in ('aggregate_output_tok_s','e2e_p50_s')}
        stable = all(x <= .02 for x in drift.values()) and tokens[names[0]] and tokens[names[-1]]
        candidates=[]
        for name in names[1:-1]:
            point=points[name]
            comparisons=[dict(throughput_ratio=point['aggregate_output_tok_s']/a['aggregate_output_tok_s'],
                              p50_speed_ratio=a['e2e_p50_s']/point['e2e_p50_s']) for a in anchors]
            eligible=stable and tokens[name] and all(r['throughput_ratio'] >= 1.03 and r['p50_speed_ratio'] >= 1.03 for r in comparisons)
            candidates.append(dict(label=name, eligible_for_confirmation=eligible, exact_returned_ids=tokens[name],
                comparisons_to_start_end= comparisons, point=point))
        eligible=sorted((r for r in candidates if r['eligible_for_confirmation']),
                        key=lambda r:r['point']['aggregate_output_tok_s'],reverse=True)
        result[str(C)]=dict(bookend_drift=drift, bookends_stable=stable, candidates=candidates,
                           ranked_for_confirmation=[x['label'] for x in eligible])
    return result


def audit(screen, execution):
    screen, execution = Path(screen).resolve(), Path(execution).resolve()
    result=dict(schema='E045-independent-screen-audit-v1', status='INCOMPLETE',
        diagnostic_only=True, full_state_qualified=False, numerical_qualified=False, performance_qualified=False,
        cycle_closed=False, resources_restoration_independently_audited_here=False,
        audit_source=ref(__file__), points={}, token_checks={}, errors=[])
    refs={}
    def receipt(path):
        refs[str(path)]=sha(path); return read(path)
    try:
        # Root writes this only after its complete finally/recovery path.
        terminal = execution/'final.json'
        if not terminal.exists(): return result
        outer = receipt(terminal)
        need(outer['schema']=='E045-config-driver-result-v1' and outer['status']=='PASS_CAPTURE_AND_RESTORE'
             and outer['errors']==[] and outer['lease_returncode']==0, 'outer cycle not PASS; preserve original failure')
        plan=receipt(execution/'execution-plan.json')
        same(receipt(screen/'plan.json'),plan,'actual runner/driver execution plan')
        registered=receipt(execution/'plan.json'); invocation=receipt(execution/'invocation.json')
        same({k:v for k,v in plan.items() if k!='physical_baseline'},registered,'registered plan plus observed baseline only')
        need(sha(execution/'plan.json')==invocation['plan_sha256'],'driver bound registered plan')
        need(plan['pins'][str(HERE/'audit.py')]==sha(__file__),'this independent auditor was preregistered')
        need(plan['outputs']['screen']==str(screen) and plan['outputs']['execution']==str(execution), 'actual output binding')
        same(outer['service_identity_after'],plan['service']['identity'],'driver reported service identity')
        for path,digest in plan['pins'].items(): need(sha(path)==digest,'prepared source/input changed: '+path)
        prepared=[receipt(Path(r['path'])) for r in plan['specs']]
        need([s['label'] for s in prepared]==list(LABELS),'registered complete trial order')
        for item,r in zip(prepared,plan['specs']): need(sha(r['path'])==r['sha256'],'prepared spec hash')
        whole = receipt(screen/'result.json')
        need(whole['schema']=='E045-concurrency-config-screen-result-v1' and whole['status']=='PASS_CAPTURE_ONLY'
             and whole['errors']==[] and [r['label'] for r in whole['trials']]==list(LABELS), 'screen matrix incomplete/failed')
        need(all(outer[k] is whole[k] is False for k in ('full_state_qualified','performance_qualified','engine_pareto_point')),
             'original diagnostic scopes retained')
        result['cycle_closed']=True
        result['cycle_closure_basis']='Source-bound root final receipt; resource/restoration audit is separate.'
        parser, capture = load('http_join.py'), load('http_capture_v2.py')
        from tokenizers import Tokenizer
        tokens={}; canonical={}; seen=set(); previous_end=None
        for index,label in enumerate(LABELS):
            folder=screen/'trials'/label; spec=receipt(folder/'spec.json'); C,mb=specification(spec,label)
            same(spec,prepared[index],'actual executed/prepared spec')
            complete=receipt(folder/'complete.json'); child=receipt(folder/'child.json'); release=receipt(folder/'release.json')
            need(complete['schema']=='concurrency-config-screen-result-v1' and complete['status']=='PASS'
                 and complete['errors']==[] and complete['label']==label and complete['mb_threshold']==mb
                 and complete['concurrency']==C and complete['repeat']==1
                 and all(complete[k] is True for k in ('count_eligible','cleanup_complete','server_reaped',
                     'cleanup_task_resolved','client_task_resolved','small_pins_rechecked','diagnostic_only'))
                 and all(complete[k] is False for k in ('numerical_qualified','full_state_qualified','performance_qualified')),
                 'complete diagnostic trial')
            same(complete['source_pins'],spec['sources'],'executed source pins')
            same(complete['base_spec'],spec['base_spec'],'executed base spec')
            need(release['pid']==child['pid'] and release['returncode'] in (0,-15)
                 and all(release[k] is True for k in ('waited_reaped','strict_absence','port_absent'))
                 and complete['started_at']<=child['time']<=release['time']<=complete['ended_at'], 'direct server release')
            if release['returncode']==-15:
                events=[json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
                need(any(e.get('label')=='server-term' and e.get('pid')==child['pid'] for e in events),
                     'SIGTERM return code requires this owner termination event')
            child_identity=receipt(folder/'child-identity.json')
            need(child_identity['returncode']==0 and child_identity['stdout'].split()[0]==str(child['pid']),
                 'recorded direct child identity')
            need(previous_end is None or previous_end<=complete['started_at'],'fresh processes sequential')
            previous_end=complete['ended_at']
            for name, info in complete['artifacts'].items():
                path=folder/name; need(path.parent==folder and path.is_file()
                     and path.stat().st_size==info['bytes'] and sha(path)==info['sha256'],'artifact inventory: '+name)
            same(receipt(folder/'launch.json'),dict(command=spec['command'],environment=spec['environment']),'actual launch')
            for name in ('ready.json','ready-after-warm.json','ready-after-formal.json'):
                need(capture.ready_identity(receipt(folder/name),spec['identity']),'actual model/config readiness')
            need(capture.owned_listener(receipt(folder/'listener.json'),child['pid'],spec['transport']['port']), 'owned listener')
            for name in ('child-absence.json','port-after.json'): need(capture.no_process(receipt(folder/name)), 'strict absence')
            tokenizer=spec['tokenizer']; need(sha(tokenizer['path'])==tokenizer['sha256'],'tokenizer source')
            decoder=Tokenizer.from_file(tokenizer['path']); groups={}; mismatches=[]
            for phase in ('warm','formal'):
                group=receipt(folder/(phase+'-group.json'))
                same(group,receipt(folder/phase/'group.json'),'retained raw group copies')
                checked=group_check(group,spec,phase,parser,lambda ids:decoder.decode(ids,skip_special_tokens=False))
                need(not (seen & set(checked['uuids'])),'UUID reused across processes'); seen.update(checked['uuids'])
                producer=receipt(folder/(phase+'-raw-check.json'))
                same(producer['point'],checked['point'],'raw checker point')
                need(producer['raw_requests_checked']==8 and producer['status']=='PASS'
                     and producer['raw_client_phase']==group['phase'] and producer['validation_view_phase']=='warm'
                     and producer['diagnostic_only'] is True
                     and all(producer[k] is False for k in ('numerical_qualified','full_state_qualified','performance_qualified')),
                     'raw checker count and diagnostic scope')
                if C not in canonical: canonical[C]=checked['outputs']
                mismatches.extend(dict(phase=phase,**x) for x in compare_tokens(canonical[C],checked['outputs']))
                groups[phase]=checked
            need(groups['warm']['close_end']<=groups['formal']['first_post'],'warm drain before screen')
            point=groups['formal']['point']; same(complete['point'],point,'terminal point')
            same(whole['trials'][index],dict(label=label,status='PASS',point=point,server_reaped=True),'matrix trial index')
            same(receipt(folder/'point.json'),dict(**point,count_eligible=True,performance_qualified=False),'saved point')
            result['points'][label]=point; tokens[label]=not mismatches
            result['token_checks'][label]=dict(equal_to_same_C_start50_warm=not mismatches,mismatches=mismatches,
                scope='All eight fixtures, complete returned raw IDs in both warm and screen. No C1-to-C8 comparison.')
        result.update(status='PASS_SCREEN_ONLY' if all(tokens.values()) else 'FAIL_TOKEN_EQUIVALENCE',
            exact_returned_ids=all(tokens.values()), requests_reparsed=160, ranking=rank(result['points'],tokens))
        for path,digest in refs.items(): need(sha(path)==digest,'input changed during audit: '+path)
    except Exception as exc:
        result['status']='FAIL'; result['errors'].append(dict(type=type(exc).__name__,message=str(exc)))
    result['input_sha256']=refs
    result['scope']='N8 screen only; no full-state proof, no formal Pareto or independently audited outer resource/restoration approval. p95 is the maximum of eight observations.'
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('screen','execution','out'): p.add_argument('--'+name,type=Path)
    args=p.parse_args(argv)
    if not any((args.screen,args.execution,args.out)):
        print(json.dumps(dict(default_action='describe only',labels=LABELS,performance_qualified=False))); return 0
    need(all((args.screen,args.execution,args.out)),'supply --screen --execution --out together')
    args.out.mkdir(parents=True,exist_ok=False)
    result=audit(args.screen,args.execution)
    (args.out/'audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],errors=result['errors'],out=str(args.out))))
    return 0 if result['status']=='PASS_SCREEN_ONLY' else 1


if __name__=='__main__': raise SystemExit(main())
