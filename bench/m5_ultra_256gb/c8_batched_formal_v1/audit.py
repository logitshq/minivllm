"""Small six-process raw timing/IDs reducer; no service or numerical replay.

Cross-schedule greedy differences are reported separately from malformed wire
output. Even a speed-gate win leaves performance/resource qualification pending.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent;B=HERE.parent
TRIAL=HERE/'trial.py';TRIAL_SHA='b6c5c8603ee877a3e68f8de4b551bd54fa71887389af117c1f1f31caa0ef5bd7'
RAW=B/'pareto_realistic/audit_native_formal.py';RAW_SHA='b57e4e9657f3dddb04069290bdfbab4c99b0d6cbfb1427e454d0217d5463c870'
METRICS=('aggregate_output_tok_s','e2e_p50_s','e2e_p95_s')
DECISION=dict(min_median_throughput_ratio=1.03,max_median_p50_ratio=.97,max_median_p95_ratio=1.02,
              all_pair_throughput_above_one=True,all_pair_p50_below_one=True,max_off_rate_spread_fraction=.02)

def need(ok,message):
    if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path,digest):
    need(sha(path)==digest,'fixed reducer support changed')
    s=importlib.util.spec_from_file_location('_e055_reducer_'+path.parent.name+'_'+path.stem,path)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def compare_ids(left,right):
    need(len(left)==len(right),'same compared request count')
    first=None;differences=0
    for a,b in zip(left,right):
        key=(a['fixture_id'],a['occurrence']);need(key==(b['fixture_id'],b['occurrence']),'same fixture/occurrence order')
        fields=('raw_ids','committed_ids','terminal_eos_id')
        if any(a[k]!=b[k] for k in fields):
            differences+=1
            if first is None:
                x,y=a['raw_ids'],b['raw_ids'];index=next((i for i in range(min(len(x),len(y))) if x[i]!=y[i]),min(len(x),len(y)))
                first=dict(fixture_id=key[0],occurrence=key[1],raw_index=index,
                    left_token=x[index] if index<len(x) else None,right_token=y[index] if index<len(y) else None,
                    left_raw_length=len(x),right_raw_length=len(y))
    return dict(requests_compared=len(left),different_trajectories=differences,first_difference=first,
        status='IDENTICAL' if not differences else 'DIFFERENT_GREEDY_TRAJECTORIES',numerical_error_inferred=False)


def reduce_pairs(points,ids,phases,decision):
    need([p['phase'] for p in points]==list(phases) and len(points)==6,'exact six ordered fresh-process points')
    need(decision==DECISION,'explicit preregistered speed gate')
    pairs=[]
    for first,second in zip(points[::2],points[1::2]):
        arms={p['arm']:p for p in (first,second)}
        need(set(arms)=={'off','on'} and first['repeat']==second['repeat'] and first['concurrency']==second['concurrency']==8,'adjacent same-repeat/C8 arms')
        for p in (first,second):
            need(p['requests']==64 and all(type(p[k]) in (int,float) and 0<p[k]<float('inf') for k in METRICS),'finite complete64 point')
            need(len(ids[p['phase']]['warm'])==8 and len(ids[p['phase']]['formal'])==64,'all actual8/64 ID sequences')
        off,on=arms['off'],arms['on']
        pairs.append(dict(repeat=off['repeat'],order=[first['arm'],second['arm']],
            throughput_ratio=on[METRICS[0]]/off[METRICS[0]],p50_ratio=on[METRICS[1]]/off[METRICS[1]],
            p95_ratio=on[METRICS[2]]/off[METRICS[2]],
            output_comparison={phase:compare_ids(ids[off['phase']][phase],ids[on['phase']][phase]) for phase in ('warm','formal')}))
    medians={key:statistics.median(p[key] for p in pairs) for key in ('throughput_ratio','p50_ratio','p95_ratio')}
    rates=[p[METRICS[0]] for p in points if p['arm']=='off'];spread=(max(rates)-min(rates))/statistics.median(rates)
    win=(all(p['throughput_ratio']>1 and p['p50_ratio']<1 for p in pairs)
         and medians['throughput_ratio']>=decision['min_median_throughput_ratio']
         and medians['p50_ratio']<=decision['max_median_p50_ratio'] and medians['p95_ratio']<=decision['max_median_p95_ratio']
         and spread<=decision['max_off_rate_spread_fraction'])
    identical=all(c['different_trajectories']==0 for p in pairs for c in p['output_comparison'].values())
    return dict(pairs=pairs,median_pair_ratios=medians,off_rate_spread_fraction=spread,speed_gate_passed=win,
        all_pair_output_ids_identical=identical,output_comparison_status='IDENTICAL' if identical else 'DIFFERENT_GREEDY_TRAJECTORIES',
        numerical_error_inferred=False,process_summary={arm:{key:dict(median=statistics.median(p[key] for p in points if p['arm']==arm),
            minimum=min(p[key] for p in points if p['arm']==arm),maximum=max(p[key] for p in points if p['arm']==arm)) for key in METRICS} for arm in ('off','on')},
        scope='Three adjacent process pairs; no pooled-request statistical claim. Different schedules/greedy trajectories are reported, not labeled arithmetic errors.')


def ids_from_raw(join,group):
    counts={};rows=[]
    for row in group['requests']:
        rid,terminal,_=join.raw_completion(row);fixture=row['fixture_id'];n=counts.get(fixture,0);counts[fixture]=n+1
        rows.append(dict(fixture_id=fixture,occurrence=n,request_id=row['request_id'],response_id=rid,
            raw_ids=terminal['benchmark_output_ids'],committed_ids=terminal['benchmark_committed_output_ids'],terminal_eos_id=terminal['benchmark_terminal_eos_id']))
    return rows


def audit(screen,plan_path,plan_sha256):
    result=dict(schema='E055-six-process-metrics-ids-v1',status='INCOMPLETE',performance_qualified=False,
        owner_resources_restore_qualified=False,formal_full_state_replayed=False,diagnostic_throughput_only=True,
        timing_valid=False,same_schedule_numerical_prerequisite=False,cross_schedule_token_equal=None,
        audit_source_sha256=sha(__file__),scope='Actual observer-OFF6process C8 counts/timing/IDs; external lifecycle/resource audit required. No automatic Pareto point.')
    try:
        trial=load(TRIAL,TRIAL_SHA);formal=trial.module(trial.FORMAL);native=trial.module(trial.NATIVE);raw=load(RAW,RAW_SHA)
        read,same=formal.read,formal.same;screen=Path(screen).resolve();plan_path=Path(plan_path).resolve()
        need(sha(plan_path)==plan_sha256,'explicit registered owner plan SHA');plan=read(plan_path)
        need(plan['schema']=='E055-c8-batched-formal-plan-v1' and len(plan['specs'])==6 and Path(plan['outputs']['screen'])==screen,'actual six-trial screen')
        actual=read(screen/'plan.json');same({k:v for k,v in actual.items() if k!='physical_baseline'},plan,'same owner execution plan')
        need('physical_baseline' in actual,'actual physical baseline receipt retained')
        terminal=read(screen/'result.json')
        need(terminal['status']=='PASS_CAPTURE_ONLY' and terminal['errors']==[] and [x['label'] for x in terminal['trials']]==list(trial.TRIALS),'allsix closed owner trials')
        for p,h in plan['pins'].items():need(sha(p)==h,'owner source/input pin changed')
        stable={str(p):sha(p) for p in (plan_path,screen/'plan.json',screen/'result.json')};points=[];ids={};seen=set();specs=[]
        tokenizer_ref=None;decode=None;join=native.load('http_join')
        for spec_ref,label in zip(plan['specs'],trial.TRIALS):
            spec=read(formal.pinned(spec_ref));trial.validate(spec);need(spec['label']==label,'registered unique label');specs.append(spec)
            folder=screen/'trials'/label;complete=read(folder/'complete.json')
            need(complete['schema']=='E055-c8-batched-formal-result-v1' and complete['status']=='PASS' and complete['errors']==[]
                 and complete['label']==label and complete['arm']==spec['arm'] and all(complete[k] is True for k in
                    ('count_eligible','server_reaped','cleanup_complete','cleanup_task_resolved','client_task_resolved','small_pins_rechecked')),
                 'complete original trial/cleanup')
            need(complete['performance_qualified'] is complete['full_state_qualified'] is complete['observer_enabled'] is False,'scope/observers retained')
            same(read(folder/'spec.json'),spec,'actual spec');same(read(folder/'launch.json'),dict(command=spec['command'],environment=spec['environment']),'actual command')
            need(not any((folder/name).exists() for name in ('trace.jsonl','route-events.jsonl','dispatch-events.jsonl')),'observer artifacts forbidden')
            for name,item in complete['artifacts'].items():
                p=folder/name;need(p.parent==folder and not p.is_symlink(),'local regular artifact')
                same(item,dict(sha256=sha(p),bytes=p.stat().st_size),'actual trial artifact')
            if decode is None:
                tokenizer_ref=spec['tokenizer'];from tokenizers import Tokenizer
                tokenizer=Tokenizer.from_file(str(formal.pinned(tokenizer_ref)));decode=lambda values:tokenizer.decode(values,skip_special_tokens=False)
            else:same(spec['tokenizer'],tokenizer_ref,'same exact tokenizer')
            ids[label]={}
            for phase,count in (('warm',8),('formal',64)):
                group=read(folder/(phase+'-group.json'));same(group,read(folder/phase/'group.json'),'original raw group duplicate')
                checked=native.check_group(group,spec,phase,decode);metrics=raw.raw_metrics(group)
                need(metrics['requests']==count and metrics['max_http_outstanding']==8,'complete actual8/64 C8 queue')
                need(not seen.intersection(metrics['response_uuids']),'response UUID reused');seen.update(metrics['response_uuids'])
                same(read(folder/(phase+'-raw-check.json')),checked,'producer raw-check')
                point=dict(concurrency=8,repeat=spec['repeat'],**{k:metrics[k] for k in ('requests','committed_output_tokens','common_window_s',*METRICS)})
                same(checked['point'],point,'independent timing/ID counts');ids[label][phase]=ids_from_raw(join,group)
                if phase=='formal':
                    same(complete['point'],point,'terminal timing');same(read(folder/'point.json'),dict(**point,count_eligible=True,performance_qualified=False),'raw unapproved point')
                    points.append(dict(phase=label,arm=spec['arm'],**point))
            same(read(formal.pinned(complete['output_ids'])),ids[label],'retained complete IDs versus raw SSE')
            stable[str(folder/'complete.json')]=sha(folder/'complete.json')
        need(len(seen)==432,'all432 actual request UUIDs')
        for a,b in zip(specs[::2],specs[1::2]):same(a['jobs'],b['jobs'],'same paired bodies/order/request IDs')
        reduction=reduce_pairs(points,ids,trial.TRIALS,plan['decision'])
        for p,h in stable.items():need(sha(p)==h,'closed evidence changed')
        result.update(status='PASS_METRICS_AND_IDS',timing_valid=True,same_schedule_numerical_prerequisite=True,
            numerical_prerequisite_scope='Source-bound E053 complete captured history; not all formal schedules.',
            cross_schedule_token_equal=reduction['all_pair_output_ids_identical'],
            cross_schedule_differences=[dict(repeat=p['repeat'],phase=phase,**value) for p in reduction['pairs']
                for phase,value in p['output_comparison'].items() if value['different_trajectories']],
            points=points,comparison=reduction,actual_request_count=len(seen),
            plan_sha256=plan_sha256,screen=str(screen),terminal_sources=stable,support_sources={str(TRIAL):TRIAL_SHA,str(RAW):RAW_SHA},
            note='Speed gate is separate from output comparison. Either result remains diagnostic until external correctness/resource review.')
    except Exception as exc:result.update(status='FAIL',error=dict(type=type(exc).__name__,message=str(exc)))
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--screen',type=Path,required=True);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--plan-sha256',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args(argv)
    need(a.out.parent.is_dir() and not a.out.exists(),'fresh output in existing directory')
    value=audit(a.screen,a.plan,a.plan_sha256)
    with a.out.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=value['status'],performance_qualified=value['performance_qualified'])))
    return 0 if value['status']=='PASS_METRICS_AND_IDS' else 1


if __name__=='__main__':raise SystemExit(main())
