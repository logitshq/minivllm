"""E055 six-process saved resource review; default inert and never live probes."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import FunctionType

R=Path('/Users/ryanli/code/minivllm'); B=R/'bench/m5_ultra_256gb'
HELPER=B/'concurrency_config_analysis_v1/resources.py'
HELPER_SHA='0f1b4c0092f5af909a9bb5e3afb87a2dead071b2c8ae4bd7ffbb43fea7d961f9'
PLAN_SHA='fc24144f4c0676ca6de3aa26f93bb6eabb2976d8addbc403ec35e228b484ed4b'
LABELS=('r1-c8-on','r1-c8-off','r2-c8-off','r2-c8-on','r3-c8-on','r3-c8-off')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def need(ok,message):
    if not ok:raise ValueError(message)

def review(execution):
    need(sha(HELPER)==HELPER_SHA,'frozen raw resource readers changed')
    source=importlib.util.spec_from_file_location('_e055_saved_resources',HELPER)
    h=importlib.util.module_from_spec(source);source.loader.exec_module(h)
    # Only the preregistered trial list changes. Raw footprint/command parsers,
    # 64/240-GiB sampled thresholds, cadence and identity/restore checks are reused.
    fn=h.audit;call=FunctionType(fn.__code__,dict(vars(h),LABELS=LABELS),fn.__name__,fn.__defaults__,fn.__closure__)
    result=call(execution);result['schema']='E055-saved-resource-review-v1'
    result['donor_review_source_sha256']=result.pop('review_source_sha256')
    result['review_source_sha256']=sha(__file__)
    def read(path):
        path=Path(path);need(not path.is_symlink(),'unexpected symlink')
        raw=path.read_bytes();result['inputs'][str(path)]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        return json.loads(raw)
    try:
        plan_path=execution/'plan.json';plan=read(plan_path)
        prereg=B/'c8_batched_formal_v1/prereg-v1/plan.json'
        need(sha(prereg)==sha(plan_path)==PLAN_SHA,'exact prepared/executed E055 plan')
        need(plan['schema']=='E055-c8-batched-formal-plan-v1' and len(plan['specs'])==6,'six E055 processes')
        screen=Path(plan['outputs']['screen']);receipt=screen/'result.json';scope=read(screen/'scope.json')
        need(scope['schema']=='E055-paired-formal-owner-scope-v1'
             and scope['inherited_result']==dict(path=str(receipt),sha256=sha(receipt),status='PASS_CAPTURE_ONLY')
             and (scope['expected_trials'],scope['warm_per_trial'],scope['formal_per_trial'])==(6,8,64)
             and scope['source_mutation'] is scope['performance_qualified'] is scope['engine_pareto_point'] is False,
             'new accurate scope must bind the unchanged legacy owner receipt')
        inventory=[]
        for label,ref in zip(LABELS,plan['specs']):
            path=Path(ref['path']);spec=read(path)
            need(sha(path)==ref['sha256']==plan['pins'][str(path)],'exact registered spec')
            need(spec['schema']=='E055-c8-batched-formal-spec-v1' and spec['label']==spec['phase']==label
                 and spec['arm']==label.rsplit('-',1)[1] and spec['concurrency']==8
                 and len(spec['jobs']['warm'])==8 and len(spec['jobs']['formal'])==64
                 and spec['observer_enabled'] is False,'registered arm and original8/64 lifecycle')
            folder=screen/'trials'/label;complete=read(folder/'complete.json')
            need(complete['schema']=='E055-c8-batched-formal-result-v1' and complete['label']==label
                 and complete['arm']==spec['arm'] and complete['phase']==label
                 and complete['observer_enabled'] is False and complete['diagnostic_throughput_only'] is True
                 and complete['cross_schedule_token_comparison']=='PENDING_PAIRED_AUDIT'
                 and all(complete[k] is False for k in ('numerical_qualified','full_state_qualified','performance_qualified')),
                 'unmodified per-process producer scope')
            for phase,count in (('warm',8),('formal',64)):
                group=read(folder/(phase+'-group.json'))
                need(group['phase']==phase and len(group['requests'])==count,'raw group inventory only')
            inventory.append(dict(label=label,arm=spec['arm'],warm_requests=8,formal_requests=64))
        for path in [prereg,*sorted((B/'c8_batched_formal_v1').glob('*.py'))]:
            digest=sha(path)
            if path!=prereg:need(digest==plan['pins'][str(path)],'changed E055 source: '+str(path))
            result['inputs'][str(path)]=dict(bytes=path.stat().st_size,sha256=digest)
        result.update(trial_inventory=inventory,warm_requests=48,formal_requests=384,
            numerical_qualified_here=False,full_state_qualified=False,http_history_qualified_here=False,
            performance_qualified=False,engine_pareto_point=False,raw_sse_and_timing_audited_here=False,
            original_legacy_owner_receipt_preserved=True,accurate_scope_receipt_verified=True,
            scope='Saved resource, six fresh process releases, and restoration facts only. Actual8warm/64formal specs and scope.json checked; original inherited owner receipt remains unchanged. Numerical histories, output IDs and formal timing require their separate audit.')
    except Exception as exc:
        result['status']='FAIL_OR_UNKNOWN';result['errors'].append(dict(type=type(exc).__name__,message=str(exc)))
    for path,ref in result['inputs'].items():
        if sha(path)!=ref['sha256']:
            result['status']='FAIL_OR_UNKNOWN';result['errors'].append(dict(type='InputChanged',message=path))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execution',type=Path);p.add_argument('--out',type=Path);a=p.parse_args()
    if a.execution is None and a.out is None:print('Inert: explicit --execution closed-directory --out fresh.json required after restoration.')
    else:
        need(a.execution is not None and a.out is not None,'both arguments required');result=review(a.execution.resolve())
        with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
        print(json.dumps({k:result[k] for k in ('status','errors')}));raise SystemExit(0 if result['status']=='PASS_SAVED_OBSERVATIONS' else 1)
