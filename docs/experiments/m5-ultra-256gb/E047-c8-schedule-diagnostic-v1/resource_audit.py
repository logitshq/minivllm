"""E047 offline saved-resource review; no live/device actions, default inert."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import FunctionType

R=Path('/Users/ryanli/code/minivllm'); B=R/'bench/m5_ultra_256gb'
HELPER=B/'concurrency_config_analysis_v1/resources.py'
HELPER_SHA='0f1b4c0092f5af909a9bb5e3afb87a2dead071b2c8ae4bd7ffbb43fea7d961f9'
LABEL='c8-mb50-two-waves'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def review(execution):
    assert sha(HELPER)==HELPER_SHA, 'frozen pure resource readers changed'
    spec=importlib.util.spec_from_file_location('_e047_saved_resources',HELPER)
    h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    # Identical raw readers, ceilings, zero-swap, cadence, identity and restoration
    # checks. The sole donor change is the explicitly preregistered one-trial list.
    namespace=dict(vars(h),LABELS=(LABEL,))
    fn=h.audit; call=FunctionType(fn.__code__,namespace,fn.__name__,fn.__defaults__,fn.__closure__)
    result=call(execution);result['schema']='E047-saved-resource-review-v1'
    result['donor_review_source_sha256']=result.pop('review_source_sha256')
    result['review_source_sha256']=sha(__file__)
    try:
        plan_path=execution/'plan.json';plan=json.loads(plan_path.read_bytes())
        assert plan['schema']=='E047-c8-schedule-plan-v1' and len(plan['specs'])==1
        prereg=B/'c8_schedule_diagnostic_v1/prereg-v1/plan.json'
        assert sha(prereg)==sha(plan_path), 'prepared/executed E047 plan mismatch'
        small=[prereg,*sorted((B/'c8_schedule_diagnostic_v1').glob('*.py'))]
        for path in small:
            if path!=prereg:assert sha(path)==plan['pins'][str(path)], 'E047 source changed: '+str(path)
            result['inputs'][str(path)]=dict(bytes=path.stat().st_size,sha256=sha(path))
        path=Path(plan['outputs']['screen'])/'trials'/LABEL/'complete.json'
        complete=json.loads(path.read_bytes())
        assert complete['schema']=='E047-c8-schedule-result-v1' and complete['label']==LABEL
        assert complete['diagnostic_only'] is True and all(complete[k] is False for k in
            ('numerical_qualified','full_state_qualified','performance_qualified','replay_input_checked'))
        result.update(numerical_qualified_here=False,full_state_qualified=False,engine_pareto_point=False,
            scope='Saved resource and closure facts only; instrumented two-wave C8 diagnosis. No timing or numerical qualification.')
    except Exception as exc:
        result['status']='FAIL_OR_UNKNOWN';result['errors'].append(dict(type=type(exc).__name__,message=str(exc)))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execution',type=Path);p.add_argument('--out',type=Path);a=p.parse_args()
    if a.execution is None and a.out is None:print('Inert: explicit --execution closed-directory --out fresh.json required.')
    else:
        assert a.execution is not None and a.out is not None
        result=review(a.execution.resolve())
        with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
        print(json.dumps({k:result[k] for k in ('status','errors')}));raise SystemExit(0 if result['status']=='PASS_SAVED_OBSERVATIONS' else 1)
