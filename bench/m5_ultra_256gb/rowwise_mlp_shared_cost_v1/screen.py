"""Default-OFF E057 adapter of the frozen E054 device loop; no owner or HTTP."""
import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback
from types import FunctionType, SimpleNamespace
import weakref

HERE=Path(__file__).resolve().parent; B=HERE.parent
BASE=B/'rowwise_native_verify_cost_v1/screen.py'; BASE_SHA='679ad62a8b2973c9ea6d767da3135782345ca09c870ff071ff1494005b914d4d'
CANDIDATE=B/'rowwise_mlp_shared/candidate.py'; CANDIDATE_SHA='02200afa33fd6daa3f0f4dac3436b717616cf620f4b6cd7e59f1d8f263658950'
CANARY=B/'rowwise_mlp_shared/canary.py'; CANARY_SHA='c1aeae682a5822a0483926fb588f3e184d307a1aeb7fc3a2380a0d16bf85ffd8'
AUDITOR=B/'rowwise_mlp_shared_analysis_v1/audit.py'; AUDITOR_SHA='3ce4f312ce3e7a4c76ea429dcd9617d23430bfa0bb4bfa7105379562502240a5'
SHARED=B/'nvfp4_multirow_shared/candidate.py'; SHARED_SHA='d1826b3c4d07d44adbae071ef2da43197348c09b2d8be1035eb1a0c2788d132e'


def need(ok,message):
    if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def load(path,digest):
    need(sha(path)==digest,'frozen source '+str(path))
    spec=importlib.util.spec_from_file_location('_e057_'+path.parent.name+'_'+path.stem,path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);return module

def pinned(ref):
    need(set(ref)=={'path','sha256'} and Path(ref['path']).is_absolute() and ref['sha256']==sha(ref['path']),'absolute exact evidence reference')
    return json.loads(Path(ref['path']).read_bytes())


def contract():
    return dict(schema='C1-MLP-shared-cost-contract-v1',enabled_by_default=False,widths=[4,8],used=1023,
        inherited_device_loop=dict(path=str(BASE),sha256=BASE_SHA),shared_head=True,shared_mlp_calls_per_forward=128,
        qualification='Six k=1,T/2,T: fullT logits + selected128 states + native continuation;32 raw files/2064 tensor pairs.',
        timing='Unchanged E054400 calls: native device AR(T) primary, host AR(T) secondary, versus MLP+head verifier/commit.',
        timed_mlp_observer=False,head_only_baseline='Historical E054 only; not a paired third arm.',
        compiled_qualified=False,full_model_qualified=False,http_qualified=False,performance_qualified=False,engine_pareto_point=False)


def prepare(e054_plan,e056_plan,e056_result,e056_audit,deadline_s=900):
    plan=dict(schema='C1-MLP-shared-cost-plan-v1',adapter_sha256=sha(__file__),e054_plan=e054_plan,
        e056_plan=e056_plan,e056_result=e056_result,e056_audit=e056_audit,deadline_s=deadline_s)
    validate(plan);return plan


def validate(plan):
    need(set(plan)=={'schema','adapter_sha256','e054_plan','e056_plan','e056_result','e056_audit','deadline_s'}
         and plan['schema']=='C1-MLP-shared-cost-plan-v1' and plan['adapter_sha256']==sha(__file__),'exact actual adapter plan')
    need(type(plan['deadline_s']) in (int,float) and math.isfinite(plan['deadline_s']) and 0<plan['deadline_s']<=1800,'bounded deadline')
    base=load(BASE,BASE_SHA);candidate=load(CANDIDATE,CANDIDATE_SHA);canary=load(CANARY,CANARY_SHA)
    need(sha(AUDITOR)==AUDITOR_SHA and sha(SHARED)==SHARED_SHA,'fixed canary auditor and shared projector')
    previous_plan=pinned(plan['e056_plan']);previous=pinned(plan['e056_result']);audit=pinned(plan['e056_audit'])
    canary.validate(previous_plan)
    need(previous_plan['e054_plan']==plan['e054_plan'],'identical original E054 reference')
    need(previous['schema']=='C1-MLP-shared-activation-canary-result-v1' and previous['status']=='PASS_PROJECTION_CANARY_ONLY'
         and previous['plan_sha256']==plan['e056_plan']['sha256'] and previous['projection_pairs']==12 and previous['raw_files']==30
         and all(previous[k] is True for k in ('projection_canary_complete','native_returning_provenance_complete','device_cleanup_completed','source_model_pins_rechecked')),'actual closed E056 canary')
    need(audit['schema']=='E056-MLP-canary-independent-v1' and audit['status']=='PASS'
         and audit['audit_source_sha256']==AUDITOR_SHA and audit['complete_sha256']==plan['e056_result']['sha256']
         and audit['projection_bytes_audited'] is True and audit['raw_files']==30 and audit['tensor_pairs']==12
         and audit['provenance_receipt_pairs']==4 and audit['provenance_receipt_arrays']==516
         and audit['raw_full_state_checked'] is False,'actual independent E056 byte/provenance audit')
    old_plan=pinned(plan['e054_plan']);q,deps,seed=base.validate(old_plan)
    need(q.HEAD==SHARED and q.HEAD_SHA==SHARED_SHA,'same real shared projector for head and MLP')
    return base,candidate,old_plan,q,deps,seed


class CountedMLP:
    """Untimed qualification receipt only; actual live-source ordering is checked."""
    def __init__(self,model,project,width):
        self.ops=[(i,name,getattr(layer.mlp,name+'_proj')) for i,layer in enumerate(model.layers) for name in ('gate','up')]
        self.project,self.width,self.rows=project,width,[]
    def __call__(self,x,w,s):
        need(len(self.rows)<128,'extra shared MLP call')
        i,name,op=self.ops[len(self.rows)]
        need(w is op.weight and s is op.scales and x.shape==(self.width,5120),'actual ordered shared MLP input/parameter handles')
        value=self.project(x,w,s)
        need(value is not None and value.shape==(self.width,17408),'explicit shared MLP rejection')
        self.rows.append(dict(layer=i,projection=name,T=self.width,K=5120,N=17408,
            weight_shape=list(w.shape),scales_shape=list(s.shape),input_dtype=str(x.dtype),output_dtype=str(value.dtype)))
        return value
    def receipt(self):
        need(len(self.rows)==128,'complete128 shared MLP calls')
        return dict(status='PASS',calls=self.rows,call_count=128,scope='Python qualification calls; not GPU dispatch counts or timed observer')


def adapted_device(base,candidate,result,out,deadline,deps):
    """Same original device code object; no mutation of imported module globals."""
    h=deps[2];parameter_context={}
    def parameters(mx,model,flatten,check_deadline):
        parameter_context.update(mx=mx,model=weakref.ref(model),flatten=flatten,deadline=check_deadline)
        return h.parameter_hashes(mx,model,flatten,check_deadline)
    helper=SimpleNamespace(**vars(h));helper.parameter_hashes=parameters
    adapted_deps=tuple(helper if i==2 else value for i,value in enumerate(deps))
    def saved(path,value):
        if Path(path)==out/'complete.json' and value.get('numerical_screen_qualified') is True and not value.get('pre_timing_parameters_unchanged'):
            need(len(value['qualification'])==6 and len(value['raw_comparisons'])==16 and len(value['mlp_qualifications'])==6,'all full raw checks before timing')
            p=parameter_context;model=p['model']();need(model is not None,'live model for pre-timing byte check')
            current=h.parameter_hashes(p['mx'],model,p['flatten'],p['deadline'])
            before=json.loads((out/'parameters-before.json').read_bytes())
            need(h.parameter_values(current)==h.parameter_values(before),'all original parameter bytes before timing')
            write(out/'parameters-before-timing.json',current);value['pre_timing_parameters_unchanged']=True
        base.write(path,value)
    def run(r,engine,q,ignored,h_arg,layout,storage,position,tokens,count,project,*,audit=False):
        need(project is not None,'explicit shared head/MLP projector')
        observation=CountedMLP(engine.model,project,len(tokens)) if audit else None
        def verify(model,ids,ctx,runtime,*,enabled,head_project):
            need(enabled is True and head_project is not None,'both shared paths explicitly enabled')
            silu=type(model.layers[0].mlp).__call__.__globals__['nn'].silu
            enriched=SimpleNamespace(**vars(runtime),silu=silu)
            return candidate.verify(model,ids,ctx,enriched,enabled=True,head_project=head_project,
                                    mlp_input_project=observation if audit else project)
        value=base.candidate_run(r,engine,q,SimpleNamespace(verify=verify),h_arg,layout,storage,position,tokens,count,project,audit=audit)
        if audit:result['mlp_qualifications'].append(dict(T=len(tokens),k=count,**observation.receipt()))
        return value
    scope=dict(vars(base),candidate_run=run,write=saved)
    device=FunctionType(base.device.__code__,scope,'e057_device',base.device.__defaults__,base.device.__closure__)
    return device,adapted_deps


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--plan',type=Path)
    p.add_argument('--plan-sha256');p.add_argument('--out',type=Path);a=p.parse_args(argv)
    if not a.run:
        need(not any((a.plan,a.plan_sha256,a.out)),'explicit --run required');print(json.dumps(contract(),indent=2));return 0
    need(all((a.plan,a.plan_sha256,a.out)),'explicit complete run arguments')
    a.out.mkdir(parents=True,exist_ok=False);started=time.monotonic();previous={}
    result=dict(schema='C1-MLP-shared-cost-result-v1',status='INCOMPLETE',pid=os.getpid(),ppid=os.getppid(),plan_sha256=a.plan_sha256,
        contract=contract(),qualification=[],raw_comparisons=[],timings=[],mlp_qualifications=[],numerical_screen_qualified=False,
        pre_timing_parameters_unchanged=False,device_cleanup_completed=False,source_model_pins_rechecked=False,
        full_model_qualified=False,http_qualified=False,performance_qualified=False,engine_pareto_point=False)
    try:
        need(sha(a.plan)==a.plan_sha256,'plan SHA');plan=json.loads(a.plan.read_bytes());write(a.out/'plan.json',plan)
        base,candidate,old_plan,q,deps,seed=validate(plan)
        result.update(seed=seed,inherited_contract=base.contract(),inherited_plan=plan['e054_plan'],
            adaptation='Original E054 device code with private candidate_run/write/helper dependencies; actual shared MLP+head candidate, not original E054 run.',
            sources={str(Path(__file__).resolve()):plan['adapter_sha256'],str(BASE):BASE_SHA,str(CANDIDATE):CANDIDATE_SHA,
                     str(CANARY):CANARY_SHA,str(AUDITOR):AUDITOR_SHA,str(SHARED):SHARED_SHA})
        write(a.out/'complete.json',result)
        def deadline():
            if time.monotonic()-started>=plan['deadline_s']:raise TimeoutError('E057 child deadline')
        def stop(signum,frame):raise InterruptedError('bounded E057 signal '+str(signum))
        need(signal.getitimer(signal.ITIMER_REAL)==(0.0,0.0),'existing alarm')
        previous={s:signal.getsignal(s) for s in (signal.SIGALRM,signal.SIGINT,signal.SIGTERM)}
        for s in previous:signal.signal(s,stop)
        signal.setitimer(signal.ITIMER_REAL,plan['deadline_s'])
        device,adapted_deps=adapted_device(base,candidate,result,a.out,deadline,deps)
        device(old_plan,q,adapted_deps,seed,a.out,deadline,result);deadline();validate(plan)
        need(all(result[k] is True for k in ('numerical_screen_qualified','pre_timing_parameters_unchanged','device_cleanup_completed','source_model_pins_rechecked'))
             and len(result['mlp_qualifications'])==6 and sum(len(t['samples']) for t in result['timings'])==400,'full numerical and cost completion')
        need(sha(a.plan)==a.plan_sha256,'unchanged actual plan');result['status']='PASS_LOCAL_COST_ONLY'
    except BaseException as error:
        result.update(status='FAIL',error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
    finally:
        if previous:
            signal.setitimer(signal.ITIMER_REAL,0)
            for s,value in previous.items():signal.signal(s,value)
        result['elapsed_wall_s']=time.monotonic()-started;write(a.out/'complete.json',result)
    print(json.dumps(dict(status=result['status'],samples=sum(len(t['samples']) for t in result['timings']))))
    return 0 if result['status']=='PASS_LOCAL_COST_ONLY' else 1


if __name__=='__main__':raise SystemExit(main())
