"""Default-OFF prepared GDN prework 37-request numerical qualification.

Without execution arguments, only describes scope; no evidence/runtime opens.
--plan PATH --plan-sha256 SHA --out NEW requires the actual independent
18-call / 54-pair / 60-raw-file canary PASS first. Exact six-field plan: schema,
qualifier_sha256, qualification:{path,sha256} (original E023), canary:{audit,
plan,complete} (three absolute pinned refs), sources, deadline_s=1800.

The private base retains its two independent models,37 jobs, Comparator,
scheduler, full logits/state and semantic ahead pairing. Only candidate GDN
instances are installed before its original Observer; teardown is reversed.
Python branch counters are submission evidence, not kernel completion counts.
Numerical completion still depends on the unchanged evaluated base checks.
Short packed calls bypassing _decode are outside this candidate's coverage.
No E024/E028, parameter replacement, live HTTP or performance qualification.
External owner must enforce the inherited hard process/resource deadline.
The stable owner passes those three arguments without --run. --run is optional;
any partial execution arguments are rejected, never interpreted as describe.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback

HERE = Path(__file__).resolve().parent
BASE = HERE.parent/'pareto_realistic/qualify_native.py'
BASE_SHA = '26964fcbed9a2e5dcefaee4888244987897f3138d4a2b7add59bf63a7461b7bf'
BASE_PLAN_SHA = '21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb'
CANDIDATE_SHA = '1bf5a837c6515ca505bc5d1beaa1718b88e1ccd52609f6814bb995afbe6c1260'
CANARY_SHA = 'f3088a98d2aca6b4f267e7d4c258efbbf9f83e1beccb6745c88ae42343f88e89'
AUDIT_SHA = 'a3e97dc368da47735d7c8e9fd2bcb40d7decaccb9c5926084d53cffb10495d3d'
FIXED = {BASE:BASE_SHA,HERE/'candidate.py':CANDIDATE_SHA,
         HERE/'qualify.py':CANARY_SHA,HERE/'audit.py':AUDIT_SHA}
CASES = ('D-C1','D-C8','lifecycle')
SCOPE = ('Original37 same-schedule numerical comparison; current native flags plus default-OFF '
         'B1 ordinary GDN adapter. Short packed bypass is not adapter coverage. No E024/E028 '
         'composition, idle/live parameter replacement, cancellation, live HTTP or performance proof.')


def need(ok,message):
    if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def same(a,b,message):
    need(json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False),message)
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def pinned(ref):
    need(type(ref) is dict and set(ref)=={'path','sha256'} and Path(ref['path']).is_absolute(),'absolute pinned ref')
    need(sha(ref['path'])==ref['sha256'],'changed prerequisite: '+ref['path']);return read(ref['path'])
def source_pins():return {**{str(p):s for p,s in FIXED.items()},str(Path(__file__).resolve()):sha(__file__)}


def canary_gate(audit,complete):
    """Receipt contract only; the pinned independent auditor owns raw-bit proof."""
    need(audit['schema']=='GDN-prework-real-layer-independent-v1'
         and audit['status']=='PASS' and audit['canary_qualified'] is True
         and audit['audit_source_sha256']==AUDIT_SHA,'actual independent canary PASS required')
    for key,value in (('raw_files',60),('full_output_state_pairs',54),('causal_state_pairs',24),
                      ('full_vocabulary_greedy_checks',6),('device_comparison_arrays',162)):
        same(audit[key],value,'canary independent coverage: '+key)
    need(complete['schema']=='GDN-prework-real-layer-result-v1' and complete['status']=='PASS'
         and complete['canary_qualified'] is True,'actual canary producer complete required')
    for key,value in (('candidate_calls',18),('candidate_compared_calls',18),('reference_chains',2),
                      ('candidate_chains',6),('device_run',True),('instance_hooks_restored',True),
                      ('source_model_pins_rechecked',True)):
        same(complete[key],value,'canary producer coverage: '+key)
    same(complete['dispatch_counters'],{str(i):dict(candidate=6,fallback=0) for i in (0,32,62)},'actual18/no-fallback canary')
    for row in (audit,complete):
        for key in ('full_model_qualified','http_qualified','performance_qualified'):
            need(row[key] is False,'canary scope must not claim full model or performance')


def bind_canary(refs):
    need(type(refs) is dict and set(refs)=={'audit','plan','complete'},'three real canary evidence refs required')
    audit,plan,complete=(pinned(refs[k]) for k in ('audit','plan','complete'))
    canary_gate(audit,complete)
    child=Path(refs['complete']['path']).parent.resolve()
    need(Path(refs['complete']['path'])==child/'complete.json' and Path(refs['plan']['path'])==child/'plan.json'
         and audit['child_directory']==str(child),'same actual canary directory')
    same(audit['plan_sha256'],refs['plan']['sha256'],'canary audit plan SHA')
    same(complete['plan_sha256'],refs['plan']['sha256'],'canary producer plan SHA')
    same(audit['complete_sha256'],refs['complete']['sha256'],'independent terminal SHA')
    need(plan['schema']=='GDN-prework-real-layer-plan-v1' and plan['qualifier_sha256']==CANARY_SHA,'canary pinned producer')
    same(plan['candidate'],dict(path=str(HERE/'candidate.py'),sha256=CANDIDATE_SHA),'same candidate canary source')
    for name,field in (('events.jsonl','ledger_sha256'),('captures.json','captures_sha256')):
        need(sha(child/name)==audit[field],'actual canary '+name+' changed')
    same(audit['source_pins'],complete['sources'],'canary audit/producer sources')
    need(complete['sources'][str(HERE/'candidate.py')]==CANDIDATE_SHA
         and complete['sources'][str(HERE/'qualify.py')]==CANARY_SHA,'canary adapter source binding')
    inherited=pinned(plan['e024_plan'])
    need(inherited['e023_plan']['sha256']==BASE_PLAN_SHA
         and sha(inherited['e023_plan']['path'])==BASE_PLAN_SHA,'same original model/workload plan')
    return dict(evidence=refs,child_directory=str(child),raw_files=60,full_output_state_pairs=54,
                candidate_calls=18,audit_source_sha256=AUDIT_SHA,
                scope='Consumes SHA-pinned independent raw canary proof; does not rerun its raw audit.')


def validate(path,digest):
    need(sha(path)==digest,'serving plan SHA')
    plan=read(path)
    need(set(plan)=={'schema','qualifier_sha256','qualification','canary','sources','deadline_s'}
         and plan['schema']=='GDN-prework-serving-plan-v1','exact serving plan')
    need(plan['qualifier_sha256']==sha(__file__) and type(plan['deadline_s']) is int
         and plan['deadline_s']==1800,'self source and inherited1800s deadline')
    need(plan['sources']==source_pins(),'exact serving/base/canary/auditor/candidate source pins')
    for name,value in plan['sources'].items():need(sha(name)==value,'source changed: '+name)
    original=pinned(plan['qualification']);need(plan['qualification']['sha256']==BASE_PLAN_SHA,'original37 plan')
    same(original['deadline_s'],1800,'unchanged numeric deadline')
    same(original['candidate'],dict(compile_decode=True,fuse_decode=True,gates_compile=True,norm_reuse=False),'unchanged current candidate flags')
    proof=bind_canary(plan['canary'])
    return plan,proof


def dispatch_delta(before,after,mode,B,used):
    need(len(before)==len(after)==48,'all48 installed layers')
    deltas=[]
    for a,b in zip(before,after):
        need(set(a)==set(b)=={'candidate','fallback'}
             and all(type(n) is int and n>=0 for n in (*a.values(),*b.values())),'integer actual counters')
        deltas.append((b['candidate']-a['candidate'],b['fallback']-a['fallback']))
    if mode=='decode' and used>256:
        expected=(1,0) if B==1 else (0,1)
        need(B>=1 and all(d==expected for d in deltas),'ordinary real call silently fell back/stale or mixed layers')
        path='B1-candidate' if B==1 else 'B>1-fallback'
    else:
        need(mode in ('decode','prefill') and all(d==(0,0) for d in deltas),'prefill/short-bypass changed adapter counters')
        path='short-packed-bypass' if mode=='decode' else 'prefill-untouched'
    return dict(path=path,candidate=sum(x[0] for x in deltas),fallback=sum(x[1] for x in deltas))


@contextmanager
def adapt(base,adapter,emit,*,enabled=False):
    """Only private base hooks; injected fake base/adapter are CPU-testable."""
    need(type(enabled) is bool,'explicit boolean enable')
    coverage=dict(enabled=enabled,layer_installs=0,layer_restores=0,observer_restores=0,
        calls=0,per_case={},final_counters=[],identity_refill_enabled=False,route_graph_enabled=False)
    if not enabled:
        yield coverage;return
    original_observer,original_run=base.Observer,base.run_case

    def close_handles(handles):
        failure=None
        for handle in reversed(handles):
            try:handle.restore();coverage['layer_restores']+=1
            except BaseException as exc:
                if failure is None:failure=exc
        handles.clear()
        if failure is not None:raise failure

    class PreworkObserver(original_observer):
        def __init__(self,engine,cmp,label):
            self.handles=[];self.closed=False;self.case='unassigned';self.step=0
            runner,scheduler=engine.runner,engine.scheduler
            old_model,old_decode=runner.model,runner.decode_model
            saved=[(runner,'run','run' in vars(runner),vars(runner).get('run')),
                   (scheduler,'schedule','schedule' in vars(scheduler),vars(scheduler).get('schedule'))]
            try:
                if label=='candidate':
                    layers=[layer.linear_attn for layer in engine.model.layers if not layer.is_attn]
                    need(len(engine.model.layers)==64 and len(layers)==48,'actual64/48 model required')
                    for layer in layers:
                        module=sys.modules[type(layer).__module__]
                        handle=adapter.install(layer,module,enabled=True)
                        need(handle is not None,'candidate install disabled')
                        self.handles.append(handle);coverage['layer_installs']+=1
                super().__init__(engine,cmp,label)
            except BaseException:
                runner.model,runner.decode_model=old_model,old_decode
                for target,name,present,value in saved:
                    if present:setattr(target,name,value)
                    else:vars(target).pop(name,None)
                try:close_handles(self.handles)
                finally:
                    self.engine=self.original_model=self.original_decode=None
                    self.original_run=self.original_schedule=None;self.closed=True
                raise

        def begin(self,expected_schedule=None):
            self.step+=1
            return super().begin(expected_schedule)

        def call(self,target,tokens,ctx,kwargs):
            if self.label!='candidate':return super().call(target,tokens,ctx,kwargs)
            need(self.case in CASES,'candidate model call outside original case')
            before=[dict(h.calls) for h in self.handles];old_keys=set(self.calls)
            row=dict(event='prework-model-call',status='FAILED',case=self.case,step=self.step,
                     mode=ctx.mode,B=tokens.shape[0],used=ctx.max_len)
            try:
                result=super().call(target,tokens,ctx,kwargs)
                after=[dict(h.calls) for h in self.handles]
                delta=dispatch_delta(before,after,ctx.mode,tokens.shape[0],ctx.max_len)
                added=set(self.calls)-old_keys;need(len(added)==1,'one original semantic model-call record')
                row.update(status='SUBMITTED',semantic_key=base.plain(next(iter(added))),**delta)
                counts=coverage['per_case'].setdefault(self.case,dict(candidate=0,fallback=0,
                    b1_calls=0,multirow_calls=0,prefill_calls=0,short_bypass_calls=0))
                for key in ('candidate','fallback'):counts[key]+=delta[key]
                field={'B1-candidate':'b1_calls','B>1-fallback':'multirow_calls',
                       'prefill-untouched':'prefill_calls','short-packed-bypass':'short_bypass_calls'}[delta['path']]
                counts[field]+=1;coverage['calls']+=1
                return result
            finally:emit(row)

        def restore(self):
            if self.closed:return
            try:super().restore()
            finally:
                if self.handles:coverage['final_counters'].append([dict(h.calls) for h in self.handles])
                try:close_handles(self.handles)
                finally:
                    self.calls.clear();self.previous_future=self.postrun=None
                    self.engine=self.original_model=self.original_decode=self.original_run=self.original_schedule=None
                    self.closed=True;coverage['observer_restores']+=1

    def run(case,engines,observers,*args,**kwargs):
        for observer in observers:observer.case=case['name'];observer.step=0
        return original_run(case,engines,observers,*args,**kwargs)

    base.Observer,base.run_case=PreworkObserver,run
    try:yield coverage
    finally:base.Observer,base.run_case=original_observer,original_run


def require_complete(result,coverage):
    need(result['status']=='PASS' and result['source_model_pins_rechecked'] is True,'original full numerical gate')
    same(result['total_requests'],37,'all37 requests')
    same([(c['name'],len(c['completed'])) for c in result['cases']],
         [('D-C1',8),('D-C8',16),('lifecycle',13)],'unchanged original cases')
    need(coverage['enabled'] is True and coverage['layer_installs']==coverage['layer_restores']==96
         and coverage['observer_restores']==4 and len(coverage['final_counters'])==2,'all actual hooks installed/restored')
    rows=coverage['per_case'];need(set(rows)==set(CASES),'all original cases observed')
    need(rows['D-C1']['b1_calls']>0 and rows['D-C1']['multirow_calls']==0
         and rows['D-C8']['multirow_calls']>0,'actual B1 candidate and C8 multirow fallback coverage')
    for row in rows.values():
        need(row['candidate']==48*row['b1_calls'] and row['fallback']==48*row['multirow_calls'],'all-layer dispatch coverage')
    for key in ('candidate','fallback'):
        need(sum(c[key] for group in coverage['final_counters'] for c in group)==sum(c[key] for c in rows.values()),'unobserved dispatch: '+key)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true')
    parser.add_argument('--plan',type=Path);parser.add_argument('--plan-sha256');parser.add_argument('--out',type=Path)
    args=parser.parse_args(argv)
    supplied=(args.plan is not None,args.plan_sha256 is not None,args.out is not None)
    if not args.run and not any(supplied):
        print(json.dumps(dict(enabled=False,device_run=False,scope=SCOPE,
            prerequisite='real pinned independent18-call/54-pair/60-raw canary PASS',
            required='--plan PATH --plan-sha256 SHA --out NEW (--run optional)')));return 0
    if not all(supplied):parser.error('execution requires plan, plan-sha256 and out together')
    plan,proof=validate(args.plan,args.plan_sha256)
    args.out.mkdir(parents=True,exist_ok=False);write(args.out/'plan.json',plan)
    result=dict(schema='GDN-prework-serving-result-v1',status='INCOMPLETE',plan_sha256=args.plan_sha256,
                canary=proof,sources=plan['sources'],scope=SCOPE,same_schedule_checks_passed=False,
                full_model_qualified=False,http_qualified=False,performance_qualified=False,
                independent_audit_required=True,source_pins_rechecked=False)
    write(args.out/'complete.json',result)
    try:
        base=load(BASE,'_gdn_prework_serving_base');adapter=load(HERE/'candidate.py','_gdn_prework_serving_candidate')
        with (args.out/'dispatch.jsonl').open('x') as events:
            def emit(row):events.write(json.dumps(row,allow_nan=False)+'\n');events.flush()
            with adapt(base,adapter,emit,enabled=True) as coverage:
                result['coverage']=coverage
                code=base.main(['--plan',plan['qualification']['path'],'--plan-sha256',BASE_PLAN_SHA,'--out',str(args.out/'base')])
        original=read(args.out/'base/complete.json')
        result['base_result']=dict(path='base/complete.json',sha256=sha(args.out/'base/complete.json'),status=original['status'],returncode=code)
        need(code==0,'original numerical qualifier failed');require_complete(original,coverage)
        same(validate(args.plan,args.plan_sha256),(plan,proof),'plan/sources/canary evidence changed')
        result.update(status='PASS',same_schedule_checks_passed=True,source_pins_rechecked=True,total_requests=37,
                      dispatch_sha256=sha(args.out/'dispatch.jsonl'))
    except BaseException as exc:
        result.update(status='FAIL',error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        if (args.out/'base/complete.json').exists():
            result['base_result']=dict(path='base/complete.json',sha256=sha(args.out/'base/complete.json'),status=read(args.out/'base/complete.json').get('status'))
    finally:write(args.out/'complete.json',result)
    print(json.dumps(dict(status=result['status'],out=str(args.out))));return int(result['status']!='PASS')


if __name__=='__main__':raise SystemExit(main())
