"""E028 full-model numerical adapter; reuses the frozen E023 37-request run.

CLI: --plan PLAN --plan-sha256 SHA --out NEW. Plan has exactly schema,
qualification:{path,sha256} (the original E023 plan), and sources:{abs:sha}
for this file, base qualifier, integration and candidate. No runtime import on
help/validation. The base owns the unchanged 1800s model/trace deadline; an
external resource owner must enforce its hard process deadline.

Only a privately loaded qualifier's Observer/pair_calls are replaced. The
instance _sync_batch observer captures identity refills after synchronization
and before decode; independent view handles retain all 64 layers/128 arrays.
This extra retention is qualification overhead, never a timing experiment.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import traceback

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'pareto_realistic/qualify_native.py'
BASE_SHA = '26964fcbed9a2e5dcefaee4888244987897f3138d4a2b7add59bf63a7461b7bf'
BASE_PLAN_SHA = '21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb'
FIXED = {BASE: BASE_SHA,
         HERE/'integration.py': 'ea0d6cb6af7c80d7580b4ab1c8319a4b76110fd1d73cf660c048c8dd45cf0c52',
         HERE/'candidate.py': 'bb9f8a461625e97c80e75140c9d9b7b8718a633595ab4f3a422267e752ded8f5'}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def load(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def validate(plan_path, expected_sha):
    need(sha(plan_path)==expected_sha,'E028 plan SHA mismatch')
    plan=json.loads(Path(plan_path).read_text())
    need(set(plan)=={'schema','qualification','sources'} and plan['schema']=='E028-model-qualification-plan-v1','E028 plan schema')
    q=plan['qualification']
    need(set(q)=={'path','sha256'} and Path(q['path']).is_absolute() and q['sha256']==BASE_PLAN_SHA
         and sha(q['path'])==BASE_PLAN_SHA,'unchanged E023 plan required')
    expected={str(p):h for p,h in FIXED.items()}
    expected[str(Path(__file__).resolve())]=sha(__file__)
    need(plan['sources']==expected,'exact adapter/base/integration/candidate source pins required')
    for p,h in expected.items():need(sha(p)==h,'source changed: '+p)
    original=json.loads(Path(q['path']).read_text())
    need(original['deadline_s']==1800,'original 1800s deadline required')
    return plan


def refill_transition(runner, decode):
    """Plain metadata only, mirroring the existing sync's membership choice."""
    old=list(runner.batch.req_ids);positions={rid:i for i,rid in enumerate(old)}
    keep=[positions[r.req_id] for r in decode if r.req_id in positions]
    new=[r for r in decode if r.req_id not in positions]
    if not old or not new or keep!=list(range(len(old))):return None
    kv=[layer for layer in runner.batch.layers if hasattr(layer,'K')]
    need(len(kv)==16 and len(runner.batch.layers)==64,'expected 16 attention/48 GDN layers at refill')
    need_tokens=max([r.num_prompt_tokens+r.sampling.max_tokens for r in new]+[kv[0].K.shape[2]])
    L=(need_tokens+runner.batch.step-1)//runner.batch.step*runner.batch.step
    if any(layer.K.shape[2]!=L or layer.V.shape[2]!=L for layer in kv):return None
    all_ids=old+[r.req_id for r in new]
    cases=[name for prefix,name in (('c1-','D-C1'),('c8-','D-C8'),('probe-','lifecycle'))
           if all(rid.startswith(prefix) for rid in all_ids)]
    need(len(cases)==1,'refill belongs to exactly one unchanged workload')
    return dict(case=cases[0],old_ids=old,old_lens=list(runner.batch.lens),keep=keep,
                new_ids=[r.req_id for r in new],decode_ids=[r.req_id for r in decode],capacity=L)


@contextmanager
def observe_refills(base, integration, emit):
    """Patch only this private base module; never a production class/global."""
    original_observer,original_pair=base.Observer,base.pair_calls
    coverage=dict(candidate_installs=0,paired_identity_refills=0,compared_arrays=0,
                  per_case={},restored_observers=0,candidate_final_counters=[])

    class RefillObserver(original_observer):
        def __init__(self,engine,comparator,label):
            if label=='candidate':
                integration.install(engine.runner,enabled=True)
                coverage['candidate_installs']+=1
            super().__init__(engine,comparator,label)
            self.refills=[];self.refill_step=0;self.refill_closed=False
            self.sync_was_instance='_sync_batch' in vars(engine.runner)
            self.original_sync=engine.runner._sync_batch
            engine.runner._sync_batch=self.sync_refill

        def begin(self,expected_schedule=None):
            need(not self.refills,'unpaired refill snapshot would be discarded')
            self.refill_step+=1
            return super().begin(expected_schedule)

        def sync_refill(self,decode):
            r=self.engine.runner;transition=refill_transition(r,decode)
            before=dict(r.batch.counters) if self.label=='candidate' else None
            result=self.original_sync(decode)
            if transition is not None:
                witness=None
                if self.label=='candidate':
                    after=dict(r.batch.counters)
                    witness=dict(before=before,after=after,path=r.batch.last_path,reason=r.batch.last_reason)
                    if (after['identity_refill']-before['identity_refill']!=1
                            or after['fallback']!=before['fallback'] or r.batch.last_path!='identity_refill'):
                        self.cmp.fail('E028 identity transition silently fell back',witness)
                snap=base.snapshot(dict(req_ids=list(r.batch.req_ids),lens=list(r.batch.lens),layers=r.batch.layers))
                if len(r.batch.layers)!=64 or len(snap['arrays'])!=128:
                    self.cmp.fail('E028 refill must capture every layer/array',dict(layers=len(r.batch.layers),arrays=len(snap['arrays'])))
                self.refills.append(dict(transition=transition,snapshot=snap,witness=witness,step=self.refill_step))
            return result

        def restore(self):
            if self.refill_closed:return
            r=self.engine.runner
            try:
                if self.label=='candidate':coverage['candidate_final_counters'].append(dict(r.batch.counters))
                super().restore()
            finally:
                if self.sync_was_instance:r._sync_batch=self.original_sync
                else:vars(r).pop('_sync_batch',None)
                self.refills.clear();self.original_sync=None;self.refill_closed=True
                # Base release_pair subsequently clears calls/postrun/future. Clear
                # all strong observer links too, even if a caller retains this wrapper.
                self.calls.clear();self.previous_future=self.postrun=None
                self.engine=self.original_model=self.original_decode=None
                self.original_run=self.original_schedule=None
                coverage['restored_observers']+=1

    def pair(reference,candidate,cmp):
        try:
            cmp.meta('E028 refill count',len(reference.refills),len(candidate.refills))
            for left,right in zip(reference.refills,candidate.refills):
                event=dict(event='identity-refill',status='FAIL',observer_step=left['step'],transition=left['transition'],candidate_witness=right['witness'])
                try:
                    cmp.meta('E028 refill transition',left['transition'],right['transition'])
                    cmp.meta('E028 refill observer step',left['step'],right['step'])
                    cmp.compare('E028-refill-predecode-state '+repr(left['transition']['decode_ids']),left['snapshot'],right['snapshot'])
                    coverage['paired_identity_refills']+=1;coverage['compared_arrays']+=128
                    case=left['transition']['case'];coverage['per_case'][case]=coverage['per_case'].get(case,0)+1
                    event.update(status='PASS',arrays=128)
                finally:emit(event)
            return original_pair(reference,candidate,cmp)
        finally:
            reference.refills.clear();candidate.refills.clear()

    base.Observer,base.pair_calls=RefillObserver,pair
    try:yield coverage
    finally:base.Observer,base.pair_calls=original_observer,original_pair


def require_complete(result,coverage):
    need(result['status']=='PASS' and result['total_requests']==37 and result['source_model_pins_rechecked'] is True,'base full 37-request gate did not pass')
    need([(c['name'],len(c['completed'])) for c in result['cases']]==[('D-C1',8),('D-C8',16),('lifecycle',13)],'unchanged complete workload required')
    hits=coverage['paired_identity_refills']
    need(hits>0 and coverage['compared_arrays']==128*hits,'no actual fully compared E028 identity refill')
    need(coverage['per_case'].get('D-C8',0)>=1 and coverage['per_case'].get('D-C1',0)==0,
         'D-C8 actual refill required; D-C1 must not claim refill coverage')
    need(coverage['candidate_installs']==2 and coverage['restored_observers']==4
         and len(coverage['candidate_final_counters'])==2,'all model pairs installed/restored')
    need(sum(c['identity_refill'] for c in coverage['candidate_final_counters'])==hits,'unobserved candidate identity branch')


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--plan-sha256',required=True)
    ap.add_argument('--out',type=Path,required=True);args=ap.parse_args(argv)
    plan=validate(args.plan,args.plan_sha256)
    args.out.mkdir(parents=True,exist_ok=False);write(args.out/'plan.json',plan)
    result=dict(schema='E028-model-qualification-result-v1',status='INCOMPLETE',plan_sha256=args.plan_sha256,
                sources=plan['sources'],numerical_only_not_http=True,performance_qualified=False,source_pins_rechecked=False)
    write(args.out/'complete.json',result)
    try:
        base=load(BASE,'_e028_private_qualifier');integration=load(HERE/'integration.py','_e028_private_integration')
        with (args.out/'refills.jsonl').open('x') as events:
            def emit(row):events.write(json.dumps(row,allow_nan=False)+'\n');events.flush()
            with observe_refills(base,integration,emit) as coverage:
                result['coverage']=coverage
                code=base.main(['--plan',plan['qualification']['path'],'--plan-sha256',BASE_PLAN_SHA,'--out',str(args.out/'base')])
        base_result=json.loads((args.out/'base/complete.json').read_text())
        result['base_result']=dict(path='base/complete.json',sha256=sha(args.out/'base/complete.json'),status=base_result['status'],returncode=code)
        need(code==0,'base qualification returned failure')
        require_complete(base_result,coverage)
        need(validate(args.plan,args.plan_sha256)==plan,'source/plan changed')
        result.update(status='PASS',source_pins_rechecked=True,total_requests=37,
                      scope='Original 37-request same-schedule eager versus current native stack plus E028; all base gates plus full refill states. Not live HTTP/performance.')
    except BaseException as error:
        result.update(status='FAIL',error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        if (args.out/'base/complete.json').is_file():
            result['base_result']=dict(path='base/complete.json',sha256=sha(args.out/'base/complete.json'),status=json.loads((args.out/'base/complete.json').read_text()).get('status'))
    finally:write(args.out/'complete.json',result)
    print(json.dumps(dict(status=result['status'],out=str(args.out))),flush=True)
    return int(result['status']!='PASS')


if __name__=='__main__':raise SystemExit(main())
