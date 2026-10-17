"""Prepared E024 serving adapter qualification; no HTTP/performance claim.

CLI --plan PATH --plan-sha256 SHA --out NEW. Exact five-field plan:
schema E024-route-serving-plan-v1, qualifier_sha256 (this file), qualification
{path,sha256} (unchanged E023 37-request plan), sources {absolute source:SHA},
deadline_s=1800. Only main executes the inherited
model loader, with its original 1800s deadline, two independent models, resource
ownership external. No model/runtime import at module load or --help.

Candidate gates ON/norm OFF/short compile+fuse ON remain the inherited setting;
reference stays eager/gates OFF. E028 remains OFF. A diagnostic idle parameter
prelude extends fixed D prompt IDs by repetition, not generation or HTTP replay.
Its tokens are each model's own prefill argmax, never recorded HTTP outputs.
The original37-request loop/Comparator/semantic ahead pairing remain unchanged.
Discarded prior ahead is reported as observed coverage, not required: this
workload does not inject cancellation or sampling-key changes.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import traceback

HERE = Path(__file__).resolve().parent
BASE = HERE.parent/'pareto_realistic/qualify_native.py'
OLD = HERE.parent/'long_decode_graph/qualify.py'
BASE_PLAN_SHA = '21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb'
FIXED = {
    BASE: '26964fcbed9a2e5dcefaee4888244987897f3138d4a2b7add59bf63a7461b7bf',
    OLD: 'c24219f431b08502a26a4f7a83c002214bddd7ea3962d6b1869b313e4c13faf7',
    HERE/'candidate.py': '6531be25acb53713d6ddcd930d2c778f6640a3b1f53304fd5375d20b044525c3',
    OLD.with_name('candidate.py'): 'eb14b671a66764fe677219762090acb495b89d384cf078f3f63b918a40aa2b84',
    OLD.with_name('model-source-pins.json'): '7143538a168bfcd8b01292c9687472c8647db393c6d2a510d628f71c670260c5',
}


def need(ok, message):
    if not ok: raise ValueError(message)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_bytes())
def write(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def load(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def source_pins():
    return {**{str(p):h for p,h in FIXED.items()},str(HERE/'integration.py'):sha(HERE/'integration.py'),
            str(Path(__file__).resolve()):sha(__file__)}


def validate(path, digest):
    need(sha(path)==digest,'serving qualification plan SHA')
    plan=read(path)
    need(set(plan)=={'schema','qualifier_sha256','qualification','sources','deadline_s'}
         and plan['schema']=='E024-route-serving-plan-v1','exact serving qualification plan')
    need(plan['qualifier_sha256']==sha(__file__),'serving qualifier self SHA')
    need(type(plan['deadline_s']) is int and plan['deadline_s']==1800,'fixed serving deadline')
    q=plan['qualification']
    need(set(q)=={'path','sha256'} and Path(q['path']).is_absolute() and q['sha256']==BASE_PLAN_SHA
         and sha(q['path'])==BASE_PLAN_SHA,'unchanged E023 workload/model/configuration plan')
    need(plan['sources']==source_pins(),'exact serving adapter and inherited source pins')
    for name,value in plan['sources'].items():need(sha(name)==value,'source changed: '+name)
    original=read(q['path'])
    need(type(original['deadline_s']) is int and original['deadline_s']==1800
         and original['candidate']==dict(compile_decode=True,fuse_decode=True,
         gates_compile=True,norm_reuse=False),'original deadline/current native candidate settings')
    return plan


class RouteWitness:
    """Qualification-only actual callable counter; never calls _eligible again."""
    def __init__(self, graph, old, emit, coverage):
        need(not graph._functions,'witness attached after graph use')
        self.graph,self.emit,self.coverage=graph,emit,coverage
        self.functions=old.WitnessFunctions();graph._functions=self.functions
        self.phase='unassigned';self.step=None;self.direct_label=None

    def __call__(self, tokens, ctx):
        before=self.functions.invocations
        capacities=[(c.K.shape[2],c.V.shape[2]) for c in ctx.caches if hasattr(c,'K')]
        expected=(1<=tokens.shape[0]<=8 and 257<=ctx.max_len<=1536
                  and len(capacities)==16 and all(k>=1536 and v>=1536 for k,v in capacities))
        row=dict(event='route-call',phase=self.phase,step=self.step,direct_label=self.direct_label,
                 status='RUNNING',B=tokens.shape[0],used=ctx.max_len,
                 capacities=capacities,expected_graph=expected)
        try:
            result=self.graph(tokens,ctx)
            count=self.functions.invocations-before
            need(count==int(expected),'actual long callable/fallback does not match registered normal input domain')
            if count:
                need(all(k==v==1536 for k,v in capacities),'serving evidence only covers physical1536')
                bound=1023 if ctx.max_len<1024 else 1536
                hit=self.coverage['route_hits'].setdefault(self.phase,{'1023':0,'1536':0})
                hit[str(bound)]+=1
            else:self.coverage['long_fallback_calls']+=1
            row.update(status='SUBMITTED',actual_invocations=count,functions=len(self.functions))
            return result
        finally:self.emit(row)


def parameter_checks(base, old, engines, observers, fixtures, cmp, deadline, emit, coverage):
    """Idle same-object graph capture test; no scheduler work or hot weight reload."""
    for engine in engines:base.assert_empty(engine,cmp)
    mx=cmp.mx
    # These modules are already loaded by the real inherited loader, not at import.
    from tutorial_code.miniengine.cache import make_cache, DecodeBatch
    from tutorial_code.miniengine.context import ForwardContext
    models=[o.original_model for o in observers]
    witness=observers[1].route_witness;witness.phase='parameter-diagnostic'
    prompts=old.diagnostic_fixtures(fixtures,1023)
    emit(dict(event='parameter-inputs',prompts=prompts,history='diagnostic repeated D IDs, not generated history'))
    caches=[[make_cache(model) for _ in range(8)] for model in models]
    previous=0;baseline={}

    def flush(label):
        emit(dict(event='parameter-comparison',label=label,arrays=cmp.rows));cmp.rows=[]

    def direct(label,B,used,tokens):
        deadline();batches=[DecodeBatch(model) for model in models];contexts=[]
        for batch,arm in zip(batches,caches):
            batch.rebuild([],[(f'parameter-row-{i}',arm[i],1536) for i in range(B)])
            contexts.append(ForwardContext('decode',batch.layers,positions=mx.array([used-1]*B),
                lens=mx.array([used]*B),max_len=used))
        cmp.compare(label+'/input',base.snapshot(contexts[0]),base.snapshot(contexts[1]))
        cmp.meta(label+'/independently-computed-tokens',tokens[0][:B],tokens[1][:B])
        left=models[0](mx.array(tokens[0][:B]),contexts[0])
        witness.direct_label=label
        try:right=observers[1].route_install.decoder(mx.array(tokens[1][:B]),contexts[1])
        finally:witness.direct_label=None
        need(len(base.snapshot(contexts[0].caches)['arrays'])==128,'all64 layers required')
        cmp.compare(label+'/logits',base.snapshot(left),base.snapshot(right))
        cmp.compare(label+'/state',base.snapshot(contexts[0].caches),base.snapshot(contexts[1].caches))
        flush(label);coverage['parameter_direct_calls']+=1
        return base.snapshot(left)

    try:
        for used in (1023,1024):
            end=used-1;tokens=[[],[]]
            for i,f in enumerate(prompts):
                deadline()
                logits=[model(mx.array(f['prompt_ids'][previous:end]),ForwardContext('prefill',arm[i]),last_only=True)
                        for model,arm in zip(models,caches)]
                cmp.compare(f'parameter-prefill-{end}-{i}/logits',base.snapshot(logits[0]),base.snapshot(logits[1]))
                cmp.compare(f'parameter-prefill-{end}-{i}/state',base.snapshot(caches[0][i]),base.snapshot(caches[1][i]))
                for arm,y in zip(tokens,logits):arm.append(int(mx.argmax(y,axis=-1).item()))
                flush(f'parameter-prefill-{end}-{i}')
            retained=[base.snapshot(arm) for arm in caches]
            for n,B in enumerate((1,8,1)):
                baseline[(used,B)]=direct(f'parameter-baseline-{used}-{n}-B{B}',B,used,tokens)
            for index,path in enumerate(old.PARAMETERS):
                with old.parameter_replacement(mx,models,path,index,cmp,base.snapshot,emit):
                    for n,B in enumerate((1,8,1)):
                        label=f'parameter-changed-{path}-{used}-{n}-B{B}'
                        actual=direct(label,B,used,tokens)
                        a,b=actual['arrays']['root'],baseline[(used,B)]['arrays']['root']
                        changed=not bool(mx.array_equal(a.view(mx.uint16),b.view(mx.uint16)).item())
                        emit(dict(event='parameter-eager-change',label=label,parameter=path,used=used,B=B,changed_eager_logits=changed))
                        need(changed,'parameter change did not change eager logits')
                for n,B in enumerate((1,8,1)):
                    actual=direct(f'parameter-restored-{path}-{used}-{n}-B{B}',B,used,tokens)
                    cmp.compare(f'parameter-restored-original-{path}-{used}-{n}',baseline[(used,B)],actual)
                    flush('restored-original')
            for arm,saved in zip(caches,retained):cmp.compare('parameter-prefill-retained',saved,base.snapshot(arm))
            flush('parameter-prefill-retained');previous=end
        need(coverage['parameter_direct_calls']==42,'complete changed/restored parameter matrix')
        coverage['parameter_checks_complete']=True
    finally:
        caches.clear();baseline.clear()
        for engine in engines:base.assert_empty(engine,cmp)


@contextmanager
def adapt(base, integration, old, fixtures, emit):
    original_observer,original_run=base.Observer,base.run_case
    coverage=dict(installs=0,restores=0,parameter_checks_complete=False,parameter_direct_calls=0,
        route_hits={},long_fallback_calls=0,cases={},identity_refill_enabled=False)

    class RouteObserver(original_observer):
        def __init__(self,engine,cmp,label):
            self.route_install=self.route_witness=None;self.case='unassigned';self.case_step=0;self.closed=False
            runner,scheduler=engine.runner,engine.scheduler
            previous_model,previous_decode=runner.model,runner.decode_model
            runner_run=('run' in vars(runner),vars(runner).get('run'))
            scheduler_call=('schedule' in vars(scheduler),vars(scheduler).get('schedule'))
            try:
                if label=='candidate':
                    self.route_install=integration.install(runner,enabled=True)
                    self.route_witness=RouteWitness(self.route_install.decoder.long,old,emit,coverage)
                    self.route_install.decoder.long=self.route_witness;coverage['installs']+=1
                super().__init__(engine,cmp,label)
            except BaseException:
                # Constructor failure may occur after CallProxy/instance-method
                # installation. No observer was returned to base.release_pair.
                runner.model=previous_model
                runner.decode_model=self.route_install.decoder if self.route_install is not None else previous_decode
                for target,name,saved in ((runner,'run',runner_run),(scheduler,'schedule',scheduler_call)):
                    if saved[0]:setattr(target,name,saved[1])
                    else:vars(target).pop(name,None)
                if self.route_install is not None:self.route_install.restore()
                self.engine=self.original_model=self.original_decode=self.original_run=self.original_schedule=None
                self.route_install=self.route_witness=None
                raise

        def begin(self,expected_schedule=None):
            super().begin(expected_schedule)
            self.case_step+=1
            if self.route_witness:
                self.route_witness.step=self.case_step
                self.route_witness.direct_label=None

        def run(self,out):
            old_ids=list(self.engine.runner.batch.req_ids);decode=[r.req_id for r in out.decode]
            self.refilled=bool(old_ids and all(rid in decode for rid in old_ids) and any(rid not in old_ids for rid in decode))
            result=super().run(out)
            self.before_retirement=list(self.engine.runner.batch.req_ids)
            return result

        def finish(self):
            result=super().finish()
            if self.label=='candidate':
                counts=coverage['cases'].setdefault(self.case,dict(consumed=0,discarded_prior=0,partial_retirement=0,refills=0))
                counts['consumed']+=int(result['consumed_prior_future'] is not None)
                counts['discarded_prior']+=int(result['discarded_prior_future'] is not None)
                after=self.engine.runner.batch.req_ids
                counts['partial_retirement']+=int(bool(after) and bool(set(self.before_retirement)-set(after)))
                counts['refills']+=int(self.refilled)
            return result

        def restore(self):
            if self.closed:return
            super().restore()
            if self.route_install is not None:self.route_install.restore()
            self.calls.clear();self.previous_future=self.postrun=None
            self.engine=self.original_model=self.original_decode=self.original_run=self.original_schedule=None
            self.route_install=self.route_witness=None;self.closed=True;coverage['restores']+=1

    def run(case,engines,observers,sampling,runner_module,cmp,ledger,deadline):
        if not coverage['parameter_checks_complete']:
            need(case['name']=='D-C1','parameter check must precede requests')
            parameter_checks(base,old,engines,observers,fixtures,cmp,deadline,emit,coverage)
        for observer in observers:
            observer.case=case['name'];observer.case_step=0
            if observer.route_witness:
                observer.route_witness.phase=case['name'];observer.route_witness.step=None
        return original_run(case,engines,observers,sampling,runner_module,cmp,ledger,deadline)

    base.Observer,base.run_case=RouteObserver,run
    try:yield coverage
    finally:base.Observer,base.run_case=original_observer,original_run


def require_complete(result,coverage):
    need(result['status']=='PASS' and result['total_requests']==37 and result['source_model_pins_rechecked'] is True,'original37-request/source/model gate')
    need([(c['name'],len(c['completed'])) for c in result['cases']]==[('D-C1',8),('D-C8',16),('lifecycle',13)],'unchanged workload completeness')
    need(coverage['installs']==2 and coverage['restores']==4 and coverage['parameter_checks_complete'] is True
         and coverage['parameter_direct_calls']==42,'installation/release/parameter coverage')
    for name in ('D-C1','D-C8'):
        need(all(coverage['route_hits'].get(name,{}).get(bound,0)>0 for bound in ('1023','1536')),'both actual graph families required in '+name)
        need(coverage['cases'].get(name,{}).get('consumed',0)>0,'actual consumed ahead required in '+name)
    need(sum(row['partial_retirement'] for row in coverage['cases'].values())>0,'actual partial retirement required')
    need(coverage['cases']['D-C8']['refills']>0,'actual C8 survivor refill required')
    discarded=sum(row['discarded_prior'] for row in coverage['cases'].values())
    return dict(discarded_ahead_observed=discarded>0,discarded_prior_ahead_count=discarded)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--plan-sha256',required=True)
    ap.add_argument('--out',type=Path,required=True);args=ap.parse_args(argv)
    plan=validate(args.plan,args.plan_sha256);args.out.mkdir(parents=True,exist_ok=False);write(args.out/'plan.json',plan)
    result=dict(schema='E024-route-serving-result-v1',status='INCOMPLETE',plan_sha256=args.plan_sha256,
        sources=plan['sources'],same_schedule_qualified=False,http_qualified=False,performance_qualified=False,
        source_pins_rechecked=False,identity_refill_enabled=False)
    write(args.out/'complete.json',result)
    try:
        base=load(BASE,'_e024_route_serving_base');old=load(OLD,'_e024_route_parameter_helpers')
        integration=load(HERE/'integration.py','_e024_route_serving_install')
        fixtures=base.pinned_json(read(plan['qualification']['path'])['fixtures'])['fixtures']
        with (args.out/'route-events.jsonl').open('x') as events:
            def emit(row):events.write(json.dumps(row,allow_nan=False)+'\n');events.flush()
            with adapt(base,integration,old,fixtures,emit) as coverage:
                result['coverage']=coverage
                code=base.main(['--plan',plan['qualification']['path'],'--plan-sha256',BASE_PLAN_SHA,'--out',str(args.out/'base')])
        path=args.out/'base/complete.json';completed=read(path)
        result['base_result']=dict(path='base/complete.json',sha256=sha(path),status=completed['status'],returncode=code)
        need(code==0,'base numerical qualification failed');observed=require_complete(completed,coverage)
        need(validate(args.plan,args.plan_sha256)==plan,'source/plan changed')
        eos=sorted(set(fixtures[0]['eos_token_ids']))
        eos_count=sum(bool(r['raw_ids']) and r['raw_ids'][-1] in eos for case in completed['cases'] for r in case['completed'])
        result.update(status='PASS',same_schedule_qualified=True,source_pins_rechecked=True,total_requests=37,
            **observed,
            observed_eos_requests=eos_count,eos_retirement_observed=eos_count>0,
            scope='Original37 same-schedule full-logit/state and actual ahead/retirement checks plus changed-value parameter diagnostic. Independent audit can reconstruct scheduling and verify complete device comparison receipts; successful tensor bytes are not saved for offline arithmetic rechecking. Route calls carry a CPU step/direct label and ordered inference, not an explicit token key. Normal EOS enabled; actual EOS and discarded-prior-ahead coverage reported. Same-schedule scope excludes unobserved ahead invalidation paths; cancellation and sampling-key changes are not injected or qualified. Not HTTP capture/replay, performance, hot reload or combined E028 qualification.')
    except BaseException as exc:
        result.update(status='FAIL',error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        if (args.out/'base/complete.json').is_file():result['base_result']=dict(path='base/complete.json',sha256=sha(args.out/'base/complete.json'),status=read(args.out/'base/complete.json').get('status'))
    finally:write(args.out/'complete.json',result)
    print(json.dumps(dict(status=result['status'],out=str(args.out))),flush=True)
    return int(result['status']!='PASS')


if __name__=='__main__':raise SystemExit(main())
