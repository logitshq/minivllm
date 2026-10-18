"""CPU mechanisms only; prepared source, not device/numerical qualification.

No real backend in these tests. Array values are tiny fakes; the server-side
observer test deliberately forbids tolist/eval. Capture uses the frozen fake
Popen backend, original spec builder and actual cleanup path. Replay uses the
real frozen GDN adapter/Observer control flow with explicitly fake model math.
"""
import asyncio
import ast
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bench.m5_ultra_256gb.gdn_prework_http import http_server as server
from bench.m5_ultra_256gb.gdn_prework_http import http_capture as cap
from bench.m5_ultra_256gb.gdn_prework_http import http_qualification as q
from bench.m5_ultra_256gb.gdn_prework_http import audit as a
from bench.m5_ultra_256gb.gdn_prework_unbound import serving
from bench.m5_ultra_256gb.gdn_prework_unbound import test_serving_cpu as fake
from bench.m5_ultra_256gb.pareto_realistic.test_http_capture_v2_cpu import Backend as OriginalBackend


def stream(rows): return io.BytesIO(b''.join((json.dumps(r)+'\n').encode() for r in rows))
def ref(path): return dict(path=str(Path(path).resolve()),sha256=q.sha(path))

def event(B=1,used=900,step=1,C=1,mode='decode'):
    path='prefill-untouched' if mode=='prefill' else 'short-packed-bypass' if used<=256 else 'B1-candidate' if B==1 else 'B>1-fallback'
    return dict(event='dispatch-call',status='SUBMITTED',phase=f'HTTP-C{C}',step=step,mode=mode,B=B,used=used,
                path=path,candidate=48 if path=='B1-candidate' else 0,fallback=48 if path=='B>1-fallback' else 0)

class PoisonTensor:
    def __init__(self,B):self.shape=(B,)
    def tolist(self):raise AssertionError('capture observer must not read array values')
    def eval(self):raise AssertionError('capture observer must not evaluate arrays')

class Backend(OriginalBackend):
    """CPU-only original capture fake; emits new actual-entry dispatch header."""
    def __init__(self,spec,*,dispatch='valid',**kwargs):super().__init__(spec,**kwargs);self.dispatch=dispatch
    def spawn(self,command,environment,stdout,stderr):
        self.started=True
        assert command[:-4]==self.spec['command'] and command[3]==str(cap.SERVER)
        assert command[-4]=='--qualification-trace' and command[-2]=='--dispatch-events'
        assert environment==self.spec['environment']
        Path(command[-3]).write_text(json.dumps(dict(event='startup',identity=self.spec['identity']))+'\n')
        if self.dispatch!='missing':
            row=dict(event='dispatch-startup',schema=cap.DISPATCH_SCHEMA,pid=self.child.pid,
                     phase=f"HTTP-C{self.spec['concurrency']}",identity=self.spec['identity'])
            if self.dispatch=='wrong-pid':row['pid']+=1
            Path(command[-1]).write_text(json.dumps(row)+'\n')
        stdout.write(b'CPU fake direct server\n');stderr.write(b'CPU fake diagnostic\n');return self.child
    async def identity(self,pid):
        return {**await super().identity(pid),'command':['/bin/ps','-p',str(pid),'-o','pid=,ppid=,lstart=,comm=']}
    async def absent(self,pid):
        return {**await super().absent(pid),'command':['/bin/ps','-p',str(pid),'-o','pid=']}
    async def port(self,port,pid=None):
        return {**await super().port(port,pid),'command':['/usr/sbin/lsof','-nP','-a']+
                (['-p',str(pid)] if pid is not None else [])+[f'-iTCP:{port}','-sTCP:LISTEN','-Fpn']}


def capture_spec(C=8):
    p=cap.donor.HERE/'prereg-qualification-v1/plan.json';n=q.read(p)
    return cap.prepare(p,q.BASE_PLAN_SHA,C,python=n['execution']['python'],environment=n['execution']['environment'],
        transport=dict(port=18181,tcp_nodelay=True,io_timeout_seconds=300),
        deadlines=dict(ready=1,request=1,group=1,close=.05,term=.05,kill=.05,overall=3,resource_interval=.01))

class ServerTests(unittest.TestCase):
    def test_off_does_not_inspect_engine_and_partial_install_restores(self):
        class Poison:
            def __getattribute__(self,name):raise AssertionError(name)
        self.assertIsNone(server.install(Poison(),Poison()))
        e=fake.Engine();e.requests={};e.runner.spec=None;adapter=fake.Adapter(fail_at=7)
        with self.assertRaisesRegex(RuntimeError,'partial install'):server.install(e,adapter,enabled=True)
        self.assertEqual(adapter.closed,list(reversed(range(7))))
        self.assertTrue(all('_decode' not in vars(l.linear_attn) for l in e.model.layers if not l.is_attn))

    def test_actual_proxy_calls_no_tensor_reads_current_future_and_multibatch(self):
        e=fake.Engine();e.requests={};e.runner.spec=None;adapter=fake.Adapter()
        installed=server.install(e,adapter,enabled=True);trace=io.StringIO();events=io.StringIO();old=e.runner.model
        donor=server.load_fixed(server.DONOR,'_gdn_http_test_donor_no_devices')
        observed=server.QualificationObserver(e,installed,donor,trace,events,'HTTP-C8',serving)
        observed.startup({'CPU':'only'});observed.trace({'event':'before_step'})
        first=e.runner.decode_model(PoisonTensor(1),fake.ctx(1,900))
        second=e.runner.decode_model(PoisonTensor(1),fake.ctx(1,901))
        self.assertEqual(first.shape,second.shape)
        observed.trace({'event':'before_step'});e.runner.decode_model(PoisonTensor(8),fake.ctx(8,902))
        e.runner.model(PoisonTensor(19),SimpleNamespace(mode='prefill',max_len=0))
        rows=[json.loads(r) for r in events.getvalue().splitlines()][1:]
        self.assertEqual([r['step'] for r in rows],[1,1,2,2])
        self.assertEqual([r['candidate'] for r in rows],[48,48,0,0])
        self.assertEqual([r['fallback'] for r in rows],[0,0,48,0])
        observed.restore();installed.restore();self.assertIs(e.runner.model,old)
        self.assertIsNone(observed.runner)

    def test_observer_constructor_rollback_and_cleanup_attempts_both_scopes(self):
        e=fake.Engine();old=e.runner.model
        def broken(engine,trace):engine.runner.run=lambda out:None;raise RuntimeError('observer setup')
        with self.assertRaisesRegex(RuntimeError,'observer setup'):
            server.QualificationObserver(e,None,SimpleNamespace(observe_runner=broken),io.StringIO(),None,'HTTP-C1',None)
        self.assertIs(e.runner.model,old);self.assertNotIn('run',vars(e.runner))
        seen=[]
        def fail():seen.append('observer');raise RuntimeError('first error')
        with self.assertRaisesRegex(RuntimeError,'first error'):
            server.restore_scopes(SimpleNamespace(restore=fail),SimpleNamespace(restore=lambda:seen.append('installation')))
        self.assertEqual(seen,['observer','installation'])

    def test_failed_forward_is_recorded_and_prefill_batch_explicitly_rejected(self):
        e=fake.Engine();e.requests={};e.runner.spec=None;installed=server.install(e,fake.Adapter(),enabled=True)
        donor=SimpleNamespace(observe_runner=lambda engine,trace:setattr(engine.runner,'run',lambda out:None));output=io.StringIO()
        observed=server.QualificationObserver(e,installed,donor,io.StringIO(),output,'HTTP-C1',serving)
        observed.trace({'event':'before_step'})
        def fail(*args,**kwargs):raise RuntimeError('actual forward failed')
        with self.assertRaisesRegex(RuntimeError,'actual forward'):observed.call(fail,PoisonTensor(1),fake.ctx(),{})
        self.assertEqual(json.loads(output.getvalue())['status'],'FAILED')
        with self.assertRaises(ValueError):
            observed.call(lambda *a,**k:None,PoisonTensor(2),SimpleNamespace(mode='prefill_batch',max_len=0),{})
        observed.restore();installed.restore()


class ReplayTests(unittest.TestCase):
    def test_http_wrapper_then_gdn_adapt_preserves_case_setup_and_original_arguments(self):
        base=fake.base_module();original=(base.Observer,base.workloads,base.run_case)
        recording={'concurrency':1,'jobs':{'CPU':'recorded tokens not model inputs'}};fixtures=['CPU fixtures'];calls=[]
        engines=[fake.Engine(),fake.Engine()];adapter=fake.Adapter()
        def recorded(*args,**kwargs):
            calls.append((args,kwargs));observers=args[2]
            self.assertEqual([o.case for o in observers],['D-C1','D-C1']);self.assertEqual([o.step for o in observers],[0,0])
            observers[1].begin();engines[1].runner.model(fake.Tensor([8]),fake.ctx(1,900))
            return 'original-replay-result'
        events=[]
        with q.adapted(base,SimpleNamespace(run_recorded_case=recorded),serving,adapter,recording,fixtures,
                       SimpleNamespace(emit=events.append)) as coverage:
            observers=[base.Observer(e,object(),label) for e,label in zip(engines,('reference','candidate'))]
            try:
                case=base.workloads(fixtures)[0]
                self.assertEqual(base.run_case(case,engines,observers,'sampling','runner','cmp','ledger','deadline'),'original-replay-result')
            finally:
                for observer in observers:observer.restore()
        self.assertEqual((base.Observer,base.workloads,base.run_case),original)
        self.assertIs(calls[0][0][0],recording);self.assertEqual(calls[0][0][3:],('sampling','runner','cmp','ledger','deadline'))
        self.assertIs(calls[0][1]['base'],base);self.assertEqual(events[0]['case'],'D-C1')
        self.assertEqual(coverage['layer_restores'],48)

    def test_actual_closed_37_receipt_is_prerequisite_not_new_http_qualification(self):
        repo=q.HERE.parents[2]
        audit_path=repo/'docs/experiments/m5-ultra-256gb/E029-gdn-prework-serving-v1/independent-numerical.json'
        original=q.read(audit_path);child=Path(original['child_directory'])
        refs=dict(audit=ref(audit_path),plan=ref(child/'plan.json'),complete=ref(child/'complete.json'))
        result=q.serving_gate(refs)
        self.assertEqual(result['audit_sha256'],q.SERVING_AUDIT_SHA)
        self.assertFalse(original['http_qualified'] or original['performance_qualified'])
        with tempfile.TemporaryDirectory() as temp:
            altered=Path(temp).resolve()/'audit.json';q.write(altered,{**original,'http_qualified':True})
            with self.assertRaisesRegex(ValueError,'immutable actual GDN37'):
                q.serving_gate({**refs,'audit':ref(altered)})

    def test_http_adapter_unwinds_selection_after_replay_error(self):
        base=fake.base_module();old=(base.Observer,base.workloads,base.run_case)
        def fail(*args,**kwargs):raise RuntimeError('real replay failed')
        with self.assertRaisesRegex(RuntimeError,'real replay'):
            with q.adapted(base,SimpleNamespace(run_recorded_case=fail),serving,fake.Adapter(),{'concurrency':8},[],SimpleNamespace(emit=lambda r:None)):
                base.run_case(base.workloads([])[0],[],[],None,None,None,None,None)
        self.assertEqual((base.Observer,base.workloads,base.run_case),old)

    def test_dispatch_c8_fallback_qualifies_routing_without_false_candidate_hit(self):
        header=dict(event='dispatch-startup',schema=server.DISPATCH_SCHEMA,pid=12345,phase='HTTP-C8',identity={})
        row=event(B=8,C=8);raw=stream([header,row]);digest=q.hashlib.sha256(raw.getvalue()).hexdigest();output=io.StringIO()
        dispatch=q.CapturedDispatch(raw,header,output,{'path':'/CPU/cap','sha256':digest},'HTTP-C8')
        actual={k:v for k,v in row.items() if k not in ('phase','event')};actual.update(event='prework-model-call',case='D-C8',semantic_key=['unused'])
        dispatch.emit(actual);stats=dispatch.finish(digest)
        self.assertEqual(stats['counts']['candidate'],0);self.assertEqual(stats['counts']['fallback'],48)
        coverage=a.expected_coverage(8,stats['counts'],1);q.require_coverage(coverage,8,stats['counts'])
        coverage['final_counters'][0][7]['fallback']=2
        with self.assertRaisesRegex(ValueError,'unobserved'):q.require_coverage(coverage,8,stats['counts'])

    def test_replay_mismatch_saved_before_stop_and_extra_capture_rejected(self):
        header=dict(event='dispatch-startup',pid=12345,phase='HTTP-C1');row=event();raw=stream([header,row,row]);out=io.StringIO()
        dispatch=q.CapturedDispatch(raw,header,out,{'CPU':True},'HTTP-C1')
        wrong={**row,'candidate':47,'case':'D-C1'}
        with self.assertRaisesRegex(ValueError,'dispatch differs'):dispatch.emit(wrong)
        self.assertEqual(json.loads(out.getvalue().splitlines()[-1])['candidate'],47)
        raw=stream([header,row,row]);dispatch=q.CapturedDispatch(raw,header,io.StringIO(),{'CPU':True},'HTTP-C1')
        dispatch.emit({**row,'case':'D-C1'})
        with self.assertRaisesRegex(ValueError,'extra'):dispatch.finish(q.hashlib.sha256(raw.getvalue()).hexdigest())


class CaptureTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_spec_jobs_identity_and_complete_capture_bind(self):
        for C,n in ((1,8),(8,16)):
            spec=capture_spec(C);cap.validate(spec)
            self.assertEqual(len(spec['jobs']),n);self.assertEqual(spec['command'][-4:],['--phase','qualification','--gdn-prework','on'])
            identity=spec['identity']['gdn_prework'];self.assertFalse(identity['route_graph_enabled'] or identity['identity_refill_enabled'])
            self.assertEqual(identity['installed_layers'],48)
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp).resolve()/'capture';backend=Backend(spec)
            async def check(*args):pass
            result=await cap.capture(spec,out,checkpoint=check,backend=backend)
            self.assertEqual(result['status'],'PASS');self.assertEqual(backend.child.actions,['TERM'])
            plan=dict(concurrency=8,qualification=spec['base_plan'],capture={k:ref(out/n) for k,n in q.CAPTURE_NAMES.items()},
                      inputs={k:ref(out/n) for k,n in q.INPUT_NAMES.items()})
            got,header=a.capture_binding(q,plan);self.assertEqual(got,spec);self.assertEqual(header['pid'],backend.child.pid)
            self.assertFalse(result['numerical_qualified'] or result['performance_qualified'])

    async def test_wrong_startup_missing_dispatch_and_resource_abort_reap(self):
        for corrupt in ('wrong-pid','missing','pressure'):
            with tempfile.TemporaryDirectory() as temp:
                spec=capture_spec();backend=Backend(spec,dispatch=corrupt,block=corrupt=='pressure')
                async def check(label,pid):
                    if corrupt=='pressure' and label=='capturing':raise RuntimeError('CPU resource abort')
                result=await cap.capture(spec,Path(temp).resolve()/'capture',checkpoint=check,backend=backend)
                self.assertEqual(result['status'],'FAIL');self.assertEqual(backend.child.actions,['TERM']);self.assertTrue(backend.waited)

    async def test_rehashed_wrong_pid_probe_cannot_authorize_capture(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp).resolve()/'capture';spec=capture_spec();backend=Backend(spec)
            async def check(*args):pass
            await cap.capture(spec,out,checkpoint=check,backend=backend)
            raw=q.read(out/'child-absence.json');raw['command'][2]=str(backend.child.pid+1);q.write(out/'child-absence.json',raw)
            terminal=q.read(out/'complete.json');terminal['artifacts']['child-absence.json']=dict(sha256=q.sha(out/'child-absence.json'),bytes=(out/'child-absence.json').stat().st_size)
            q.write(out/'complete.json',terminal)
            plan=dict(concurrency=8,qualification=spec['base_plan'],capture={k:ref(out/n) for k,n in q.CAPTURE_NAMES.items()},
                      inputs={k:ref(out/n) for k,n in q.INPUT_NAMES.items()})
            with self.assertRaisesRegex(ValueError,'actual PID/port probe command'):a.capture_binding(q,plan)


class AuditTests(unittest.TestCase):
    def dual(self,rows,C=8):
        header=dict(event='dispatch-startup',schema=server.DISPATCH_SCHEMA,pid=12345,phase=f'HTTP-C{C}',identity={})
        source=dict(path='/CPU/captured.jsonl',sha256='a'*64)
        replay=dict(event='replay-dispatch-startup',schema='GDN-prework-http-replay-events-v1',pid=12346,
                    phase=header['phase'],captured_dispatch_events=source)
        return a.DualDispatch(stream([header,*rows]),stream([replay,*rows]),header,source)

    def test_identically_wrong_dispatch_does_not_pass_independent_expectations(self):
        good=event(B=8,C=8);key=['decode',['x']*8,[899]*8,[7]*8]
        for field,value in (('candidate',48),('fallback',0),('B',1),('used',901),('step',True),('path','B1-candidate'),('status','RUNNING')):
            dispatch=self.dual([{**good,field:value}])
            with self.subTest(field=field),self.assertRaises(ValueError):dispatch.call('HTTP-C8',1,key)
        dispatch=self.dual([good]);dispatch.call('HTTP-C8',1,key);self.assertEqual(dispatch.finish(8)['calls'],1)

    def test_consumed_ahead_skips_only_consumed_actual_invocation(self):
        class Parent:
            def check_arrays(self,*args):pass
        current=['decode',['r'],[899],[7]];future=['decode',['r'],[900],[8]]
        for consumed in (None,current):
            events=[] if consumed else [event()];events.append(event(used=901))
            dispatch=self.dual(events,1);coverage={};case=a.dispatch_case_type(SimpleNamespace(CaseAudit=Parent),dispatch,coverage)()
            case.name='HTTP-C1';case.step=1
            row=dict(ahead={'candidate':dict(consumed_prior_future=consumed,discarded_prior_future=None)},
                     schedule={'decode':['r']},after={'engine':{'batch_rids':['r']}})
            case.check_arrays(row,[current,future],set(),True,1)
            self.assertEqual(dispatch.finish(1)['calls'],1 if consumed else 2)

    def test_inherited_full64_bits_finite_and_missing_release_reject(self):
        from bench.m5_ultra_256gb.pareto_realistic import test_audit_http_replay_v2_cpu as numeric
        http=a.load(a.HTTP_AUDIT,a.HTTP_AUDIT_SHA)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'steps.jsonl'
            for corrupt in (None,'bits','finite','layer','release'):
                base,recording,fixtures,config,claimed,rows=numeric.synthetic(8)
                dispatch=SimpleNamespace(call=lambda *args:None)
                facade=SimpleNamespace(**{**vars(base),'CaseAudit':a.dispatch_case_type(base,dispatch,{})})
                if corrupt=='bits':rows[2]['arrays'][0]['bit_equal']=False
                elif corrupt=='finite':rows[2]['arrays'][0]['candidate_finite']=False
                elif corrupt=='layer':rows[2]['arrays'].pop()
                elif corrupt=='release':rows.pop()
                path.write_bytes(stream(rows).getvalue())
                if corrupt:
                    with self.subTest(corrupt=corrupt),self.assertRaises(ValueError):http.stream_ledger(path,recording,fixtures,config,claimed,facade)
                else:self.assertEqual(http.stream_ledger(path,recording,fixtures,config,claimed,facade)['requests'],16)

    def test_active_or_failed_terminal_never_opens_ledger(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'base').mkdir();(root/'base/steps.jsonl').write_text('never open this\n')
            for status,expected in (('INCOMPLETE','INCOMPLETE'),('PREPARED','INCOMPLETE'),('FAIL','FAIL'),('PASS','INCOMPLETE')):
                q.write(root/'complete.json',dict(status=status))
                with patch.object(a,'load',side_effect=AssertionError('premature ledger/helper')):result=a.audit(root,'none')
                self.assertEqual(result['status'],expected);self.assertFalse(result['qualified'] or result['ledger_opened'])

if __name__=='__main__':unittest.main()
