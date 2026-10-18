"""Source-prepared CPU contracts. Execute later under the no-backend guard.

The callable under test is the actual frozen ModelRunner.run AST, with scalar
array/backend fakes. These tests establish control flow, not device exactness.
"""
import ast
import gc
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

from bench.m5_ultra_256gb.ahead_submit_once import candidate as c
from bench.m5_ultra_256gb.c1_pipeline_profile import instrumentation as profile

RUNNER = Path(__file__).resolve().parents[3]/'tutorial_code/miniengine/engine/model_runner.py'


class Array:
    def __init__(self, values): self.values=list(values)
    @property
    def shape(self): return (len(self.values),)
    def astype(self,dtype): return Array(self.values)
    def __getitem__(self,key):
        value=self.values[key]
        return Array(value if isinstance(value,list) else [value])
    def tolist(self): return list(self.values)
    def item(self): return self.values[0]


class Backend:
    int32='int32'
    def __init__(self): self.events=[];self.fail_next=False
    def array(self,value): return Array(value)
    def async_eval(self,*roots,**kwargs):
        if self.fail_next: self.fail_next=False;raise RuntimeError('injected asynchronous submission failure')
        self.events.append(('async',tuple(roots),kwargs));return None
    def eval(self,*roots,**kwargs): self.events.append(('eval',tuple(roots),kwargs))


class KV:
    def __init__(self,K,V): self.K,self.V=K,V
class State:
    def __init__(self,conv,S): self.conv,self.S=conv,S
class Context:
    def __init__(self,mode,caches,**kwargs):self.mode,self.caches=mode,caches;vars(self).update(kwargs)


class Batch:
    def __init__(self):
        self.req_ids=['r'];self.lens=[900]
        self.layers=[KV(Array([1]),Array([2])),State(Array([3]),Array([4]))]
    @property
    def size(self): return len(self.req_ids)
    def arrays(self):
        return [a for layer in self.layers for a in ((layer.K,layer.V) if isinstance(layer,KV) else (layer.conv,layer.S))]
    def clear(self):self.req_ids=[];self.layers=[];self.lens=[]


class Model:
    def __init__(self): self.calls=[]
    def __call__(self,tokens,ctx,**kwargs):
        self.calls.append((ctx.mode,tuple(tokens.values),getattr(ctx,'max_len',None)))
        if ctx.mode=='decode':
            for layer in ctx.caches:
                if isinstance(layer,KV):layer.K,layer.V=Array([tokens.values[0]+1]),Array([tokens.values[0]+2])
                else:layer.conv,layer.S=Array([tokens.values[0]+3]),Array([tokens.values[0]+4])
        return Array([value+1 for value in tokens.values[-1:]] if ctx.mode=='prefill' else [value+1 for value in tokens.values])


def request(rid='r'):
    return SimpleNamespace(req_id=rid,num_computed_tokens=900,num_tokens=901,
        num_prompt_tokens=900,prompt_token_ids=[5]*900,all_token_ids=[5]*900+[10],
        output_token_ids=[10],is_finished=False,
        sampling=SimpleNamespace(temperature=0,top_p=1.0,top_k=0,max_tokens=512,ignore_eos=False,stop_token_ids=()))


def fixture():
    backend=Backend()
    tree=ast.parse(RUNNER.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ModelRunner')
    run=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    namespace=dict(mx=backend,BatchKV=KV,BatchState=State,ForwardContext=Context,
        Request=object,SchedulerOutput=object,prefill_batch_backend_supported=lambda *a:False,
        make_cache=lambda model:[],sample=lambda logits,params:logits,__file__=str(RUNNER))
    exec(compile(ast.Module(body=[run],type_ignores=[]),str(RUNNER),'exec'),namespace)
    class Runner:
        def _sync_batch(self,decode):
            if self.batch.req_ids != [r.req_id for r in decode]:
                self._ahead=None;self.batch.req_ids=[r.req_id for r in decode]
    Runner.run=namespace['run'];namespace['ModelRunner']=Runner;namespace['DecodeBatch']=Batch
    runner=Runner();runner.batch=Batch();runner.model=Model();runner.decode_model=runner.model
    runner.spec=None;runner._ahead=None;runner.max_len=2048;runner.staging={};runner.feats={}
    req=request();out=SimpleNamespace(decode=[req],prefill=[],finished_ids=[])
    return runner,req,out,backend


def install_fake(runner):return c.Scope(runner,runner,'run',runner.run)
def advance(req,result):
    req.output_token_ids.append(result[req.req_id]);req.num_computed_tokens+=1;req.num_tokens+=1

def signature(events):return [(kind,tuple(tuple(root.values) for root in roots),kwargs) for kind,roots,kwargs in events]


class SubmitOnceTests(unittest.TestCase):
    def test_default_off_has_no_source_or_runner_access(self):
        with patch.object(c,'_source',side_effect=AssertionError('file touched')):
            self.assertIsNone(c.install(object()))
        with self.assertRaises(TypeError):c.install(object(),enabled=1)

    def test_actual_run_first_step_ordinary_then_exact_hit_one_less_submit(self):
        runner,req,out,mx=fixture();scope=install_fake(runner)
        first=runner.run(out);advance(req,first)
        self.assertEqual([event[0] for event in mx.events],['async','async','eval'])
        self.assertEqual(scope.counts['omitted_current_submits'],0)
        mx.events.clear();second=runner.run(out)
        self.assertEqual([event[0] for event in mx.events],['async','eval'])
        self.assertEqual(scope.counts['omitted_current_submits'],1)
        self.assertEqual(len(runner.model.calls),3) # first current+future, then only future
        self.assertEqual(second['r'],12);scope.restore()

    def test_outputs_committed_and_ahead_roots_equal_original_fake_run(self):
        left,lr,lo,lmx=fixture();right,rr,ro,rmx=fixture();scope=install_fake(right)
        for step in range(4):
            left_result=left.run(lo);right_result=right.run(ro)
            self.assertEqual(left_result,right_result)
            self.assertEqual([x.values for x in left.batch.arrays()],[x.values for x in right.batch.arrays()])
            self.assertEqual(left._ahead[0],right._ahead[0])
            self.assertEqual(left._ahead[1].values,right._ahead[1].values)
            self.assertEqual([x.values for x in scope.roots(left._ahead)],[x.values for x in scope.roots(right._ahead)])
            advance(lr,left_result);advance(rr,right_result)
        self.assertEqual(left.model.calls,right.model.calls)
        self.assertEqual(sum(x[0]=='async' for x in lmx.events)-sum(x[0]=='async' for x in rmx.events),3)
        scope.restore()

    def test_no_preinstall_certificate_never_skips_claimed_ahead(self):
        runner,req,out,mx=fixture();result=runner.run(out);advance(req,result)
        scope=install_fake(runner);mx.events.clear();runner.run(out)
        self.assertEqual([x[0] for x in mx.events],['async','async','eval'])
        self.assertEqual(scope.counts['omitted_current_submits'],0);scope.restore()

    def test_key_new_request_finished_and_known_terminal_are_ordinary(self):
        for variant in ('key','new','finished','terminal','finished_ids','batch2'):
            with self.subTest(variant=variant):
                runner,req,out,mx=fixture();scope=install_fake(runner)
                result=runner.run(out);advance(req,result);mx.events.clear()
                if variant=='key':req.sampling.top_k=1
                elif variant=='new':req.req_id='new'
                elif variant=='finished':req.is_finished=True
                elif variant=='terminal':scope.terminal_ids=frozenset([req.output_token_ids[-1]])
                elif variant=='finished_ids':out.finished_ids=['old']
                else:out.decode.append(request('other'))
                runner.run(out)
                self.assertEqual(scope.counts['omitted_current_submits'],0)
                self.assertEqual([x[0] for x in mx.events],['async','async','eval']);scope.restore()

    def test_mixed_prefill_and_cap_boundaries_keep_original_eval_paths(self):
        for variant in ('prefill','cap','max_len'):
            with self.subTest(variant=variant):
                runner,req,out,mx=fixture();scope=install_fake(runner)
                result=runner.run(out);advance(req,result);mx.events.clear()
                if variant=='prefill':
                    fresh=request('fresh');fresh.num_computed_tokens=0;fresh.num_prompt_tokens=1;fresh.all_token_ids=[7]
                    out.prefill=[(fresh,1)]
                elif variant=='cap':req.sampling.max_tokens=len(req.output_token_ids)+1
                else:runner.max_len=req.num_tokens+1
                runner.run(out)
                self.assertEqual(scope.counts['omitted_current_submits'],0)
                self.assertEqual([x[0] for x in mx.events],['eval']);scope.restore()

    def test_one_changed_root_or_token_does_not_inherit_certificate(self):
        for variant in ('token','state'):
            with self.subTest(variant=variant):
                runner,req,out,mx=fixture();scope=install_fake(runner)
                result=runner.run(out);advance(req,result);mx.events.clear()
                key,token,layers=runner._ahead
                if variant=='token':runner._ahead=(key,Array(token.values),layers)
                else:layers[0].K=Array(layers[0].K.values)
                runner.run(out)
                self.assertEqual(scope.counts['omitted_current_submits'],0)
                self.assertEqual([x[0] for x in mx.events],['async','async','eval']);scope.restore()

    def test_backend_exception_cannot_certify_or_retry_advanced_state(self):
        runner,req,out,mx=fixture();scope=install_fake(runner);mx.fail_next=True
        with self.assertRaisesRegex(RuntimeError,'injected'):runner.run(out)
        self.assertIsNone(scope.certificate);self.assertFalse(scope.active)
        self.assertEqual(scope.counts['errors'],1);self.assertEqual(len(runner.model.calls),1)
        self.assertEqual(mx.events,[]);scope.restore();self.assertNotIn('run',vars(runner))

    def test_explicit_profile_chain_retains_real_eval_events_and_reverse_restore(self):
        runner,req,out,mx=fixture();recorder=profile.Recorder()
        runner.batch.req_ids=[]
        ps=profile.Session(runner,type(runner).run,recorder,[],{})
        with patch.object(c,'_source',side_effect=lambda path,digest:Path(path).resolve()),patch.dict(sys.modules,{'mlx.core':mx}):
            scope=c.install(runner,enabled=True,profile_session=ps)
        runner.batch.req_ids=['r']
        result=runner.run(out);advance(req,result);runner.run(out)
        self.assertEqual(scope.counts['omitted_current_submits'],1)
        self.assertEqual(sum(row['kind']=='existing_async_eval' for row in recorder.rows),3)
        self.assertEqual(sum(row['kind']=='existing_eval' for row in recorder.rows),2)
        self.assertEqual(sum(row['kind']=='runner_step' for row in recorder.rows),2)
        self.assertEqual(sum(row['kind']=='sample_graph' for row in recorder.rows),3)
        scope.restore();ps.restore();self.assertNotIn('run',vars(runner))

    def test_restore_refuses_foreign_wrapper_then_releases_runner_references(self):
        runner,req,out,mx=fixture();scope=install_fake(runner);installed=runner.run
        runner.run=lambda out:None
        with self.assertRaisesRegex(ValueError,'outer observer'):scope.restore()
        runner.run=installed;scope.restore();scope.restore()
        ref=weakref.ref(runner);del installed,runner;gc.collect()
        self.assertIsNone(ref());self.assertIsNone(scope.certificate)

    def test_real_install_rejects_unknown_existing_wrapper(self):
        runner,req,out,mx=fixture();runner.batch.req_ids=[]
        runner.run=MethodType(lambda self,out:None,runner)
        with patch.object(c,'_source',side_effect=lambda path,digest:Path(path).resolve()),patch.dict(sys.modules,{'mlx.core':mx}):
            with self.assertRaisesRegex(ValueError,'unknown existing runner wrapper'):
                c.install(runner,enabled=True)

    def test_dead_root_reference_never_matches_reused_value(self):
        value=Array([1]);refs=c.references([value]);self.assertTrue(c.identical(refs,[value]))
        del value;gc.collect();self.assertFalse(c.identical(refs,[Array([1])]))
        self.assertIsNone(c.references([object()]))


if __name__=='__main__':unittest.main()
