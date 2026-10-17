"""E028 qualifier adapter mechanisms only; no MLX/model/service operations."""
import gc
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
import weakref

HERE=Path(__file__).resolve().parent


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


q=load(HERE/'model_qualification.py','e028_model_test')


class Array:
    def __init__(self,shape,value):self.shape,self.value,self.dtype,self.ndim=shape,value,'bf16',len(shape)
    def __getitem__(self,key):return Array(self.shape,self.value)
    def view(self,dtype):return self


class KV:
    def __init__(self,B,value):self.K,self.V=Array((B,1,256,1),value),Array((B,1,256,1),value)


class State:
    def __init__(self,B,value):self.conv,self.S=Array((B,3,2),value),Array((B,1,2,2),value)


class Batch:
    step=256
    def __init__(self):self.req_ids=[];self.lens=[];self.layers=[]


class Runner:
    def __init__(self):
        self.model=SimpleNamespace(layers=[None]*64);self.decode_model=self.model
        self.batch=Batch();self.corrupt=0;self.fallback=False;self.stale_witness=False
    def run(self,out):return {}
    def _sync_batch(self,decode):
        old=self.batch.req_ids;new=[r for r in decode if r.req_id not in old]
        if hasattr(self.batch,'counters'):
            path='fallback' if self.fallback else 'identity_refill'
            if not self.stale_witness:self.batch.counters[path]+=1
            self.batch.last_path=path
            self.batch.last_reason='disabled' if self.fallback else 'identity_keep_unchanged_capacity'
        self.batch.req_ids=[r.req_id for r in decode]
        self.batch.lens=[r.num_prompt_tokens for r in decode]
        self.batch.layers=[KV(len(decode),i+self.corrupt) if i<16 else State(len(decode),i+self.corrupt) for i in range(64)]


class Scheduler:
    def schedule(self):return None


class Engine:
    def __init__(self):self.runner=Runner();self.scheduler=Scheduler()


class Integration:
    def __init__(self):self.calls=0
    def install(self,runner,*,enabled):
        assert enabled is True and not runner.batch.req_ids
        assert type(runner.model) is SimpleNamespace  # Before base CallProxy.
        self.calls+=1;runner.batch=Batch()
        runner.batch.counters=dict(identity_refill=0,fallback=0)


class Comparator:
    def __init__(self):self.rows=[];self.first_failure=None
    def fail(self,label,details):self.first_failure=label;raise RuntimeError(label)
    def meta(self,label,left,right):
        if left!=right:self.fail(label,{})
    def compare(self,label,left,right):
        self.meta(label,left['metadata'],right['metadata']);self.meta(label,set(left['arrays']),set(right['arrays']))
        for name,a in left['arrays'].items():
            b=right['arrays'][name]
            if (a.shape,a.dtype,a.value)!=(b.shape,b.dtype,b.value):self.fail(label+'/'+name,{})
            self.rows.append(dict(check=label,name=name))


def requests():
    return [SimpleNamespace(req_id=rid,num_prompt_tokens=10,sampling=SimpleNamespace(max_tokens=20)) for rid in ('c8-0','c8-1')]


def seed(engine):
    b=engine.runner.batch;b.req_ids=['c8-0'];b.lens=[10]
    b.layers=[KV(1,i) if i<16 else State(1,i) for i in range(64)]


def full_base():
    return dict(status='PASS',total_requests=37,source_model_pins_rechecked=True,
                cases=[dict(name=n,completed=[None]*k) for n,k in (('D-C1',8),('D-C8',16),('lifecycle',13))])


def full_coverage():
    return dict(paired_identity_refills=1,compared_arrays=128,candidate_installs=2,restored_observers=4,
                per_case={'D-C8':1},
                candidate_final_counters=[dict(identity_refill=0,fallback=2),dict(identity_refill=1,fallback=5)])


class Tests(unittest.TestCase):
    def fresh(self):
        return q.load(q.BASE,'_e028_base_cpu'),Integration(),Comparator(),[Engine(),Engine()],[]

    def test_cli_help_and_unknown_are_runtime_free(self):
        script='''import importlib.abc,sys,runpy
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*a):
  if name.split('.')[0] in ('mlx','numpy','torch','tutorial_code'):raise AssertionError(name)
sys.meta_path.insert(0,Deny());sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name='__main__')
'''
        for args,rc in ((['--help'],0),(['--unknown'],2)):
            p=subprocess.run([sys.executable,'-I','-B','-c',script,str(HERE/'model_qualification.py'),*args],capture_output=True,text=True,timeout=5)
            self.assertEqual(p.returncode,rc,p.stderr)

    def test_exact_small_plan_and_pin_validation(self):
        original=HERE.parents[2]/'.artifacts/m5-ultra-256gb/realistic-qualification-v1-inputs/plan.json'
        plan=dict(schema='E028-model-qualification-plan-v1',qualification=dict(path=str(original),sha256=q.BASE_PLAN_SHA),
                  sources={**{str(p):h for p,h in q.FIXED.items()},str(HERE/'model_qualification.py'):q.sha(HERE/'model_qualification.py')})
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'plan.json';path.write_text(json.dumps(plan))
            self.assertEqual(q.validate(path,q.sha(path)),plan)
            for mutation in ('extra','changed-base','missing-source'):
                bad=json.loads(json.dumps(plan))
                if mutation=='extra':bad['deadline_s']=1
                if mutation=='changed-base':bad['qualification']['sha256']='0'*64
                if mutation=='missing-source':bad['sources'].pop(str(q.BASE))
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):q.validate(path,q.sha(path))

    def test_installs_before_proxy_and_captures_before_decode_handle_writes(self):
        base,integration,cmp,engines,events=self.fresh();original=base.Observer;native=Runner._sync_batch
        with q.observe_refills(base,integration,events.append) as coverage:
            observers=[base.Observer(e,cmp,arm) for e,arm in zip(engines,('reference','candidate'))]
            self.assertEqual(integration.calls,1)
            for e,o in zip(engines,observers):
                seed(e);o.begin();e.runner._sync_batch(requests())
                self.assertEqual(len(o.refills[0]['snapshot']['arrays']),128)
                e.runner.batch.layers[0].K.value=999  # Simulate a later decode handle write.
            self.assertEqual(base.pair_calls(*observers,cmp),[])
            self.assertEqual(coverage['paired_identity_refills'],1)
            self.assertEqual(coverage['compared_arrays'],128)
            self.assertEqual(len(cmp.rows),128)
            self.assertEqual(events[0]['transition']['new_ids'],['c8-1'])
            self.assertTrue(all(not o.refills for o in observers))
            for o in observers:o.restore()
            self.assertEqual(coverage['restored_observers'],2)
        self.assertIs(base.Observer,original);self.assertIs(Runner._sync_batch,native)
        self.assertTrue(all('_sync_batch' not in vars(e.runner) for e in engines))

    def test_real_identity_counter_required_not_metadata_or_state_equality(self):
        base,integration,cmp,engines,events=self.fresh()
        with q.observe_refills(base,integration,events.append):
            o=base.Observer(engines[1],cmp,'candidate');seed(engines[1]);o.begin();engines[1].runner.fallback=True
            with self.assertRaisesRegex(RuntimeError,'silently fell back'):engines[1].runner._sync_batch(requests())
            o.restore();self.assertEqual(o.refills,[])

    def test_stale_identity_last_path_without_counter_progress_is_rejected(self):
        base,integration,cmp,engines,events=self.fresh()
        with q.observe_refills(base,integration,events.append):
            o=base.Observer(engines[1],cmp,'candidate');seed(engines[1]);o.begin()
            engines[1].runner.stale_witness=True
            engines[1].runner.batch.last_path='identity_refill'
            with self.assertRaisesRegex(RuntimeError,'silently fell back'):engines[1].runner._sync_batch(requests())
            o.restore();self.assertEqual(o.refills,[])

    def test_refill_difference_fails_and_clears_both_snapshots(self):
        base,integration,cmp,engines,events=self.fresh()
        with q.observe_refills(base,integration,events.append) as coverage:
            observers=[base.Observer(e,cmp,arm) for e,arm in zip(engines,('reference','candidate'))]
            engines[1].runner.corrupt=1
            for e,o in zip(engines,observers):seed(e);o.begin();e.runner._sync_batch(requests())
            with self.assertRaisesRegex(RuntimeError,'E028-refill-predecode-state'):base.pair_calls(*observers,cmp)
            self.assertEqual(coverage['paired_identity_refills'],0);self.assertEqual(events[0]['status'],'FAIL')
            self.assertTrue(all(not o.refills for o in observers))
            for o in observers:o.restore()

    def test_unpaired_snapshot_cannot_disappear_on_next_step(self):
        base,integration,cmp,engines,events=self.fresh()
        with q.observe_refills(base,integration,events.append):
            o=base.Observer(engines[0],cmp,'reference');seed(engines[0]);o.begin();engines[0].runner._sync_batch(requests())
            with self.assertRaisesRegex(ValueError,'unpaired'):o.begin()
            o.restore();self.assertEqual(o.refills,[])

    def test_exception_restores_private_module_and_prior_instance_method(self):
        base,integration,cmp,engines,events=self.fresh();old_observer,old_pair=base.Observer,base.pair_calls
        r=engines[0].runner;r._sync_batch=r._sync_batch;prior=r._sync_batch
        with self.assertRaisesRegex(RuntimeError,'test abort'):
            with q.observe_refills(base,integration,events.append):
                o=base.Observer(engines[0],cmp,'reference')
                try:raise RuntimeError('test abort')
                finally:o.restore()
        self.assertIs(r._sync_batch,prior);self.assertIs(base.Observer,old_observer);self.assertIs(base.pair_calls,old_pair)

    def test_closed_observer_and_coverage_do_not_retain_engine(self):
        base,integration,cmp,engines,events=self.fresh();engine=engines.pop();ref=weakref.ref(engine)
        with q.observe_refills(base,integration,events.append) as coverage:
            o=base.Observer(engine,cmp,'candidate');seed(engine);o.begin();engine.runner._sync_batch(requests());o.restore();o.restore()
        del engine;gc.collect()
        self.assertIsNone(ref());self.assertEqual(coverage['restored_observers'],1)
        self.assertIsNone(o.original_sync);self.assertEqual(o.refills,[])

    def test_original_full_pass_alone_and_partial_coverage_are_rejected(self):
        q.require_complete(full_base(),full_coverage())
        for name in ('no-hit','lifecycle-only','unexpected-c1','unobserved-hit','no-restore','partial-workload','base-fail'):
            result,cov=full_base(),full_coverage()
            if name=='no-hit':cov.update(paired_identity_refills=0,compared_arrays=0)
            if name=='lifecycle-only':cov['per_case']={'lifecycle':1}
            if name=='unexpected-c1':cov['per_case']['D-C1']=1
            if name=='unobserved-hit':cov['candidate_final_counters'][1]['identity_refill']=2
            if name=='no-restore':cov['restored_observers']=3
            if name=='partial-workload':result['cases'][1]['completed'].pop()
            if name=='base-fail':result['status']='FAIL'
            with self.subTest(name=name),self.assertRaises(ValueError):q.require_complete(result,cov)

    def test_growth_empty_and_reorder_are_not_identity_refill_witnesses(self):
        engine=Engine();seed(engine)
        self.assertIsNone(q.refill_transition(engine.runner,[requests()[0]]))
        reqs=requests();reqs[1].sampling.max_tokens=300
        self.assertIsNone(q.refill_transition(engine.runner,reqs))
        engine.runner.batch.req_ids=['c8-0','c8-1'];engine.runner.batch.lens=[10,10]
        third=SimpleNamespace(req_id='c8-2',num_prompt_tokens=10,sampling=SimpleNamespace(max_tokens=20))
        self.assertIsNone(q.refill_transition(engine.runner,[requests()[1],requests()[0],third]))


if __name__=='__main__':unittest.main()
