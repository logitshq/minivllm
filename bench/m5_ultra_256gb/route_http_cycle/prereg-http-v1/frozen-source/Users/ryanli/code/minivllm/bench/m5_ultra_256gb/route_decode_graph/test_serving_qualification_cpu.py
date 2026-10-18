"""Prepared CPU-only lifecycle and coverage tests; no arrays or device proof.

Use the frozen real Observer/CallProxy and actual WitnessFunctions bookkeeping
with explicitly fake runners/compiler callables. These tests never run main,
load a model, import MLX or authorize device correctness/performance.
"""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bench.m5_ultra_256gb.route_decode_graph import integration as install_module
from bench.m5_ultra_256gb.route_decode_graph import serving_qualification as q


def base_module():return q.load(q.BASE,'_serving_cpu_frozen_observer')
def old_module():return q.load(q.OLD,'_serving_cpu_frozen_witness')


class Model:
    def __call__(self,*args,**kwargs):raise AssertionError('CPU lifecycle test called a model')


class Runner:
    def __init__(self):
        self.model=Model();self.decode_model=self.model
        self.batch=SimpleNamespace(req_ids=[]);self._ahead=None
    def run(self,out):return {}


class Scheduler:
    def schedule(self):return SimpleNamespace(decode=[],prefill=[])


def engine():return SimpleNamespace(runner=Runner(),scheduler=Scheduler())


class Graph:
    """Fake compiled callable, solely to test counter evidence and rejection."""
    def __init__(self):
        self._functions={};self.calls=0;self.actual_calls=0;self.invokes=1;self.fallback=False
        self.output=object()
    def _eligible(self,*args):raise AssertionError('witness must not duplicate eligibility')
    def __call__(self,tokens,ctx):
        self.calls+=1
        if self.fallback:return self.output
        if 'CPU-compiled' not in self._functions:
            def compiled(*args):self.actual_calls+=1;return self.output
            self._functions['CPU-compiled']=compiled
        for _ in range(self.invokes):self._functions['CPU-compiled'](tokens,ctx)
        return self.output


class FakeIntegration:
    def __init__(self):self.handles=[];self.graphs=[]
    def install(self,runner,*,enabled):
        if enabled is not True:raise AssertionError('expected explicit ON candidate only')
        graph=Graph();self.graphs.append(graph)
        dispatch=install_module._Dispatch(runner.decode_model,graph)
        handle=install_module.Installation(runner,runner.decode_model,dispatch)
        self.handles.append(handle);runner.decode_model=dispatch;return handle


def context(B=1,used=1023,capacity=1536):
    return SimpleNamespace(max_len=used,caches=[SimpleNamespace(
        K=SimpleNamespace(shape=(B,4,capacity,256)),V=SimpleNamespace(shape=(B,4,capacity,256))) for _ in range(16)])


def coverage_fixture():
    result=dict(status='PASS',total_requests=37,source_model_pins_rechecked=True,
        cases=[dict(name=name,completed=[None]*n) for name,n in [('D-C1',8),('D-C8',16),('lifecycle',13)]])
    coverage=dict(installs=2,restores=4,parameter_checks_complete=True,parameter_direct_calls=42,
        route_hits={name:{'1023':1,'1536':1} for name in ('D-C1','D-C8')},
        cases={name:dict(consumed=1,discarded_prior=0,
            partial_retirement=int(name=='lifecycle'),refills=int(name=='D-C8'))
            for name in ('D-C1','D-C8','lifecycle')})
    return result,coverage


class Tests(unittest.TestCase):
    def test_plan_binds_qualifier_and_fixed_inherited_deadline(self):
        path=Path('/prepared/serving-plan.json');inherited=Path('/prepared/original-plan.json')
        own=str(Path(q.__file__).resolve());sources={own:'self-source-sha'}
        plan=dict(schema='E024-route-serving-plan-v1',qualifier_sha256=sources[own],
            qualification=dict(path=str(inherited),sha256=q.BASE_PLAN_SHA),sources=sources,deadline_s=1800)
        original=dict(deadline_s=1800,candidate=dict(compile_decode=True,fuse_decode=True,gates_compile=True,norm_reuse=False))
        digests={str(path):'plan-sha',str(inherited):q.BASE_PLAN_SHA,own:sources[own]}
        def validate(candidate,base):
            with patch.object(q,'sha',side_effect=lambda p:digests[str(p)]), \
                 patch.object(q,'read',side_effect=lambda p:candidate if Path(p)==path else base), \
                 patch.object(q,'source_pins',return_value=sources):
                return q.validate(path,'plan-sha')
        self.assertEqual(validate(plan,original),plan)
        for field,value in [('qualifier_sha256','wrong'),('deadline_s',1799),('deadline_s',1801),
                            ('deadline_s',1800.0),('deadline_s',True),('unknown',1)]:
            with self.subTest(field=field,value=value):
                broken=copy.deepcopy(plan);broken[field]=value
                with self.assertRaises(ValueError):validate(broken,original)
        missing=copy.deepcopy(plan);missing.pop('qualifier_sha256')
        with self.assertRaises(ValueError):validate(missing,original)
        for deadline in (1799,1801,1800.0,True):
            broken=copy.deepcopy(original);broken['deadline_s']=deadline
            with self.assertRaises(ValueError):validate(plan,broken)

    def test_real_observer_wraps_after_install_and_restores_before_handle(self):
        base=base_module();old=old_module();installed=FakeIntegration();events=[]
        original_observer,original_run=base.Observer,base.run_case
        e=engine();model=e.runner.model;decoder=e.runner.decode_model
        with q.adapt(base,installed,old,[],events.append) as counts:
            observed=base.Observer(e,object(),'candidate')
            handle=installed.handles[0]
            self.assertIsInstance(e.runner.decode_model,base.CallProxy)
            self.assertIs(e.runner.decode_model.target,handle.decoder)
            self.assertIs(e.runner.model.target,model)
            self.assertIsInstance(handle.decoder.long,q.RouteWitness)
            self.assertIs(handle.decoder.long.graph,installed.graphs[0])
            observed.begin();self.assertEqual(observed.case_step,1)
            self.assertEqual(handle.decoder.long.step,1)
            observed.begin();self.assertEqual(handle.decoder.long.step,2)
            observed.restore();observed.restore()
            self.assertIs(e.runner.model,model);self.assertIs(e.runner.decode_model,decoder)
            self.assertNotIn('run',vars(e.runner));self.assertNotIn('schedule',vars(e.scheduler))
            self.assertTrue(handle.closed);self.assertIsNone(observed.engine)
            self.assertIsNone(observed.original_model);self.assertIsNone(observed.route_witness)
            self.assertEqual((counts['installs'],counts['restores']),(1,1))
        self.assertIs(base.Observer,original_observer);self.assertIs(base.run_case,original_run)

    def test_reference_never_installs_adapter_and_instance_methods_restore_exactly(self):
        base=base_module();e=engine();original=lambda out:{};schedule=lambda:None
        e.runner.run=original;e.scheduler.schedule=schedule
        installed=FakeIntegration()
        with q.adapt(base,installed,old_module(),[],lambda row:None):
            observed=base.Observer(e,object(),'reference');observed.restore()
        self.assertEqual(installed.handles,[])
        self.assertIs(e.runner.run,original);self.assertIs(e.scheduler.schedule,schedule)

    def test_failed_witness_construction_restores_decoder_and_original_error(self):
        base=base_module();e=engine();before=dict(vars(e.runner));installed=FakeIntegration()
        with q.adapt(base,installed,old_module(),[],lambda row:None):
            with patch.object(q,'RouteWitness',side_effect=RuntimeError('witness construction failed')):
                with self.assertRaisesRegex(RuntimeError,'witness construction failed'):base.Observer(e,object(),'candidate')
        self.assertTrue(installed.handles[0].closed)
        self.assertEqual(set(vars(e.runner)),set(before))
        for key,value in before.items():self.assertIs(vars(e.runner)[key],value)

    def test_partial_base_constructor_failure_rolls_back_own_proxies_and_methods(self):
        for label in ('reference','candidate'):
            with self.subTest(label=label):
                base=base_module();original=base.Observer;e=engine();before=dict(vars(e.runner));installed=FakeIntegration()
                original_schedule=lambda:None;e.scheduler.schedule=original_schedule
                class Broken(original):
                    def __init__(self,*args):
                        super().__init__(*args)
                        raise RuntimeError('base failed after proxy installation')
                base.Observer=Broken
                with q.adapt(base,installed,old_module(),[],lambda row:None):
                    with self.assertRaisesRegex(RuntimeError,'base failed after proxy installation'):base.Observer(e,object(),label)
                self.assertEqual(set(vars(e.runner)),set(before))
                for key,value in before.items():self.assertIs(vars(e.runner)[key],value)
                self.assertIs(e.scheduler.schedule,original_schedule)
                if label=='candidate':self.assertTrue(installed.handles[0].closed)

    def test_actual_callable_counts_not_family_creation_and_fallback_is_not_hit(self):
        old=old_module();graph=Graph();events=[];counts={'route_hits':{},'long_fallback_calls':0}
        witness=q.RouteWitness(graph,old,events.append,counts);witness.phase='D-C8'
        witness.step=7
        tokens=SimpleNamespace(shape=(8,));ctx=context(8)
        self.assertIs(witness(tokens,ctx),graph.output);self.assertIs(witness(tokens,ctx),graph.output)
        self.assertEqual((graph.calls,graph.actual_calls,witness.functions.invocations,len(witness.functions)),(2,2,2,1))
        self.assertEqual(counts['route_hits'],{'D-C8':{'1023':2,'1536':0}})
        self.assertTrue(all(row['status']=='SUBMITTED' for row in events))
        self.assertTrue(all(row['step']==7 and row['direct_label'] is None for row in events))
        graph.fallback=True;self.assertIs(witness(tokens,context(8,1023,1024)),graph.output)
        self.assertEqual(counts['long_fallback_calls'],1)
        self.assertEqual(counts['route_hits']['D-C8']['1023'],2)

    def test_missing_or_multiple_actual_invocations_and_unqualified_capacity_reject(self):
        for invokes,capacity in ((0,1536),(2,1536),(1,1792)):
            with self.subTest(invokes=invokes,capacity=capacity):
                graph=Graph();graph.invokes=invokes;events=[];counts={'route_hits':{},'long_fallback_calls':0}
                witness=q.RouteWitness(graph,old_module(),events.append,counts)
                with self.assertRaises(ValueError):witness(SimpleNamespace(shape=(1,)),context(capacity=capacity))
                self.assertEqual(counts['route_hits'],{});self.assertEqual(len(events),1)
                self.assertEqual(events[0]['status'],'RUNNING')

    def test_completion_requires_each_bound_and_actual_ahead_refill_retirement(self):
        result,counts=coverage_fixture()
        self.assertEqual(q.require_complete(result,counts),
            dict(discarded_ahead_observed=False,discarded_prior_ahead_count=0))
        with_discard=copy.deepcopy(counts);with_discard['cases']['lifecycle']['discarded_prior']=1
        self.assertEqual(q.require_complete(result,with_discard),
            dict(discarded_ahead_observed=True,discarded_prior_ahead_count=1))
        changes=[]
        for case in ('D-C1','D-C8'):
            for bound in ('1023','1536'):
                changes.append(lambda c,case=case,bound=bound:c['route_hits'][case].__setitem__(bound,0))
            changes.append(lambda c,case=case:c['cases'][case].__setitem__('consumed',0))
        changes.extend([lambda c:[row.__setitem__('partial_retirement',0) for row in c['cases'].values()],
            lambda c:c['cases']['D-C8'].__setitem__('refills',0),
            lambda c:c.__setitem__('parameter_checks_complete',False),lambda c:c.__setitem__('parameter_direct_calls',41),
            lambda c:c.__setitem__('installs',1),lambda c:c.__setitem__('restores',3)])
        for change in changes:
            broken=copy.deepcopy(counts);change(broken)
            with self.assertRaises(ValueError):q.require_complete(result,broken)

    def test_base_failure_or_incomplete_original_workload_cannot_gain_qualification(self):
        result,counts=coverage_fixture()
        for key,value in (('status','FAIL'),('total_requests',36),('source_model_pins_rechecked',False)):
            broken=copy.deepcopy(result);broken[key]=value
            with self.assertRaises(ValueError):q.require_complete(broken,counts)
        broken=copy.deepcopy(result);broken['cases'][1]['completed'].pop()
        with self.assertRaises(ValueError):q.require_complete(broken,counts)


if __name__=='__main__':unittest.main()
