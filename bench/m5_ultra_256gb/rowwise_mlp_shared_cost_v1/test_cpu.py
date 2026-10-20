"""Small CPU adapter mechanisms. No model/runtime imports or device arrays."""
from contextlib import redirect_stdout
import gc
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref
from bench.m5_ultra_256gb.rowwise_mlp_shared_cost_v1 import screen as s
from bench.m5_ultra_256gb.rowwise_mlp_shared import test_cpu as f

nn=SimpleNamespace(silu=object())
class MLP:
    def __call__(self,x):return nn.silu
class Model:pass

def base_device(*args):raise AssertionError('never device')


class Tests(unittest.TestCase):
    def test_default_inert_and_partial_run_rejected(self):
        with patch.object(s,'validate',side_effect=AssertionError('validate')),redirect_stdout(io.StringIO()):
            self.assertEqual(s.main([]),0)
            with self.assertRaises(ValueError):s.main(['--run'])
        self.assertFalse(s.contract()['timed_mlp_observer'])

    def test_frozen_adapter_sources_load_without_backend(self):
        self.assertEqual(s.sha(s.BASE),s.BASE_SHA);self.assertEqual(s.sha(s.CANDIDATE),s.CANDIDATE_SHA)
        self.assertEqual(s.sha(s.CANARY),s.CANARY_SHA);self.assertEqual(s.sha(s.AUDITOR),s.AUDITOR_SHA)
        base=s.load(s.BASE,s.BASE_SHA);self.assertTrue(callable(base.device))

    def test_ordered_live128_calls_and_rejections(self):
        model=f.Model();x=f.Array((4,5120),'bf16',[1]*4);counter=s.CountedMLP(model,model.projector,4)
        first=model.layers[0].mlp.gate_proj
        with self.assertRaises(ValueError):counter(x,model.layers[0].mlp.up_proj.weight,first.scales)
        for _,_,op in counter.ops:counter(x,op.weight,op.scales)
        self.assertEqual(counter.receipt()['call_count'],128)
        with self.assertRaises(ValueError):counter(x,first.weight,first.scales)
        empty=s.CountedMLP(model,lambda *args:None,4)
        with self.assertRaises(ValueError):empty(x,first.weight,first.scales)
        self.assertEqual(empty.rows,[])

    def fixture(self,folder):
        calls=[]
        def hashes(mx,model,flatten,deadline):calls.append(model.version);return {'weight':model.version}
        helper=SimpleNamespace(parameter_hashes=hashes,parameter_values=lambda value:value)
        base=SimpleNamespace(device=base_device,write=s.write)
        result=dict(mlp_qualifications=[],qualification=[],raw_comparisons=[],numerical_screen_qualified=False)
        device,deps=s.adapted_device(base,None,result,folder,lambda:None,(None,None,helper))
        return base,result,device,deps,calls

    def test_private_code_clone_and_parameter_precheck_before_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);base,result,device,deps,calls=self.fixture(out)
            self.assertIs(device.__code__,base.device.__code__);self.assertNotIn('candidate_run',base.__dict__)
            self.assertNotIn('candidate_run',base.device.__globals__)
            model=Model();model.version='original';before=deps[2].parameter_hashes(None,model,None,lambda:None)
            s.write(out/'parameters-before.json',before)
            result.update(qualification=[{}]*6,raw_comparisons=[{}]*16,mlp_qualifications=[{}]*6,numerical_screen_qualified=True)
            device.__globals__['write'](out/'complete.json',result)
            self.assertEqual(calls,['original','original']);self.assertTrue(result['pre_timing_parameters_unchanged'])
            self.assertEqual(json.loads((out/'parameters-before-timing.json').read_text()),before)

    def test_changed_parameters_stop_before_qualified_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);_,result,device,deps,_=self.fixture(out);model=Model();model.version='original'
            before=deps[2].parameter_hashes(None,model,None,lambda:None);s.write(out/'parameters-before.json',before)
            model.version='mutated';result.update(qualification=[{}]*6,raw_comparisons=[{}]*16,mlp_qualifications=[{}]*6,numerical_screen_qualified=True)
            with self.assertRaisesRegex(ValueError,'parameter bytes'):device.__globals__['write'](out/'complete.json',result)
            self.assertFalse((out/'complete.json').exists())

    def test_parameter_context_does_not_retain_model_during_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,_,_,deps,_=self.fixture(Path(tmp));model=Model();model.version='x';ref=weakref.ref(model)
            deps[2].parameter_hashes(None,model,None,lambda:None);del model;gc.collect();self.assertIsNone(ref())

    def test_actual_callback_and_silu_injection_untimed_only(self):
        model=Model();fake=f.Model();model.layers=fake.layers
        for layer in model.layers:
            old=layer.mlp;new=MLP();new.gate_proj=old.gate_proj;new.up_proj=old.up_proj;layer.mlp=new
        runtime=SimpleNamespace(mx=f.MX);events=[]
        def verify(model,ids,ctx,r,*,enabled,head_project,mlp_input_project):
            self.assertIs(r.silu,nn.silu);self.assertFalse(hasattr(runtime,'silu'));self.assertTrue(enabled)
            x=f.Array((4,5120),'bf16',[1]*4)
            for layer in model.layers:
                for op in (layer.mlp.gate_proj,layer.mlp.up_proj):mlp_input_project(x,op.weight,op.scales)
            head_project('head');return 'output'
        def original(r,engine,q,candidate,h,layout,storage,position,tokens,count,project,*,audit):
            candidate.verify(engine.model,None,None,r,enabled=True,head_project=lambda x:events.append(x))
            return {'head':'original head receipt'}
        helper=SimpleNamespace(parameter_hashes=lambda *a:{},parameter_values=lambda x:x)
        base=SimpleNamespace(device=base_device,write=s.write,candidate_run=original)
        result={'mlp_qualifications':[]};device,_=s.adapted_device(base,SimpleNamespace(verify=verify),result,Path('/unused'),lambda:None,(None,None,helper))
        run=device.__globals__['candidate_run'];args=(runtime,SimpleNamespace(model=model),None,None,helper,None,None,1022,[1]*4,4,fake.projector)
        run(*args,audit=True);self.assertEqual(result['mlp_qualifications'][0]['call_count'],128)
        run(*args,audit=False);self.assertEqual(len(result['mlp_qualifications']),1);self.assertEqual(events,['head','head'])

    def test_reject_unsealed_or_failed_canary_before_any_device(self):
        plan=dict(schema='C1-MLP-shared-cost-plan-v1',adapter_sha256='current',e054_plan={},e056_plan={},e056_result={},e056_audit={},deadline_s=900)
        with patch.object(s,'sha',return_value='different'),patch.object(s,'load',side_effect=AssertionError('source load')):
            with self.assertRaisesRegex(ValueError,'actual adapter plan'):s.validate(plan)
        for deadline in (0,1801,True,float('nan')):
            with patch.object(s,'sha',return_value='current'),patch.object(s,'load',side_effect=AssertionError('source load')):
                with self.assertRaisesRegex(ValueError,'deadline'):s.validate(dict(plan,deadline_s=deadline))
        loaded={s.BASE:SimpleNamespace(validate=lambda p:(_ for _ in ()).throw(AssertionError('old validator reached'))),
                s.CANDIDATE:None,s.CANARY:SimpleNamespace(validate=lambda p:None)}
        values=iter([{'e054_plan':{}},{'schema':'C1-MLP-shared-activation-canary-result-v1','status':'FAIL'},{}])
        digests={s.AUDITOR:s.AUDITOR_SHA,s.SHARED:s.SHARED_SHA}
        with patch.object(s,'sha',side_effect=lambda p:digests.get(p,'current')),patch.object(s,'load',side_effect=lambda p,h:loaded[p]),patch.object(s,'pinned',side_effect=lambda ref:next(values)):
            with self.assertRaisesRegex(ValueError,'closed E056'):s.validate(plan)


if __name__=='__main__':unittest.main()
