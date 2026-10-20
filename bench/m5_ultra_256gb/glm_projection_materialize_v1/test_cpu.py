"""Tiny injected ownership/order tests; no backend/device or model payload."""
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import weakref

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_glm_projection_candidate',HERE/'candidate.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)

class Array:
    def __init__(self,shape,dtype):
        self.shape=shape;self.dtype=dtype;self.nbytes=math.prod(shape)*(4 if dtype=='U32' else 2);self.parents=[]

class Backend:
    array=Array;uint32='U32';float16='F16'
    def __init__(self,fail=False):self.events=[];self.fail=fail;self.on_eval=lambda:None
    def eval(self,*values):
        self.on_eval();self.events.append(('eval',len(values)))
        if self.fail:raise RuntimeError('injected eval failure')
        for v in values:v.parents.clear()
    def synchronize(self):self.events.append(('sync',))
    def clear_cache(self):self.events.append(('clear',))
    def get_active_memory(self):return 0
    def get_cache_memory(self):return 0
    def get_peak_memory(self):return 0
    def get_memory_limit(self):return 0

class Model:
    def __init__(self):
        self.language_model=SimpleNamespace(args=SimpleNamespace(num_hidden_layers=45,n_routed_experts=288,
            first_k_dense_replace=3,num_nextn_predict_layers=1,hidden_size=4096,moe_intermediate_size=2048))
        self.values={}
        for prefix in [f'language_model.model.layers.{i}.mlp.switch_mlp' for i in range(3,45)]+['language_model.mtp.0.block.mlp.switch_mlp']:
            for projection in c.PROJECTIONS:
                down=projection=='down_proj';width=4096 if down else 2048
                shapes=((288,width,320 if down else 512),(288,width,32 if down else 64),(288,width,32 if down else 64))
                for suffix,shape,dtype in zip(c.SUFFIXES,shapes,('U32','F16','F16')):
                    self.values[prefix+'.'+projection+'.'+suffix]=Array(shape,dtype)
        self.values['language_model.model.norm.weight']=Array((4096,),'F16')
    def parameters(self):return self.values

def flatten(values):return list(values.items())

class Tests(unittest.TestCase):
    def test_off_is_inert(self):self.assertIsNone(c.build(None,None,None))
    def test_complete_original_order_and_handles(self):
        model=Model();mx=Backend();rows=[];before=dict(model.values)
        value=c.materialize(model,mx,flatten,rows.append)
        self.assertEqual(value['parameter_keys'],388);self.assertEqual(value['expert_projection_groups'],129)
        self.assertEqual(rows[0]['keys'][0],'language_model.model.layers.3.mlp.switch_mlp.gate_proj.weight')
        self.assertEqual(rows[1]['keys'][0],'language_model.model.layers.3.mlp.switch_mlp.down_proj.weight')
        self.assertEqual(rows[128]['keys'][0],'language_model.mtp.0.block.mlp.switch_mlp.up_proj.weight')
        self.assertEqual([e[0] for e in mx.events],['eval','sync','clear']*130)
        self.assertTrue(all(model.values[k] is v for k,v in before.items()))
    def test_bad_shape_missing_mtp_and_extra_expert_refuse_before_eval(self):
        for mode in ('shape','missing','extra'):
            model=Model();mx=Backend();key=next(iter(model.values))
            if mode=='shape':model.values[key].shape=(1,)
            elif mode=='missing':model.values.pop('language_model.mtp.0.block.mlp.switch_mlp.up_proj.biases')
            else:model.values['language_model.model.layers.3.mlp.experts.0.gate_proj.weight']=Array((1,),'U32')
            with self.assertRaises(ValueError):c.materialize(model,mx,flatten,lambda r:None)
            self.assertEqual(mx.events,[])
    def test_large_group_refuses_before_eval(self):
        model=Model();mx=Backend();model.values[next(iter(model.values))].nbytes=c.MAX_GROUP_BYTES
        with self.assertRaises(ValueError):c.materialize(model,mx,flatten,lambda r:None)
        self.assertEqual(mx.events,[])
    def test_original_lazy_once_and_outer_dictionary_gone(self):
        class Owned(dict):pass
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder).resolve();mx=Backend();calls=[];refs=[];rows=[];model=Model();processor=object()
            def original(p,**kwargs):
                calls.append((p,kwargs));owned=Owned(model.values);refs.append(weakref.ref(owned));return model,processor
            mx.on_eval=lambda:self.assertIsNone(refs[0]())
            with patch.object(c,'ORIGINAL',path):
                load=c.build(original,mx,flatten,enabled=True,admit=lambda p,f:self.assertIs(f,original),emit=rows.append)
                self.assertEqual(load(path),(model,processor))
            self.assertEqual(calls,[(str(path),dict(adapter_path=None,lazy=True,strict=True,trust_remote_code=False))])
            self.assertEqual(rows[-1]['event'],'projection-materialization-complete')
    def test_unsupported_options_and_offload_refuse_before_load(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder).resolve();calls=[]
            with patch.object(c,'ORIGINAL',path):
                load=c.build(lambda *a,**k:calls.append(1),Backend(),flatten,enabled=True,admit=lambda p,f:None,emit=lambda r:None)
                for kwargs in (dict(adapter_path='x'),dict(lazy=True),dict(strict=False),dict(force_download=True),dict(trust_remote_code=True)):
                    with self.assertRaises(ValueError):load(path,**kwargs)
                (path/'offload_index.json').write_text('{}')
                with self.assertRaises(ValueError):load(path)
            self.assertEqual(calls,[])
    def test_eval_failure_has_no_complete_and_no_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder).resolve();calls=[];rows=[];mx=Backend(fail=True)
            def original(*a,**k):calls.append(1);return Model(),object()
            with patch.object(c,'ORIGINAL',path):
                load=c.build(original,mx,flatten,enabled=True,admit=lambda p,f:None,emit=rows.append)
                with self.assertRaisesRegex(RuntimeError,'injected eval failure'):load(path)
            self.assertEqual(calls,[1]);self.assertNotIn('projection-materialization-complete',[r['event'] for r in rows])
            self.assertEqual(mx.events,[('eval',3)])

if __name__=='__main__':unittest.main()
