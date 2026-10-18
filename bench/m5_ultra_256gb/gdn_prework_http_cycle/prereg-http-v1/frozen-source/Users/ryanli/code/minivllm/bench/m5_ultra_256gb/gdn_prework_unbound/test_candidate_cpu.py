"""Small CPU fake contracts only; no numerical/device/MLX execution claim."""
import builtins
import gc
import runpy
from types import SimpleNamespace, MethodType
import unittest
from unittest.mock import patch
import weakref

from bench.m5_ultra_256gb.gdn_prework_unbound import candidate as c


class Array:
    def __init__(self,shape,dtype,value=None): self.shape,self.dtype,self.value=shape,dtype,value
    def reshape(self,*shape): return Array(shape,self.dtype,self.value)
    def astype(self,dtype): return self if dtype==self.dtype else Array(self.shape,dtype,self.value)


class Linear:
    def __init__(self,name,K,N,events):
        self.name,self.in_features,self.out_features,self.events=name,K,N,events
        self.q=dict(c.QUANT); self.weight=Array((N,K//8),'u32',name+'-w'); self.scales=Array((N,K//16),'u8',name+'-s')
    def __call__(self,x):
        self.events.append((self.name,self.weight,self.scales,x))
        return Array((1,self.out_features),'bf16',(self.weight.value,self.scales.value))


class BatchState:
    def __init__(self): self.conv=Array((1,3,10240),'bf16','old-conv'); self.S=Array((1,48,128,128),'fp32','old-S')


class GDN:
    Hk,Hv,Dk,Dv,conv_dim=16,48,128,128,10240
    def __init__(self):
        self.events=[]; self._norm_reuse_enabled=False; self._gates_compile_enabled=True
        self.conv1d_weight=Array((10240,4),'bf16','w'); self.A_log=Array((48,),'fp32','A'); self.dt_bias=Array((48,),'fp32','dt')
        for name,K,N in (('in_proj_qkv',5120,10240),('in_proj_z',5120,6144),('in_proj_b',5120,48),('in_proj_a',5120,48),('out_proj',6144,5120)):
            setattr(self,name,Linear(name,K,N,self.events))
    def _decode(self,x,st,**flags): self.events.append(('original',x,st,flags)); return 'original'
    def _gates(self,b,a):
        self.events.append(('gates',self.A_log,self.dt_bias,b,a))
        return Array((1,48),'fp32','beta'),Array((1,48),'fp32','g')
    def _decode_gates(self,b,a,x,st): self.events.append(('compiled-gates',)); return self._gates(b,a)
    def norm(self,o,z): self.events.append(('norm',o,z)); return o


class Tests(unittest.TestCase):
    def setUp(self):
        self.layer=GDN(); self.state=BatchState(); self.x=Array((1,5120),'bf16','x'); self.fused_calls=[]
        self.mx=SimpleNamespace(bfloat16='bf16',float32='fp32',uint32='u32',uint8='u8',gpu='gpu',default_device=lambda:'gpu')
        self.module=SimpleNamespace(mx=self.mx,GatedDeltaNet=GDN,BatchState=BatchState,Linear=Linear,_m5_ultra_decode_identity=lambda:True)
        def fused(*args):
            self.fused_calls.append(args)
            return Array((1,48,128),'fp32','o'),Array((1,48,128,128),'fp32','new-S'),Array((1,3,10240),'bf16','new-conv')
        self.fused=fused
    def call(self,fn,**flags):
        return fn(self.layer,self.x,self.state,**dict({'_compile_gates':True,'_norm_reuse_normal':True},**flags))

    def test_import_and_off_are_inert_and_preserve_original_handles(self):
        original=builtins.__import__
        def guarded(name,*args,**kwargs):
            if name=='mlx' or name.startswith('mlx.'):raise AssertionError('runtime import')
            return original(name,*args,**kwargs)
        with patch('builtins.__import__',side_effect=guarded):runpy.run_path(c.__file__)
        self.assertIsNone(c.install(object(),object()))
        fn=c.make_decode(self.module,self.fused); self.assertEqual(self.call(fn),'original')
        self.assertIs(self.layer.events[0][1],self.x); self.assertIs(self.layer.events[0][2],self.state)
        self.assertEqual(self.fused_calls,[])

    def test_five_original_projections_gate_branch_and_state_commit_order(self):
        fn=c.make_decode(self.module,self.fused,enabled=True); oldconv,oldS=self.state.conv,self.state.S
        self.call(fn)
        names=[x[0] for x in self.layer.events]
        self.assertEqual(names,['in_proj_qkv','in_proj_z','in_proj_b','in_proj_a','compiled-gates','gates','norm','out_proj'])
        args=self.fused_calls[0];self.assertIs(args[1],oldconv);self.assertIs(args[5],oldS)
        self.assertEqual((self.state.conv.value,self.state.S.value),('new-conv','new-S'))
        self.assertEqual(fn.calls,dict(candidate=1,fallback=0))
        self.layer.events.clear();self.layer._gates_compile_enabled=False;self.call(fn)
        self.assertNotIn('compiled-gates',[e[0] for e in self.layer.events])

    def test_capture_unknown_state_batch_quant_and_backend_fall_back(self):
        fn=c.make_decode(self.module,self.fused,enabled=True)
        for flags in ({'_norm_reuse_normal':False},{'_compile_gates':False}):self.assertEqual(self.call(fn,**flags),'original')
        self.state.record=True;self.assertEqual(self.call(fn),'original');del self.state.record
        self.x.shape=(8,5120);self.assertEqual(self.call(fn),'original');self.x.shape=(1,5120)
        self.layer._norm_reuse_enabled=True;self.assertEqual(self.call(fn),'original');self.layer._norm_reuse_enabled=False
        self.layer.in_proj_qkv.q={'bits':8};self.assertEqual(self.call(fn),'original');self.layer.in_proj_qkv.q=dict(c.QUANT)
        self.module._m5_ultra_decode_identity=lambda:False;self.assertEqual(self.call(fn),'original')
        self.assertEqual(self.fused_calls,[])

    def test_replacement_uses_current_actual_parameter_objects(self):
        fn=c.make_decode(self.module,self.fused,enabled=True);self.call(fn)
        weight=Array((10240,640),'u32','changed-value');conv=Array((10240,4),'bf16','changed-conv')
        self.layer.in_proj_qkv.weight=weight;self.layer.conv1d_weight=conv
        self.layer.A_log=Array((48,),'fp32','changed-A');self.layer.events.clear();self.call(fn)
        self.assertIs(self.layer.events[0][1],weight);self.assertIs(self.fused_calls[-1][2],conv)
        gate=next(e for e in self.layer.events if e[0]=='gates');self.assertIs(gate[1],self.layer.A_log)

    def test_fused_error_does_not_commit_retry_or_hide_error(self):
        conv,S=self.state.conv,self.state.S
        def fail(*args):raise RuntimeError('device preparation failed')
        fn=c.make_decode(self.module,fail,enabled=True)
        with self.assertRaisesRegex(RuntimeError,'preparation'):self.call(fn)
        self.assertIs(self.state.conv,conv);self.assertIs(self.state.S,S)
        self.assertNotIn('original',[e[0] for e in self.layer.events]);self.assertEqual(fn.calls['candidate'],0)

    def test_scoped_restore_exception_foreign_refusal_and_release(self):
        with patch.object(c,'_load_fused',return_value=self.fused):
            handle=c.install(self.layer,self.module,enabled=True)
            with self.assertRaisesRegex(RuntimeError,'caller'):
                with handle:raise RuntimeError('caller failure')
            self.assertTrue(handle.closed);self.assertNotIn('_decode',vars(self.layer))
            other=c.install(self.layer,self.module,enabled=True);wrapper=self.layer._decode
            object.__setattr__(self.layer,'_decode',MethodType(lambda self,*a,**kw:None,self.layer))
            with self.assertRaisesRegex(ValueError,'observer'):other.restore()
            object.__setattr__(self.layer,'_decode',wrapper);other.restore();del wrapper
        witness=weakref.ref(self.layer);del self.layer;gc.collect()
        self.assertIsNone(witness());self.assertIsNone(handle.layer);self.assertIsNone(other.layer)

    def test_invalid_enable_and_overlapping_instance_hook_reject(self):
        for value in (1,None,'on'):
            with self.assertRaises(TypeError):c.install(self.layer,self.module,enabled=value)
        with patch.object(c,'_load_fused',return_value=self.fused):
            handle=c.install(self.layer,self.module,enabled=True)
            with self.assertRaisesRegex(ValueError,'untouched'):c.install(self.layer,self.module,enabled=True)
            handle.restore()


if __name__=='__main__':unittest.main()
