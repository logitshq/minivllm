"""Tiny source/dispatch/index tests, not a Metal arithmetic or performance oracle."""
import ast
from collections import Counter
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

P=Path(__file__).with_name('candidate.py')
s=importlib.util.spec_from_file_location('shared_candidate_cpu',P);c=importlib.util.module_from_spec(s);s.loader.exec_module(c)


class Fake:
    bfloat16,uint32,uint8,gpu='bf16','u32','u8','gpu'
    def __init__(self):
        self.device='gpu';self.arch='applegpu_g17d';self.name='Apple M5 Ultra';self.builds=[];self.calls=[];self.error=None
        self.fast=SimpleNamespace(metal_kernel=self.kernel);self.metal=SimpleNamespace(device_info=lambda:dict(architecture=self.arch))
    def default_device(self):return self.device
    def device_info(self):return dict(device_name=self.name)
    def kernel(self,**kwargs):
        self.builds.append(kwargs)
        def run(**kwargs):
            self.calls.append(kwargs)
            if self.error:raise self.error
            return ['output']
        return run


def arrays(T=2,K=5120,N=48):
    return [SimpleNamespace(shape=s,dtype=d) for s,d in (((T,K),'bf16'),((N,K//8),'u32'),((N,K//16),'u8'))]


class Tests(unittest.TestCase):
    def build(self,mx):
        with patch('importlib.metadata.version',return_value='0.32.3'):return c.build(mx,enabled=True)

    def test_off_is_inert_and_import_only_defines_sources_and_functions(self):
        class Poison:
            def __getattribute__(self,name):raise AssertionError(name)
        self.assertIsNone(c.build(Poison()));self.assertIsNone(c.build(Poison(),enabled=False))
        for value in (1,None,'on'):
            with self.assertRaises(TypeError):c.build(Poison(),enabled=value)
        self.assertTrue(all(isinstance(n,(ast.Assign,ast.FunctionDef)) for n in ast.parse(P.read_text()).body[1:]))

    def test_device_and_version_guards_build_nothing(self):
        mx=Fake()
        with patch('importlib.metadata.version',return_value='0.32.4'):self.assertIsNone(c.build(mx,enabled=True))
        for key,value in [('device','cpu'),('arch','other'),('name','other')]:
            before=getattr(mx,key);setattr(mx,key,value);self.assertIsNone(self.build(mx));setattr(mx,key,before)
        self.assertEqual(mx.builds,[])

    def test_all_widths_use_parallel_groups_and_unchanged_inputs(self):
        mx=Fake();fn=self.build(mx);build=mx.builds[0]
        self.assertTrue(build['ensure_row_contiguous']);self.assertEqual(build['compile_options'],dict(math_mode='safe'))
        for T in (2,4,8):
            args=arrays(T,N=248320);args[0].strides=(10240,2);before=[vars(a).copy() for a in args]
            self.assertEqual(fn(*args),'output');call=mx.calls[-1]
            self.assertEqual(call['grid'],(32,248320*T,1));self.assertEqual(call['threadgroup'],(32,2*T,1))
            self.assertEqual(call['template'],[('DT','bf16'),('ROWS',T)]);self.assertEqual(call['output_shapes'],[(T,248320)])
            self.assertTrue(all(a is b for a,b in zip(args,call['inputs'])));self.assertEqual([vars(a) for a in args],before)
        self.assertEqual(len(mx.calls),3)

    def test_rejected_domains_never_dispatch_and_kernel_error_does_not_retry(self):
        mx=Fake();fn=self.build(mx)
        for T,K,N in [(1,5120,4),(3,5120,4),(16,5120,4),(2,512,4),(2,5632,4),(2,5120,6),(2,5120,838864)]:
            self.assertIsNone(fn(*arrays(T,K,N)))
        for i,key,value in [(0,'shape',(True,5120)),(1,'dtype','bf16'),(2,'dtype','bf16'),(1,'shape',(48,639)),(2,'shape',(48,319))]:
            args=arrays();setattr(args[i],key,value);self.assertIsNone(fn(*args))
        mx.device='cpu';self.assertIsNone(fn(*arrays()));self.assertEqual(mx.calls,[])
        mx.device='gpu';mx.error=RuntimeError('Metal error')
        with self.assertRaisesRegex(RuntimeError,'Metal error'):fn(*arrays())
        self.assertEqual(len(mx.calls),1)

    def test_shared_copy_has_exactly_one_writer_and_barrier_is_unconditional(self):
        for T in (2,4,8):
            threads=64*T;self.assertLessEqual(threads,512)
            for size in (1280,640):
                written=Counter(i for tid in range(threads) for i in range(tid,size,threads))
                self.assertEqual(written,Counter(range(size)))
            outputs=[(sg//2,sg%2) for sg in range(2*T)]
            self.assertEqual(set(outputs),{(t,n) for t in range(T) for n in range(2)})
            self.assertEqual(len(outputs),2*T)
        self.assertEqual(c.SOURCE.count('threadgroup_barrier('),1)
        prefix=c.SOURCE.split('threadgroup_barrier(')[0];self.assertNotIn('return',prefix);self.assertNotIn('if (',prefix)
        self.assertIn('threadgroup uint shared_w[1280]',c.SOURCE);self.assertIn('threadgroup uchar shared_s[640]',c.SOURCE)
        self.assertEqual(1280*4+640,5760)

    def test_every_local_weight_scale_and_activation_index_is_in_bounds(self):
        for local_row in (0,1):
            weight=[local_row*1280+k//4+lane*4+j for k in range(0,5120,512) for lane in range(32) for j in range(4)]
            scale=[local_row*320+k//16+lane for k in range(0,5120,512) for lane in range(32)]
            self.assertEqual(sorted(weight),list(range(local_row*1280,(local_row+1)*1280)))
            self.assertEqual(sorted(scale),list(range(local_row*320,(local_row+1)*320)))
        for T in (2,4,8):
            for token in range(T):
                x=[token*5120+k+lane*16+4*j+i for k in range(0,5120,512) for lane in range(32) for j in range(4) for i in range(4)]
                self.assertEqual(sorted(x),list(range(token*5120,(token+1)*5120)))
            for N in (4,17408,248320):
                last_group=N//2-1
                self.assertEqual(max(t*N+last_group*2+r for t in range(T) for r in (0,1)),T*N-1)

    def test_scalar_reduction_and_conversion_match_original_source(self):
        old=P.parent.parent/'nvfp4_multirow_reuse/candidate.py'
        assignments={n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(old.read_text()).body if isinstance(n,ast.Assign)}
        self.assertEqual(c.HEADER,assignments['HEADER'])
        self.assertIn('float result = 0;',c.SOURCE);self.assertIn('float accum = 0;',c.SOURCE)
        self.assertNotIn('result[',c.SOURCE);self.assertNotIn('accum[',c.SOURCE);self.assertNotIn('for (uint t',c.SOURCE)
        self.assertIn('result += scale * accum;',c.SOURCE);self.assertIn('result = simd_sum(result);',c.SOURCE)
        self.assertIn('for (uint k = 0; k < K; k += 512)',c.SOURCE);self.assertIn('for (uint j = 0; j < 4; ++j)',c.SOURCE)
        self.assertIn('#pragma clang fp contract(off)',c.HEADER)


if __name__=='__main__':unittest.main()
