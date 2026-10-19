"""Tiny native-M8 dispatch/index checks; never a numeric or performance oracle."""
import ast
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

P=Path(__file__).with_name('candidate.py')
s=importlib.util.spec_from_file_location('m8_shared_candidate_cpu',P);c=importlib.util.module_from_spec(s);s.loader.exec_module(c)

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

def arrays(M=8,K=5120,N=17408):
    return [SimpleNamespace(shape=s,dtype=d) for s,d in (((M,K),'bf16'),((N,K//8),'u32'),((N,K//16),'u8'))]

class Tests(unittest.TestCase):
    def build(self,mx):
        with patch('importlib.metadata.version',return_value='0.32.3'):return c.build(mx,enabled=True)

    def test_default_off_does_not_inspect_backend_and_module_is_inert(self):
        class Poison:
            def __getattribute__(self,name):raise AssertionError(name)
        self.assertIsNone(c.build(Poison()));self.assertIsNone(c.build(Poison(),enabled=False))
        for value in (1,None,'on'):
            with self.assertRaises(TypeError):c.build(Poison(),enabled=value)
        self.assertTrue(all(isinstance(n,(ast.Assign,ast.FunctionDef)) for n in ast.parse(P.read_text()).body[1:]))

    def test_fixed_runtime_guards_never_compile_unknown_backend(self):
        mx=Fake()
        with patch('importlib.metadata.version',return_value='0.32.4'):self.assertIsNone(c.build(mx,enabled=True))
        for key,value in [('device','cpu'),('arch','other'),('name','other')]:
            old=getattr(mx,key);setattr(mx,key,value);self.assertIsNone(self.build(mx));setattr(mx,key,old)
        self.assertEqual(mx.builds,[])

    def test_one_m8_dispatch_and_no_mutation_or_retry(self):
        mx=Fake();fn=self.build(mx)
        self.assertTrue(mx.builds[0]['ensure_row_contiguous']);self.assertEqual(mx.builds[0]['compile_options'],dict(math_mode='safe'))
        for N in (4,48,10240,17408,248320):
            args=arrays(N=N);args[0].strides=(10240,2);before=[vars(a).copy() for a in args]
            self.assertEqual(fn(*args),'output');v=mx.calls[-1]
            self.assertEqual(v['grid'],(32,N*4,1));self.assertEqual(v['threadgroup'],(32,16,1))
            self.assertEqual(v['output_shapes'],[(8,N)]);self.assertEqual(v['template'],[('DT','bf16')])
            self.assertTrue(all(x is y for x,y in zip(v['inputs'],args)));self.assertEqual([vars(a) for a in args],before)
        mx.error=RuntimeError('device failure')
        with self.assertRaisesRegex(RuntimeError,'device failure'):fn(*arrays())
        self.assertEqual(len(mx.calls),6)

    def test_all_non_m8_and_wrong_formats_refuse_without_dispatch(self):
        mx=Fake();fn=self.build(mx)
        for M,K,N in [(1,5120,48),(2,5120,48),(4,5120,48),(9,5120,48),(8,17408,5120),(8,5120,6),(8,5120,838864),(8,0,4)]:
            self.assertIsNone(fn(*arrays(M,K,N)))
        for i,key,value in [(0,'shape',(True,5120)),(0,'shape',[8,5120]),(0,'dtype','f32'),(1,'dtype','bf16'),(2,'dtype','bf16'),(1,'shape',(17408,639)),(2,'shape',(17408,319))]:
            args=arrays();setattr(args[i],key,value);self.assertIsNone(fn(*args))
        mx.device='cpu';self.assertIsNone(fn(*arrays()));self.assertEqual(mx.calls,[])

    def test_cooperative_tile_copy_has_one_writer_and_no_barrier_divergence(self):
        for size in (2560,1280):
            self.assertEqual(Counter(i for tid in range(512) for i in range(tid,size,512)),Counter(range(size)))
        self.assertEqual(2560*4+1280,11520)
        self.assertEqual(c.SOURCE.count('threadgroup_barrier('),1)
        before=c.SOURCE.split('threadgroup_barrier(')[0]
        self.assertNotIn('if (',before);self.assertNotIn('return',before)
        self.assertIn('threadgroup uint shared_w[2560]',before);self.assertIn('threadgroup uchar shared_s[1280]',before)

    def test_exact_output_ownership_global_and_local_index_bounds(self):
        outputs=[(sg//2,(sg%2)*2+lane//16) for sg in range(16) for lane in (0,16)]
        self.assertEqual(Counter(outputs),Counter((m,n) for m in range(8) for n in range(4)))
        for n in range(4):
            packed=[n*1280+g*4+i for lane in range(16) for g in range(lane,320,16) for i in range(4)]
            self.assertEqual(sorted(packed),list(range(n*1280,(n+1)*1280)))
            scales=[n*320+g for lane in range(16) for g in range(lane,320,16)]
            self.assertEqual(sorted(scales),list(range(n*320,(n+1)*320)))
        for m in range(8):
            indices=[m*5120+g*16+j*4+i for lane in range(16) for g in range(lane,320,16) for j in range(4) for i in range(4)]
            self.assertEqual(sorted(indices),list(range(m*5120,(m+1)*5120)))
        for N in (4,48,17408,248320):
            last=N//4-1
            self.assertEqual(max(m*N+4*last+n for m,n in outputs),8*N-1)
            self.assertLess((4*last+3)*640+639,N*640)
            self.assertLess((4*last+3)*320+319,N*320)

    def test_native_kl16_shuffle_leaders_do_not_mix_half_simd_rows(self):
        # Independent lane ancestry simulation: only stored leaders are relevant.
        deps=[{lane} for lane in range(32)]
        for distance in (8,4,2,1):
            before=deps;deps=[before[i]|(before[i+distance] if i+distance<32 else {-1}) for i in range(32)]
        self.assertEqual(deps[0],set(range(16)));self.assertEqual(deps[16],set(range(16,32)))
        self.assertNotIn('simd_sum(',c.SOURCE)
        calls=[n.strip() for n in c.SOURCE.splitlines() if 'simd_shuffle_down' in n]
        self.assertEqual(calls,[f'result += simd_shuffle_down(result, {n});' for n in (8,4,2,1)])

    def test_pinned_native_source_and_vector_dot_order_are_retained(self):
        refs=json.loads(P.with_name('references.json').read_text())
        for name,h in refs['paths'].items():self.assertEqual(hashlib.sha256(Path(name).read_bytes()).hexdigest(),h)
        native=next(Path(name).read_text() for name in refs['paths'] if name.endswith('/fp_quantized.h'))
        wide=native.split('METAL_FUNC void fp_qmv_wide_impl(',1)[1].split('template <',1)[0]
        for expression in ('float4 w4[','acc += dot(w4[j], float4(xv4[j]));'):
            self.assertIn(expression,wide);self.assertIn(expression,c.SOURCE)
        self.assertIn('g += k_lanes',wide);self.assertIn('g += 16',c.SOURCE)
        self.assertIn('result[v] += s * acc;',wide);self.assertIn('result += s * acc;',c.SOURCE)
        self.assertNotIn('fp contract(off)',c.HEADER)
        self.assertFalse(refs['binary_compile_options_verified']);self.assertFalse(refs['numerical_qualified'])
        self.assertIn('NEVER independent M1',P.read_text())

if __name__=='__main__':unittest.main()
