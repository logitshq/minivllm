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
s=importlib.util.spec_from_file_location('m8_register_candidate_cpu',P);c=importlib.util.module_from_spec(s);s.loader.exec_module(c)

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
            self.assertEqual(v['grid'],(32,N//2,1));self.assertEqual(v['threadgroup'],(32,2,1))
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

    def test_installed_helper_and_conversion_types_are_verbatim(self):
        header=c.native_header();q=(Path(c.HEADER_ROOT)/'fp_quantized.h').read_text()
        start=q.index('template <typename T, int group_size, int bits, int vecs_per_tg, int k_lanes>')
        stop=q.index('template <typename T, int group_size, int bits, bool has_global_scale = false>',start)
        self.assertIn(q[start:stop],header)
        for name in ('fp4.h','fp8.h'):self.assertIn((Path(c.HEADER_ROOT)/name).read_text(),header)
        self.assertEqual(header.count('METAL_FUNC void fp_qmv_wide_impl('),1)
        self.assertNotIn('threadgroup_barrier',header);self.assertNotIn('threadgroup float',header)
        self.assertNotIn('fp contract(off)',header)
        self.assertIn('fp_qmv_wide_impl<DT, 16, 4, 8, 16>',c.SOURCE)
        self.assertIn('x_shape[1], scales_shape[0], x_shape[0]',c.SOURCE)
        self.assertIn('threadgroup_position_in_grid, simdgroup_index_in_threadgroup',c.SOURCE)
        self.assertIn('thread_index_in_simdgroup',c.SOURCE)
        refs=json.loads(P.with_name('references.json').read_text())
        for path,h in refs['paths'].items():self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(),h)
        self.assertFalse(refs['numerical_qualified']);self.assertFalse(refs['performance_qualified'])

    def test_native_header_mismatch_fails_before_compile(self):
        mx=Fake()
        with patch.object(Path,'read_bytes',return_value=b'changed pinned source'):
            with self.assertRaisesRegex(ValueError,'native header SHA mismatch'):self.build(mx)
        self.assertEqual(mx.builds,[]);self.assertEqual(mx.calls,[])

    def test_64_threads_cover_all_rows_and_preserve_per_row_group_order(self):
        owners=[(v,sg*2+lane//16) for sg in range(2) for lane in (0,16) for v in range(8)]
        self.assertEqual(Counter(owners),Counter((v,n) for v in range(8) for n in range(4)))
        for lane in range(16):
            def schedule(nv):
                return {(tile*nv+v):[(g,j) for g in range(lane,320,16) for j in range(4)]
                        for tile in range(8//nv) for v in range(nv)}
            self.assertEqual(schedule(4),schedule(8))
        for N in (4,48,17408,248320):
            self.assertEqual((N//2)//2,N//4)
            self.assertLess((N-1)*2560+2559,1<<31) # native signed-int packed-row offset
            self.assertLess((N-1)*320+319,1<<31)
            self.assertLess(8*N-1,1<<31)
            self.assertEqual(max(v*N+N-4+n for v,n in owners),8*N-1)

    def test_native_kl16_shuffle_leaders_do_not_mix_output_columns(self):
        deps=[{lane} for lane in range(32)]
        for distance in (8,4,2,1):
            before=deps;deps=[before[i]|(before[i+distance] if i+distance<32 else {-1}) for i in range(32)]
        self.assertEqual(deps[0],set(range(16)));self.assertEqual(deps[16],set(range(16,32)))
        header=c.native_header()
        for distance in (8,4,2,1):self.assertIn(f'result[v] += simd_shuffle_down(result[v], {distance});',header)
        self.assertIn('if constexpr (k_lanes >= 32)',header)
        self.assertIn('if (k_lane == 0 && out_row < out_vec_size)',header)

if __name__=='__main__':unittest.main()
