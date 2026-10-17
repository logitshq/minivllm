"""M5 shared-W structural/lifecycle tests; fake backend, never MLX/GPU."""
import gc
import hashlib
import importlib.util
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import weakref

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('m5_shared_w_cpu_candidate',HERE/'candidate.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


class Tensor:
    def __init__(self,shape,dtype):self.shape,self.dtype=shape,dtype
    @property
    def ndim(self):return len(self.shape)


class Backend:
    gpu, cpu, uint32, uint8, bfloat16='gpu','cpu','uint32','uint8','bfloat16'
    def __init__(self):
        self.device=self.gpu;self.name='Apple M5 Ultra';self.arch='applegpu_g17d'
        self.builds=[];self.calls=[];self.build_error=None;self.call_error=None
        self.fast=NS(metal_kernel=self.build)
        self.metal=NS(device_info=lambda:{'architecture':self.arch})
    def default_device(self):return self.device
    def device_info(self):return {'device_name':self.name}
    def build(self,**kw):
        self.builds.append(kw)
        if self.build_error:raise self.build_error
        def call(**kwargs):
            self.calls.append(kwargs)
            if self.call_error:raise self.call_error
            return [Tensor(kwargs['output_shapes'][0],kwargs['output_dtypes'][0])]
        return call


def operator(n=48,k=5120):
    return NS(weight=Tensor((n,k//8),'uint32'),scales=Tensor((n,k//16),'uint8'),q=dict(c.Q),biases=None)


def prepare(mx,op,**overrides):
    k=op.weight.shape[1]*8
    args=dict(enabled=True,weight_strides=(k//8,1),scale_strides=(k//16,1),
              runtime_version='0.32.3',device_name='Apple M5 Ultra',architecture='applegpu_g17d')
    args.update(overrides)
    return c.prepare(mx,op,**args)


class CandidateTests(unittest.TestCase):
    def test_default_off_touches_neither_backend_operator_nor_files(self):
        class Poison:
            def __getattribute__(self,name):raise AssertionError('touched '+name)
        with patch.object(Path,'read_bytes',side_effect=AssertionError('read')):
            self.assertIsNone(c.prepare(Poison(),Poison()))
        for bad in (None,0,1,'yes'):
            with self.subTest(bad=bad),self.assertRaises(TypeError):c.prepare(None,None,enabled=bad)

    def test_actual_device_and_declared_runtime_guard(self):
        for changes in ({'runtime_version':'0.32.4'},{'device_name':'Apple M4 Max'},
                        {'architecture':'applegpu_g16s'},{'runtime_version':None}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):prepare(Backend(),operator(),**changes)
        for field,value in (('device','cpu'),('name','Apple M4 Max'),('arch','applegpu_g16s')):
            mx=Backend();setattr(mx,field,value)
            with self.subTest(field=field),self.assertRaises(ValueError):prepare(mx,operator())

    def test_all_registered_geometry_and_rows_typed_strides(self):
        for n,k in c.SHAPES:
            c.validate_geometry(n,k,(k//8,1),(k//16,1))
            for t in c.ROWS:c.validate_input((t,k),(k,1),k)
        for shape,strides in (((1,5120),(5120,1)),((3,5120),(5120,1)),((6,5120),(5120,1)),
                              ((2,1,5120),(5120,1)),((2,5120),(0,1)),((2,5120),(5120,2)),
                              ((2,5120),(5120,True)),((2,5120),(5120.0,1)),((2,6144),(6144,1))):
            with self.subTest(shape=shape,strides=strides),self.assertRaises(ValueError):c.validate_input(shape,strides,5120)
        for args in ((47,5120,(640,1),(320,1)),(48,5121,(640,1),(320,1)),
                     (48,5120,(641,1),(320,1)),(48,5120,(640,1),(0,1)),(48.0,5120,(640,1),(320,1))):
            with self.subTest(args=args),self.assertRaises(ValueError):c.validate_geometry(*args)

    def test_quantization_dtype_rank_and_bias_guard(self):
        for change in ('group','bits','mode','floatbits','extraq','bias','weightdtype','scaledtype','scalerank','scaleshape'):
            mx,op=Backend(),operator()
            if change=='group':op.q['group_size']=32
            if change=='bits':op.q['bits']=8
            if change=='mode':op.q['mode']='affine'
            if change=='floatbits':op.q['bits']=4.0
            if change=='extraq':op.q['unknown']=None
            if change=='bias':op.biases=object()
            if change=='weightdtype':op.weight.dtype='float32'
            if change=='scaledtype':op.scales.dtype='bfloat16'
            if change=='scalerank':op.scales.shape=(1,48,320)
            if change=='scaleshape':op.scales.shape=(47,320)
            with self.subTest(change=change),self.assertRaises(ValueError):prepare(mx,op)

    def test_exact_published_source_and_qdot_expression(self):
        contents=c._read_sources()
        for name,expected in c.SOURCE_PINS.items():
            self.assertEqual(hashlib.sha256((c.SOURCE/name).read_bytes()).hexdigest(),expected)
        original=contents['kernel_header.metal'].split('inline U qdot(',1)[1].split('METAL_FUNC void fp_qmv_fast_impl',1)[0]
        shared=contents['shared_impl.metal']
        expression=lambda value:re.sub(r'\s+','',re.search(r'accum\s*\+=.*?;',value,re.S).group())
        self.assertEqual(expression(original),expression(shared))
        for fragment in ('return scale * accum;','result[token][row] += v124_qdot_cached(cached, x_thread, scale);',
                         'result[token][row] = simd_sum(result[token][row]);','static_cast<T>(result[token][row])'):
            self.assertIn(fragment,shared)
        self.assertLess(shared.index('cached[i] = wl[i]'),shared.index('for (int token = 0; token < VERIFY_T; token++)'))
        for banned in ('fma(','simd_shuffle','simdgroup_matrix','threadgroup_barrier','fast_math'):
            self.assertNotIn(banned,shared)

    def test_modified_or_symlinked_published_source_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            for name in c.SOURCE_PINS:(directory/name).write_bytes((c.SOURCE/name).read_bytes())
            with patch.object(c,'SOURCE',directory):
                c._read_sources();(directory/'shared_impl.metal').write_text('changed')
                with self.assertRaisesRegex(ValueError,'SHA'):c._read_sources()
                (directory/'shared_impl.metal').unlink()
                (directory/'shared_impl.metal').symlink_to(HERE/'candidate.py')
                with self.assertRaisesRegex(ValueError,'publication'):c._read_sources()

    def test_lazy_callable_reuse_exact_launch_and_original_inputs(self):
        mx,op=Backend(),operator();p=prepare(mx,op)
        self.assertEqual(mx.builds,[]);self.assertEqual((p.kernel_build_count,p.launch_count),(0,0))
        for t in c.ROWS:
            x=Tensor((t,5120),'bfloat16');y=p.project(x,observed_x_strides=(5120,1))
            self.assertEqual(y.shape,(t,48));call=mx.calls[-1]
            self.assertIs(call['inputs'][0],op.weight);self.assertIs(call['inputs'][1],op.scales);self.assertIs(call['inputs'][2],x)
            self.assertEqual(call['template'],[('T','bfloat16'),('VERIFY_T',t),('K_SIZE',5120),('N_SIZE',48)])
            self.assertEqual(call['grid'],(64,6,1));self.assertEqual(call['threadgroup'],(64,1,1))
            self.assertEqual(call['output_shapes'],[(t,48)]);self.assertEqual(call['output_dtypes'],['bfloat16'])
        self.assertEqual(len(mx.builds),1);self.assertEqual((p.kernel_build_count,p.launch_count),(1,3))
        build=mx.builds[0]
        self.assertIs(build['ensure_row_contiguous'],False)
        self.assertEqual(set(build),{'name','input_names','output_names','source','header','ensure_row_contiguous'})
        self.assertEqual(build['source'],(c.SOURCE/'kernel_body.metal').read_text())
        self.assertEqual(build['header'],(c.SOURCE/'kernel_header.metal').read_text()+'\n'+(c.SOURCE/'shared_impl.metal').read_text())

    def test_prepared_operator_replacement_metadata_and_device_changes_fail(self):
        for change in ('weight','scales','shape','dtype','q','bias','device'):
            mx,op=Backend(),operator();p=prepare(mx,op)
            if change=='weight':op.weight=Tensor((48,640),'uint32')
            if change=='scales':op.scales=Tensor((48,320),'uint8')
            if change=='shape':op.weight.shape=(48,639)
            if change=='dtype':op.scales.dtype='uint32'
            if change=='q':op.q['bits']=8
            if change=='bias':op.biases=object()
            if change=='device':mx.device='cpu'
            with self.subTest(change=change),self.assertRaises(RuntimeError):p.project(Tensor((2,5120),'bfloat16'),observed_x_strides=(5120,1))
            self.assertEqual(mx.builds,[])

    def test_invalid_input_and_backend_errors_never_fallback(self):
        mx,op=Backend(),operator();p=prepare(mx,op)
        with self.assertRaises(ValueError):p.project(Tensor((2,5120),'float32'),observed_x_strides=(5120,1))
        mx.build_error=RuntimeError('compile error')
        with self.assertRaisesRegex(RuntimeError,'compile error'):p.project(Tensor((2,5120),'bfloat16'),observed_x_strides=(5120,1))
        self.assertEqual((p.kernel_build_count,p.launch_count),(0,0))
        mx.build_error=None;mx.call_error=RuntimeError('call error')
        with self.assertRaisesRegex(RuntimeError,'call error'):p.project(Tensor((2,5120),'bfloat16'),observed_x_strides=(5120,1))
        self.assertEqual((p.kernel_build_count,p.launch_count),(1,0))

    def test_prepared_does_not_keep_input_or_output(self):
        mx,op=Backend(),operator();p=prepare(mx,op)
        x=Tensor((2,5120),'bfloat16');y=p.project(x,observed_x_strides=(5120,1))
        xr,yr=weakref.ref(x),weakref.ref(y);mx.calls.clear();del x,y;gc.collect()
        self.assertIsNone(xr());self.assertIsNone(yr())
        self.assertIs(p.weight,op.weight);self.assertIs(p.scales,op.scales)

    def test_integer_index_order_and_final_tile_coverage(self):
        for n,k in c.SHAPES:
            covered=[]
            for lane in range(32):
                sequence=[block+lane*16+i for block in range(0,k,512) for i in range(16)]
                self.assertEqual(sequence,sorted(set(sequence)));covered.extend(sequence)
            self.assertEqual(sorted(covered),list(range(k)))
            for tile in (0,n//8-1):
                outputs=[tile*8+sg*4+row for sg in range(2) for row in range(4)]
                self.assertEqual(outputs,list(range(tile*8,tile*8+8)))
                for outrow in outputs:
                    for lane in (0,31):
                        for block in range(0,k,512):
                            weight=outrow*(k//2)+lane*8+block//2
                            scale=outrow*(k//16)+lane+block//16
                            self.assertLessEqual(weight+8,n*k//2);self.assertLess(scale,n*k//16)
                            for i in range(4):self.assertEqual(weight+2*i,outrow*(k//2)+(block+lane*16)//2+2*i)
                            for t in c.ROWS:
                                for token in (0,t-1):
                                    lo=token*k+block+lane*16
                                    self.assertGreaterEqual(lo,token*k);self.assertLess(lo+15,(token+1)*k)

    def test_import_and_default_off_are_inert(self):
        code='''
import importlib.abc,runpy,socket,sys
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if name.split('.')[0] in {'mlx','numpy','torch'}: raise AssertionError(name)
sys.meta_path.insert(0,Deny());socket.socket=lambda *a,**k: (_ for _ in ()).throw(AssertionError('socket'))
m=runpy.run_path(sys.argv[1]);assert m['prepare'](None,None) is None
assert m['contract']()['performance_qualified'] is False
'''
        result=subprocess.run([sys.executable,'-I','-B','-c',code,str(HERE/'candidate.py')],capture_output=True,text=True,timeout=5)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
