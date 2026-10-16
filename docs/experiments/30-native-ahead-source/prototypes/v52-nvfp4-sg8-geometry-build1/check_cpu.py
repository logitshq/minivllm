"""Pure stdlib source/geometry/fake-backend checks; never imports MLX."""
import ast
import importlib.util
from pathlib import Path
import sys
import unittest
D=Path(__file__).resolve().parent
P=D.parent
spec=importlib.util.spec_from_file_location("v52_geometry_cpu",D/"geometry.py")
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
class Array:
    def __init__(self,shape,dtype):self.shape,self.dtype=shape,dtype
class FakeMX:
    uint32,uint8,bfloat16,gpu="u32","u8","bf16","gpu"
    def __init__(self):self.fast=self;self.factories=[];self.calls=[]
    def default_device(self):return self.gpu
    def metal_kernel(self,**kwargs):
        self.factories.append(kwargs)
        def call(**kw):self.calls.append(kw);return ["sentinel-output"]
        return call
class Contract(unittest.TestCase):
    def test_header_single_geometry_change_and_guard(self):
        old=(P/"v22-nvfp4-qmv-r2-build1/kernel_header.metal").read_text().rsplit("\nconstant int v22_in_vec_size = 5120;",1)[0].rstrip()+"\n"
        new=(D/"kernel_header.metal").read_text()
        restored=new.replace("constexpr int num_simdgroups = 8;","constexpr int num_simdgroups = 2;").replace("  if (out_row >= out_vec_size) return;  // Whole-SIMD padded-row guard; N is divisible by R4.\n","")
        self.assertEqual(restored,old)
        self.assertNotIn("math_mode",(D/"geometry.py").read_text())
        # Installed helper/state exact source segments, not a new arithmetic formulation.
        q=Path("/Users/ryanli/code/minivllm/.venv/lib/python3.13/site-packages/mlx/include/mlx/backend/metal/kernels/fp_quantized.h").read_text()
        segment=q[q.index("template <typename U, int values_per_thread, int bits>\ninline U qdot("):q.index("template <typename U, int values_per_thread, int bits>\ninline U\nqdot_safe")]
        self.assertIn(segment,old)
    def test_no_copy_lazy_grid_all_declared_shapes(self):
        for n,k in g.ELIGIBLE_SHAPES:
            mx=FakeMX();w=Array((n,k//8),mx.uint32);s=Array((n,k//16),mx.uint8)
            p=g.prepare(mx,w,s,runtime_version="0.32.3",device_name="Apple M4 Max",architecture="applegpu_g16s")
            self.assertIs(p.inputs[0],w);self.assertIs(p.inputs[1],s);self.assertEqual(mx.factories,[])
            self.assertEqual(p.project(Array((1,k),mx.bfloat16)),"sentinel-output")
            self.assertEqual(mx.calls[0]["grid"],(256,(n+31)//32,1));self.assertEqual(mx.calls[0]["threadgroup"],(256,1,1))
            self.assertEqual(mx.calls[0]["template"],[("T",mx.bfloat16),("R",4)])
            self.assertTrue(mx.factories[0]["ensure_row_contiguous"])
    def test_row_ownership_and_padded_whole_simd_guard(self):
        for n in [4,12,20,*[n for n,k in g.ELIGIBLE_SHAPES]]:
            out=[]
            for tile in range((n+31)//32):
                for sg in range(8):
                    start=tile*32+sg*4
                    if start>=n:continue
                    out.extend(range(start,start+4))
            self.assertEqual(out,list(range(n)))
    def test_reject_wrong_shape_dtype_runtime(self):
        mx=FakeMX();w=Array((17408,640),mx.uint32);s=Array((17408,320),mx.uint8)
        with self.assertRaises(ValueError):g.prepare(mx,w,s,runtime_version="0.32.4",device_name="Apple M4 Max",architecture="applegpu_g16s")
        with self.assertRaises(ValueError):g.prepare(mx,w,Array((17408,320),mx.uint32),runtime_version="0.32.3",device_name="Apple M4 Max",architecture="applegpu_g16s")
        with self.assertRaises(ValueError):g.prepare(mx,Array((15,640),mx.uint32),s,runtime_version="0.32.3",device_name="Apple M4 Max",architecture="applegpu_g16s")
        p=g.prepare(mx,w,s,runtime_version="0.32.3",device_name="Apple M4 Max",architecture="applegpu_g16s")
        with self.assertRaises(ValueError):p.project(Array((2,5120),mx.bfloat16))
    def test_stationary_timer_and_boundary_identical_v32(self):
        old=ast.parse((P/"v32-compiled-qmv-factorfold-build1/micro.py").read_text());new=ast.parse((D/"micro.py").read_text())
        def find(t,name):return next(n for n in ast.walk(t)if isinstance(n,ast.FunctionDef)and n.name==name)
        self.assertEqual(ast.dump(find(old,"invoke")),ast.dump(find(new,"invoke")))
        def sched(t):return next(n for n in ast.walk(t)if isinstance(n,ast.For)and ast.unparse(n.iter)=="enumerate(SCHEDULE)")
        self.assertEqual(ast.dump(sched(old)),ast.dump(sched(new)))
        self.assertFalse(any(n=="mlx"or n.startswith("mlx.")for n in sys.modules))
if __name__=="__main__":unittest.main()
