"""Pure stdlib V30 source/metadata checks; no MLX/GPU."""
import ast
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
D=Path(__file__).resolve().parent
P=D.parent
BASE=P/"candidate-v17-gdn-gates-build1-root/tutorial_code/miniengine/kernels/gdn_metal.py"
class Checks(unittest.TestCase):
    def test_exact_source_delta(self):
        original=(D/"baseline.metal").read_text();new=(D/"gdn_register_cache.metal").read_text()
        t=ast.parse(BASE.read_text());actual=next(ast.literal_eval(n.value) for n in t.body if isinstance(n,ast.Assign) and any(isinstance(a,ast.Name) and a.id=="_SOURCE" for a in n.targets))
        self.assertEqual(original,actual)
        first="""    float held[32];
    #pragma unroll 32
    for (uint j = 0; j < 32; j++) {
        uint d = d0 + j;
        held[j] = S[base + d * Dv + dv];
        kv += ks * kh[d] * held[j];
    }"""
        second="""    #pragma unroll 32
    for (uint j = 0; j < 32; j++) {
        uint d = d0 + j;
        float s = ex * held[j] + ks * kh[d] * delta;"""
        restored=new.replace(first,'    for (uint d = d0; d < d0 + seg; d++) kv += ks * kh[d] * S[base + d * Dv + dv];',1).replace(second,'    for (uint d = d0; d < d0 + seg; d++) {\n        float s = ex * S[base + d * Dv + dv] + ks * kh[d] * delta;',1)
        self.assertEqual(restored,original)
        self.assertEqual(new.count("S[base + d * Dv + dv]"),1)
        self.assertEqual(new.count("threadgroup_barrier"),original.count("threadgroup_barrier"))
    def test_default_compile_policy_and_metadata(self):
        spec=importlib.util.spec_from_file_location("v30_inert_candidate",D/"gdn_register_cache.py");m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        class A:
            def __init__(self,shape,dtype):self.shape=shape;self.dtype=dtype
            def astype(self,dtype):return A(self.shape,dtype)
        calls=[]
        def factory(**kw):
            calls.append(kw)
            def kernel(**args):
                calls.append(args)
                return [A(shape,dtype) for shape,dtype in zip(args["output_shapes"],args["output_dtypes"])]
            return kernel
        mx=NS(float32="f32",bfloat16="bf16",fast=NS(metal_kernel=factory))
        q=A((1,48,128),"bf16");g=A((1,48),"f32");S=A((1,48,128,128),"f32")
        o,Sn=m.gated_delta_step_cached(q,q,q,g,g,S,backend=mx)
        self.assertEqual(set(calls[0]),{"name","input_names","output_names","source"})
        self.assertEqual(calls[1]["grid"],(128,192,1));self.assertEqual(calls[1]["threadgroup"],(32,4,1))
        self.assertEqual(Sn.shape,(1,48,128,128));self.assertEqual(o.dtype,"f32")
        with self.assertRaises(ValueError):m.gated_delta_step_cached(q,q,q,g,g,A((1,48,64,128),"f32"),backend=mx)
    def test_canonical_ownership(self):
        addresses=[]
        for tile in range(4):
            for dvlocal in range(32):
                dv=tile*32+dvlocal
                for sp in range(4):
                    for j in range(32):addresses.append((sp*32+j)*128+dv)
        self.assertEqual(sorted(addresses),list(range(128*128)))
    def test_compile_and_lazy_imports(self):
        for name in ("numeric.py","gdn_register_cache.py","check_cpu.py"):
            text=(D/name).read_text();compile(ast.parse(text),str(D/name),"exec")
            for node in ast.parse(text).body:
                if isinstance(node,ast.Import):self.assertFalse(any(n.name.startswith(("mlx","numpy")) for n in node.names))
        self.assertIn("Permission is hereby granted",(D/"LICENSE-MLX-MIT").read_text())
if __name__=="__main__":unittest.main()
