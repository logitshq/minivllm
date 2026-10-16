"""V40 stdlib source/order and inert API checks; no GPU/MLX imports."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
D=Path(__file__).resolve().parent
P=D.parent
OLD=P/'candidate-v19-gdn-norm-reuse-build1-root/tutorial_code/miniengine/kernels/gdn_norm_reuse_metal.py'

def values(path):
    return {t.id:ast.literal_eval(n.value) for n in ast.parse(path.read_text()).body if isinstance(n,ast.Assign) and isinstance(n.value,(ast.Constant,ast.Dict)) for t in n.targets if isinstance(t,ast.Name)}

class Checks(unittest.TestCase):
    def test_only_recurrent_load_delta_and_ascending_order(self):
        old,new=values(OLD),values(D/'gdn_norm_reuse_cached_metal.py')
        for key in ('HEADER','_PREWORK_SOURCE','_NORM_BODY','_NORM_PROBE_SOURCE','COMPILE_OPTIONS'):
            self.assertEqual(old[key],new[key])
        first='''    float held[32];
    #pragma unroll 32
    for (uint j = 0; j < 32; j++) {
        uint d = d0 + j;
        held[j] = S[base + d * Dv + dv];
        kv += ks * kh[d] * held[j];
    }'''
        second='''    #pragma unroll 32
    for (uint j = 0; j < 32; j++) {
        uint d = d0 + j;
        float s = ex * held[j] + ks * kh[d] * delta;'''
        restored=new['_STATE_SOURCE'].replace(first,'    for (uint d = d0; d < d0 + seg; d++) kv += ks * kh[d] * S[base + d * Dv + dv];',1).replace(second,'    for (uint d = d0; d < d0 + seg; d++) {\n        float s = ex * S[base + d * Dv + dv] + ks * kh[d] * delta;',1)
        self.assertEqual(restored,old['_STATE_SOURCE'])
        self.assertEqual(new['_STATE_SOURCE'].count('S[base + d * Dv + dv]'),1)

    def test_lazy_api_metadata_and_unchanged_safe_policy(self):
        spec=importlib.util.spec_from_file_location('v40_cpu',D/'gdn_norm_reuse_cached_metal.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        class A:
            def __init__(self,shape,dtype):self.shape,self.dtype=shape,dtype
        calls=[]
        def factory(**kw):
            calls.append(('factory',kw))
            def kernel(**args):
                calls.append(('call',args));return [A(s,t) for s,t in zip(args['output_shapes'],args['output_dtypes'])]
            return kernel
        mx=NS(bfloat16='bf16',float32='f32',fast=NS(metal_kernel=factory));m._runtime=lambda:mx
        u=A((1,10240),'bf16');g=A((1,48),'f32');S=A((1,48,128,128),'f32')
        o,s=m.gated_delta_step_norm_reuse_metal(u,g,g,S)
        factories=[kw for kind,kw in calls if kind=='factory'];launches=[kw for kind,kw in calls if kind=='call']
        self.assertEqual(len(factories),2)
        self.assertTrue(all(kw['compile_options']=={'math_mode':'safe'} for kw in factories))
        self.assertEqual(launches[1]['grid'],(128,192,1));self.assertEqual(launches[1]['threadgroup'],(32,4,1))
        self.assertEqual((o.shape,s.shape),((1,48,128),(1,48,128,128)))
        with self.assertRaises(ValueError):m.gated_delta_step_norm_reuse_metal(u,g,g,A((1,48,64,128),'f32'))

    def test_syntax_and_inert_top_level(self):
        for name in ('numeric.py','gdn_norm_reuse_cached_metal.py','check_cpu.py'):
            tree=ast.parse((D/name).read_text());compile(tree,str(D/name),'exec')
            for node in tree.body:
                if isinstance(node,ast.Import):self.assertFalse(any(n.name.startswith(('mlx','numpy')) for n in node.names))
        self.assertIn('Permission is hereby granted',(D/'LICENSE-MLX-MIT').read_text())

if __name__=='__main__':unittest.main()
