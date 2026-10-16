"""V44 stdlib checks: no activation in prework, exact V40 state/policy, lazy API."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
D=Path(__file__).resolve().parent
OLD=D.parent/'v40-gdn-normreuse-cache-build1/gdn_norm_reuse_cached_metal.py'

def vals(path):
    return {t.id:ast.literal_eval(n.value) for n in ast.parse(path.read_text()).body if isinstance(n,ast.Assign) and isinstance(n.value,(ast.Constant,ast.Dict)) for t in n.targets if isinstance(t,ast.Name)}

def func(path,name):return next(x for x in ast.parse(path.read_text()).body if isinstance(x,ast.FunctionDef) and x.name==name)

class Checks(unittest.TestCase):
    def test_exact_state_and_only_three_activated_load_changes(self):
        old,new=vals(OLD),vals(D/'gdn_activated_norm_reuse_cached_metal.py')
        for key in ('_STATE_SOURCE','HEADER','_NORM_BODY','_NORM_PROBE_SOURCE','COMPILE_OPTIONS'):
            self.assertEqual(old[key],new[key])
        self.assertEqual(ast.dump(func(OLD,'_state'),include_attributes=False),ast.dump(func(D/'gdn_activated_norm_reuse_cached_metal.py','_state'),include_attributes=False))
        source=new['_PREWORK_SOURCE']
        self.assertNotIn('v18_bf16_activation(',source)
        for index in ('h * 128 + lane','2048 + h * 128 + lane','4096 + h * 128 + lane'):
            source=source.replace('float(activated['+index+'])','float(v18_bf16_activation(u['+index+']).activated)',1)
        self.assertEqual(source,old['_PREWORK_SOURCE'])

    def test_activated_api_and_unchanged_kernel_metadata(self):
        spec=importlib.util.spec_from_file_location('v44_cpu',D/'gdn_activated_norm_reuse_cached_metal.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        class A:
            def __init__(self,shape,dtype):self.shape,self.dtype=shape,dtype
        factories=[];launches=[]
        def factory(**kw):
            factories.append(kw)
            def kernel(**args):
                launches.append(args);return [A(s,t) for s,t in zip(args['output_shapes'],args['output_dtypes'])]
            return kernel
        mx=NS(bfloat16='bf16',float32='f32',fast=NS(metal_kernel=factory));m._runtime=lambda:mx
        activated=A((1,10240),'bf16');g=A((1,48),'f32');S=A((1,48,128,128),'f32')
        m.gated_delta_step_activated_norm_reuse_metal(activated,g,g,S)
        self.assertEqual(factories[0]['input_names'],['activated'])
        self.assertEqual(factories[1]['name'],'gdn_norm_reuse_cached_state_v40_build1')
        self.assertTrue(all(x['compile_options']=={'math_mode':'safe'} for x in factories))
        self.assertEqual(launches[1]['grid'],(128,192,1));self.assertEqual(launches[1]['threadgroup'],(32,4,1))
        with self.assertRaises(ValueError):m.gated_delta_step_activated_norm_reuse_metal(A((1,10240),'f32'),g,g,S)

    def test_syntax_inert_and_license(self):
        for name in ('numeric.py','gdn_activated_norm_reuse_cached_metal.py','check_cpu.py'):
            tree=ast.parse((D/name).read_text());compile(tree,str(D/name),'exec')
            for n in tree.body:
                if isinstance(n,ast.Import):self.assertFalse(any(x.name.startswith(('mlx','numpy')) for x in n.names))
        self.assertIn('Permission is hereby granted',(D/'LICENSE-MLX-MIT').read_text())

if __name__=='__main__':unittest.main()
