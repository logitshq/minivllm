"""Pure stdlib source/API/geometry contracts; injected fake backend, no MLX/GPU."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import sys,unittest
D=Path(__file__).resolve().parent;P=D.parent
VERSION=60
HELPER='gdn_activated_exp48_metal.py'
OLD=P/'v44-gdn-activated-normreuse-build1/gdn_activated_norm_reuse_cached_metal.py'
def vals(path):
 return {t.id:ast.literal_eval(n.value) for n in ast.parse(path.read_text()).body if isinstance(n,ast.Assign)and isinstance(n.value,(ast.Constant,ast.Dict))for t in n.targets if isinstance(t,ast.Name)}
def fun(path,name):return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)and n.name==name)
class Check(unittest.TestCase):
 def test_only_declared_source_change(self):
  old,new=vals(OLD),vals(D/HELPER)
  for k in ['HEADER','_NORM_BODY','_NORM_PROBE_SOURCE','COMPILE_OPTIONS','SPLIT']:self.assertEqual(old[k],new[k])
  if VERSION==60:
   self.assertEqual(new['VALUE_TILE'],32)
   self.assertEqual(new['_STATE_SOURCE'].replace('float ex = exp_ready[bh];','float ex = metal::exp(g[bh]);'),old['_STATE_SOURCE'])
   self.assertEqual(new['_PREWORK_SOURCE'].replace('        if (lane == 0) exp_ready[h] = metal::exp(g[h]);\n',''),old['_PREWORK_SOURCE'])
   self.assertNotIn('metal::exp(',new['_STATE_SOURCE'])
   self.assertIn('float ex = metal::exp(g[bh]);',new['_EXP_REFERENCE_PROBE_SOURCE'])
  else:
   self.assertEqual(new['VALUE_TILE'],64)
   for k in ['_PREWORK_SOURCE','_STATE_SOURCE']:self.assertEqual(old[k],new[k])
   self.assertEqual(ast.dump(fun(OLD,'_prepare')),ast.dump(fun(D/HELPER,'_prepare')))
   newfn=ast.unparse(fun(D/HELPER,'_state')).replace('gdn_tile64_norm_reuse_cached_state_v61_build1','gdn_norm_reuse_cached_state_v40_build1')
   self.assertEqual(newfn,ast.unparse(fun(OLD,'_state')))
 def test_lazy_injected_api_and_safe_launch(self):
  spec=importlib.util.spec_from_file_location('candidate_cpu',D/HELPER);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  class A:
   def __init__(self,shape,dtype):self.shape,self.dtype=shape,dtype
  factories=[];launches=[]
  def factory(**kw):
   factories.append(kw)
   def kernel(**args):launches.append(args);return [A(s,t)for s,t in zip(args['output_shapes'],args['output_dtypes'])]
   return kernel
  mx=NS(bfloat16='bf16',float32='f32',fast=NS(metal_kernel=factory));m._runtime=lambda:mx
  activated=A((1,10240),'bf16');g=A((1,48),'f32');S=A((1,48,128,128),'f32')
  out=m.gated_delta_step_activated_norm_reuse_metal(activated,g,g,S)
  self.assertEqual(len(out),2);self.assertEqual(launches[1]['grid'],(128,192,1));self.assertEqual(launches[1]['threadgroup'],(32 if VERSION==60 else 64,4,1))
  self.assertEqual(launches[1]['output_shapes'],[(1,48,128),(1,48,128,128)])
  self.assertEqual(factories[0]['input_names'],['activated','g']if VERSION==60 else['activated'])
  if VERSION==60:
   self.assertEqual(launches[0]['output_shapes'][-1],(1,48));self.assertEqual(len(launches[0]['output_shapes']),5)
   self.assertIsNot(launches[1]['inputs'][4],g);self.assertEqual(launches[1]['inputs'][4].shape,(1,48))
   probe=m._exp_reference_probe(g,mx);self.assertEqual(probe.shape,(1,48,128,4));self.assertEqual(launches[-1]['grid'],(128,192,1));self.assertEqual(launches[-1]['threadgroup'],(32,4,1))
  else:self.assertIs(launches[1]['inputs'][4],g)
  self.assertTrue(all(x['compile_options']=={'math_mode':'safe'}for x in factories))
  self.assertIn('v'+str(VERSION)+'_build1',factories[1]['name'])
  with self.assertRaises(ValueError):m.gated_delta_step_activated_norm_reuse_metal(A((1,10240),'f32'),g,g,S)
 def test_canonical_value_ownership_and_ascending_serial_loops(self):
  v=vals(D/HELPER);tile=v['VALUE_TILE'];allvalues=[group*tile+local for group in range(128//tile)for local in range(tile)]
  self.assertEqual(allvalues,list(range(128)));self.assertEqual([sp*32+j for sp in range(4)for j in range(32)],list(range(128)))
  self.assertEqual(v['_STATE_SOURCE'].count('#pragma unroll 32'),2);self.assertIn('for (uint i = 0; i < SPLIT; i++) kv += red[i * VALUE_TILE + dv_local];',v['_STATE_SOURCE'])
  self.assertIn('for (uint i = 0; i < SPLIT; i++) acc += red[i * VALUE_TILE + dv_local];',v['_STATE_SOURCE'])
 def test_stdlib_inert_and_full_mit(self):
  for f in [D/HELPER,D/'numeric.py',D/'check_cpu.py']:
   tree=ast.parse(f.read_text());compile(tree,str(f),'exec')
   for n in tree.body:
    if isinstance(n,ast.Import):self.assertFalse(any(x.name.startswith(('mlx','numpy'))for x in n.names))
  self.assertIn('Permission is hereby granted',(D/'LICENSE-MLX-MIT').read_text());self.assertFalse(any(n=='mlx'or n.startswith('mlx.')for n in sys.modules))
class FailureMetadata(unittest.TestCase):
 def test_real_except_ast_preserves_original_error_and_counts_completed_rows(self):
  import traceback
  t=ast.parse((D/'numeric.py').read_text());handler=next(x for x in ast.walk(t) if isinstance(x,ast.ExceptHandler) and x.name=='error')
  self.assertFalse(any(isinstance(n,ast.Assert) for st in handler.body for n in ast.walk(st)))
  # Execute the actual exception handler statements in a stdlib-only wrapper.
  fn=ast.parse('def simulate(cases):\n r={"cases":cases}\n try: raise ValueError("ORIGINAL_GPU_ERROR_SENTINEL")\n except BaseException as error:\n  pass\n return r\n').body[0]
  fn.body[1].handlers[0].body=handler.body[:-1]
  code=compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'actual_exception_handler','exec')
  ns={'traceback':traceback};exec(code,ns)
  for cases,expected in [([],0),([{"exp_equality":{"strict_equal":True}}],1),([{}, {"exp_equality":{"strict_equal":False}}, {"exp_equality":{"strict_equal":True}}],1)]:
   result=ns['simulate'](cases)
   self.assertEqual(result['strict_exp_bank_checks'],expected);self.assertEqual(result['status'],'FAIL_STOP')
   self.assertIn('ORIGINAL_GPU_ERROR_SENTINEL',result['error']);self.assertIn('ValueError',result['traceback'])
  self.assertEqual((D/'gdn_activated_exp48_metal.py').read_bytes(),(P/'v60-gdn-exp48-build1/gdn_activated_exp48_metal.py').read_bytes())

if __name__=='__main__':unittest.main()
