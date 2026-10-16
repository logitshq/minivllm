"""CPU-only adapter ownership/source proof, no model/runtime simulation."""
import ast
import copy
from pathlib import Path
import runpy
import sys
import unittest
D=Path(__file__).resolve().parent;P=D.parent
S=(D/'native_runner.py').read_text();T=ast.parse(S)
def named(t,name):return next(n for n in ast.walk(t) if isinstance(n,ast.FunctionDef) and n.name==name)
def dump(n):return ast.dump(n,include_attributes=False)
class Restore(ast.NodeTransformer):
 def visit_Tuple(self,n):
  n=self.generic_visit(n)
  if len(n.elts)==5 and [x.value if isinstance(x,ast.Constant) else None for x in n.elts[:4]]==[131,132,131,128]:n.elts[-1]=ast.Constant(value=256)
  return n
class Checks(unittest.TestCase):
 def test_default_off_import_and_bool_guard(self):
  d=runpy.run_path(str(D/'native_runner.py'),run_name='v102_cpu_only')
  self.assertFalse(d['configure_physical_cap64']()['enabled'])
  for v in (None,0,1,'true',[],{}):
   with self.assertRaises(TypeError):d['configure_physical_cap64'](enabled=v)
  self.assertFalse(any(n=='mlx' or n.startswith(('mlx.','numpy')) for n in sys.modules))
 def test_inverse_build_exact41(self):
  old=ast.parse((P/'v41-native-runner-build1/native_runner.py').read_text())
  b=copy.deepcopy(named(T,'build'));self.assertEqual(len(b.body[-4:]),4)
  b.body=b.body[2:-4];b=Restore().visit(b)
  self.assertEqual(dump(b),dump(named(old,'build')))
 def test_FIFO_run_begin_drop_inherited_unmodified(self):
  c=next(n for n in T.body if isinstance(n,ast.ClassDef));self.assertEqual(ast.unparse(c.bases[0]),'old.NativeRunner')
  names={n.name for n in c.body if isinstance(n,ast.FunctionDef)};self.assertEqual(names,{'__init__','build'})
  self.assertIn('v64-native-direct-async-runner-build1/native_runner.py',S)
  for x in ('self.call("v55_enable_pipeline", 2)','self.capacity = 64','(AttnCache.step, DecodeBatch.step) == (64, 64)','(1, 4, 64, 256)'):
   self.assertIn(x,S)
 def test_setup_no_tensor_creation_or_weight_patch(self):
  n=named(T,'configure_physical_cap64');text=ast.unparse(n)
  self.assertNotIn('mx.',text);self.assertNotIn('.model',text)
  self.assertIn('AttnCache.step = 64',text);self.assertIn('DecodeBatch.step = 64',text)
  self.assertNotIn('DYLD',S);self.assertNotIn('mx.compile',S)
if __name__=='__main__':unittest.main()
