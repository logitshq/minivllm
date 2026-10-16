"""Pure source/AST checks of the new protocol; no model/backend import or simulation."""
from pathlib import Path
import ast,hashlib,json,sys,unittest
D=Path(__file__).resolve().parent;P=D.parent;S=P/'v52-nvfp4-sg8-geometry-build1'
find=lambda t,name:next(n for n in ast.walk(t) if isinstance(n,ast.FunctionDef) and n.name==name)
dump=lambda n:ast.dump(n,include_attributes=False)
class Contract(unittest.TestCase):
 def test_exact_kernel_helper_and_compile_numeric_unchanged(self):
  for name in ('geometry.py','kernel_header.metal','kernel_body.metal','LICENSE-MLX.txt','checkpoint-headers.json'):
   self.assertEqual((D/name).read_bytes(),(S/name).read_bytes())
  before=ast.parse((S/'micro.py').read_text());after=ast.parse((D/'micro.py').read_text())
  for name in ('eager_original','compare','check_readonly','a_fn','b_fn','invoke'):
   self.assertEqual(dump(find(before,name)),dump(find(after,name)),name)
  def schedule(t):return next(n for n in ast.walk(t) if isinstance(n,ast.For) and ast.unparse(n.iter)=='enumerate(SCHEDULE)')
  a,b=schedule(before),schedule(after)
  for n in ast.walk(b):
   if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='longprime_block':n.func=ast.Attribute(value=ast.Name(id='old',ctx=ast.Load()),attr='block',ctx=ast.Load())
  self.assertEqual(dump(a),dump(b))
 def test_exact_timed_loop_and_two_synchronizations(self):
  old=find(ast.parse((P/'v22-nvfp4-qmv-r2-build1/micro.py').read_text()),'block')
  new=find(ast.parse((D/'micro.py').read_text()),'longprime_block')
  timed=lambda f:next(n for n in ast.walk(f) if isinstance(n,ast.For) and ast.unparse(n.iter)=='range(REPEATS)')
  self.assertEqual(dump(timed(old)),dump(timed(new)))
  calls=[ast.unparse(n.func) for n in ast.walk(new) if isinstance(n,ast.Call)]
  self.assertEqual(calls.count('mx.synchronize'),2)
  self.assertFalse(any(x in calls for x in ('write','open','hashlib.sha256','probe')))
  self.assertEqual({ast.unparse(n.func) for n in ast.walk(timed(new)) if isinstance(n,ast.Call)},{'range','invoke'})
 def test_fixed_four_chunks_and_gate_metadata(self):
  source=(D/'micro.py').read_text();tree=ast.parse(source)
  self.assertIn('PRIME, REPEATS = 8192, 2048',source)
  self.assertIn('PRIME_CHUNKS, PRIME_CHUNK_REPEATS = 4, 2048',source)
  block=find(tree,'longprime_block');outer=next(n for n in block.body if isinstance(n,ast.For));inner=next(n for n in outer.body if isinstance(n,ast.For))
  self.assertEqual(ast.unparse(outer.iter),'range(PRIME_CHUNKS)');self.assertEqual(ast.unparse(inner.iter),'range(PRIME_CHUNK_REPEATS)')
  self.assertEqual(ast.unparse(inner.body[0].value),'invoke(variant, chunk * PRIME_CHUNK_REPEATS + j)')
  # Chunk records cannot alter the loop bound, early-stop, or measured iteration count.
  self.assertFalse(any(isinstance(n,(ast.Break,ast.Continue,ast.While)) for n in ast.walk(block)))
  p=json.loads((D/'protocol.json').read_text());old=json.loads((S/'protocol.json').read_text())
  for k in ('repeats','schedule','minimum_reduction','minimum_wins','control_each','control_median','rotating_inputs','every_output_eval','timeout_s'):self.assertEqual(p['timing'][k],old['timing'][k])
  self.assertEqual(p['math'],old['math']);self.assertEqual(p['numeric'],old['numeric']);self.assertEqual(p['resources'],old['resources'])
  self.assertEqual(p['timing']['B'],'mx.compile(same R4 math SG8 helper,shapeless=False)')
 def test_no_gpu_import_or_execution_in_this_check(self):
  self.assertFalse(any(n=='mlx' or n.startswith(('mlx.','numpy','tutorial_code')) for n in sys.modules))
  for f in ('micro.py','geometry.py','check_cpu.py'):compile((D/f).read_text(),str(D/f),'exec')
if __name__=='__main__':unittest.main()
