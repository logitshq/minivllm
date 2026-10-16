"""CPU clock/source checks only; no MLX/NumPy/dylib execution."""
import ast
from pathlib import Path
import unittest
D=Path(__file__).resolve().parent;P=D.parent
S=(D/"screen.py").read_text();OLD=(P/"v44-native-decode-screen-build1/screen.py").read_text()
def named(s,name):return next(n for n in ast.walk(ast.parse(s)) if isinstance(n,ast.FunctionDef) and n.name==name)
def dump(n):return ast.dump(n,include_attributes=False)
def clock(s):
 r=next(n for n in ast.walk(ast.parse(s)) if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple) and ast.unparse(n.target)=='(index, phase)')
 start=next(i for i,n in enumerate(r.body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='t0')
 end=next(i for i,n in enumerate(r.body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='t2')
 return ast.Module(body=r.body[start:end+1],type_ignores=[])
class Checks(unittest.TestCase):
 def test_whole_clock_exact44(self):
  self.assertEqual(dump(clock(S)),dump(clock(OLD)))
  calls=[ast.unparse(n.func) for n in ast.walk(clock(S)) if isinstance(n,ast.Call)]
  for forbidden in ('canonical_output','sha256','originals_readonly','write','sample','report'):self.assertFalse(any(forbidden in x for x in calls))
 def test_fresh_setup_exact44(self):self.assertEqual(dump(named(S,'setup')),dump(named(OLD,'setup')))
 def test_true_capacity_and_terminal_proof(self):
  self.assertIn('AttnCache.step=CAP;DecodeBatch.step=CAP',S);self.assertIn('max_len=CAP',S);self.assertIn('native.call("v34_seal_feedback",128,CAP)',S)
  self.assertIn('base.canonical_output(physical',S);self.assertIn('sum(v["physical_tail_zero_checked"] for v in checks)==32',S)
  self.assertIn('terminal_canonical_hash_checks"]==792',S)
  self.assertEqual(S.count('base.originals_readonly('),2)
  self.assertIn('"physical_KV_capacity":CAP',S)
  self.assertIn('P/"v89-cap64-native-fullqual-build1/fullqual.py"',S)
 def test_inert_boundary_and_hooks(self):
  t=ast.parse(S);main=named(S,'main')
  inert=next(n for n in main.body if isinstance(n,ast.If) and ast.unparse(n.test)=='args.inert')
  self.assertTrue(any(isinstance(n,ast.Return) for n in inert.body))
  self.assertFalse(any(isinstance(n,(ast.Import,ast.ImportFrom)) and any(v.name.startswith(('mlx','numpy')) for v in n.names) for n in t.body))
  self.assertEqual(dump(named(S,'capture_norm_engagement')),dump(named(OLD,'capture_norm_engagement')))
  self.assertIn('"schedule":["warm"]*3+["measure"]*3',S)
  self.assertEqual(S.count('timing_raw=timing_fn()'),1)
if __name__=='__main__':unittest.main()
