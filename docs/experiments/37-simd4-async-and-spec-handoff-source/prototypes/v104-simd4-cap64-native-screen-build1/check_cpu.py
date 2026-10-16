"""CPU-only clock/source/qualification gates; never import MLX or load native library."""
import ast
import json
from pathlib import Path
import runpy
import sys
import unittest
D=Path(__file__).resolve().parent;P=D.parent
S=(D/'screen.py').read_text();OLD=(P/'v90-cap64-native-decode-screen-build1/screen.py').read_text();F99=(P/'v99-simd4-cap64-fullqual-build1/fullqual.py').read_text()
def named(s,name):return next(n for n in ast.walk(ast.parse(s)) if isinstance(n,ast.FunctionDef) and n.name==name)
def dump(n):return ast.dump(n,include_attributes=False)
def clock(s):
 r=next(n for n in ast.walk(ast.parse(s)) if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple) and ast.unparse(n.target)=='(index, phase)')
 start=next(i for i,n in enumerate(r.body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='t0')
 end=next(i for i,n in enumerate(r.body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='t2')
 return ast.Module(body=r.body[start:end+1],type_ignores=[])
class Checks(unittest.TestCase):
 def test_clock_exact90_and44(self):
  for ref in (OLD,(P/'v44-native-decode-screen-build1/screen.py').read_text()):self.assertEqual(dump(clock(S)),dump(clock(ref)))
  calls=[ast.unparse(n.func) for n in ast.walk(clock(S)) if isinstance(n,ast.Call)]
  for forbidden in ('canonical_output','sha256','originals_readonly','write','sample','report','engagement'):self.assertFalse(any(forbidden in x for x in calls))
 def test_setup_schedule_capacity(self):
  self.assertEqual(dump(named(S,'setup')),dump(named(OLD,'setup')))
  for t in ['"schedule":["warm"]*3+["measure"]*3','AttnCache.step=CAP;DecodeBatch.step=CAP','max_len=CAP','native.call("v34_seal_feedback",128,CAP)','terminal_canonical_hash_checks"]==792']:self.assertIn(t,S)
  self.assertEqual(S.count('timing_raw=timing_fn()'),1)
 def test_capture_hook_exact99_and_finally(self):
  self.assertEqual(dump(named(S,'capture_norm_engagement')),dump(named(F99,'capture_norm_engagement')))
  for t in ['helper_capture_calls==norm_capture_calls and not active_gdn','hooks_removed_before_replay','g.set_simd4_segment(True)','assert g._simd4_segment_enabled is False']:self.assertIn(t,S)
 def test_native_uniform_and_fullschema(self):
  self.assertIn('P / "v99-native-simd4-cap64-build1/libv99replay.dylib"',S)
  self.assertIn('P/"v99-simd4-cap64-fullqual-build1/fullqual.py"',S)
  self.assertIn('os.environ.get("DYLD_INSERT_LIBRARIES")==str(LIB)',S)
  self.assertNotIn('v89-native-capacity64-build1/lib',S)
  for t in ['actual==predicted["active_arguments"]==qualified["active_arguments"]','len(pairs)==48','byname["S"]["access"]==0','byname["S_new"]["access"]==1','len(expected_source)==28']:self.assertIn(t,S)
 def test_canonical_gold_readonly_outside_clock(self):
  self.assertIn('len(byold)==3',S);self.assertIn('base.canonical_output(physical',S)
  self.assertIn('sum(v["physical_tail_zero_checked"] for v in checks)==32',S)
  self.assertEqual(S.count('base.originals_readonly('),2)
  self.assertIn('ids==w["continuation_ids"]',S);self.assertIn('"physical_KV_capacity":CAP',S)
 def test_inert_and_hold_refusal_before_gpu(self):
  m=runpy.run_path(str(D/'screen.py'));pins,q,expected=m['source_check']()
  self.assertEqual(len(expected),132);self.assertEqual(q['strict_output_checks'],12276)
  b=m['BINDING'];oldstatus=b['status'];b['status']='HOLD_TEST'
  try:
   with self.assertRaisesRegex(AssertionError,'HOLD'):m['source_check']()
   self.assertEqual(m['source_check'](allow_hold=True),(pins,None,None))
  finally:b['status']=oldstatus
  self.assertFalse(any(x=='mlx' or x.startswith('mlx.') or x=='numpy' for x in sys.modules))
if __name__=='__main__':unittest.main()
