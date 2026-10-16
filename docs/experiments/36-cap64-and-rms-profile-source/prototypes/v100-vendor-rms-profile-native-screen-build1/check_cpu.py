"""CPU-only V100 source/clock boundaries; no model or numerical simulation."""
import ast
from pathlib import Path
import runpy
import sys
import unittest
D=Path(__file__).resolve().parent;P=D.parent
S=(D/'screen.py').read_text();T=ast.parse(S)
OLD=(P/'v90-cap64-native-decode-screen-build1/screen.py').read_text();OT=ast.parse(OLD)
def named(t,name):return next(n for n in ast.walk(t) if isinstance(n,ast.FunctionDef) and n.name==name)
def dump(n):return ast.dump(n,include_attributes=False)
def request(t):return next(n for n in ast.walk(t) if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple) and [v.id for v in n.target.elts]==['index','phase'])
def clock(t):
 body=request(t).body
 def idx(name):return next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id==name)
 return dump(ast.Module(body=body[idx('t0'):idx('t2')+1],type_ignores=[]))
class Checks(unittest.TestCase):
 def test_inert_module_import(self):
  runpy.run_path(str(D/'screen.py'),run_name='v100_import_only')
  self.assertFalse(any(n=='mlx' or n.startswith(('mlx.','numpy')) for n in sys.modules))
 def test_exact90_t0_through_t2(self):self.assertEqual(clock(T),clock(OT))
 def test_original_setup_resources_unchanged(self):
  for name in ('setup','sample','layers','states'):self.assertEqual(dump(named(T,name)),dump(named(OT,name)),name)
 def test_preparation_outside_any_token_and_clock(self):
  prepare=S.index('profile=vendor.prepare_model(');firstsetup=S.index('warm,token=setup()');timing=S.index('t0=time.perf_counter_ns()')
  self.assertLess(prepare,firstsetup);self.assertLess(prepare,timing)
  self.assertIn('len(profile.entries)==161',S);self.assertIn('len(gamma_holders)==322',S)
  self.assertIn('Araw.address+Araw.nbytes<=Bcached.address or Bcached.address+Bcached.nbytes<=Araw.address',S)
  self.assertIn('base.metadata_finite(np,Araw)',S);self.assertNotIn('\n            am,bm=metadata_finite(',S)
 def test_161_gamma_pins_RO_and_hook_removal(self):
  for text in ('for h in cached_gamma_holders:native.call("v33_pin"','len(native_gamma_proof)==161','len(plain_capture_calls)==161','(48,48,83)','gamma322_readonly_final'):
   self.assertIn(text,S)
  n=named(T,'capture_norm_engagement');b=next(n for n in n.body if isinstance(n,ast.Try));self.assertTrue(b.finalbody)
  final=ast.unparse(ast.Module(body=b.finalbody,type_ignores=[]));self.assertIn('PlainRMSNorm.__call__ = candidate_plain',final)
  self.assertLess(S.index('assert PlainRMSNorm.__call__ is candidate_plain'),S.index('for index,phase in enumerate'))
 def test_only_actual97_new_profile_reference(self):
  self.assertIn('DIAGNOSTIC_COMPLETED_NEW_REFERENCE_STRICT_NO_ADOPTION',S)
  self.assertIn('expected_ids=qualification["requests"][0]["A_ids"]',S)
  self.assertNotIn('w["continuation_ids"]',S);self.assertNotIn('OLD44_SHA',S)
  self.assertIn('canonical256 final output differs from actual97 independent newA',S)
  self.assertIn('COMPLETED_V100_VENDOR_PLAIN_RMS_B_ONLY_EXPLORATORY_SCREEN_NO_ADOPTION',S)
  self.assertIn('sha(INDEPENDENT_REVIEW97)==INDEPENDENT_REVIEW97_SHA',S)
 def test_hashes_RO_probes_outside_clock(self):
  f=request(T);b=f.body
  t2=next(i for i,n in enumerate(b) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='t2')
  tail=ast.unparse(ast.Module(body=b[t2+1:],type_ignores=[]))
  for marker in ('sample(', 'hashlib.sha256(', 'base.originals_readonly(', 'profile.validate_current_bindings()', 'timing_fn()'):self.assertIn(marker,tail)
  self.assertIn('"terminal_canonical_hash_checks"]==792',S)
if __name__=='__main__':unittest.main()
