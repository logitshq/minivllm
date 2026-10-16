"""V97 diagnostic source boundaries; CPU stdlib only, no numerical stand-ins."""
import ast
from pathlib import Path
import runpy
import sys
import unittest
D=Path(__file__).resolve().parent;P=D.parent
S=(D/'fullqual.py').read_text();T=ast.parse(S)
OLD=(P/'v93-vendor-rms-cap64-fullqual-build1/fullqual.py').read_text();OT=ast.parse(OLD)
def named(t,name):return next(n for n in ast.walk(t) if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
def dump(n):return ast.dump(n,include_attributes=False)
class Checks(unittest.TestCase):
 def test_import_no_MLX_numpy(self):
  runpy.run_path(str(D/'fullqual.py'),run_name='v97_cpu_import')
  self.assertFalse(any(n=='mlx' or n.startswith(('mlx.','numpy')) for n in sys.modules))
 def test_native_and_numeric_helpers_exact93(self):
  for name in ('Native','canonical_output','metadata_finite','originals_readonly','assert_icb','ctx_layers','state_of','outputs','raw_gamma_reference','capture_norm_engagement'):
   self.assertEqual(dump(named(T,name)),dump(named(OT,name)),name)
 def test_independent_prefill_and_reference_scope(self):
  s=ast.unparse(named(T,'setup'))
  for text in ('with raw_gamma_reference() if reference else nullcontext()', 'make_cache(model)', 'mx.eval(*seed)', 'mx.synchronize()', 'shared(v, label'):
   self.assertIn(text,s)
  pair=ast.unparse(named(T,'independent_prefill_pair'))
  self.assertIn('A = setup(prompt, True',pair);self.assertIn('B = setup(prompt, False',pair)
  self.assertIn('native.lib.memcmp',pair);self.assertIn("assert A['first'] == B['first'] and A['first'] not in eos",pair)
  scopes=[n for n in ast.walk(T) if isinstance(n,ast.With) and any(isinstance(i.context_expr,ast.Call) and isinstance(i.context_expr.func,ast.Name) and i.context_expr.func.id=='raw_gamma_reference' for i in n.items)]
  self.assertEqual(len(scopes),1);text=ast.unparse(scopes[0])
  for c in ('logits_a = model','mx.eval(*expected_values)','mx.synchronize()','expected = [shared'):self.assertIn(c,text)
 def test_no_old_ID_gate_new_pair_strict(self):
  for old in ('assert warm_ids == w["continuation_ids"]','assert a_token==archived_cases[name]["A_ids"][step]','assert ids_a==archived_cases[name]["A_ids"]','assert A["first"]==B["first"]==archived_cases'):
   self.assertNotIn(old,S)
  self.assertIn('assert a_token==b_token',S);self.assertIn('assert ids_a==ids_b',S)
  self.assertIn('assert len(ids_a)==len(ids_b)==32',S);self.assertIn('assert all(t not in eos for t in ids_a+ids_b)',S)
  self.assertIn('sum(len(r["A_ids"])',S);self.assertIn('==96',S)
  self.assertIn('"firsttoken_new_A_B_strict_checks":3',S);self.assertNotIn('"firsttoken_oldID_checks":3',S)
 def test_132strict_gamma_RO_and_no_future_B_feeding(self):
  for c in ('native.call("v34_compare_output",original.address,original.nbytes,ctypes.c_void_p(reference.address))',
   '"seed_uploads_this_step":0','native.call("v34_write_input",original.address,original.nbytes,ctypes.c_void_p(current.address))',
   'seed_values,first=seed_pair["B"]["seed"],seed_pair["B"]["first"]',
   'a_flat=tuple(v[:] for v in a_seed_values[3:])','case["gamma322_readonly"]','"gamma322_readonly_final"',
   'len(native_gamma_proof)==161','len(plain_capture_calls)==161','(48,48,83)'):
   self.assertIn(c,S)
  self.assertEqual(S.count('native.call("v34_write_input"'),1)
  self.assertNotIn('allclose',S)
 def test_warm_saved_before_further_work_and_donor_bound(self):
  warm=S.index('receipt["ordinary_warm32_diagnostic"]')
  save=S.index('write(args.out/"ordinary-warm32-ids-diagnostic.json"',warm)
  capture=S.index('capture_pair=independent_prefill_pair',save)
  self.assertLess(warm,save);self.assertLess(save,capture)
  self.assertIn('"donor_visible_A":id_difference(ids_a,donor_visible) if canonical else None',S)
  self.assertIn('"donor_reference_is_raw_generation_ids":False',S)
  self.assertIn('DIAGNOSTIC_COMPLETED_NEW_REFERENCE_STRICT_NO_ADOPTION',S)
  self.assertNotIn('status="PASS_V93',S)
 def test_actual_ID_difference_function(self):
  ns={};node=named(T,'id_difference');exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'v97-ids','exec'),ns)
  f=ns['id_difference'];self.assertTrue(f([1,2],[1,2])['equal']);r=f([1,3],[1,2]);self.assertFalse(r['equal']);self.assertEqual(r['first_difference_index0'],1);self.assertEqual(r['different_aligned_positions'],1)
  self.assertEqual(f([1],[1,2])['first_difference_index0'],1);self.assertEqual(f([],[])['first_difference_index0'],None)
  self.assertIs(f([1],[2])['gate'],False)
if __name__=='__main__':unittest.main()
