"""V93 focused CPU source proofs only. No MLX/NumPy/native import/execution."""
import ast
from pathlib import Path
import runpy
import sys
import unittest
D=Path(__file__).resolve().parent;P=D.parent
S=(D/'fullqual.py').read_text();T=ast.parse(S)
OLD=(P/'v89-cap64-native-fullqual-build1/fullqual.py').read_text()
def named(t,name):return next(n for n in ast.walk(t) if isinstance(n,ast.FunctionDef) and n.name==name)
def dump(n):return ast.dump(n,include_attributes=False)
class Checks(unittest.TestCase):
 def test_import_inert_and_canonical_unchanged(self):
  runpy.run_path(str(D/'fullqual.py'),run_name='cpu_import_only')
  self.assertFalse(any(n=='mlx' or n.startswith(('mlx.','numpy')) for n in sys.modules))
  self.assertEqual(dump(named(T,'canonical_output')),dump(named(ast.parse(OLD),'canonical_output')))
  self.assertEqual(dump(named(T,'assert_icb')),dump(named(ast.parse(OLD),'assert_icb')))
 def test_raw_reference_no_legacy_gdn(self):
  n=named(T,'raw_gamma_reference');text=ast.unparse(n)
  self.assertIn('with profile.reference_context()',text);self.assertNotIn('legacy',text)
  self.assertFalse(any(isinstance(a,ast.Assign) and any(isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and t.value.id=='GDN' for t in a.targets) for a in ast.walk(n)))
  self.assertNotIn('legacy_decode',S);self.assertNotIn('old_decode',S)
 def test_independent_prefill_and_B_seed(self):
  n=named(T,'independent_prefill_pair');text=ast.unparse(n)
  self.assertIn("A = setup(prompt, True",text);self.assertIn("B = setup(prompt, False",text)
  self.assertIn('native.lib.memcmp',text);self.assertIn('separate_array_handles',text)
  setup=named(T,'setup');w=next(n for n in setup.body if isinstance(n,ast.With))
  calls={ast.unparse(n.func) for n in ast.walk(w) if isinstance(n,ast.Call)}
  self.assertTrue({'make_cache','model','mx.eval','mx.synchronize','shared'}<=calls)
  self.assertIn('seed_values,first=seed_pair["B"]["seed"],seed_pair["B"]["first"]',S)
  self.assertIn('a_flat=tuple(v[:] for v in a_seed_values[3:])',S)
  self.assertIn('"seed131_strict_checks":393',S);self.assertIn('"last_logits_strict_checks":3',S)
 def test_decode_A_graph_inside_context(self):
  scopes=[n for n in ast.walk(T) if isinstance(n,ast.With) and any(isinstance(v.context_expr,ast.Call) and isinstance(v.context_expr.func,ast.Name) and v.context_expr.func.id=='raw_gamma_reference' for v in n.items)]
  self.assertEqual(len(scopes),1);text=ast.unparse(scopes[0])
  for call in ('logits_a = model','mx.eval(*expected_values)','mx.synchronize()','expected = [shared'):self.assertIn(call,text)
 def test_gamma_and_capture(self):
  self.assertIn('e.raw_gamma is not e.cached_gamma',S);self.assertIn('Araw.address+Araw.nbytes<=Bcached.address or Bcached.address+Bcached.nbytes<=Araw.address',S)
  self.assertIn('native.lib.memcmp(Araw.address,Bcached.address,Araw.nbytes)==0',S)
  self.assertIn('for h in cached_gamma_holders:native.call("v33_pin"',S)
  n=named(T,'capture_norm_engagement');block=next(n for n in n.body if isinstance(n,ast.Try));self.assertTrue(block.finalbody)
  final=ast.unparse(ast.Module(body=block.finalbody,type_ignores=[]))
  self.assertIn('PlainRMSNorm.__call__ = candidate_plain',final);self.assertIn('GDN._decode_conv_norm_reuse_cache = candidate_norm',final)
  self.assertIn('len(plain_capture_calls)==161',S);self.assertIn('len(native_gamma_proof)==161',S)
 def test_old_hash_diagnostic_IDS_hard(self):
  self.assertIn('old_profile_equal=canonical_digest==archived["A_sha256"]',S)
  self.assertNotIn('assert canonical_digest==archived["A_sha256"]',S)
  self.assertIn('assert a_token==archived_cases[name]["A_ids"][step]',S)
  self.assertIn('assert ids_a==archived_cases[name]["A_ids"] and len(case["steps"])==31',S)
  self.assertIn('==96',S);self.assertIn('"old_profile_pass":False',S)
  self.assertIn('native.call("v34_compare_output",original.address,original.nbytes,ctypes.c_void_p(reference.address))',S)
  self.assertNotIn('allclose',S);self.assertNotIn('tolerance',S)
if __name__=='__main__':unittest.main()
