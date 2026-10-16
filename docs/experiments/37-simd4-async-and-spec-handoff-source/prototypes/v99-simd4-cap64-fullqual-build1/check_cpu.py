"""V99 CPU source/guard checks, no MLX/NumPy/dylib/GPU execution."""
import ast
import hashlib
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
import unittest
D=Path(__file__).resolve().parent;P=D.parent;N=P/'v99-native-simd4-cap64-build1';C=P/'candidate-v99-simd4-segment-build1-root';Q=C/'tutorial_code/miniengine';OLD=P/'candidate-v44-conv-normreuse-build1-root/tutorial_code/miniengine'
S=(D/'fullqual.py').read_text();T=ast.parse(S)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def named(t,name):return next(n for n in ast.walk(t) if isinstance(n,ast.FunctionDef) and n.name==name)
def dump(n):return ast.dump(n,include_attributes=False)
class Checks(unittest.TestCase):
 def test_candidate_parent_and_helper(self):
  files=list(Q.rglob('*.py'));self.assertEqual(len(files),28)
  for p in OLD.rglob('*.py'):
   if p.relative_to(OLD).as_posix()!='model/gdn.py':self.assertEqual(p.read_bytes(),(Q/p.relative_to(OLD)).read_bytes())
  self.assertEqual(sha(Q/'kernels/gdn_activated_simd4_segment_metal.py'),'7f2d0136301079436839fbdee494a5c8849ffdd1f8861c9f78be05029819c5a6')
  for p in files:compile(p.read_text(),str(p),'exec')
  oldtree=ast.parse((OLD/'model/gdn.py').read_text());newtree=ast.parse((Q/'model/gdn.py').read_text())
  for name in ('__call__','_decode','_decode_dispatch','_prefill','_prefill_batch','_gates','_decode_gates'):
   self.assertEqual(dump(named(oldtree,name)),dump(named(newtree,name)),name)
 def test_actual_strict_bool_default(self):
  tree=ast.parse((Q/'model/gdn.py').read_text());fn=named(tree,'set_simd4_segment');ns={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'setter','exec'),ns)
  obj=SimpleNamespace();setter=ns['set_simd4_segment']
  for v in (False,True):setter(obj,v);self.assertIs(obj._simd4_segment_enabled,v)
  for v in (0,1,None,[],{},'true'):
   with self.assertRaises(TypeError):setter(obj,v)
  self.assertIn('self.set_simd4_segment(False)',(Q/'model/gdn.py').read_text())
 def test_exact_native_allowlist_and_guard(self):
  new=(N/'replay.mm').read_text();old=(P/'v89-native-capacity64-build1/replay.mm').read_text()
  start=old.index('extern "C" int v36_register_pingpong');end=old.index('extern "C" const char*v36_timing_json')
  before=old[start:end];after=new[new.index('extern "C" int v36_register_pingpong'):new.index('extern "C" const char*v36_timing_json')]
  after=after.replace(' const std::string expected_v95="custom_kernel_gdn_simd4_segment_cached_state_v95_build1_float_float_float_float_float_float_float_float_float";\n','').replace('const bool known=actual==expected||((actual==expected_v40||actual==expected_v95)&&ar.index==6&&br.index==9);','const bool known=actual==expected||(actual==expected_v40&&ar.index==6&&br.index==9);')
  self.assertEqual(after,before)
  predicate='token<0||pos<0||pos>=(int)F().capacity||len!=pos+1||len>(int)F().capacity'
  self.assertIn(predicate,new)
  self.assertIn(predicate.replace('F().capacity','capacity'),(N/'guard_cpu.cpp').read_text())
  self.assertIn('original.args',new);self.assertIn('@"active_arguments":args',new)
 def test_predicted_donor_schema_exact(self):
  p=json.loads((N/'predicted-schema.json').read_text());self.assertEqual(p['status'],'PREDICTED_NOT_OBSERVED')
  self.assertEqual(sha(Path(p['donor_actual_schema_path'])),p['donor_actual_schema_sha256'])
  r=json.loads(Path(p['donor_actual_schema_path']).read_text());donor=next(v for v in r['dispatch_records'] if v['function']==p['donor_actual_function'])
  expected=[{'index':a['index'],'name':a['name'],'type':0,'access':a['reflection_access']} for a in donor['actual_active_arguments']]
  self.assertEqual(p['active_arguments'],expected);self.assertEqual([x['access'] for x in expected],[0]*8+[1,1])
  self.assertIn('assert actual_schema==predicted["active_arguments"]',S)
  self.assertIn('byname["S_new"]["access"]==1',S)
 def test_fullqual_oldhardgates_and_hooks(self):
  old=ast.parse((P/'v89-cap64-native-fullqual-build1/fullqual.py').read_text())
  self.assertEqual(dump(named(T,'canonical_output')),dump(named(old,'canonical_output')))
  self.assertIn('canonical_digest!=archived["A_sha256"]',S);self.assertIn('archived44==archived',S)
  self.assertIn('assert a_token==archived_cases[name]["A_ids"][step]',S);self.assertIn('==96',S)
  self.assertIn('helper_capture_calls==norm_capture_calls',S)
  for name in ('baseline_decode','capture_norm_engagement'):
   n=named(T,name);self.assertTrue(any(isinstance(x,ast.Try) and x.finalbody for x in n.body))
  n=named(T,'baseline_decode');text=ast.unparse(n);self.assertIn('g.set_simd4_segment(False)',text);self.assertIn('g.set_simd4_segment(flag)',text)
  self.assertNotIn('vendor_rms',S);self.assertNotIn('allclose',S)
 def test_inert_import_and_uint32_guard(self):
  ns=runpy.run_path(str(D/'fullqual.py'),run_name='cpu_source_only')
  self.assertFalse(any(n=='mlx' or n.startswith(('mlx.','numpy')) for n in sys.modules))
  native=object.__new__(ns['Native']);seen=[];native.lib=SimpleNamespace(v34_seal_feedback=lambda *a:seen.append(a) or 0)
  native.call('v34_seal_feedback',128,64);self.assertEqual(seen,[(128,64)])
  for cap in (0,1,32,63,65,128,257,-1,1<<32,(1<<32)+64):
   with self.assertRaises(AssertionError):native.call('v34_seal_feedback',128,cap)
if __name__=='__main__':unittest.main()
