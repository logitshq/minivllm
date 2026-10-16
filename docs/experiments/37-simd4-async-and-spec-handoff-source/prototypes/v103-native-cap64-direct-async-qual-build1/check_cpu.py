"""V103 CPU source/byte-boundary checks only; no MLX/NumPy/native runtime."""
import ast
from pathlib import Path
import runpy
import sys
import unittest
D=Path(__file__).resolve().parent;P=D.parent
S=(D/'fullqual.py').read_text();T=ast.parse(S)
def named(name):return next(n for n in ast.walk(T) if isinstance(n,ast.FunctionDef) and n.name==name)
class Checks(unittest.TestCase):
 def test_import_is_inert(self):
  runpy.run_path(str(D/'fullqual.py'),run_name='v103_cpu_only')
  self.assertFalse(any(n=='mlx' or n.startswith(('mlx.','numpy')) for n in sys.modules))
 def test_startup_actual64_before_engine_or_token(self):
  self.assertLess(S.index('allocation=runtime.configure_physical_cap64(enabled=True)'),S.index('e = LLMEngine('))
  self.assertLess(S.index('e = LLMEngine('),S.index('ordinary = e.generate('))
  self.assertIn('(AttnCache.step,DecodeBatch.step)==(64,64)',S)
  self.assertIn('allocation_bucket_restored_finally',S)
  self.assertNotIn('v92-',S);self.assertNotIn('v95-',S)
 def test_diag_after_wait_before_drop(self):
  f=named('observe_delivered_run');text=ast.unparse(f)
  self.assertLess(text.index('result = original_adapter_run(out)'),text.index('rows = snapshot_rows('))
  self.assertIn('step = len(request.output_token_ids)',text);self.assertNotIn('.drop(',text)
  self.assertIn('api=\'v34_copy_output\'',text)
  self.assertIn('adapter.run=observe_delivered_run',S)
  branch=next(n for n in ast.walk(T) if isinstance(n,ast.If) and ast.unparse(n.test)=='diagnostic' and any(isinstance(k,ast.FunctionDef) and k.name=='observe_delivered_run' for k in n.body))
  self.assertTrue(branch)
 def test_counts_no_discarded_tensor_claim(self):
  self.assertEqual((93+31+6+31)*132,21252);self.assertEqual(5*132,660)
  for x in ('(21252 if diagnostic else 660)','[6,31,31,31,31,31]','"discarded_future_all132_tensor_checks"]=0','completed=162','discarded=1','banks=[84, 78]','"all132_output_checks": 6*132 if diagnostic else 0'):
   self.assertIn(x,S)
  self.assertNotIn('12540',S)
 def test_original_FIFO_adapter_source_is_separate(self):
  self.assertIn('v102-native-cap64-direct-async-runner-build1/native_runner.py',S)
  self.assertIn('v101-native-cap64-direct-async-build1/libv101replay.dylib',S)
  self.assertEqual(S.count('DYLD_INSERT_LIBRARIES'),1)
  self.assertNotIn('v34_write_input',ast.unparse(named('observe_delivered_run')))
  self.assertNotIn('v55_submit_step',ast.unparse(named('observe_delivered_run')))
 def test_real_canonical_matches89_signedzero_and_tail(self):
  own=runpy.run_path(str(D/'fullqual.py'),run_name='v103_canonical')['canonical_output']
  old=runpy.run_path(str(P/'v89-cap64-native-fullqual-build1/fullqual.py'),run_name='v89_canonical')['canonical_output']
  shape=[1,4,64,256];dtype='mlx.core.bfloat16';L=20
  head=b'\x00\x80'*(L*256)+bytes((64-L)*256*2);raw=head*4
  a=own(raw,shape,dtype,True,L);b=old(raw,shape,dtype,True,L);self.assertEqual(a,b);self.assertEqual(a[0][:2],b'\x00\x80')
  bad=bytearray(raw);bad[L*256*2+1]=0x80
  with self.assertRaises(AssertionError):own(bad,shape,dtype,True,L)
 def test_gold_and_pins_are_oldmath(self):
  self.assertIn('fullmodel-v89native-attempt1',S);self.assertIn('fullmodel-v38native-attempt1',S);self.assertIn('fullmodel-v44native-attempt1',S)
  self.assertIn('physical_digest==prior["physical_A_sha256"]',S)
  self.assertIn('digest==prior["A_sha256"]',S);self.assertIn('same1349_parameter_identities_preserved',S)
  self.assertIn('(48,83)',S);self.assertIn('7143436',S)
if __name__=='__main__':unittest.main()
