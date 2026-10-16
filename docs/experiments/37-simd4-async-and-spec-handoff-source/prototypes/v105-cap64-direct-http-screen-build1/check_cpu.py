"""V105 CPU AST/source boundary proofs; no server, model, MLX or GPU execution."""
import ast
from pathlib import Path
import unittest
D=Path(__file__).resolve().parent;P=D.parent;OLD=P/'v58-native-http-screen-build1'
DT=ast.parse((D/'driver.py').read_text());ST=ast.parse((D/'server.py').read_text())
OD=ast.parse((OLD/'driver.py').read_text());OS=ast.parse((OLD/'server.py').read_text());OFF=ast.parse((P/'m4-final-native15-official-attempt2/client.py').read_text())
def named(t,name):return next(n for n in ast.walk(t) if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
def dump(n):return ast.dump(n,include_attributes=False)
class Checks(unittest.TestCase):
 def test_request_group_exact58_and_official(self):
  for name in ('request','group'):
   self.assertEqual(dump(named(DT,name)),dump(named(OD,name)))
   self.assertEqual(dump(named(DT,name)),dump(named(OFF,name)))
 def test_driver_main_plan_stats_and_owner_cleanup_exact58(self):
  self.assertEqual(dump(named(DT,'main')),dump(named(OD,'main')))
  new=named(DT,'run_arm');old=named(OD,'run_arm')
  a=next(n for n in ast.walk(new) if isinstance(n,ast.Try) and n.finalbody)
  b=next(n for n in ast.walk(old) if isinstance(n,ast.Try) and n.finalbody)
  self.assertEqual(dump(ast.Module(body=a.finalbody,type_ignores=[])),dump(ast.Module(body=b.finalbody,type_ignores=[])))
 def test_POST_GET_SSE_exact58(self):self.assertEqual(dump(named(ST,'Handler')),dump(named(OS,'Handler')))
 def test_pipeline_both_sides_exact(self):self.assertEqual(dump(named(DT,'assert_pipeline')),dump(named(ST,'assert_pipeline')))
 def test_startup64_before_engine_anytoken(self):
  s=(D/'server.py').read_text()
  self.assertLess(s.index('allocation = module.configure_physical_cap64(enabled=True)'),s.index('e = LLMEngine('))
  self.assertLess(s.index('e = LLMEngine('),s.index('warm = e.generate('))
  self.assertIn("diagnostic=False",s);self.assertIn('fixed_cap=64',s)
  self.assertIn("actual_imported_sources=imported_sources",s)
  self.assertIn('module.old.old.old',s)
 def test_no_new_resources_inside_original_groups(self):
  def scope(t):
   n=named(t,'run_arm')
   return next(n for n in ast.walk(n) if isinstance(n,ast.For) and ast.unparse(n.target)=='phase')
  self.assertEqual(dump(scope(DT)),dump(scope(OD)))
  for a in (DT,ST):
   compile(a,filename='CPU-source-only',mode='exec')
 def test_actual103_both_modes_exact_pins_required(self):
  s=(D/'prepare_config.py').read_text()
  self.assertIn('PASS_V103_CAP64_DIRECT_ASYNC_DIAGNOSTIC_161_DELIVERED_21252_OUTPUT_HASHES',s)
  self.assertIn('PASS_V103_CAP64_DIRECT_ASYNC_PRODUCTION_161_IDS_660_FINAL_OUTPUT_HASHES',s)
  self.assertIn('source_pins]==qualification_source_pins',s.replace("'",''))
  self.assertNotIn('v92-',s);self.assertNotIn('v95-',s)
  for text in ('q[\'physical_capacity\']==64',"q['early_stop']['all132_output_checks']==(792 if mode=='diagnostic' else 0)","released['GPU_released']"):
   self.assertIn(text,s)
if __name__=='__main__':unittest.main()
