"""Stdlib only; never imports MLX/NumPy/native libraries."""
from pathlib import Path
import ast,contextlib,importlib.util,io,json,math,os,subprocess,sys,unittest
D=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('v124_cpu_q',D/'localqual.py');q=importlib.util.module_from_spec(s);s.loader.exec_module(q)
s=importlib.util.spec_from_file_location('v124_cpu_launch',D/'launch.py');launch=importlib.util.module_from_spec(s);s.loader.exec_module(launch)
h=q.source_helper()
class CPU(unittest.TestCase):
 def test_imports_inert(self):
  self.assertFalse(any(n=='mlx' or n.startswith('mlx.') or n=='numpy' for n in sys.modules))
  tree=ast.parse((D/'localqual.py').read_text());gpu=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='gpu')
  outside=[n for n in tree.body if n is not gpu]
  self.assertFalse(any(isinstance(n,(ast.Import,ast.ImportFrom)) and ('mlx' in ast.unparse(n) or 'numpy' in ast.unparse(n)) for t in outside for n in ast.walk(t)))
 def test_source_pins(self):self.assertEqual(q.identity()['status'],'PREIMPORT_PINS_PASS')
 def test_supported_and_rejected_boundaries(self):
  for n,k in h.SHAPES:h.validate_geometry(n,k,(k//8,1),(k//16,1))
  for t in q.ROWS:h.validate_input((t,5120),(5120,1),5120)
  for n in (49,56):
   with self.assertRaises(ValueError):h.validate_geometry(n,5120,(640,1),(320,1))
  with self.assertRaises(ValueError):h.validate_input((3,5120),(5120,1),5120)
  with self.assertRaises(ValueError):h.validate_geometry(48,5120,(1,48),(320,1))
 def test_plan_and_memory_bound(self):
  p=json.loads((D/'protocol.json').read_text());self.assertEqual(len(q.SHAPES)*len(q.BANKS)*len(q.ROWS)+len(q.SHAPES),40);self.assertEqual(sum(q.ROWS)*len(q.BANKS)*len(q.SHAPES)+5*len(q.SHAPES),152)
  self.assertEqual(max(n*k//2+n*k//16 for n,k in q.SHAPES),50135040);self.assertLess(50135040,1<<30);self.assertEqual(set(map(tuple,p['uncovered_actual_shapes_N_K'])),set(h.SHAPES)-set(q.SHAPES))
 def test_codes_and_finite_fixture_bound(self):
  words=(0x76543210,0xfedcba98);self.assertEqual({(w>>(4*i))&15 for w in words for i in range(8)},set(range(16)))
  self.assertTrue(all((x&127)!=127 for x in q.SCALE_CODES));self.assertEqual(math.ldexp(1.,(0x5380>>7)-127),2.**40)
  self.assertLess(6*448*2.**40*max(k for n,k in q.SHAPES),3.4e38)
 def test_clean_launch_unknown_selected_keys(self):
  env={'MLX_FUTURE_CONTROL':'not-logged','DYLD_SOMETHING':'not-logged','METAL_X':'not-logged','PYTHONPATH':'not-logged','PYTHONDONTWRITEBYTECODE':'1','SAFE':'preserved'}
  out=launch.environment(env);self.assertEqual(q.selected(out),q.CONTROLS);self.assertEqual(out['SAFE'],'preserved');self.assertIn('MLX_FUTURE_CONTROL',env)
  tree=ast.parse((q.P/'m4-final-native15-official-attempt2/common.py').read_text());original=next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=='selected');actual=next(x for x in ast.parse((D/'localqual.py').read_text()).body if isinstance(x,ast.FunctionDef)and x.name=='selected');self.assertEqual(ast.dump(original,include_attributes=False),ast.dump(actual,include_attributes=False))
 def test_first_failure_and_no_perf_contract(self):
  text=(D/'localqual.py').read_text();self.assertIn("compare(native,c,'original-expression-control-vs-native-M1'",text);self.assertIn("compare(native,b,'shared-W-vs-native-M1'",text);self.assertIn("finally:ro('case-finally')",text);self.assertNotIn('perf_counter',text);self.assertNotIn('clear_cache(',text)
 def test_default_inert_entrypoints(self):
  for n in ('localqual.py','launch.py'):
   p=subprocess.run([sys.executable,'-B',str(D/n)],capture_output=True,text=True);self.assertEqual(p.returncode,0,p.stderr);self.assertFalse(json.loads(p.stdout)['GPU_used'])
if __name__=='__main__':unittest.main()
