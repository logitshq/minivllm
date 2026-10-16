"""CPU-only real timer/invoke AST lifetime, schedule, controls and provenance checks."""
import ast,gc,importlib.util,json,sys,unittest
from pathlib import Path
D=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('stationary_cpu',D/'micro.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Checks(unittest.TestCase):
 def test_fixed_protocol_and_control_gate(self):
  self.assertEqual((m.PRIME,m.REPEATS),(2048,8192));self.assertEqual(m.TIMING_BANKS,m.REAL[:8]);self.assertEqual(len(m.SCHEDULE),10)
  self.assertEqual(sum(a!=b for a,b in m.SCHEDULE),6);self.assertEqual(m.SCHEDULE[2:8],[('A','B'),('B','A')]*3)
  self.assertTrue(m.controls_pass({'A':[.995,1.005],'B':[1.0,1.01]}));self.assertFalse(m.controls_pass({'A':[.97416598,.98559784],'B':[1.004876,.996719]}));self.assertFalse(m.controls_pass({'A':[1.,1.],'B':[.896073,.996989]}))
 def test_actual_block_calls_and_bounded_python_output_lifetime(self):
  class O:
   alive=0;peak=0
   def __init__(self):O.alive+=1;O.peak=max(O.peak,O.alive)
   def __del__(self):O.alive-=1
  calls=0
  def invoke(v,j):
   nonlocal calls
   calls+=1;return O(),O()
  class MX:
   sync=0
   def synchronize(self):self.sync+=1
  mx=MX();record,values=m.block(mx,invoke,'A')
  self.assertEqual(calls,10240);self.assertEqual(mx.sync,2);self.assertEqual(len(values),2);self.assertLessEqual(O.peak,4);del values;gc.collect();self.assertEqual(O.alive,0)
 def test_real_invoke_consumes_two_fresh_outputs_same_ready_seed(self):
  t=ast.parse((D/'micro.py').read_text());f=next(n for n in ast.walk(t)if isinstance(n,ast.FunctionDef)and n.name=='invoke');seen=[];evaluated=[]
  class MX:
   def eval(self,*v):evaluated.append(v)
  bank=object()
  def call(arm,b):seen.append((arm,b));return object(),object()
  ns={'mx':MX(),'call':call,'banks':{n:bank for n in m.TIMING_BANKS},'TIMING_BANKS':m.TIMING_BANKS};exec(compile(ast.fix_missing_locations(ast.Module(body=[f],type_ignores=[])),'realinvoke','exec'),ns)
  a=ns['invoke']('A',3);b=ns['invoke']('B',3);self.assertIs(seen[0][1],seen[1][1]);self.assertEqual([len(x)for x in evaluated],[2,2]);self.assertIsNot(a[1],b[1]);self.assertNotIn('silu',ast.unparse(f));self.assertNotIn('append',ast.unparse(f))
 def test_no_large_outputs_stored_in_timer_or_feedback(self):
  t=ast.parse((D/'micro.py').read_text());block=next(n for n in t.body if isinstance(n,ast.FunctionDef)and n.name=='block');s=ast.unparse(block)
  self.assertNotIn('.append(',s);self.assertNotIn('compile(',s);self.assertNotIn('silu',s)
  cfg=json.loads((D/'config.json').read_text());self.assertIn('/v44-gdn-activated-normreuse-build1/',cfg['A_helper']);self.assertNotIn('candidate-v17',cfg['A_helper']);self.assertNotEqual(cfg['A_helper'],cfg['B_helper'])
  src=(D/'micro.py').read_text();self.assertIn('mx.set_memory_limit(512<<20);mx.set_cache_limit(64<<20)',src);self.assertIn('mx.eval(*bank.values())',src);self.assertLess(src.index('bank["activated"]=nn.silu'),src.index('for index,order in enumerate(SCHEDULE)'))
  self.assertFalse(any(n=='mlx'or n.startswith(('mlx.','numpy'))for n in sys.modules))
if __name__=='__main__':unittest.main()
