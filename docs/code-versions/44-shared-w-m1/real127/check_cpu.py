"""CPU source and injected-behavior checks; no array/numeric/GPU simulation."""
from pathlib import Path
import ast,importlib.util,json,sys,types,unittest
D=Path(__file__).resolve().parent;P=D.parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(D))
from router import ProjectionRouter
from capture import Capture

def load_helper():
 p=P/'v124-shared-weight-m1-source-build1/shared_m1.py';s=importlib.util.spec_from_file_location('v127_test_source',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
class Tests(unittest.TestCase):
 def setUp(self):
  self.helper=load_helper();self.calls=[]
  self.op=types.SimpleNamespace(weight=types.SimpleNamespace(shape=(48,640)),scales=object())
  self.x=types.SimpleNamespace(shape=(5,5120));self.fallback=lambda op,x,mx:self.calls.append((op,x)) or 'fallback'
 def router(self,helper=None):return ProjectionRouter(None,helper or self.helper,self.fallback,{})
 def test_default_off_and_strictbool(self):
  r=self.router();self.assertFalse(r.enabled)
  y,meta=r.project(self.op,self.x,weight_strides=(640,1),scale_strides=(320,1),x_strides=(5120,1));self.assertEqual(y,'fallback');self.assertEqual(meta['reason'],'default_OFF')
  for v in (1,0,None,[],object()):
   with self.assertRaises(TypeError):r.set_enabled(v)
 def test_unsupported_T_and_geometry_use_existing_fallback(self):
  r=self.router();r.set_enabled(True)
  for t in (1,3,6,8):
   y,meta=r.project(self.op,types.SimpleNamespace(shape=(t,5120)),weight_strides=(640,1),scale_strides=(320,1),x_strides=(5120,1));self.assertEqual(y,'fallback')
  op=types.SimpleNamespace(weight=types.SimpleNamespace(shape=(56,640)),scales=object())
  y,meta=r.project(op,self.x,weight_strides=(640,1),scale_strides=(320,1),x_strides=(5120,1));self.assertEqual(meta['route'],'qualified113_fallback')
 def test_layout_unsupported_does_not_hide_candidate_failure(self):
  r=self.router();r.set_enabled(True)
  y,meta=r.project(self.op,self.x,weight_strides=(640,1),scale_strides=(320,1),x_strides=(5121,1));self.assertEqual(y,'fallback')
  helper=types.SimpleNamespace(validate_geometry=lambda *x:None,validate_input=lambda *x:None,prepare=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('kernel failure')))
  r=self.router(helper);r.set_enabled(True)
  with self.assertRaisesRegex(RuntimeError,'kernel failure'):r.project(self.op,self.x,weight_strides=(640,1),scale_strides=(320,1),x_strides=(5120,1))
  self.assertEqual(len(self.calls),1)
 def test_capture_delegates_once_and_restores(self):
  class Linear:
   def __call__(self,x):return ('original',x)
  class Embedding:
   def as_linear(self,x):return ('embedding',x)
  op=Linear();layers=types.SimpleNamespace(Linear=Linear,Embedding=Embedding);model=types.SimpleNamespace(named_modules=lambda:[('test',op)])
  original=Linear.__call__;cap=Capture(model,layers,5);cap.install();x=types.SimpleNamespace(ndim=2,shape=(5,5120));cap.active=True
  y=op(x);self.assertEqual(y,('original',x));self.assertIs(cap.records[0]['x'],x);self.assertIs(cap.records[0]['y'],y);self.assertEqual(len(cap.records),1)
  cap.close();cap.close();self.assertIs(Linear.__call__,original)
 def test_actual_original_model_calls_unchanged(self):
  current=ast.parse((D/'qualify.py').read_text());old=ast.parse((P/'v113-broadcast-m1-diagnostic-build1/diagnose.py').read_text())
  def calls(tree):return {ast.dump(x,include_attributes=False) for x in ast.walk(tree) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='model'}
  self.assertEqual(calls(current),calls(old));self.assertEqual(len(calls(current)),2)
  text=(D/'capture.py').read_text();self.assertNotIn('project(',text);self.assertNotIn('quantized_matmul',text)
 def test_source_import_inert_and_planned_coverage(self):
  spec=importlib.util.spec_from_file_location('v127_qual_cpu',D/'qualify.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  self.assertFalse(any(n=='mlx' or n.startswith(('mlx.','numpy','tutorial_code.miniengine')) for n in sys.modules))
  p=json.loads((D/'protocol.json').read_text());self.assertEqual(sum(p['expected_geometry_operator_counts'].values()),497);self.assertEqual(len(p['expected_geometry_operator_counts']),9);self.assertEqual(p['draft_loads'],0)
  self.assertEqual((D/'layout_probe.metal').read_bytes(),(P/'v68-native-fullqual-build1/layout_probe.metal').read_bytes())
if __name__=='__main__':unittest.main()
