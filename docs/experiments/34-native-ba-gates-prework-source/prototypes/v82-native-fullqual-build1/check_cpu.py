"""CPU-only source, metadata, fallback and restoration checks; no MLX/GPU."""
import ast
import copy
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
import types
import unittest
sys.dont_write_bytecode=True
P=Path(__file__).resolve().parent.parent;D=Path(__file__).resolve().parent
B=P/'candidate-v82-ba-gates-prework-build1-root/tutorial_code/miniengine'
OLD=P/'candidate-v44-conv-normreuse-build1-root/tutorial_code/miniengine'


def method(tree,name):return next(n for c in tree.body if isinstance(c,ast.ClassDef) and c.name=='GatedDeltaNet' for n in c.body if isinstance(n,ast.FunctionDef) and n.name==name)
def load_core():
 spec=importlib.util.spec_from_file_location('v82_cpu_core',B/'kernels/v76_gates_prework/gdn_gates_prework_metal.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def metadata():
 core=load_core();mx=types.SimpleNamespace(bfloat16='bf16',float32='f32',uint32='u32',uint8='u8')
 values=[types.SimpleNamespace(shape=shape,dtype=dtype) for shape,dtype in zip(core.SHAPES,('bf16','bf16','u32','u8','u32','u8','f32','f32'))]
 strides=[(shape[-1],1) if len(shape)==2 else (1,) for shape in core.SHAPES]
 return core,mx,values,strides


class Checks(unittest.TestCase):
 def test_parent_AST_and_exact76_copy(self):
  a=ast.parse((OLD/'model/gdn.py').read_text());b=ast.parse((B/'model/gdn.py').read_text())
  for name in ('__call__','_decode','_decode_conv_norm_reuse_cache','_prefill','_prefill_batch','_gates','_decode_gates'):
   self.assertEqual(ast.dump(method(a,name)),ast.dump(method(b,name)),name)
  changed=[str(p.relative_to(OLD)) for p in OLD.rglob('*.py') if p.read_bytes()!=(B/p.relative_to(OLD)).read_bytes()]
  self.assertEqual(changed,['model/gdn.py']);self.assertEqual(len(list(B.rglob('*.py'))),29)
  for name in ('gdn_gates_prework_metal.py','qmv_header.metal','gates_header.metal','LICENSE-MLX-MIT'):
   self.assertEqual((B/'kernels/v76_gates_prework'/name).read_bytes(),(P/'v76-gdn-gates-prework-build1'/name).read_bytes())
  target=method(b,'_decode_ba_gates_prework');calls=[n.func.attr for n in ast.walk(target) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
  self.assertTrue(all(name not in calls for name in ('in_proj_b','in_proj_a','_gates','_decode_gates')))
  self.assertIn('prework',calls);self.assertIn('state',calls)

 def test_prepare_sameowners_singleton_and_rejections(self):
  core,mx,values,strides=metadata();tree=ast.parse((B/'model/gdn_ba_prework.py').read_text());tree.body=[n for n in tree.body if not isinstance(n,ast.ImportFrom)]
  ns={'core':core};exec(compile(tree,'<CPU-prepared-adapter>','exec'),ns)
  ranges=[(1000000+i*1000000,n) for i,n in enumerate((122880,15360,122880,15360,192,192))]
  kw=dict(observed_parameter_strides=strides[2:],observed_x=values[1],observed_x_strides=(0,1),observed_activated=values[0],observed_activated_strides=(0,1),observed_parameter_ranges=ranges,runtime_version='0.32.3',device_name='Apple M4 Max',architecture='applegpu_g16s',input_layout_contract='qualified-original-input-layernorm-B1-and-conv23')
  helper=ns['prepare'](mx,values[2:],**kw);self.assertTrue(all(x is y for x,y in zip(helper.operands,values[2:])))
  self.assertEqual(helper.x_strides,(0,1));self.assertFalse(hasattr(helper,'__dict__'))
  for change in ({'observed_x_strides':(10240,2)},{'runtime_version':'0.32.2'},{'observed_parameter_ranges':[(1000000,122880),(1000010,15360)]+ranges[2:]}):
   with self.assertRaises(ValueError):ns['prepare'](mx,values[2:],**dict(kw,**change))

 def test_flag_unprepared_stale_and_scope(self):
  core,mx,values,strides=metadata();tree=ast.parse((B/'model/gdn.py').read_text())
  class Linear:pass
  defs=[copy.deepcopy(method(tree,name)) for name in ('set_ba_gates_prework','_ba_gates_prework_operands','_ba_gates_prework_ops_valid','_ba_gates_prework_eligible')]
  cls=ast.ClassDef(name='GDN',bases=[],keywords=[],body=defs,decorator_list=[]);ns={'mx':mx,'Linear':Linear};exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),'<CPU-gdn-scope>','exec'),ns)
  g=ns['GDN']();g._gates_compile_enabled=True
  for name,weight,scale in [('in_proj_b',values[2],values[3]),('in_proj_a',values[4],values[5])]:
   op=Linear();op.q={'group_size':16,'bits':4,'mode':'nvfp4'};op.in_features=5120;op.out_features=48;op.weight=weight;op.scales=scale;setattr(g,name,op)
  g.A_log,g.dt_bias=values[6:];g._ba_gates_prework_helper=None;g.set_ba_gates_prework(False)
  self.assertFalse(g._ba_gates_prework_eligible(values[1]));g.set_ba_gates_prework(True)
  self.assertFalse(g._ba_gates_prework_eligible(values[1]));g._ba_gates_prework_helper=types.SimpleNamespace(operands=tuple(values[2:]))
  self.assertTrue(g._ba_gates_prework_eligible(values[1]));g._gates_compile_enabled=False
  self.assertFalse(g._ba_gates_prework_eligible(values[1]));g._gates_compile_enabled=True
  g.in_proj_b.weight=types.SimpleNamespace(shape=(48,640),dtype='u32')
  self.assertFalse(g._ba_gates_prework_eligible(values[1]));self.assertIsNone(g._ba_gates_prework_helper)
  with self.assertRaises(TypeError):g.set_ba_gates_prework(1)

 def test_A_all_OFF_finally_restore(self):
  tree=ast.parse((D/'fullqual.py').read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');node=copy.deepcopy(next(n for n in ast.walk(main) if isinstance(n,ast.FunctionDef) and n.name=='baseline_decode'))
  class G:
   def set_gates_compile(self,v):self._gates_compile_enabled=v
   def set_conv_norm_reuse_cache(self,v):self._conv_norm_reuse_cache_enabled=v
   def set_ba_gates_prework(self,v):self._ba_gates_prework_enabled=v
  gdns=[G() for _ in range(48)]
  for g in gdns:g.set_gates_compile(True);g.set_conv_norm_reuse_cache(True);g.set_ba_gates_prework(True)
  a,b,legacy=object(),object(),object();G._decode_dispatch=a;G._decode_conv_norm_reuse_cache=b;events=[]
  ns={'contextmanager':contextmanager,'GDN':G,'candidate_decode':a,'candidate_norm':b,'legacy_decode':legacy,'gdns':gdns,'baseline_calls':events};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'<CPU-baseline>','exec'),ns)
  with self.assertRaises(ValueError):
   with ns['baseline_decode']():
    self.assertTrue(all((g._gates_compile_enabled,g._conv_norm_reuse_cache_enabled,g._ba_gates_prework_enabled)==(False,False,False) for g in gdns));raise ValueError('CPU-only exception')
  self.assertTrue(events[-1]['all_flags_restored']);self.assertIs(G._decode_dispatch,a);self.assertIs(G._decode_conv_norm_reuse_cache,b)

 def test_qualified_replay_loop_and_inert_sources(self):
  tree=ast.parse((D/'fullqual.py').read_text());old=ast.parse((P/'v68-native-fullqual-build1/fullqual.py').read_text())
  for name in ('Native','metadata_finite','originals_readonly','assert_icb'):
   self.assertEqual(ast.dump(next(n for n in tree.body if getattr(n,'name',None)==name)),ast.dump(next(n for n in old.body if getattr(n,'name',None)==name)))
  def loop(t):return next(n for n in ast.walk(t) if isinstance(n,ast.For) and ast.unparse(n.target)=='(name, prompt, canonical)')
  self.assertEqual(ast.dump(loop(tree)),ast.dump(loop(old)))
  for p in list(B.rglob('*.py'))+[D/'fullqual.py']:compile(p.read_text(),str(p),'exec')
  self.assertFalse(any(n=='mlx' or n=='numpy' or n.startswith('mlx.') for n in sys.modules))

if __name__=='__main__':unittest.main()
