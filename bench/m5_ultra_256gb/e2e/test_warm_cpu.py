"""Actual N2 helper AST with CPU callable fakes; never imports model/MLX."""
import ast,copy,importlib.util,json,sys,types,unittest,tempfile
from pathlib import Path
from unittest import mock
from warm_routes import WarmRoutes,expected_counts
D=Path(__file__).resolve().parent;P=D.parent
class Tensor:
 def __init__(self,shape,dtype):self.shape,self.dtype=tuple(shape),dtype
class Warm(unittest.TestCase):
 def fixture(self,enabled):
  mx=types.SimpleNamespace(bfloat16='bf16',float32='fp32');factory_raw=[];compiled_raw=[];fail=[False]
  def compile_fake(fn,*a,**kw):
   def raw(*a,**kw):return fn(*a,**kw)
   compiled_raw.append(raw);return raw
  mx.compile=compile_fake
  def factory(mx,name,*a,**kw):
   def raw(**kw):
    if fail[0]:raise RuntimeError('fixture-dispatch-error')
    return [Tensor(shape,dtype) for shape,dtype in zip(kw['output_shapes'],kw['output_dtypes'])]
   factory_raw.append(raw);return raw
  norm=types.ModuleType('CPU_N2_real_helpers');norm.__dict__.update(_prework_kernel=None,_state_kernel=None,_norm_probe_kernel=None,_make_kernel=factory,_runtime=lambda:mx,_PREWORK_SOURCE='FROZEN_AST_TEST_SOURCE',_STATE_SOURCE='FROZEN_AST_TEST_SOURCE',SPLIT=4,VALUE_TILE=32)
  path=D.parents[2]/'tutorial_code/miniengine/kernels/gdn_norm_reuse_metal.py';tree=ast.parse(path.read_text())
  names={'_validate_shapes','_validate_dtypes','_prepare','_state','gated_delta_step_norm_reuse_metal'}
  for fn in tree.body:
   if isinstance(fn,ast.FunctionDef) and fn.name in names:exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),norm.__dict__)
  class BatchState:
   def __init__(self):self.conv=Tensor((1,3,10240),'bf16');self.S=Tensor((1,48,128,128),'fp32')
  gdn=types.SimpleNamespace(_gates_explicit=lambda *a:None)
  class GDN:
   def __init__(self):self._compiled_decode_gates=None
   def _decode_gates(self):
    if self._compiled_decode_gates is None:self._compiled_decode_gates=mx.compile(gdn._gates_explicit)
    return self._compiled_decode_gates()
   def _decode_norm_reuse(self,x,st):return norm.gated_delta_step_norm_reuse_metal(Tensor((1,10240),'bf16'),Tensor((1,48),'fp32'),Tensor((1,48),'fp32'),st.S)
   def _decode(self,x,st,**kw):
    self._decode_gates()
    if enabled:return self._decode_norm_reuse(x,st)
  gdn.GatedDeltaNet=GDN;layers=[GDN() for _ in range(48)];states=[BatchState() for _ in layers]
  warm=WarmRoutes(gdn,norm,mx,layers,BatchState,enabled)
  return warm,norm,layers,states,Tensor((1,5120),'bf16'),factory_raw,compiled_raw,fail
 def test_actual_helpers_dispatch_and_warmed_raw_identity(self):
  for enabled in (False,True):
   warm,norm,layers,states,x,raw,compiled,fail=self.fixture(enabled)
   try:
    for request in range(3):
     for step in range(31):
      for layer,state in zip(layers,states):layer._decode(x,state,_compile_gates=True,_norm_reuse_normal=True)
    metadata=warm.restore();self.assertEqual(metadata['counts'],expected_counts(enabled));self.assertEqual(len(compiled),48)
    self.assertEqual(len(raw),2 if enabled else 0)
    if enabled:self.assertIs(norm._prework_kernel,raw[0]);self.assertIs(norm._state_kernel,raw[1])
    for layer,fn in zip(layers,compiled):self.assertIs(layer._compiled_decode_gates,fn)
    before=copy.deepcopy(warm.counts);layers[0]._decode(x,states[0],_compile_gates=True,_norm_reuse_normal=True)
    self.assertEqual(warm.counts,before);self.assertEqual(len(compiled),48);self.assertEqual(len(raw),2 if enabled else 0)
   finally:warm.restore(validate=False)
 def test_exception_finally_preserves_created_raw_cache(self):
  warm,norm,layers,states,x,raw,compiled,fail=self.fixture(True);fail[0]=True
  try:
   with self.assertRaisesRegex(RuntimeError,'fixture-dispatch-error'):layers[0]._decode(x,states[0],_compile_gates=True,_norm_reuse_normal=True)
  finally:metadata=warm.restore(validate=False)
  self.assertIs(norm._prework_kernel,raw[0]);self.assertIsNone(norm._state_kernel);self.assertIs(layers[0]._compiled_decode_gates,compiled[0]);self.assertEqual(metadata['cached_wrappers_remaining'],0)
  fail[0]=False;before=dict(warm.counts);layers[0]._decode(x,states[0],_compile_gates=True,_norm_reuse_normal=True);self.assertEqual(warm.counts,before)

if __name__=='__main__':unittest.main()
