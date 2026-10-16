"""CPU-only V76 metadata/source/address-coverage checks; no numerical GPU claim."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys
import types
import unittest

D=Path(__file__).resolve().parent;P=D.parent
OLD=P/'v44-gdn-activated-normreuse-build1/gdn_activated_norm_reuse_cached_metal.py'
K=Path('/Users/ryanli/code/minivllm/.venv/lib/python3.13/site-packages/mlx/include/mlx/backend/metal/kernels')

def literal(tree,name):return ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)))
def first_struct(text,name):
 start=text.index('struct '+name+' {');pos=text.index('operator()',start);brace=text.index('{',pos);depth=1;i=brace+1
 while depth:
  depth+=(text[i]=='{')-(text[i]=='}');i+=1
 return text[start:i]+(';' if text[i:i+1]==';' else '')+'\n};\n'

def load_helper():
 spec=importlib.util.spec_from_file_location('v76_CPU_only',D/'gdn_gates_prework_metal.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


class Checks(unittest.TestCase):
 def test_exact_original_state_norm_and_policy(self):
  old=ast.parse(OLD.read_text());new=ast.parse((D/'gdn_gates_prework_metal.py').read_text())
  for key in ('_STATE_SOURCE','HEADER','COMPILE_OPTIONS','SPLIT','VALUE_TILE'):
   self.assertEqual(literal(old,key),literal(new,key),key)
  for name in ('_state','_make_kernel'):
   a=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name)
   b=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name)
   self.assertEqual(ast.dump(a),ast.dump(b),name)
  a=literal(old,'_PREWORK_SOURCE');b=literal(new,'_PREWORK_SOURCE');prefix=a[:a.index('    } else {')]
  self.assertTrue(b.startswith(prefix));self.assertEqual(literal(old,'_NORM_BODY').strip(),literal(new,'_PREWORK_SOURCE')[literal(new,'_PREWORK_SOURCE').index('    float qn'):literal(new,'_PREWORK_SOURCE').index('            scales[h * 2]')].strip())
  self.assertIn('gdn_norm_reuse_cached_state_v40_build1',(D/'gdn_gates_prework_metal.py').read_text())

 def test_exact_qdot_and_fp32_functor_bodies(self):
  self.assertEqual((D/'qmv_header.metal').read_bytes(),(P/'v66-gdn-four-projection-build2/kernel_header.metal').read_bytes())
  gates=(D/'gates_header.metal').read_text()
  for filename,names in [('unary_ops.h',('Sigmoid','Exp','Negative')),('binary_ops.h',('Add','Multiply','LogAddExp'))]:
   for name in names:
    observed=first_struct(gates,'V76Native'+name).replace('struct V76Native'+name+' {','struct '+name+' {').replace('v76_native_log1p(','log1p(')
    self.assertEqual(observed,first_struct((K/filename).read_text(),name),name)
  source=(K/'utils.h').read_text();start=source.index('inline float log1p(float x) {');end=source.index('\n}',start)+2
  expected=source[start:end].replace('log1p(','v76_native_log1p(').replace('Limits<float>::max','V76FloatLimits::max')
  self.assertIn(expected,gates);self.assertIn('metal::numeric_limits<float>::infinity()',gates)

 def test_geometry_exact_ranges_and_single_writer(self):
  x=[k+lane*16+j for k in range(0,5120,512) for lane in range(32) for j in range(16)]
  self.assertEqual(sorted(x),list(range(5120)))
  for head in range(48):
   w=[head*2560+step*256+lane*8+j for step in range(10) for lane in range(32) for j in range(8)]
   scales=[head*320+step*32+lane for step in range(10) for lane in range(32)]
   self.assertEqual(sorted(w),list(range(head*2560,(head+1)*2560)))
   self.assertEqual(sorted(scales),list(range(head*320,(head+1)*320)))
  self.assertEqual([group-16 for group in range(16,64)],list(range(48)))
  m=load_helper();self.assertEqual((len(m.INPUT_NAMES),len(m.OUTPUT_NAMES)),(8,6))
  self.assertIn('if (lane < 32)',m._PREWORK_SOURCE);self.assertIn('if (simd_lid == 0)',m._PREWORK_SOURCE)

 def test_strict_metadata_and_inert_import(self):
  m=load_helper();mx=types.SimpleNamespace(bfloat16='bf16',uint32='u32',uint8='u8',float32='f32')
  dtypes=('bf16','bf16','u32','u8','u32','u8','f32','f32')
  values=[types.SimpleNamespace(shape=shape,dtype=dtype) for shape,dtype in zip(m.SHAPES,dtypes)]
  strides=[(shape[-1],1) if len(shape)==2 else (1,) for shape in m.SHAPES]
  m._validate(values,strides,mx)
  for index,field,bad in [(1,'shape',(2,5120)),(6,'dtype','bf16'),(2,'shape',(48,639))]:
   original=getattr(values[index],field);setattr(values[index],field,bad)
   with self.assertRaises(ValueError):m._validate(values,strides,mx)
   setattr(values[index],field,original)
  bad=list(strides);bad[1]=(10240,2)
  with self.assertRaises(ValueError):m._validate(values,bad,mx)
  for p in D.glob('*.py'):compile(p.read_text(),str(p),'exec')
  self.assertFalse(any(n=='mlx' or n=='numpy' or n.startswith('mlx.') for n in sys.modules))


if __name__=='__main__':unittest.main()
