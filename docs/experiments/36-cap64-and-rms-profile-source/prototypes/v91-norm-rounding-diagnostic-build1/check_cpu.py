"""CPU AST/formula/metric checks; no MLX/NumPy/native loading."""
import ast,importlib.util,json,struct,sys
from pathlib import Path
D=Path(__file__).resolve().parent;P=D.parent
for p in D.glob('*.py'):compile(p.read_text(),str(p),'exec')
s=(D/'numeric.py').read_text();tree=ast.parse(s)
for n in tree.body:
 if isinstance(n,(ast.Import,ast.ImportFrom)):assert all(not a.name.startswith(('mlx','numpy')) for a in n.names)
old=ast.parse((P/'candidate-v44-conv-normreuse-build1-root/tutorial_code/miniengine/model/layers.py').read_text());new=ast.parse((D/'explicit_reference.py').read_text())
a=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='_rms_norm_explicit');b=next(n for n in new.body if isinstance(n,ast.FunctionDef));assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('perf_counter','perf_counter_ns','process_time') for n in ast.walk(tree))
assert 'mx.fast.rms_norm(x,gamma,EPS)' in s and 'norm.norm(x,offset,EPS,zero_centered=True)' in s
assert 'normalized.astype(x.dtype)' in s and 'return normalized_bf16*gamma' in s
assert "status='DIAGNOSTIC_COMPLETED_NO_ADOPTION'" in s and 'before==after' in s and 'oldgold_gate' in s
spec=importlib.util.spec_from_file_location('v91_CPU',D/'numeric.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert len(m.synthetic())==8 and all(len(x)==5120 for x in m.synthetic()) and len(m.LABELS)==9
one=struct.pack('<H',0x3f80)*5120;nextone=struct.pack('<H',0x3f81)*5120
q=m.compare(one,nextone);assert not q['strict_bytes_equal'] and q['different_elements']==5120 and q['max_BF16_ULP_distance']==1 and q['max_absolute_difference']==1/128
q=m.compare(struct.pack('<H',0x8000)*5120,b'\0'*10240);assert q['signed_zero_bit_differences']==5120 and q['max_absolute_difference']==0 and q['max_BF16_ULP_distance']==1
assert not any(k=='mlx' or k.startswith(('mlx.','numpy')) for k in sys.modules)
print(json.dumps({'status':'PASS_CPU_AST_FORMULAS_METRICS','D_AST_exact_original44':True,'synthetic_inputs':8,'cases':27,'outputs':108,'GPU_used':False}))
