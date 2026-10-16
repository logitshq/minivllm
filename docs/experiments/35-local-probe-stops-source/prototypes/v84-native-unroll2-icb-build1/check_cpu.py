"""Small CPU-only V84 identity/timing/schema checks. No MLX, NumPy or native load."""
import ast,hashlib,importlib.util,json,sys
from pathlib import Path
D=Path(__file__).resolve().parent;P=D.parent
for p in D.glob('*.py'):compile(p.read_text(),str(p),'exec')
s=(D/'micro.py').read_text();t=ast.parse(s)
for n in t.body:
 if isinstance(n,(ast.Import,ast.ImportFrom)):assert all(not a.name.startswith(('mlx','numpy')) for a in n.names)
old=ast.parse((P/'v80-native-half2-icb-build1/micro.py').read_text().replace('v80','v84'))
a=next(n for n in ast.walk(old) if isinstance(n,ast.For) and ast.unparse(n.target)=='(pi, order)')
b=next(n for n in ast.walk(t) if isinstance(n,ast.For) and ast.unparse(n.target)=='(pi, order)')
assert ast.dump(a)==ast.dump(b),'timing schedule/clock/consume/control/resource block must remain exact72'
assert s.index('if args.qualify_only:return 0')<s.index('for pi,order in enumerate(SCHEDULE)')
main=s[s.index('def main'):];assert main.index('args.qualified_receipt and args.qualified_sha')<main.index('return gpu(')
assert 'unroll2.inputs[0] is w and unroll2.inputs[1] is s' in s
assert "w.shape==(N,K//8) and w.dtype==mx.uint32 and s.shape==(N,K//16) and s.dtype==mx.uint8 and x.shape==(1,K) and x.dtype==mx.bfloat16" in s
assert 'v69-nvfp4' not in s and 'v79-nvfp4' not in s and 'half2' not in s and 'PASS_V72_' not in s and 'libv72' not in s
native=(D/'replay.mm').read_text();assert native.startswith('// Frozen base is included verbatim.')
assert '#include "'+str(P/'v40-native-icb-build2/replay.mm')+'"' in native
assert native.count('extern "C" int v84_')==5 and native.count('extern "C" const char*v84_report_json')==1
assert native.count('restoreArena()')==1 and 'encoded+=8;[e memoryBarrierWithScope:MTLBarrierScopeBuffers];' in native
assert 'if(!v84_exact_argument(' in native and 'original typed operand full range unresolved' in native
spec=importlib.util.spec_from_file_location('v84_cpu',D/'micro.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert m.PRIME==128 and m.REPEATS==1024 and len(m.SCHEDULE)==10 and m.LAYERS==(0,8,16,24,32,40,48,56)
assert 'mlx.core' not in sys.modules and 'numpy' not in sys.modules
print(json.dumps({'status':'PASS_CPU_AST_EXACT72_TIMING_SOURCE_GUARDS','native_loaded':False,'GPU_used':False,'B_name':'EXPECTED_FROM_SOURCE_NOT_OBSERVED'}))
