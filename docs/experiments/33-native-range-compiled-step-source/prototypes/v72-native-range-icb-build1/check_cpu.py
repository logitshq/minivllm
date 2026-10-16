"""Small CPU source-contract checks, no candidate/native runtime loading."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
D=Path(__file__).resolve().parent;P=D.parent
for p in D.glob('*.py'):compile(p.read_text(),str(p),'exec')
source=(D/'micro.py').read_text();t=ast.parse(source)
assert all(not isinstance(n,(ast.Import,ast.ImportFrom)) or all(not a.name.startswith(('mlx','numpy')) for a in n.names) for n in t.body)
gpu=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='gpu')
assert any(isinstance(n,ast.Import) and any(a.name=='mlx.core' for a in n.names) for n in gpu.body)
assert source.index("if args.qualify_only:return 0")<source.index('for pi,order in enumerate(SCHEDULE)')
assert "initial_input_hashes==[hashlib.sha256(h.bytes_view).hexdigest() for h in owners]" in source
assert "r8.inputs[0] is w and r8.inputs[1] is s" in source
assert "r8.inputs==(w,s)" not in source
assert source.index("args.qualified_receipt and args.qualified_sha",source.index('def main'))<source.index('return gpu(',source.index('def main'))
native=(D/'replay.mm').read_text()
assert native.startswith('// Frozen base is included verbatim.')
assert '#include "'+str(P/'v40-native-icb-build2/replay.mm')+'"' in native
assert native.count('extern "C" int v72_')==5 and 'extern "C" const char*v72_report_json' in native
assert 'encoded+=8;[e memoryBarrierWithScope:MTLBarrierScopeBuffers];' in native
assert 'restoreArena();v.restored=true' in native and native.count('restoreArena()')==1
spec=importlib.util.spec_from_file_location('v72_cpu_test',D/'micro.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert m.PRIME==128 and m.REPEATS==1024 and len(m.SCHEDULE)==10
assert 'mlx.core' not in sys.modules and 'numpy' not in sys.modules
print(json.dumps({'status':'PASS_CPU_AST_AND_STAGE_CONTRACT','MLX_imported':False,'native_loaded':False,'GPU_used':False}))
