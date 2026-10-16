"""CPU-only test of the actual backend eligibility AST."""
import ast,json,types
from functools import lru_cache
from pathlib import Path
p=Path(__file__).resolve().parents[2]/'tutorial_code/miniengine/model/gdn.py'
tree=ast.parse(p.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_m5_ultra_decode_identity','gdn_decode_backend_supported')]
assert len(nodes)==2
rows=[]
for legacy,ver,arch,name,device,want in [(True,'0.32.3','applegpu_g16s','Apple M4 Max','gpu',True),(False,'0.32.3','applegpu_g17d','Apple M5 Ultra','gpu',True),(False,'0.32.3','applegpu_g17d','Apple M5 Ultra','cpu',False),(False,'0.32.4','applegpu_g17d','Apple M5 Ultra','gpu',False),(False,'0.32.3','applegpu_g17s','Apple M5 Max','gpu',False),(False,'0.32.3','applegpu_g17d','unknown','gpu',False)]:
 mx=types.SimpleNamespace(default_device=lambda:device,gpu='gpu',device_info=lambda:{'device_name':name})
 ns={'mx':mx,'lru_cache':lru_cache,'_prefill_backend_identity':lambda:(ver,arch),'prefill_batch_backend_supported':lambda:legacy};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
 got=ns['gdn_decode_backend_supported']();assert got is want
 rows.append({'legacy':legacy,'version':ver,'architecture':arch,'device':name,'default':device,'supported':got})
print(json.dumps({'CPU_ONLY':True,'cases':rows},indent=2))
