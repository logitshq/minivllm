from pathlib import Path
import hashlib,json
D=Path(__file__).resolve().parent
rows=json.loads((D/'source-map.json').read_text())['files']
for r in rows:
 p=D/r['published'];b=p.read_bytes();assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
 if p.suffix=='.py':compile(b,str(p),'exec')
print(json.dumps({'status':'PASS_EXACT_SOURCE_HASHES_AND_SYNTAX_ONLY','files':len(rows),'MLX_imported':False,'GPU_executed':False,'runtime_qualification':False}))
