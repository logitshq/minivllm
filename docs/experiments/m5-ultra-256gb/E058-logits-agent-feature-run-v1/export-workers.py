from pathlib import Path
import json,hashlib,shutil
R=Path('/Users/ryanli/code/minivllm'); A=R/'.artifacts/m5-ultra-256gb'; W=A/'logits-build-e058-2026-10-21-v1'
D=R/'docs/experiments/m5-ultra-256gb/E058-logits-agent-feature-run-v1';D.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=[];stats=[]
for name in ('implementation','independent_audit','timing'):
 root=W/name
 for folder in [root,*sorted(root.glob('attempt-*'))]:
  for p in sorted(folder.iterdir()):
   if not p.is_file():continue
   rows.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
  label=name+'-'+('initial' if folder==root else folder.name)
  export=D/'workers'/label;export.mkdir(parents=True,exist_ok=True)
  for p in folder.iterdir():
   if p.is_file() and (p.name in ('final.md','result.json','root-intervention.json','integration-review.md') or p.name.startswith(('device-audit','device-timing','gpu-feedback','recursion-erratum')) or (folder!=root and p.suffix=='.py')):
    target=export
    if folder==root and (p.name.startswith(('device-','gpu-feedback','recursion-erratum')) or p.name=='integration-review.md'):
     target=D/'workers'/(name+'-post-device');target.mkdir(parents=True,exist_ok=True)
    if p.suffix=='.md':
     s='\n'.join(x.rstrip() for x in p.read_text().splitlines()).rstrip()+'\n';(target/p.name).write_text(s)
    else:shutil.copyfile(p,target/(p.name+'.txt' if p.suffix=='.py' else p.name))
  result=json.loads((folder/'result.json').read_text()) if (folder/'result.json').exists() else {'status':'pending'}
  logs=[]
  if (folder/'stdout.jsonl').exists():
   for line in (folder/'stdout.jsonl').read_text().splitlines():
    try:logs.append(json.loads(line))
    except json.JSONDecodeError:pass
  commands=[x['item'] for x in logs if x.get('type')=='item.completed' and x.get('item',{}).get('type')=='command_execution']
  stats.append(dict(worker=name,attempt=folder.name,result=result,completed_commands=len(commands),failed_commands=sum(x.get('exit_code') not in (0,None) for x in commands)))
(D/'worker-evidence-manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
(D/'worker-summary.json').write_text(json.dumps(stats,indent=2)+'\n')
print(json.dumps(stats,indent=2))
