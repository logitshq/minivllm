from pathlib import Path
import json,hashlib
R=Path('/Users/ryanli/code/minivllm');A=R/'.artifacts/m5-ultra-256gb';D=R/'docs/experiments/m5-ultra-256gb/E058-logits-agent-feature-run-v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
base='e058-agent-features-v1-';N=A/(base+'numerical');O=A/(base+'owner');E=A/(base+'execution');L=A/(base+'lease')
complete=read(N/'complete.json');final=read(E/'final.json');owner=read(O/'result.json');lease=read(L/'lease.json');resume=read(L/'resume.json')
physical=[read(p) for p in sorted((O/'resources').glob('physical-*/sample.json'))]
limits=read(E/'execution-plan.json')['physical_baseline']
assert physical and all(p['status']=='MEASURED' for p in physical)
assert all(p['parsed']['system_bytes']<=240*(1<<30) and max(p['parsed'][k] for k in ('top_bytes','aux_current_bytes','aux_peak_bytes'))<=64*(1<<30) for p in physical)
assert all(p['parsed']['swap_bytes']<=limits['swap_bytes'] and p['parsed']['swapouts']==limits['swapouts'] for p in physical)
assert lease['closed'] is True and resume['identity_match'] is True and 'T' not in resume['state_after']
assert owner['child_reaped'] is True
result={'schema':'E058-saved-resource-review-v1','status':'PASS_SAVED_OBSERVATIONS','recorded_service_restored':True,'physical_samples':len(physical),
'max_system_bytes':max(p['parsed']['system_bytes'] for p in physical),'max_child_historical_peak_bytes':max(p['parsed']['aux_peak_bytes'] for p in physical),
'max_gap_seconds':max(b['started_monotonic']-a['ended_monotonic'] for a,b in zip(physical,physical[1:])),
'swap_growth_bytes':max(p['parsed']['swap_bytes'] for p in physical)-limits['swap_bytes'],
'swapouts_growth':max(p['parsed']['swapouts'] for p in physical)-limits['swapouts'],
'parent_status':final['status'],'owner_status':owner['status'],'numerical_status':complete['status'],
'live_checks_performed':False,'limitation':'Saved sampled observations do not bound inter-sample allocation or prove engine performance.'}
assert result['max_gap_seconds']<=12
D.mkdir(exist_ok=True)
(D/'resource-review.json').write_text(json.dumps(result,indent=2)+'\n')
for src,name in ((N/'complete.json','complete.json'),(N/'pairs.json','pairs.json'),(E/'final.json','execution-final.json'),(O/'result.json','owner-result.json'),(L/'resume.json','resume.json')):
 if src.exists():(D/name).write_bytes(src.read_bytes())
rows=[]
for root in (N,O,E,L):
 for p in sorted(root.rglob('*')):
  if p.is_file():rows.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
(D/'device-evidence-manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(result,indent=2));print(json.dumps(complete.get('timing_summary',{}),indent=2))
