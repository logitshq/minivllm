from pathlib import Path
import hashlib,json,importlib.util
R=Path('/Users/ryanli/code/minivllm');A=R/'.artifacts/m5-ultra-256gb';H=R/'bench/m5_ultra_256gb/gdn_graph_combined_formal'
P=A/'gdn-graph-combined-formal-prepare-v1';J=A/'gdn-graph-combined-formal-v1-cold-journal'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
pre=read(P/'result.json');plan=read(P/'owner-plan.json');journal=read(J/'journal.json')
assert pre['status']=='PASS' and pre['source_unchanged'] is True and len(pre['specs'])==12 and pre['denied_operations']==[]
assert journal['state']=='PREPARED' and journal['owner_plan']==dict(path=str(P/'owner-plan.json'),sha256=sha(P/'owner-plan.json'))
assert journal['controller_sha256']==sha(H/'controller.py') and journal['owner_source']==plan['execution']['owner']
pins={**plan['execution']['pins'],str(H/'execute.py'):sha(H/'execute.py'),str(P/'owner-plan.json'):sha(P/'owner-plan.json'),str(P/'prepare.py'):sha(P/'prepare.py')}
assert all(sha(p)==h for p,h in pins.items())
value=dict(schema='GDN-graph-combined-formal-prereg-receipt-v1',status='PREPARED',owner_plan_sha256=sha(P/'owner-plan.json'),prepared_journal_sha256=sha(J/'journal.json'),source_pins=pins,actual_cpu_prepare=dict(path=str(P/'result.json'),sha256=sha(P/'result.json')),device_run=False,performance_qualified=False)
with (P/'receipt.json').open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
(P/'register.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(dict(receipt_sha256=sha(P/'receipt.json'),owner_plan_sha256=value['owner_plan_sha256'],pins=len(pins))))
