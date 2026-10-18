"""Build exact 12 process timing specs after real combined HTTP closure. CPU only."""
from pathlib import Path
import datetime, hashlib, importlib.abc, importlib.util, json, sys
from types import SimpleNamespace
R=Path('/Users/ryanli/code/minivllm');A=R/'.artifacts/m5-ultra-256gb';B=R/'bench/m5_ultra_256gb';H=B/'gdn_graph_combined_formal'
OUT=A/'gdn-graph-combined-formal-prepare-v1';ARCHIVE=R/'docs/experiments/m5-ultra-256gb/E031-gdn-graph-combined-http-v1'
CYCLE=A/'gdn-graph-combined-http-v1-analysis';OLD=B/'route_http_formal/prereg-formal-v2/owner-plan.json'
FORMAL_SHA='4b567f8ed6090175f3639aa109584ca26c1426165c03de3167c3ebde74891a89'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def ref(p):return dict(path=str(p.resolve()),sha256=sha(p))
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
for p in ('independent-cycle.json','restoration-audit.json','independent-c1.json','independent-c8.json'):
 assert read(CYCLE/p)['status']=='PASS' and sha(ARCHIVE/p)==sha(CYCLE/p),p
assert read(ARCHIVE/'summary.json')['status']=='PASS' and read(ARCHIVE/'archive-verification.json')['status']=='PASS'
assert sha(H/'formal.py')==FORMAL_SHA
OUT.mkdir();(OUT/'specs').mkdir();(OUT/'prepare.py').write_bytes(Path(__file__).read_bytes())
record=dict(schema='GDN-graph-combined-formal-actual-prepare-v1',status='RUNNING',started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),specs=[],denied_operations=[],device_run=False,performance_qualified=False)
def save(): (OUT/'result.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
def deny(event,args):
 record['denied_operations'].append(dict(event=event,args=str(args)));raise RuntimeError('CPU preparation forbids '+event)
def guard(event,args):
 if event in ('subprocess.Popen','os.system','os.fork','os.forkpty','os.posix_spawn','os.exec','os.spawn','os.kill','os.killpg','socket.connect','socket.bind'):deny(event,args)
 if event=='open' and args and isinstance(args[0],(str,bytes)):
  p=str(args[0])
  if p.endswith(('.safetensors','.gguf','.bin','steps.jsonl')):deny(event,args)
class Imports(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'mlx','omlx','mlx_lm','mlx_vlm','torch','numpy','tutorial_code','monolith'}:deny('backend import',fullname)
  return None
def load(p,name):
 spec=importlib.util.spec_from_file_location(name,p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
save();sys.meta_path.insert(0,Imports());sys.addaudithook(guard)
try:
 formal=load(H/'formal.py','_actual_combined_formal');owner=load(H/'owner.py','_actual_combined_formal_owner')
 before={**formal.source_pins(),**owner.source_pins(),str(H/'controller.py'):sha(H/'controller.py'),str(H/'audit_restore.py'):sha(H/'audit_restore.py')}
 baseline=read(OLD);record['baseline_owner_plan']=ref(OLD)
 bindings={str(C):dict(audit=ref(CYCLE/f'independent-c{C}.json'),replay_plan=ref(A/f'gdn-graph-combined-http-v1-owner/phases/c{C}-replay/plan.json'),child_directory=str(A/f'gdn-graph-combined-http-v1-owner/phases/c{C}-replay/qualification'),cycle_audit=ref(CYCLE/'independent-cycle.json')) for C in (1,8)}
 specs={}
 for phase in owner.PHASES:
  r,c,arm=phase.split('-');C=int(c[1:]);spec=formal.prepare(baseline['qualification'],C,int(r[1:]),arm=arm,baseline_qualification=baseline['qualifications'][str(C)],route_qualification=baseline['route_qualifications'][str(C)],combined_qualification=bindings[str(C)],**baseline['http'])
  formal.validate(spec);p=OUT/'specs'/(phase+'.json');write(p,spec);specs[phase]=ref(p);record['specs'].append(dict(phase=phase,**ref(p)));save()
 cold=dict(journal_path=str(A/'gdn-graph-combined-formal-v1-cold-journal'),controller=ref(H/'controller.py'),approved_contract='cold-restart-persisted-visible-v1',proposal_plan_sha256='e96a4a4d5be6f6a69d597387bbc566eadedae81065bfa879445352d986515cb8',predecessor={**ref(ARCHIVE/'restored-journal.json'), 'audit':ref(ARCHIVE/'restoration-audit.json')})
 plan=owner.plan_template(baseline['qualification'],cold,baseline['http'],baseline['qualifications'],baseline['route_qualifications'],bindings,specs)
 write(OUT/'owner-plan.json',plan);owner.validate_plan(SimpleNamespace(plan=OUT/'owner-plan.json',plan_sha256=sha(OUT/'owner-plan.json'),out=A/'gdn-graph-combined-formal-v1-owner'))
 assert all(sha(p)==h for p,h in before.items())
 for p in before:
  target=OUT/'frozen-source'/p.lstrip('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(p).read_bytes())
 record.update(status='PASS',owner_plan=ref(OUT/'owner-plan.json'),source_pins=before,source_unchanged=True,total_requests=864,formal_requests=768,warm_requests=96)
except BaseException as e:
 import traceback
 record.update(status='FAIL',error=str(e),traceback=traceback.format_exc());raise
finally:
 record['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
print(json.dumps(dict(status=record['status'],specs=len(specs),owner_plan=record['owner_plan'])))
