"""Prepare the retry plan using unchanged original12 timing specs and actual E032 restoration. CPU only."""
from pathlib import Path
import datetime, hashlib, importlib.abc, importlib.util, json, sys
from types import SimpleNamespace
R=Path('/Users/ryanli/code/minivllm');A=R/'.artifacts/m5-ultra-256gb';B=R/'bench/m5_ultra_256gb';H=B/'gdn_graph_combined_formal_retry_v2'
OUT=A/'gdn-graph-combined-formal-prepare-v2';ARCHIVE=R/'docs/experiments/m5-ultra-256gb/E032-gdn-graph-combined-formal-v1';OLD=A/'gdn-graph-combined-formal-prepare-v1/owner-plan.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def ref(p):return dict(path=str(p.resolve()),sha256=sha(p))
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
assert read(ARCHIVE/'summary.json')['status']=='FAILED_BEFORE_EXPERIMENT_CHILD'
assert read(ARCHIVE/'restoration-audit.json')['status']=='PASS' and read(ARCHIVE/'archive-verification.json')['status']=='PASS'
OUT.mkdir();(OUT/'prepare.py').write_bytes(Path(__file__).read_bytes())
record=dict(schema='GDN-graph-combined-formal-actual-prepare-v2',status='RUNNING',started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),specs=[],denied_operations=[],device_run=False,performance_qualified=False)
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
 owner=load(H/'owner.py','_actual_retry_formal_owner')
 before=owner.source_pins();baseline=read(OLD);record['original_owner_plan']=ref(OLD)
 assert sha(OLD)=='3e65c89baecc67b14e8f97c2063d4647085a8ca86c2cc8e6c48dd03577bd6d4b'
 cold=dict(journal_path=str(A/'gdn-graph-combined-formal-v2-cold-journal'),controller=ref(H/'controller.py'),approved_contract='cold-restart-persisted-visible-v1',proposal_plan_sha256='e96a4a4d5be6f6a69d597387bbc566eadedae81065bfa879445352d986515cb8',predecessor={**ref(ARCHIVE/'restored-journal.json'),'audit':ref(ARCHIVE/'restoration-audit.json')})
 plan=owner.plan_template(baseline['qualification'],cold,baseline['http'],baseline['qualifications'],baseline['route_qualifications'],baseline['combined_qualifications'],baseline['prepared_specs'])
 write(OUT/'owner-plan.json',plan)
 _,_,_,_,specs=owner.validate_plan(SimpleNamespace(plan=OUT/'owner-plan.json',plan_sha256=sha(OUT/'owner-plan.json'),out=A/'gdn-graph-combined-formal-v2-owner'))
 for phase,spec in zip(owner.PHASES,specs):
  expected=baseline['prepared_specs'][phase];assert sha(expected['path'])==expected['sha256'] and read(expected['path'])==spec
  record['specs'].append(dict(phase=phase,**expected))
 assert len(specs)==12 and all(sha(p)==h for p,h in before.items())
 for p in before:
  target=OUT/'frozen-source'/p.lstrip('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(p).read_bytes())
 record.update(status='PASS',owner_plan=ref(OUT/'owner-plan.json'),source_pins=before,source_unchanged=True,original_specs_unchanged=True,total_requests=864,formal_requests=768,warm_requests=96)
except BaseException as e:
 import traceback
 record.update(status='FAIL',error=str(e),traceback=traceback.format_exc());raise
finally:
 record['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
print(json.dumps(dict(status=record['status'],specs=len(specs),owner_plan=record['owner_plan'])))
