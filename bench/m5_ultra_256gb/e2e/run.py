"""Target-machine orchestration around the unchanged historical E2E protocol."""
import argparse,ast,concurrent.futures,copy,errno,hashlib,importlib.metadata,json,os,secrets,signal,socket,statistics,subprocess,sys,threading,time,traceback,types
from pathlib import Path
D=Path(__file__).resolve().parent;REPO=D.parents[2]
P=REPO/'.artifacts/m5-ultra-256gb/fullqual-build2'
HARNESS=REPO/'.artifacts/m5-ultra-256gb/build2-harness'
os.environ['MINIVLLM_EXPERIMENT_DOMAIN']=str(P)
sys.path.insert(0,str(HARNESS))
import driver as qual
import runtime
sys.path.insert(0,str(D))
PYTHON=qual.PYTHON;CHILD=D/'child.py';CANDIDATE=REPO/'.artifacts/m5-ultra-256gb/source-m5-decode-v2/v19/tutorial_code/miniengine'
INPUTS=REPO/'.artifacts/m5-ultra-256gb/e2e-inputs-build1'
SOURCE=INPUTS/'candidate-manifest.json';FROZEN=INPUTS/'frozen.json'
CONFIGS=[('pair1-A-off',False),('pair1-B-on',True),('pair2-B-on',True),('pair2-A-off',False),('pair3-A-off',False),('pair3-B-on',True)]
sha=runtime.digest

def write_new(path,value):
 with Path(path).open('x') as stream:json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False)
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def verify_source():
 f=json.loads(qual.FROZEN.read_text());actual={str(p.relative_to(CANDIDATE)):sha(p) for p in CANDIDATE.rglob('*.py')}
 assert actual==f['candidate_sources']
 return {'root':str(CANDIDATE),'source_files_sha256':actual,'source_count':26,'source_manifest_sha256':sha(SOURCE)}
def verify_frozen():
 ef=json.loads(FROZEN.read_text())
 for path,want in ef['files'].items():assert sha(path)==want,'E2E pin changed '+path
 f=qual.verify_frozen(ef['fullqual_frozen_sha256']);verify_source();return f
def actual_preimport_identity():
 f=json.loads(qual.FROZEN.read_text());runtime.reject_preloaded(sys.modules)
 return {**runtime.preflight(f),'source':verify_source()}
def environment_record(env):return {k:v for k,v in env.items() if k.startswith(('MLX_','METAL_','DYLD_','OMLX_')) or k in ('PYTHONPATH','PYTHONHOME','PYTHONDONTWRITEBYTECODE')}
boot=types.SimpleNamespace(environment_record=environment_record)
def child_environment(inherited):
 expected=json.loads(qual.FROZEN.read_text())['selected_environment']
 assert environment_record(inherited)==expected,'exact four-controls domain required'
 return dict(inherited)
def pid_absence(pid):
 r=subprocess.run(['/bin/ps','-p',str(pid),'-o','pid=,ppid=,rss=,comm='],text=True,capture_output=True)
 return {'pid':pid,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'absent':r.returncode==1 and r.stdout=='' and r.stderr==''}
def phase_guard(samples,power,pid):
 import resources
 resources.guard(samples);s=samples[-1]
 assert s['pressure_level_value']==1 and s['power_metadata']['status']=='known' and s['power_metadata']['source']=='AC Power'
 assert s['processinfo_thermal_metadata']['status']=='known' and s['processinfo_thermal_metadata']['code'] in (0,1)
 if pid:assert s['exact_owned_pid_metadata']['status']=='known'
# Execute only frozen, named protocol function definitions, with this target context.
exec(compile((D/'protocol.py').read_bytes(),str(D/'protocol.py'),'exec'),globals())

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');a=ap.parse_args()
 attempt=REPO/'.artifacts/m5-ultra-256gb/e2e-build1'
 if a.prepare:
  INPUTS.mkdir()
  f=json.loads(qual.FROZEN.read_text());write_new(SOURCE,{'files':f['candidate_sources']})
  proof=json.loads((P/'replay.json').read_text());release=json.loads((P/'GPU-RELEASE.json').read_text())
  assert proof['verdict']=='PASS_FULLMODEL_BOUNDED_SCOPE' and release['status']=='GPU_RELEASED'
  files=list(D.glob('*.py'))+[SOURCE,P/'replay.json',P/'GPU-RELEASE.json',P/'frozen.json']
  write_new(FROZEN,{'fullqual_frozen_sha256':sha(qual.FROZEN),'files':{str(p):sha(p) for p in files},'configs':CONFIGS,'adoption':{'positive_y_pairs':3,'aggregate_y_min_percent':1.,'aggregate_x_min_percent':0.}})
  print(sha(FROZEN));return
 f=verify_frozen();child_environment(os.environ)
 # Separate stdlib replay before any model process; previous actual STOP remains archived.
 spec=importlib.util.spec_from_file_location('target_fullqual_replay',HARNESS/'replay.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 qualification=module.verify(P/'gpu-fullmodel-run1');assert qualification['verdict']=='PASS_FULLMODEL_BOUNDED_SCOPE'
 attempt.mkdir();go_path=attempt/'GO.json';go={'owner_pid':os.getpid(),'frozen_sha256':sha(FROZEN),'scope':'V19 M5 decode OFF/ON internal C1 only','order':CONFIGS};write_new(go_path,go)
 write_new(attempt/'attempt-consumed.json',{'GO_sha256':sha(go_path),'owner_pid':os.getpid()})
 shared={};completed=[];failure=None;cleanup=[];samples=[]
 try:
  probe(samples,attempt/'owner-resource-derived.json','owner-before-first-child','AC')
  for label,enabled in CONFIGS:
   print('START '+label,flush=True);completed.append(run_one(label,enabled,attempt,'AC',shared,go_path));print(json.dumps(completed[-1]),flush=True)
  write_new(attempt/'suite-complete.json',{'configs':completed,'actual_requests':36,'actual_completion_tokens':1152,'measured_requests':18,'measured_tokens':576,'power':'AC','GPU_released':False})
 except BaseException as e:
  failure={'type':type(e).__name__,'text':str(e),'traceback':traceback.format_exc()};write_new(attempt/'suite-failure.json',failure)
 finally:
  try:probe(samples,attempt/'owner-resource-derived.json','owner-finally','AC')
  except BaseException as e:cleanup.append(str(e))
  write_new(attempt/'owner-cleanup.json',{'pid':os.getpid(),'first_error':failure,'cleanup_errors':cleanup,'completed':completed,'GPU_released':False})
 if failure or cleanup:raise SystemExit(1)
if __name__=='__main__':main()
