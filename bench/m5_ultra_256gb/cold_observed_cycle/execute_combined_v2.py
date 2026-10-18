"""Versioned driver fix: waited owner return permits no-experiment recovery.

Original v1 and its live execution are unchanged. A real PREPARED journal is
still mandatory; this does not resume or retry an already started cycle.
"""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
PREP=ROOT/'.artifacts/m5-ultra-256gb/gdn-graph-combined-prepare-v1'
PLAN=PREP/'owner-plan.json'
PLAN_SHA='a2af16324fb323dcd9a93e4b9f07d4f127415bcaa0a12f04ca01792caf2696d9'
PROPOSAL='e96a4a4d5be6f6a69d597387bbc566eadedae81065bfa879445352d986515cb8'
CONTRACT='cold-restart-persisted-visible-v1'
PYTHON=ROOT/'.venv/bin/python'
OUT=ROOT/'.artifacts/m5-ultra-256gb/gdn-graph-combined-v2-execution'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,data):
 p=OUT/name
 with p.open('x') as f:json.dump(data,f,indent=2);f.write('\n');f.flush()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');args=p.parse_args()
 if not args.run:print('DESCRIPTION_ONLY: fixed stop -> combined numerical owner -> finally recover.');return 0
 if sha(PLAN)!=PLAN_SHA:raise ValueError('fixed owner plan changed')
 plan=json.loads(PLAN.read_bytes());manifest=json.loads(Path(plan['lifecycle']['path']).read_bytes())
 if sha(plan['lifecycle']['path'])!=plan['lifecycle']['sha256']:raise ValueError('lifecycle changed')
 for row in manifest['sources'].values():
  if sha(row['path'])!=row['sha256']:raise ValueError('lifecycle source changed')
 journal=Path(plan['outputs']['journal']);jp=journal/'journal.json';prepared=json.loads(jp.read_bytes())
 if prepared['state']!='PREPARED' or prepared['owner_plan']!={'path':str(PLAN),'sha256':PLAN_SHA}:raise ValueError('actual PREPARED binding required')
 OUT.mkdir(exist_ok=False);(OUT/'prepared-journal.json').write_bytes(jp.read_bytes());(OUT/'execute.py').write_bytes(Path(__file__).read_bytes())
 common=[str(PYTHON),'-I','-B',str(HERE/'controller.py')]
 tail=['--journal',str(journal),'--plan-sha256',PROPOSAL,'--approved-contract',CONTRACT]
 commands={'stop':common+['stop']+tail,'owner':[str(PYTHON),'-I','-B',str(HERE/'owner.py'),'--plan',str(PLAN),'--plan-sha256',PLAN_SHA,'--out',plan['outputs']['out']], 'recover':common+['recover']+tail+['--watchdog-seconds','3600']}
 write('registration.json',{'plan_sha256':PLAN_SHA,'prepared_journal_sha256':sha(jp),'commands':commands,'authorization':'Prior explicit user cold-stop/restore authorization; same original persisted-visible contract'})
 invoked=False;returned=False;codes={};errors=[]
 def call(phase,command):
  write(phase+'-command.json',{'command':command,'started_at':time.time()})
  with (OUT/(phase+'.stdout')).open('x') as out,(OUT/(phase+'.stderr')).open('x') as err:
   value=subprocess.run(command,stdout=out,stderr=err).returncode
  write(phase+'-result.json',{'returncode':value,'ended_at':time.time()});return value
 try:
  codes['stop']=call('stop',commands['stop'])
  if codes['stop']==0:
   invoked=True;write('owner-invocation.json',{'attempted':True,'time':time.time()})
   codes['owner']=call('owner',commands['owner'])
   returned=True;write('owner-returned.json',{'returned':True,'returncode':codes['owner'],'time':time.time()})
 except BaseException as exc:errors.append({'type':type(exc).__name__,'message':str(exc),'time':time.time()})
 finally:
  try:
   state=json.loads(jp.read_bytes())['state']
   if state!='PREPARED':
    recovery=commands['recover']+(['--no-experiment'] if (not invoked or returned) and not Path(plan['outputs']['out']).exists() else [])
    codes['recover']=call('recover',recovery)
  except BaseException as exc:errors.append({'phase':'recover','type':type(exc).__name__,'message':str(exc),'time':time.time()})
  write('final.json',{'codes':codes,'errors':errors,'owner_invocation_attempted':invoked,'owner_invocation_returned':returned,'journal_state':json.loads(jp.read_bytes())['state'],'time':time.time(),'independent_audits_required':True})
 return 0 if codes=={'stop':0,'owner':0,'recover':0} and not errors and json.loads(jp.read_bytes())['state']=='RESTORED' else 1
if __name__=='__main__':raise SystemExit(main())
