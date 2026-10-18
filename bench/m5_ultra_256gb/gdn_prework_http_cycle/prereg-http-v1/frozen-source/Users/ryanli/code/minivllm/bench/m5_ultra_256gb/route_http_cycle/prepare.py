"""Register source/metadata and read-only service preparation; explicit --run.

Must follow actual formal-v2 RESTORED, its independent audit and archive. This
builder cannot stop/load a service or launch a model. The separate execute.py
uses the saved receipt only after review, tests and Git preregistration.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
ART=ROOT/'.artifacts/m5-ultra-256gb'
REG=HERE/'prereg-http-v1'
OLD=HERE.parent/'pareto_realistic/prereg-http-baseline-v2/owner-plan.json'
SERVING=HERE.parent/'route_decode_graph/prereg-serving-v1/serving-plan.json'
ARCHIVE=ROOT/'docs/experiments/m5-ultra-256gb/E023-omlx-d512-formal-v2'
JOURNAL=ART/'route-http-v1-cold-journal'
OWNER=ART/'route-http-v1-owner'
CPU=ART/'route-http-v1-cpu-prepare'
PY=ROOT/'.venv/bin/python'

def need(ok,message):
    if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def ref(path):return dict(path=str(path),sha256=sha(path))
def save(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def module(name):
    path=HERE/(name+'.py');spec=importlib.util.spec_from_file_location('_route_http_prepare_'+name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

def prepare():
    need(Path(sys.executable)==PY and sys.flags.isolated==1 and sys.dont_write_bytecode is True,'actual outer Python -I -B required')
    final=read(ART/'omlx-d512-formal-v2-execution/final.json');j=read(ARCHIVE/'restored-journal.json');a=read(ARCHIVE/'restoration-audit.json')
    need(final['journal_state']==j['state']=='RESTORED' and final['recovery_returncode']==0 and a['status']=='PASS' and a['qualified'] is True and a['journal_sha256']==sha(ARCHIVE/'restored-journal.json'),'closed independently restored actual predecessor')
    need(read(ARCHIVE/'archive-verification.json')['status']=='PASS','completed preserved predecessor archive')
    need(not any(p.exists() for p in (REG,JOURNAL,OWNER,CPU,ART/'route-http-v1-execution')),'fresh new cycle paths required')
    # Unsealed source constants reject here, before any live service inspection.
    c=module('controller');o=module('owner');predecessor={**ref(ARCHIVE/'restored-journal.json'),'audit':ref(ARCHIVE/'restoration-audit.json')}
    c.predecessor(predecessor)
    old=read(OLD);cold=dict(journal_path=str(JOURNAL),controller=ref(HERE/'controller.py'),approved_contract=c.CONTRACT,proposal_plan_sha256=c.PLAN_SHA,predecessor=predecessor)
    serving=dict(plan=ref(SERVING),audit=ref(ROOT/'docs/experiments/m5-ultra-256gb/E024-serving-v1/independent-numerical.json'))
    plan=o.plan_template(old['qualification'],cold,old['http'],serving)
    REG.mkdir();save(REG/'owner-plan.json',plan);plan_sha=sha(REG/'owner-plan.json')
    commands=dict(cpu_prepare=[str(PY),'-I','-B',str(HERE/'owner.py'),'--prepare-only','--plan',str(REG/'owner-plan.json'),'--plan-sha256',plan_sha,'--out',str(CPU)],
        readonly_service_prepare=[str(PY),'-I','-B',str(HERE/'controller.py'),'prepare','--journal',str(JOURNAL),'--owner-directory',str(OWNER),'--owner-plan',str(REG/'owner-plan.json'),'--owner-plan-sha256',plan_sha,'--plan-sha256',c.PLAN_SHA],
        execute=[str(PY),'-I','-B',str(HERE/'execute.py'),'--run'])
    save(REG/'commands.json',commands)
    for name in ('cpu_prepare','readonly_service_prepare'):
        cmd=commands[name];start=time.monotonic();record_path=REG/(name+'-execution.json')
        record=dict(argv=cmd,status='STARTED',started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),timeout_seconds=120)
        save(record_path,record)
        try:
            with (REG/(name+'.stdout')).open('xb') as out,(REG/(name+'.stderr')).open('xb') as err:
                result=subprocess.run(cmd,cwd=ROOT,stdout=out,stderr=err,timeout=120,check=False)
            record.update(status='RETURNED',returncode=result.returncode)
        except BaseException as exc:
            record.update(status='ERROR',error=dict(type=type(exc).__name__,message=str(exc)))
            raise
        finally:
            record['elapsed_seconds']=time.monotonic()-start
            record_path.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        need(result.returncode==0,name+' failed; preserve attempted registration')
    prepared=read(JOURNAL/'journal.json');need(prepared['state']=='PREPARED' and prepared['owner_plan']['sha256']==plan_sha,'actual read-only PREPARED binding')
    shutil.copyfile(JOURNAL/'journal.json',REG/'prepared-journal.json')
    save(REG/'capture-specs.json',read(CPU/'prepared.json')['capture_specs'])
    # Explicit small-source superset only. The owner performs original full
    # runtime/model verification before and after the actual experiment.
    paths={Path(p) for p in plan['execution']['pins']}
    paths.update(Path(p) for p in c.source_pins(plan))
    paths.update(HERE.parent/'omlx_d512_formal_retry'/name for name in ('controller.py','audit_restore.py'))
    for name in ('route_http_cycle','pareto_realistic','omlx_cold_restart','cold_cycle'):
        paths.update((HERE.parent/name).glob('*.py'))
    paths.update(HERE.parent/n for n in ('resources.py','resources_v2.py','resource_base.py','cold_cycle/lifecycle.json'))
    paths.update((HERE.parent/'route_decode_graph').glob('*.py'))
    paths.update([REG/'owner-plan.json',Path(plan['qualification']['path']),SERVING,Path(serving['audit']['path']),Path(predecessor['path']),Path(predecessor['audit']['path'])])
    pins={}
    for path in sorted(paths):
        need(path.is_absolute() and path.is_file() and not path.is_symlink() and path.is_relative_to(ROOT) and path.stat().st_size<32*(1<<20),'small regular source only')
        raw=path.read_bytes();pin=hashlib.sha256(raw).hexdigest();pins[str(path)]=pin
        dest=REG/'frozen-source'/str(path).lstrip('/');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
    need(all(sha(p)==h for p,h in pins.items()),'source changed while preparing')
    save(REG/'source-manifest.json',dict(schema='route-http-registered-small-source-v1',source_pins=pins,scope='Explicit source-only superset; no model payload or recursive runtime claim.'))
    receipt=dict(schema='route-http-execution-receipt-v1',created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),owner_plan_sha256=plan_sha,prepared_journal_sha256=sha(REG/'prepared-journal.json'),source_count=len(pins),source_pins=pins,
        source_metadata_prepare_pass=True,readonly_service_prepare_pass=True,service_stopped=False,actual_http_executed=False,numerical_qualified=False,performance_qualified=False)
    save(REG/'receipt.json',receipt);print(json.dumps({k:v for k,v in receipt.items() if k!='source_pins'}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');a=p.parse_args()
    if a.run:prepare()
    else:print('DESCRIPTION_ONLY: no service or subprocess operation; actual archived predecessor required.')
