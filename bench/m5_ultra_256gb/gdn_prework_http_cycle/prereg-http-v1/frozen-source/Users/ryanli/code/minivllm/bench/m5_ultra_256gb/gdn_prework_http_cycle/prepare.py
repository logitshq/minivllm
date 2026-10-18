"""Register GDN-only C1/C8 source/metadata; never stop, load, or inspect services.

--run writes a fresh registration and runs only owner --prepare-only in-process.
The saved readonly_service_prepare command is for the outer operator. Only after
that actual command has produced PREPARED may --seal-prepared create the driver
receipt. Empty invocation is inert. No model payload or runtime is imported.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
ART=ROOT/'.artifacts/m5-ultra-256gb'
REG=HERE/'prereg-http-v1'
OLD=HERE.parent/'pareto_realistic/prereg-http-baseline-v2/owner-plan.json'
ARCHIVE=ROOT/'docs/experiments/m5-ultra-256gb/E024-route-http-formal-v2-reparse-v1'
JOURNAL=ART/'gdn-http-v1-cold-journal'
OWNER=ART/'gdn-http-v1-owner'
CPU=ART/'gdn-http-v1-cpu-prepare'
EXECUTION=ART/'gdn-http-v1-execution'
PY=ROOT/'.venv/bin/python'


def need(ok,message):
    if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def ref(path):return dict(path=str(path),sha256=sha(path))
def save(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def module(name):
    path=HERE/(name+'.py');s=importlib.util.spec_from_file_location('_gdn_cycle_registration_'+name,path)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def collect_sources(plan,c):
    # Explicit small-file closure. No recursive model/runtime tree or payload.
    paths={Path(p) for p in c.source_pins(plan)}
    for folder in ('gdn_prework_http_cycle','gdn_prework_http','gdn_prework_unbound',
                   'pareto_realistic','route_http_cycle','route_http_formal',
                   'omlx_cold_restart','cold_cycle','long_decode_graph','long_decode_isolation',
                   'route_decode_graph','identity_refill'):
        paths.update((HERE.parent/folder).glob('*.py'))
    paths.update(HERE.parent/p for p in ('resources.py','resources_v2.py','resource_base.py','cold_cycle/lifecycle.json'))
    paths.update((REG/'owner-plan.json',Path(plan['qualification']['path'])))
    paths.update(Path(x['path']) for x in plan['serving'].values())
    predecessor=plan['cold_restart']['predecessor']
    paths.update((Path(predecessor['path']),Path(predecessor['audit']['path'])))
    pins={}
    for path in sorted(paths):
        need(path.is_absolute() and path.is_file() and not path.is_symlink() and path.is_relative_to(ROOT)
             and path.stat().st_size<32*(1<<20),'small regular source only')
        raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest();pins[str(path)]=digest
        dest=REG/'frozen-source'/str(path).lstrip('/');dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as stream:stream.write(raw)
    need(all(sha(p)==h for p,h in pins.items()),'source changed while registering')
    return pins


def prepare():
    need(Path(sys.executable)==PY and sys.flags.isolated==1 and sys.dont_write_bytecode is True,
         'actual outer Python -I -B required')
    need(not any(p.exists() for p in (REG,JOURNAL,OWNER,CPU,EXECUTION)),'fresh cycle paths required')
    c,o=module('controller'),module('owner')
    predecessor={**ref(ARCHIVE/'restored-journal.json'),'audit':ref(ARCHIVE/'restoration-audit.json')}
    prior,_=c.predecessor(predecessor)
    need(read(ARCHIVE/'archive-verification.json')['status']=='PASS','completed predecessor archive required')
    final=read(ART/'route-http-formal-v2-execution/final.json')
    need(final['journal_state']==prior['state']=='RESTORED' and final['recovery_returncode']==0,
         'actual prior direct driver closed and restored')
    old=read(OLD);audit=read(o.SERVING_AUDIT);child=Path(audit['child_directory'])
    serving={k:ref(p) for k,p in [('plan',child/'plan.json'),('complete',child/'complete.json'),('audit',o.SERVING_AUDIT)]}
    cold=dict(journal_path=str(JOURNAL),controller=ref(HERE/'controller.py'),approved_contract=c.CONTRACT,
              proposal_plan_sha256=c.PLAN_SHA,predecessor=predecessor)
    plan=o.plan_template(old['qualification'],cold,old['http'],serving)
    REG.mkdir();save(REG/'owner-plan.json',plan);plan_digest=sha(REG/'owner-plan.json')
    commands=dict(cpu_prepare=[str(PY),'-I','-B',str(HERE/'owner.py'),'--prepare-only','--plan',str(REG/'owner-plan.json'),
          '--plan-sha256',plan_digest,'--out',str(CPU)],
        readonly_service_prepare=[str(PY),'-I','-B',str(HERE/'controller.py'),'prepare','--journal',str(JOURNAL),
          '--owner-directory',str(OWNER),'--owner-plan',str(REG/'owner-plan.json'),'--owner-plan-sha256',plan_digest,'--plan-sha256',c.PLAN_SHA],
        execute=[str(PY),'-I','-B',str(HERE/'execute.py'),'--run'],
        seal_prepared=[str(PY),'-I','-B',str(HERE/'prepare.py'),'--seal-prepared'])
    save(REG/'commands.json',commands)
    pins=collect_sources(plan,c)
    save(REG/'source-manifest.json',dict(schema='GDN-http-registered-source-v1',source_pins=pins))
    started=time.monotonic();result=dict(schema='GDN-http-registration-v1',status='INCOMPLETE',
        owner_plan_sha256=plan_digest,argv=commands['cpu_prepare'],source_pins=pins,
        readonly_service_prepare_executed=False,service_stopped=False,http_executed=False,
        numerical_qualified=False,performance_qualified=False)
    try:
        # The exact saved CLI is interpreted by the actual owner. No subprocess.
        code=o.main(commands['cpu_prepare'][4:])
        result['returncode']=code
        need(code==0,'CPU owner preparation failed')
        prepared=read(CPU/'prepared.json')
        need(prepared['owner_plan_sha256']==plan_digest and prepared['status']=='PREPARED','actual CPU preparation binding')
        save(REG/'capture-specs.json',prepared['capture_specs'])
        need(all(sha(p)==h for p,h in pins.items()),'source changed after CPU preparation')
        result.update(status='PASS',prepared=ref(CPU/'prepared.json'))
    except BaseException as exc:
        result.update(status='FAIL',error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
    finally:
        result['elapsed_seconds']=time.monotonic()-started;save(REG/'registration-result.json',result)
    return int(result['status']!='PASS')


def seal_prepared():
    result=read(REG/'registration-result.json');need(result['status']=='PASS','actual CPU registration PASS required')
    pins=read(REG/'source-manifest.json')['source_pins']
    need(result['source_pins']==pins and all(sha(p)==h for p,h in pins.items()),'registered source changed')
    raw=(JOURNAL/'journal.json').read_bytes();j=json.loads(raw);plan=read(REG/'owner-plan.json')
    c=module('controller');c.owner_plan(REG/'owner-plan.json',result['owner_plan_sha256'],JOURNAL)
    need(j['schema']==c.JOURNAL_SCHEMA and j['state']=='PREPARED' and j['controller_sha256']==sha(HERE/'controller.py')
         and j['owner_plan']==dict(path=str(REG/'owner-plan.json'),sha256=result['owner_plan_sha256'])
         and j['owner_directory']==str(OWNER) and j['owner_source']==plan['execution']['owner']
         and j['phases']==list(c.PHASES) and j['predecessor']==plan['cold_restart']['predecessor']
         and not OWNER.exists() and not EXECUTION.exists(),'actual untouched PREPARED journal required')
    with (REG/'prepared-journal.json').open('xb') as f:f.write(raw)
    receipt=dict(schema='GDN-http-execution-receipt-v1',created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        owner_plan_sha256=result['owner_plan_sha256'],prepared_journal_sha256=hashlib.sha256(raw).hexdigest(),
        source_count=len(pins),source_pins=pins,source_metadata_prepare_pass=True,readonly_service_prepare_pass=True,
        service_stopped=False,actual_http_executed=False,numerical_qualified=False,performance_qualified=False)
    save(REG/'receipt.json',receipt)
    return 0


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);mode=p.add_mutually_exclusive_group()
    mode.add_argument('--run',action='store_true');mode.add_argument('--seal-prepared',action='store_true');a=p.parse_args(argv)
    if a.run:return prepare()
    if a.seal_prepared:return seal_prepared()
    print('DESCRIPTION_ONLY: no service action; --run only registers and CPU-prepares. Actual read-only controller prepare is separate.')
    return 0

if __name__=='__main__':raise SystemExit(main())
