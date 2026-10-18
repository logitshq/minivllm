"""Build immutable twelve-trial plans from the already validated actual specs.

Default describes only. --prepare writes new small evidence; it neither imports
MLX nor performs model/runtime payload checks, processes, service, or HTTP I/O.
A future actual GDN37 RESTORED journal + independent audit must be explicit.
--seal-receipt only binds a real controller-created PREPARED journal afterward.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
ART=ROOT/'.artifacts/m5-ultra-256gb'
PREPARED=ART/'route-http-formal-prepare-v1'
P=HERE/'prereg-formal-v2'
NAME='route-http-formal-v2'
FIXED={
    'owner.py':'455683cb8dc9d09b1e7c85a4ae586d8ffadf58bf6a637d23334975fabec3121d',
    'controller.py':'98d48eb4ad608be5d8d65f6a28622126483ee96ff9a312b3cbd16ee4c63817f3',
    'audit_restore.py':'90a980d0dbe6a61fc1bab2ac6b2edaa9bf012bb60726cd0742801d11410eb0b1',
    'audit_formal.py':'ed8ad64c3eb5fed052effd05981a386006f4ea1f752463bb0ada0c2f0e4ea5e0',
    'formal.py':'0e590191d32b12027bc8b5f4f76335ed42a633a4179d90431427aa42c6e18f30',
    'execute.py':'15deb40c8a72eaa08c677e9a68f30261e106b1c9efe0465b481656f075e9b9fa',
}


def need(ok,message):
    if not ok:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):return json.loads(Path(path).read_bytes())

def ref(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))

def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def modules():
    for name,digest in FIXED.items():need(sha(HERE/name)==digest,'unsealed/changed actual source: '+name)
    def load(name):
        s=importlib.util.spec_from_file_location('route_formal_registration_'+name,HERE/(name+'.py'))
        m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
    return load('owner'),load('controller')


def prepare(predecessor):
    owner,c=modules();prior,_=c.predecessor(predecessor)
    need(prior['state']=='RESTORED','actual restoration required')
    need(not P.exists(),'new preregistration directory required')
    for suffix in ('owner','cold-journal','execution','analysis','cpu-prepare'):
        need(not (ART/(NAME+'-'+suffix)).exists(),'fresh experiment outputs required')
    receipt=read(PREPARED/'result.json')
    need(receipt['status']=='PASS' and receipt['prepared_specs']==12
         and receipt['source_unchanged'] is True and receipt['inputs_unchanged'] is True
         and receipt['device_run'] is False and receipt['execution_registered'] is False,
         'actual CPU prepared12 spec receipt required')
    specs={phase:read(PREPARED/'specs'/(phase+'.json')) for phase in owner.PHASES}
    first=specs[owner.PHASES[0]]
    http={k:first[k] for k in ('transport','deadlines','environment')}
    q=first['base_plan'];baseline={};route={}
    for C in (1,8):
        spec=next(x for x in specs.values() if x['concurrency']==C)
        baseline[str(C)]=spec['baseline_qualification'];route[str(C)]=spec['route_qualification']
    cold=dict(journal_path=str(ART/(NAME+'-cold-journal')),controller=ref(HERE/'controller.py'),
        approved_contract=c.CONTRACT,proposal_plan_sha256=c.PLAN_SHA,predecessor=predecessor)
    plan=owner.plan_template(q,cold,http,baseline,route,
        {phase:ref(PREPARED/'specs'/(phase+'.json')) for phase in owner.PHASES})
    P.mkdir();write(P/'owner-plan.json',plan);plan_digest=sha(P/'owner-plan.json')
    owner.validate_plan(SimpleNamespace(plan=P/'owner-plan.json',plan_sha256=plan_digest))
    c.owner_plan(P/'owner-plan.json',plan_digest,Path(cold['journal_path']))
    pins=c.source_pins(plan)
    extra=[Path(__file__).resolve(),HERE/'execute.py',P/'owner-plan.json',PREPARED/'result.json',
           *(PREPARED/'specs'/(p+'.json') for p in owner.PHASES),
           Path(predecessor['path']),Path(predecessor['audit']['path'])]
    for name in ('test_owner_cpu.py','test_controller_cpu.py','test_audit_restore_cpu.py','test_audit_formal_cpu.py','test_execute_cpu.py','test_preregister_cpu.py'):
        need((HERE/name).is_file(),'actual reviewed test source required');extra.append(HERE/name)
    for path in extra:pins[str(path)]=sha(path)
    frozen=P/'frozen-source';frozen.mkdir()
    for source,source_digest in pins.items():
        raw=Path(source).read_bytes();need(hashlib.sha256(raw).hexdigest()==source_digest,'source changed during registration')
        dest=frozen/Path(source).relative_to('/');dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as f:f.write(raw)
    python=read(q['path'])['execution']['python'];out=ART/(NAME+'-owner');journal=Path(cold['journal_path'])
    common=['--journal',str(journal),'--approved-contract',c.CONTRACT,'--plan-sha256',c.PLAN_SHA]
    commands=dict(
        cpu_prepare=[python,'-I','-B',str(HERE/'owner.py'),'--prepare-only','--plan',str(P/'owner-plan.json'),
                     '--plan-sha256',plan_digest,'--out',str(ART/(NAME+'-cpu-prepare'))],
        readonly_service_prepare=[python,'-I','-B',str(HERE/'controller.py'),'prepare','--journal',str(journal),
            '--owner-directory',str(out),'--owner-plan',str(P/'owner-plan.json'),'--owner-plan-sha256',plan_digest,'--plan-sha256',c.PLAN_SHA],
        seal_receipt=[python,'-I','-B',str(Path(__file__).resolve()),'--seal-receipt'],
        execute=[python,'-I','-B',str(HERE/'execute.py'),'--run'],
        stop=[python,'-I','-B',str(HERE/'controller.py'),'stop',*common],
        owner=[python,'-I','-B',str(HERE/'owner.py'),'--plan',str(P/'owner-plan.json'),'--plan-sha256',plan_digest,'--out',str(out)],
        recover=[python,'-I','-B',str(HERE/'controller.py'),'recover',*common,'--watchdog-seconds','3600'],
        restore_audit=[python,'-I','-B',str(HERE/'audit_restore.py'),'--journal',str(journal),'--out',str(ART/(NAME+'-analysis')/'restoration-audit.json')],
        independent_formal_audit=[python,'-I','-B',str(HERE/'audit_formal.py'),'--owner',str(out),'--journal',str(journal),
            '--plan-sha256',plan_digest,'--out',str(ART/(NAME+'-analysis')/'independent-formal.json')],
        scope='Stop/owner/finally-recover requires the supervising execution driver; never run owner without recovery supervision.')
    write(P/'commands.json',commands)
    write(P/'preparation.json',dict(schema='E024-route-formal-prereg-build-v1',status='PLAN_BUILT',
        owner_plan_sha256=plan_digest,source_pins=pins,predecessor=predecessor,prepared_specs=plan['prepared_specs'],
        planned_trials=plan['trials'],warm_requests=96,formal_requests=768,hard_work_seconds=plan['deadline_seconds'],
        runtime_weights_verified=False,service_prepared=False,performance_qualified=False))
    return dict(status='PLAN_BUILT',owner_plan_sha256=plan_digest,commands=str(P/'commands.json'))


def seal_receipt():
    owner,c=modules();preparation=read(P/'preparation.json');pins=dict(preparation['source_pins'])
    for p,h in pins.items():need(sha(p)==h,'prepared source changed: '+p)
    plan_digest=preparation['owner_plan_sha256'];plan=read(P/'owner-plan.json')
    owner.validate_plan(SimpleNamespace(plan=P/'owner-plan.json',plan_sha256=plan_digest))
    jpath=Path(plan['cold_restart']['journal_path'])/'journal.json';j=read(jpath)
    prior,_=c.predecessor(plan['cold_restart']['predecessor'])
    need(j['schema']==c.JOURNAL_SCHEMA and j['state']=='PREPARED' and j['original']==prior['replacement']
         and j['owner_plan']==dict(path=str(P/'owner-plan.json'),sha256=plan_digest)
         and j['owner_source']==plan['execution']['owner'] and j['controller_sha256']==sha(HERE/'controller.py')
         and j['predecessor']==plan['cold_restart']['predecessor'] and j['trials']==plan['trials']
         and j['owner_directory']==str(ART/(NAME+'-owner')) and j['plan_sha256']==c.PLAN_SHA,
         'actual read-only PREPARED journal required')
    cpu=ART/(NAME+'-cpu-prepare/prepared.json')
    need(read(cpu)['owner_plan_sha256']==plan_digest and read(cpu)['status']=='PREPARED','actual owner CPU preparation receipt')
    for path in (P/'commands.json',P/'preparation.json',cpu):pins[str(path)]=sha(path)
    value=dict(schema='E024-route-formal-prereg-receipt-v1',status='PREPARED',owner_plan_sha256=plan_digest,
        prepared_journal_sha256=sha(jpath),source_pins=pins,performance_qualified=False)
    write(P/'receipt.json',value);return value


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);mode=p.add_mutually_exclusive_group()
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--seal-receipt',action='store_true')
    p.add_argument('--predecessor',type=Path);p.add_argument('--predecessor-sha256')
    p.add_argument('--restoration-audit',type=Path);p.add_argument('--restoration-audit-sha256');args=p.parse_args(argv)
    if args.prepare:
        need(all((args.predecessor,args.predecessor_sha256,args.restoration_audit,args.restoration_audit_sha256)),
             'explicit future archived RESTORED journal/audit refs required')
        result=prepare(dict(path=str(args.predecessor.resolve()),sha256=args.predecessor_sha256,
            audit=dict(path=str(args.restoration_audit.resolve()),sha256=args.restoration_audit_sha256)))
    elif args.seal_receipt:result=seal_receipt()
    else:result=dict(status='DESCRIPTION_ONLY',trials=12,warm_requests=96,formal_requests=768,
        requirement='Actual GDN37 RESTORED archive dual refs; frozen sources; explicit --prepare. No service action here.')
    print(json.dumps(result,indent=2,allow_nan=False));return 0


if __name__=='__main__':raise SystemExit(main())
