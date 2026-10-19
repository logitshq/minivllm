"""Build actual three-mode specs; separate real owner/controller preparation and sealing."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from bench.m5_ultra_256gb.c1_profile_cycle_v1 import _reuse as reuse
_original=reuse.install(globals(),'glm_flash_cycle/preregister.py')


def build_commands(path,digest,plan,analysis,python):
    commands=_original.build_commands(path,digest,plan,analysis,python)
    # Rebuild actual adjacent entries; the imported donor's dictionary is untouched.
    c=plan['cold_restart'];common=['--journal',c['journal_path'],'--plan-sha256',c['proposal_plan_sha256']]
    commands['closed_audit']=[python,'-I','-B',str(HERE/'audit_closed.py'),'--journal',c['journal_path'],'--plan-sha256',digest,'--out',str(analysis)]
    commands['owner_prepare']=[python,'-I','-B',str(HERE/'owner.py'),'--prepare-only','--plan',str(path),'--plan-sha256',digest,'--out',str(path.parent/'owner-prepare')]
    commands['restoration_audit']=[python,'-I','-B',str(HERE/'audit_restore.py'),'--journal',c['journal_path'],'--out',str(Path(analysis).with_name('restoration-audit.json'))]
    commands['sequence']='owner_prepare; controller_prepare; seal actual PREPARED; stop; direct owner; finally recover; independent audits'
    return commands


def prepare(a):
    owner=reuse.local('owner.py');c=reuse.local('controller.py');binding=c.predecessor_manifest()['binding'];c.predecessor(binding)
    out,owner_out,journal,analysis=[Path(getattr(a,k)).resolve() for k in ('out','owner','journal','analysis')]
    validate_outputs([out,owner_out,journal,analysis]);need(all(not p.exists() and p.parent.is_dir() for p in (out,owner_out,journal,analysis)),'fresh output paths')
    formal=read(owner.SPEC);python=python_entry(a.python);capture=reuse.fixed('c1_pipeline_profile/capture.py');glue=reuse.fixed('c1_pipeline_profile/owner.py')
    import secrets
    recipe=glue.prepare_modes(dict(path=str(owner.SPEC),sha256=owner.SPEC_SHA),owner_out,{m:secrets.token_hex(32) for m in owner.PHASES},
        capture_source=ref(HERE.parent/'c1_pipeline_profile/capture.py'),server_sha256=sha(capture.SERVER))
    out.mkdir();prepared={}
    for spec in recipe['specs']:
        path=out/(spec['mode']+'-spec.json');write(path,spec);prepared[spec['mode']]=ref(path)
    plan=dict(schema=owner.PLAN_SCHEMA,execution=dict(owner=ref(HERE/'owner.py'),pins=owner.source_pins()),qualification=formal['base_plan'],
        formal_spec=dict(path=str(owner.SPEC),sha256=owner.SPEC_SHA),prepared_specs=prepared,trials=list(owner.PHASES),
        cold_restart=dict(journal_path=str(journal),controller=ref(HERE/'controller.py'),predecessor=binding,approved_contract=c.CONTRACT,proposal_plan_sha256=c.PLAN_SHA),
        http={k:formal[k] for k in ('transport','deadlines','environment')},outputs=dict(owner=str(owner_out)),deadline_seconds=owner.work_seconds(list(owner.PHASES),formal['deadlines']))
    path=out/'owner-plan.json';write(path,plan);digest=sha(path)
    try:
        owner.validate_plan(SimpleNamespace(plan=path,plan_sha256=digest,out=owner_out))
        commands=build_commands(path,digest,plan,analysis,python);write(out/'commands.json',commands)
        pins={**c.source_pins(plan),str(path):digest,str(out/'commands.json'):sha(out/'commands.json'),python:sha(python)}
        for r in (binding,binding['audit'],*prepared.values()):pins[r['path']]=r['sha256']
        write(out/'source-manifest.json',pins)
        result=dict(schema='C1-profile-plan-build-v1',status='CPU_PLAN_ONLY',owner_plan_sha256=digest,owner_plan=str(path),outer_python=python,source_pins=pins,
            outputs=dict(owner=str(owner_out),journal=str(journal),analysis=str(analysis)),service_prepared=False,experiment_started=False)
        write(out/'plan-build.json',result);return result
    except BaseException as exc:
        write(out/'plan-build-failure.json',dict(status='FAIL',error_type=type(exc).__name__,error=str(exc)));raise


# Preserve original PREPARED identity/settings/source checks; require actual CLI prepare first.
SEAL_SOURCE=reuse.variant(globals(),_original.seal_prepared,[("data['trials']==[c.PHASE]","data['trials']==list(c.PHASES)"),
    ("schema='GLM-cold-load-prepared-receipt-v1'","schema='C1-profile-prepared-receipt-v1'"),
    ("    receipt=dict(","    prepared=read(folder/'owner-prepare/prepared.json')\n    need(prepared['schema']=='C1-profile-owner-prepare-v1' and prepared['status']=='PREPARED' and prepared['owner_plan_sha256']==build['owner_plan_sha256'] and prepared['sources']==plan['execution']['pins'],'actual owner CLI preparation required')\n    receipt=dict(")])

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);m=p.add_mutually_exclusive_group();m.add_argument('--run',action='store_true');m.add_argument('--seal-prepared',type=Path)
    for name in ('out','owner','journal','analysis','python'):p.add_argument('--'+name,type=Path)
    a=p.parse_args(argv)
    if a.seal_prepared:
        need(not any(getattr(a,k) for k in ('out','owner','journal','analysis','python')),'seal only accepts registration folder');seal_prepared(a.seal_prepared);return 0
    if not a.run:
        need(not any(getattr(a,k) for k in ('out','owner','journal','analysis','python')),'explicit --run required');print('DESCRIPTION_ONLY: actual GLM predecessor remains required.');return 0
    need(all(getattr(a,k) for k in ('out','owner','journal','analysis','python')),'all explicit paths required');prepare(a);return 0
if __name__=='__main__':raise SystemExit(main())
