"""Explicit CPU-only GLM plan builder; does not prepare a service or start a child.

No arguments describe. --run requires a pre-existing pinned original probe plan,
62-shard/runtime inventory, explicit child ceiling and a future actual archived
route-formal RESTORED journal/audit pair. --seal-prepared is separate and only
accepts bytes actually emitted by controller prepare. No future receipt/PID is
invented and no Qwen experiment preflight is used.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
OWNER_SHA='37f47300c14d8da721d6c30fcc71f52325132bad3dc3ba902c1cc08af4326b40'
GUARD_SHA='908e0dc74812fd82b65c63a22e03ed8c5692e812bdbf3d6957e219aa52b14166'
PROBE_SHA='cd9df050384af83d92f0321d95f5bba70dca349defbedf0bb7d1d1e8b57385fc'


def need(ok,message):
    if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def ref(path):
    path=Path(path).resolve();need(path.is_file() and not path.is_symlink(),'existing real file required')
    return dict(path=str(path),sha256=sha(path))
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def module(path,digest,name):
    need(sha(path)==digest,'fixed source changed: '+str(path))
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m

def build_commands(plan_path,plan_digest,plan,analysis,python):
    c=plan['cold_restart'];controller=c['controller']['path'];owner=plan['execution']['owner']['path']
    common=['--journal',c['journal_path'],'--plan-sha256',c['proposal_plan_sha256']]
    approved=['--approved-contract',c['approved_contract']]
    return dict(
        controller_prepare=[python,'-I','-B',controller,'prepare',*common,'--owner-directory',plan['outputs']['owner'],
            '--owner-plan',str(plan_path),'--owner-plan-sha256',plan_digest],
        controller_stop=[python,'-I','-B',controller,'stop',*common,*approved],
        owner=[python,'-I','-B',owner,'--run','--plan',str(plan_path),'--plan-sha256',plan_digest,'--out',plan['outputs']['owner']],
        controller_recover=[python,'-I','-B',controller,'recover',*common,*approved,'--watchdog-seconds','3600'],
        closed_audit=[python,'-I','-B',str(HERE/'audit_closed.py'),'--journal',c['journal_path'],
            '--plan-sha256',plan_digest,'--out',str(analysis)],
        sequence='Actual read-only prepare, separately seal PREPARED bytes, then stop; owner; recover in finally. '
                 'Only a proven pre-Popen non-start may use recover --no-experiment. Unknown launch never gets that flag.')


def validate_outputs(paths):
    need(all(p.is_absolute() and p==p.resolve() for p in paths),'canonical output paths')
    need(len(set(paths))==len(paths) and not any(a.is_relative_to(b) or b.is_relative_to(a)
        for i,a in enumerate(paths) for b in paths[i+1:]),'distinct non-overlapping output paths')


def python_entry(path):
    path=Path(path).absolute()
    need(path.is_file() and path.parent.resolve()/path.name==path,'actual canonical-parent interpreter entry')
    return str(path)  # Preserve venv symlink entry; hashes may resolve its target.


def prepare(args):
    out=Path(args.out).resolve();owner_out=Path(args.owner).resolve();journal=Path(args.journal).resolve()
    analysis=Path(args.analysis).resolve()
    validate_outputs([out,owner_out,journal,analysis])
    need(not out.exists() and out.parent.is_dir(),'new registration directory')
    need(len({out,owner_out,journal})==3 and not owner_out.exists() and not journal.exists()
         and owner_out.parent.is_dir() and journal.parent.is_dir(),'distinct fresh owner/journal outputs')
    need(not analysis.exists() and analysis.parent.is_dir(),'new explicit analysis path')
    owner=module(HERE/'owner.py',OWNER_SHA,'glm_registered_frozen_owner')
    guard=module(HERE.parent/'glm_flash/physical_guard.py',GUARD_SHA,'glm_registered_guard')
    controller=module(HERE/'controller.py',sha(HERE/'controller.py'),'glm_actual_registration_controller')
    probe_ref=dict(path=str(Path(args.probe_plan).resolve()),sha256=args.probe_plan_sha256)
    inventory_ref=dict(path=str(Path(args.verification).resolve()),sha256=args.verification_sha256)
    probe=read(owner.reference(probe_ref));owner.reference(inventory_ref)
    binding=dict(path=str(Path(args.predecessor_journal).resolve()),sha256=args.predecessor_journal_sha256,
        audit=dict(path=str(Path(args.predecessor_audit).resolve()),sha256=args.predecessor_audit_sha256))
    controller.predecessor(binding)
    plan=dict(schema=owner.SCHEMA,execution={},probe_plan=probe_ref,verification=inventory_ref,
        physical_policy=guard.policy(child_limit_bytes=args.child_limit_bytes,work_seconds=probe['deadlines']['work_seconds'],
                                    interval_seconds=args.interval_seconds,probe_seconds=args.probe_seconds),
        cold_restart=dict(journal_path=str(journal),controller=ref(HERE/'controller.py'),predecessor=binding,
                          approved_contract=owner.CONTRACT,proposal_plan_sha256=owner.PROPOSAL),
        outputs=dict(owner=str(owner_out),probe=str(owner_out/'phases'/owner.PHASE/'probe')),
        budgets=dict(work_seconds=1830,prepare_seconds=300,postflight_seconds=120,term_seconds=15,kill_seconds=15,checkpoint_seconds=5))
    plan['execution']=dict(owner=ref(HERE/'owner.py'),pins=owner.source_pins(plan))
    # Actual explicit interpreter and GLM model/runtime inventory remain mandatory
    # at owner admission; the builder validates shape/source, not payloads.
    out.mkdir();path=out/'owner-plan.json';write(path,plan);plan_digest=sha(path)
    try:
        owner.validate_plan(SimpleNamespace(plan=path,plan_sha256=plan_digest,out=owner_out))
        python=python_entry(args.python)
        pins=controller.source_pins(plan)
        for p in (HERE/'preregister.py',HERE/'prepare_inputs.py',HERE/'audit_restore.py',HERE/'audit_closed.py',HERE/'execute.py',Path(python),path,
                  HERE.parent/'omlx_cold_restart/audit_restore_v4.py',HERE.parent/'pareto_realistic/audit_absent_qualification.py',
                  Path(probe_ref['path']),Path(inventory_ref['path']),Path(binding['path']),Path(binding['audit']['path'])):
            pins[str(p)]=sha(p)
        for source,source_digest in pins.items():need(sha(source)==source_digest,'source changed during registration')
        write(out/'commands.json',build_commands(path,plan_digest,plan,analysis,python))
        pins[str(out/'commands.json')]=sha(out/'commands.json')
        write(out/'source-manifest.json',pins)
        result=dict(schema='GLM-cold-load-plan-build-v1',status='CPU_PLAN_ONLY',owner_plan_sha256=plan_digest,
            owner_plan=str(path),outer_python=python,source_pins=pins,outputs=dict(owner=str(owner_out),journal=str(journal),analysis=str(analysis)),
            inventory_scope='supplied GLM/runtime inventory; owner will recheck hashes/stat; no recursive closure or new weight hash',
            service_prepared=False,experiment_started=False)
        write(out/'plan-build.json',result);return result
    except BaseException as error:
        write(out/'plan-build-failure.json',dict(status='FAIL',error_type=type(error).__name__,error=str(error),owner_plan_sha256=plan_digest))
        raise


def seal_prepared(folder):
    folder=Path(folder).resolve();build=read(folder/'plan-build.json');path=folder/'owner-plan.json'
    need(build['status']=='CPU_PLAN_ONLY' and sha(path)==build['owner_plan_sha256'],'actual built plan SHA')
    for p,h in build['source_pins'].items():need(sha(p)==h,'source changed before seal')
    plan=read(path);c=module(HERE/'controller.py',plan['cold_restart']['controller']['sha256'],'glm_seal_actual_controller')
    c.owner_plan(path,build['owner_plan_sha256'],plan['cold_restart']['journal_path'])
    journal=Path(plan['cold_restart']['journal_path'])/'journal.json';data=read(journal)
    prior,_=c.predecessor(plan['cold_restart']['predecessor'])
    need(data['schema']==c.JOURNAL_SCHEMA and data['state']=='PREPARED' and data['controller_sha256']==sha(HERE/'controller.py')
         and data['owner_plan']==dict(path=str(path),sha256=build['owner_plan_sha256'])
         and data['owner_source']==plan['execution']['owner'] and data['owner_directory']==plan['outputs']['owner']
         and data['original']==prior['replacement'] and data['trials']==[c.PHASE]
         and data['predecessor']==plan['cold_restart']['predecessor'] and data['support_sha256']==c.BASE_SHA
         and data['plan_sha256']==c.PLAN_SHA and data['original_status']==prior['final_status']
         and data['global_settings_stat']==prior['global_settings_stat'] and data['model_metadata']==prior['model_metadata'],
         'actual read-only controller preparation required')
    need(not Path(data['owner_directory']).exists(),'owner already appeared before seal')
    receipt=dict(schema='GLM-cold-load-prepared-receipt-v1',owner_plan_sha256=build['owner_plan_sha256'],
        source_pins=build['source_pins'],prepared_journal_sha256=sha(journal),experiment_started=False,outer_python=build['outer_python'])
    write(folder/'receipt.json',receipt);return receipt


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);mode=p.add_mutually_exclusive_group()
    mode.add_argument('--run',action='store_true');mode.add_argument('--seal-prepared',type=Path)
    for name in ('out','owner','journal','analysis','python','probe-plan','verification','predecessor-journal','predecessor-audit'):
        p.add_argument('--'+name,type=Path)
    for name in ('probe-plan-sha256','verification-sha256','predecessor-journal-sha256','predecessor-audit-sha256'):p.add_argument('--'+name)
    p.add_argument('--child-limit-bytes',type=int);p.add_argument('--interval-seconds',type=float,default=1.0)
    p.add_argument('--probe-seconds',type=float,default=5.0);a=p.parse_args(argv)
    if a.seal_prepared:
        need(not any(getattr(a,k) for k in ('out','owner','journal','analysis','python','probe_plan','verification','predecessor_journal','predecessor_audit','child_limit_bytes')),'seal only accepts actual registration folder')
        seal_prepared(a.seal_prepared);return 0
    required=('out','owner','journal','analysis','python','probe_plan','verification','predecessor_journal','predecessor_audit',
              'probe_plan_sha256','verification_sha256','predecessor_journal_sha256','predecessor_audit_sha256','child_limit_bytes')
    if not a.run:
        need(not any(getattr(a,k) for k in required),'explicit --run needed for build arguments')
        print('DESCRIPTION_ONLY: explicit existing GLM probe/inventory and actual archived route-formal RESTORED dual refs; no service action.');return 0
    need(all(getattr(a,k) for k in required),'all explicit GLM inputs, paths, SHA values and physical ceiling required')
    prepare(a);return 0


if __name__=='__main__':raise SystemExit(main())
