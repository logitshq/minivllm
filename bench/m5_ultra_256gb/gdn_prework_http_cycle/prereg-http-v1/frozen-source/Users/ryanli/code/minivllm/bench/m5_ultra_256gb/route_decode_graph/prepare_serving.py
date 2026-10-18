"""Prepare only new E024 serving plans; no service, process, model or GPU action.

Default: describe inputs/scope without runtime imports, hashing or writes. Explicit
--run verifies a real immutable archived predecessor and creates a NEW prereg
directory. It never calls controller.prepare/stop/recover or owner.prepare/run.
The separate root-owned cycle must first finish CPU review/tests and source
freeze, then execute the saved commands with finally-recover semantics.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
LIFECYCLE=HERE.parent/'cold_cycle_v2'
INHERITED=HERE.parent/'long_decode_graph/prereg-v1/owner-plan.json'
INHERITED_SHA='4dff5f46d0b35d038ffbb0e32ebb6197b047f4da6e847b3160f3ae8685caf6a4'
NUMERIC_SHA='21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb'
BASE_AUDITOR=HERE.parent/'pareto_realistic/audit_qualification.py'
BASE_AUDITOR_SHA='b6d90413911f2b17e0951a2fcfdb5fe00666aa06d4382b603e29e5dc4659d4f1'
RESOURCE_AUDITOR=HERE.parent/'resource_reparse.py'
RESOURCE_AUDITOR_SHA='5948ccbf7baeb21d1e71d9e8f7169b1bd9f84e65eb492746aba86fe377806c23'
PHASE='e024-route-serving'
ARTIFACTS=ROOT/'.artifacts/m5-ultra-256gb'


def need(ok,message):
    if not ok:raise ValueError(message)
def read(path):return json.loads(Path(path).read_bytes())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def reference(path):
    path=Path(path).resolve();return dict(path=str(path),sha256=sha(path))
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def prepare(args):
    out=args.out.resolve();journal=args.journal.resolve();owner=args.owner.resolve()
    need(not out.exists() and out.parent.is_dir(),'new prereg directory with an existing parent required')
    need(out.is_relative_to(HERE) and out!=HERE,'prereg output must be a new directory inside route_decode_graph')
    for path in (journal,owner):
        need(not path.exists() and path.parent.is_dir() and path.is_relative_to(ARTIFACTS),
             'new experiment outputs under existing artifact parent required')
    need(not owner.is_relative_to(journal) and not journal.is_relative_to(owner),'separate owner/journal outputs')
    # Verify the adjacent lifecycle before importing its read-only validation
    # functions. No backend System is constructed or controller main invoked.
    lifecycle_ref=reference(LIFECYCLE/'lifecycle.json');manifest=read(lifecycle_ref['path'])
    need(set(manifest)=={'schema','sources'} and manifest['schema']=='cold-single-child-lifecycle-v2'
         and set(manifest['sources'])=={'controller','owner','audit_restore'},'exact v2 lifecycle manifest')
    for name,item in manifest['sources'].items():
        path=LIFECYCLE/(name+'.py')
        need(set(item)=={'path','sha256'} and item['path']==str(path) and not path.is_symlink()
             and sha(path)==item['sha256'],'reviewed lifecycle source changed: '+name)
    controller=load(LIFECYCLE/'controller.py','_prepare_serving_cold_protocol')
    predecessor=dict(path=str(args.predecessor.resolve()),sha256=args.predecessor_sha256,
        audit=dict(path=str(args.restoration_audit.resolve()),sha256=args.restoration_audit_sha256))
    # This checks archived exact bytes, approved restoration protocol/source,
    # RESTORED, independent recovery PASS, replacement and raw-log identity.
    # A prior numerical result need not pass. No present service probe occurs.
    prior,_=controller.predecessor(predecessor,lifecycle_ref)
    need(sha(INHERITED)==INHERITED_SHA,'fixed inherited owner plan')
    inherited=read(INHERITED);numeric_ref=inherited['inherited_plan']
    need(numeric_ref['sha256']==NUMERIC_SHA==sha(numeric_ref['path']),'fixed original37 numerical plan')
    numeric=read(numeric_ref['path'])
    need(sha(BASE_AUDITOR)==BASE_AUDITOR_SHA,'frozen original numerical auditor')
    need(sha(RESOURCE_AUDITOR)==RESOURCE_AUDITOR_SHA,'frozen single-child resource auditor')
    qualifier_path=HERE/'serving_qualification.py'
    qualifier=load(qualifier_path,'_prepare_serving_child_contract')
    child_sources=qualifier.source_pins()
    need(len(child_sources)==7,'exact seven child sources')
    for path,digest in child_sources.items():need(sha(path)==digest,'child source changed: '+path)
    child=dict(schema='E024-route-serving-plan-v1',qualifier_sha256=child_sources[str(qualifier_path)],
        qualification=dict(numeric_ref),sources=child_sources,deadline_s=1800)
    # Auditor/preparer are outer evidence pins only; the child's exact seven
    # sources remain untouched. No circular plan/source hashes are introduced.
    outer_sources={**child_sources,str(BASE_AUDITOR):BASE_AUDITOR_SHA,
        str(RESOURCE_AUDITOR):RESOURCE_AUDITOR_SHA,
        str(HERE/'audit_serving.py'):sha(HERE/'audit_serving.py'),str(Path(__file__).resolve()):sha(__file__)}
    child_path=out/'serving-plan.json';outer_path=out/'owner-plan.json'
    out.mkdir(exist_ok=False)
    try:
        write(child_path,child);child_ref=reference(child_path)
        qualifier.validate(child_path,child_ref['sha256'])
        outer=dict(schema='cold-single-child-run-plan-v2',lifecycle=lifecycle_ref,predecessor=predecessor,
            experiment=dict(phase=PHASE,qualifier=reference(qualifier_path),plan=child_ref,source_pins=outer_sources),
            inherited_owner_plan=dict(path=str(INHERITED),sha256=INHERITED_SHA),
            outputs=dict(journal=str(journal),out=str(owner)),
            budgets=dict(work_seconds=1830,child_seconds=1800,term_seconds=15,kill_seconds=15))
        write(outer_path,outer);outer_ref=reference(outer_path)
        checked,checked_manifest=controller.run_plan(outer_path,outer_ref['sha256'])
        need(checked==outer and checked_manifest==manifest,'actual controller run-plan validation')
        # Preserve all real source references plus both exact plan bytes. The
        # source manifest deliberately does not try to hash itself.
        pins=controller.source_pins(outer,manifest)
        pins.update({outer_ref['path']:outer_ref['sha256'],predecessor['path']:predecessor['sha256'],
                     predecessor['audit']['path']:predecessor['audit']['sha256']})
        write(out/'source-manifest.json',dict(schema='E024-route-serving-prereg-sources-v1',files=pins))
        python=numeric['execution']['python'];prefix=[python,'-I','-B']
        control=prefix+[str(LIFECYCLE/'controller.py')]
        fixed=['--journal',str(journal),'--plan-sha256',controller.PLAN_SHA]
        authorization=['--approved-contract',controller.CONTRACT]
        analysis=owner.parent/(owner.name.removesuffix('-owner')+'-analysis')
        commands=dict(
            read_only_service_prepare=control+['prepare',*fixed,'--owner-directory',str(owner),
                '--owner-plan',str(outer_path),'--owner-plan-sha256',outer_ref['sha256']],
            stop=control+['stop',*fixed,*authorization],
            owner=prefix+[str(LIFECYCLE/'owner.py'),'--plan',str(outer_path),'--plan-sha256',outer_ref['sha256'],'--out',str(owner)],
            recover=control+['recover',*fixed,*authorization],
            restoration_audit=prefix+[str(LIFECYCLE/'audit_restore.py'),'--journal',str(journal),
                '--out',str(analysis/'restoration-audit.json')],
            resource_audit=prefix+[str(RESOURCE_AUDITOR),'--owner',str(owner),
                '--out',str(analysis/'independent-resources.json')],
            numerical_audit=prefix+[str(HERE/'audit_serving.py'),'--child',str(owner/'phases'/PHASE/'qualification'),
                '--plan-sha256',child_ref['sha256'],'--out',str(analysis/'independent-numerical.json')])
        preparation=dict(schema='E024-route-serving-plan-build-v1',status='PLANS_VALIDATED_NOT_EXECUTED',
            owner_plan=outer_ref,child_plan=child_ref,lifecycle=lifecycle_ref,predecessor=predecessor,
            predecessor_replacement_receipt=prior['replacement'],source_manifest=reference(out/'source-manifest.json'),
            source_pins=pins,commands=commands,
            final_receipt_contract=dict(filename='receipt.json',created_by='root after actual read-only controller prepare',
                required_keys=['owner_plan_sha256','source_pins','prepared_journal_sha256'],
                source_pins_note='Copy these pins and add the final execute_serving.py source SHA; recheck all before sealing.',
                prepared_journal_sha256_note='Hash the actual newly PREPARED journal bytes. Never use a predecessor SHA or a placeholder.'),
            execution_order='After CPU review/freeze/commit: read-only service prepare, stop, direct owner and wait, finally recover; after terminal RESTORED create the analysis directory and independently audit recovery, raw resource samples and numerical receipts.',
            failure_policy='Owner exit0 is execution only. Child nonzero, numerical FAIL/INCOMPLETE or resource failure gives no qualification. Recover after the owner is reaped even when numerical failure occurs. --no-experiment is allowed only if the owner directory never appeared; never resolve unknown Popen by assertion.',
            analysis_directory=str(analysis),analysis_directory_created=False,
            service_probed=False,service_mutated=False,owner_prepared=False,model_loaded=False,
            scope='Plans and small source/protocol checks only. Original37 plus42 parameter diagnostic; no HTTP or performance result. No current service identity observation.')
        need(all(sha(path)==digest for path,digest in pins.items()),'source/plan/predecessor changed during preparation')
        write(out/'plan-build.json',preparation)
        return preparation
    except BaseException as exc:
        write(out/'failure.json',dict(status='PREPARATION_FAILED',error_type=type(exc).__name__,error=str(exc),
            scope='Only new plan files may exist; no owner/service/model operation was attempted.'))
        raise


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='store_true')
    p.add_argument('--predecessor',type=Path);p.add_argument('--predecessor-sha256')
    p.add_argument('--restoration-audit',type=Path);p.add_argument('--restoration-audit-sha256')
    p.add_argument('--out',type=Path,default=HERE/'prereg-serving-v1')
    p.add_argument('--journal',type=Path,default=ARTIFACTS/'route-decode-serving-v1-cold-journal')
    p.add_argument('--owner',type=Path,default=ARTIFACTS/'route-decode-serving-v1-owner')
    args=p.parse_args(argv)
    if not args.run:
        print(json.dumps(dict(status='DESCRIPTION_ONLY',writes=False,service_operations=False,
            required=['--run','--predecessor ARCHIVED_RESTORED_JOURNAL','--predecessor-sha256 SHA',
                      '--restoration-audit ARCHIVED_INDEPENDENT_RECOVERY_PASS','--restoration-audit-sha256 SHA'],
            out=str(args.out),journal=str(args.journal),owner=str(args.owner),
            scope='Generate a five-field serving plan and v2 single-child plan/manifest only. Never substitute an in-progress journal or synthesize recovery PASS.'),indent=2))
        return 0
    if not all((args.predecessor,args.predecessor_sha256,args.restoration_audit,args.restoration_audit_sha256)):
        p.error('explicit archived predecessor and independent restoration audit paths and SHAs are required')
    result=prepare(args)
    print(json.dumps(dict(status=result['status'],owner_plan=result['owner_plan'],child_plan=result['child_plan']),indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
