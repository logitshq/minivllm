"""Existing cold lifecycle, three actual modes, UNSEALED until normal GLM v3 closure."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from bench.m5_ultra_256gb.c1_profile_cycle_v1 import _reuse as reuse
_original=reuse.install(globals(),'gdn_graph_combined_formal/controller.py')
OWNER_SHA=reuse.sha(HERE/'owner.py')
OWNER_SCHEMA='C1-profile-owner-plan-v1'
JOURNAL_SCHEMA='C1-profile-cold-journal-v1'
PHASES=('off','runner','full')
PREDECESSOR_FOLDER='glm_flash_cycle_v3'
PREDECESSOR_CONTROLLER_SHA=reuse.FIXED[PREDECESSOR_FOLDER+'/controller.py']
PREDECESSOR_AUDITOR_SHA=reuse.FIXED[PREDECESSOR_FOLDER+'/audit_restore.py']
BOUND_MANIFEST_SHA='ad1254e01dbd610d29f8abdebdd80db9c416f2c34ec5d8df61990bc54df84075'


def predecessor_manifest():
    need(type(BOUND_MANIFEST_SHA) is str and len(BOUND_MANIFEST_SHA)==64 and all(x in '0123456789abcdef' for x in BOUND_MANIFEST_SHA),'actual predecessor remains UNSEALED')
    path=HERE/'predecessor-manifest.json';need(not path.is_symlink() and b.sha(path)==BOUND_MANIFEST_SHA,'sealed predecessor changed')
    value=b.read_json(path);need(set(value)=={'schema','status','binding'} and value['schema']=='C1-profile-GLM-v3-predecessor-v1' and value['status']=='SEALED','actual predecessor manifest')
    return value

def predecessor(binding):
    need(binding==predecessor_manifest()['binding'] and set(binding)=={'path','sha256','audit'} and set(binding['audit'])=={'path','sha256'},'exact archived GLM pair')
    raw=[]
    for ref,name in ((binding,'restored-journal.json'),(binding['audit'],'restoration-audit.json')):
        path=Path(ref['path']);need(path.is_absolute() and path==path.resolve() and path.is_relative_to(ARCHIVE_ROOT.resolve()) and path.name==name and not path.is_symlink(),'real immutable GLM archive path')
        data=path.read_bytes();need(hashlib.sha256(data).hexdigest()==ref['sha256'],'actual archived bytes');raw.append(data)
    prior,audit=map(json.loads,raw);pc=reuse.fixed('glm_flash_cycle_v3/controller.py')
    need(prior.get('schema')=='GLM-cold-load-journal-v3' and prior.get('state')=='RESTORED' and prior.get('trials')==['glm-official-load']
         and prior.get('controller_sha256')==PREDECESSOR_CONTROLLER_SHA and prior.get('support_sha256')==BASE_SHA
         and prior.get('approved_contract')==CONTRACT and prior.get('plan_sha256')==PLAN_SHA,'normal GLM v3 RESTORED predecessor')
    need(prior.get('owner_source')==dict(path=str(HERE.parent/'glm_flash_cycle_v3/owner.py'),sha256='9d69c5a64dc50b7d65330ad130b8af9f757b91e884aa197d1e828775b27c88a8'),'actual GLM owner')
    need(audit.get('schema')=='cold-service-restoration-audit-v1' and audit.get('status')=='PASS' and audit.get('qualified') is True
         and audit.get('journal_sha256')==binding['sha256'] and audit.get('controller_sha256')==PREDECESSOR_CONTROLLER_SHA
         and audit.get('audit_source_sha256')==PREDECESSOR_AUDITOR_SHA and audit.get('owner_plan_sha256')==prior['owner_plan']['sha256'],'independent prior restoration')
    need(not prior.get('post_monitor_failed') and not any(e.get('event') in ('post_monitor_incomplete','post_worker_cleanup_failed') for e in prior.get('events',[])),'prior recovery monitor failure')
    owner=Path(prior['owner_directory']);attempt=b.read_json(owner/'attempt.json');paths,pids=pc.phase_reaps(SimpleNamespace(data=prior),owner,attempt)
    need(prior['reap'].get('no_experiment') is False and prior['reap'].get('no_child_started') is False and prior['reap'].get('owner_pid')==attempt['owner_pid'],'actual observed GLM child reap')
    expected=[owner/'attempt.json',owner/'stopped-journal.json',*paths]
    need(prior['reap']['receipts']==[dict(path=str(p),sha256=b.sha(p)) for p in expected],'actual model/monitor reap receipts')
    need(audit['owner_reap']['status']=='ALL_OBSERVED_CHILDREN_REAPED' and audit['owner_reap']['children']==[dict(phase='glm-official-load',pid=pids[0])],'independent GLM child proof')
    restored=audit['service_restore']
    need(restored['status']=='PERSISTED_VISIBLE_RECEIPTS_CONSISTENT' and restored['replacement']==prior['replacement']
         and restored['raw_log_sha256']==prior['log_window']['sha256'] and restored['raw_log_bytes']==prior['log_window']['bytes']
         and restored['http_worker_returncode']==0,'actual successful restore/log/load-worker')
    fixed_identity(prior['replacement'])
    need(prior['final_status']==dict(version='0.7.0',default_model=b.TARGET,loaded=[b.TARGET],loading=0,target_loading=False,path=b.MODEL,pinned=False,engine_type='vlm',config_model_type='qwen4_exp')
         and prior['exact_runtime_signature_restored'] is False and prior['warm_kv_restored'] is False,'persisted-visible restoration scope')
    return prior,raw


def trial_names(value):
    need(type(value) is list and value==list(PHASES),'exact three diagnostic modes');return value
# These original algorithms read the actual registered list, including partial-prefix failure cleanup.
stop=_bound('stop',dict(authorize=authorize))
phase_reaps=_bound('phase_reaps',dict(registered_trials=registered_trials))
reaped=_bound('reaped',dict(registered_trials=registered_trials,phase_reaps=phase_reaps))
recover=_bound('recover',dict(authorize=authorize,reaped=reaped,monitored_load=monitored_load))
_original_main=main

def main(argv=None):
    if argv==[] or (argv is None and len(sys.argv)==1):print('DESCRIPTION_ONLY: sealed actual GLM v3 restoration required.');return 0
    return _original_main(argv)
if __name__=='__main__':raise SystemExit(main())
