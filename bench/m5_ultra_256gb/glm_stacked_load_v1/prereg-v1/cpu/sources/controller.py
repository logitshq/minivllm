"""Current E035 predecessor; frozen GLM stop/recover and dual monitor closure."""
import importlib.util as _importlib
from pathlib import Path as _Path
_path=_Path(__file__).resolve().with_name('_reuse.py')
_spec=_importlib.spec_from_file_location('_glm_stacked_reuse',_path)
_adapter=_importlib.module_from_spec(_spec);_spec.loader.exec_module(_adapter)

_v3=_adapter.install(globals(),'glm_flash_cycle_v3/controller.py')
OWNER_SHA='bfcba6459dc167996833e2582e6caf04753765efd1fc0cf2f5e47f83a9059386'
PRIOR_PATH=HERE.parent/'c1_profile_cycle_v1/controller.py'
PRIOR_SHA=_adapter.FIXED['c1_profile_cycle_v1/controller.py']
PRIOR_AUDIT_PATH=PRIOR_PATH.with_name('audit_restore.py')
PRIOR_AUDIT_SHA=_adapter.FIXED['c1_profile_cycle_v1/audit_restore.py']
PRIOR_OWNER_PATH=PRIOR_PATH.with_name('owner.py')
PRIOR_OWNER_SHA=_adapter.FIXED['c1_profile_cycle_v1/owner.py']
PRIOR_ARCHIVE=ARCHIVE_ROOT/'E035-c1-pipeline-profile'
PRIOR_PHASES=('off','runner','full')
BOUND_MANIFEST_SHA='b0413b459e8a99ff2f49522ea34d5a2f9a3ea29b4c843d9d5b83e3d47086d5ff'
def predecessor_manifest():
    path=HERE/'predecessor-manifest.json'
    need(not path.is_symlink() and b.sha(path)==BOUND_MANIFEST_SHA,'actual E035 binding changed')
    v=b.read_json(path);need(set(v)=={'schema','status','binding'} and v['schema']=='GLM-stacked-E035-predecessor-v1' and v['status']=='SEALED','actual E035 sealed pair')
    return v
_original_predecessor=_adapter.fixed('glm_flash_cycle/controller.py').predecessor
_adapter.variant(globals(),_adapter.inspect.getsource(_original_predecessor),[("'E024-route-http-formal-cold-journal-v1'","'C1-profile-cold-journal-v1'")])
_e035_base=predecessor
def predecessor(binding):
    need(binding==predecessor_manifest()['binding'] and Path(binding['path'])==PRIOR_ARCHIVE/'restored-journal.json' and
         Path(binding['audit']['path'])==PRIOR_ARCHIVE/'restoration-audit.json','exact actual E035 archive pair')
    prior,raw=_e035_base(binding);audit=json.loads(raw[1])
    need(prior['owner_source']==dict(path=str(PRIOR_OWNER_PATH),sha256=PRIOR_OWNER_SHA),'actual predecessor owner source')
    pc=_adapter.fixed('c1_profile_cycle_v1/controller.py');owner=Path(prior['owner_directory']);attempt=b.read_json(owner/'attempt.json')
    paths,pids=pc.phase_reaps(SimpleNamespace(data=prior),owner,attempt)
    need(prior['reap']==dict(no_experiment=False,no_child_started=False,owner_pid=attempt['owner_pid'],
         receipts=[dict(path=str(p),sha256=b.sha(p)) for p in (owner/'attempt.json',owner/'stopped-journal.json',*paths)]),'actual complete predecessor reap chain')
    need(audit['owner_reap']['status']=='ALL_OBSERVED_CHILDREN_REAPED' and
         audit['owner_reap']['children']==[dict(phase=n,pid=pid) for n,pid in zip(PRIOR_PHASES,pids)] and len(pids)==3,'actual three phase reap audit')
    return prior,raw
def source_pins(plan):
    pins=_original_source_pins(plan);pins.update(_adapter.pins());pins.update(_reuse.support_pins())
    pins[str(HERE/'predecessor-manifest.json')]=BOUND_MANIFEST_SHA;predecessor_manifest()
    return pins
stop=_bound('stop',dict(authorize=authorize))
reaped=_bound('reaped',dict(registered_trials=registered_trials,phase_reaps=phase_reaps,no_popen_proof=no_popen_proof))
recover=_bound('recover',dict(authorize=authorize,reaped=reaped,monitored_load=monitored_load))
if __name__=='__main__':raise SystemExit(main())
