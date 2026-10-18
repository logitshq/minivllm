"""GLM single-child recovery after an actual closed GDN formal cycle.

Only the predecessor schema/source allowlist and true namespace change. Twelve
GDN trial identities remain a prerequisite history, never GLM execution phases.
No hypothetical receipt, old-source impersonation or service action on import.
"""
import importlib.util as _importlib
from pathlib import Path as _Path
import hashlib as _hashlib

_adapter_path=_Path(__file__).resolve().parent/'_reuse.py'
if _hashlib.sha256(_adapter_path.read_bytes()).hexdigest()!='fbf523c586629ce6e5e2030d629b00c9f1cd99dfec9edef6c35ac8e054468f97':
    raise ValueError('fixed GLM v2 reuse source changed')
_adapter_spec=_importlib.spec_from_file_location('_glm_v2_code_reuse',_adapter_path)
_adapter=_importlib.module_from_spec(_adapter_spec);_adapter_spec.loader.exec_module(_adapter)

_original=_adapter.install(globals(),'controller.py',dict(
    OWNER_SHA='f8e32b6532ed0d22b21b27ef83231ae7a1d7423a4269ea32b5e4aa20ff2a0657',
    PRIOR_PATH=_Path(__file__).resolve().parent.parent/'gdn_prework_formal/controller.py',
    PRIOR_SHA='38a6e904d928813efa8ba1f6de724693dccce196322401c4f44b0442ed4dcd35',
    PRIOR_AUDIT_PATH=_Path(__file__).resolve().parent.parent/'gdn_prework_formal/audit_restore.py',
    PRIOR_AUDIT_SHA='8ece5748f8328a110651981f40d3c7ba158194a9479052c544083d45a5621fc0'))
_original_source_pins=source_pins

def source_pins(plan):
    pins=_original_source_pins(plan)
    pins.update(_adapter.support_pins())
    return pins

def predecessor(binding):
    """Only actual archived GDN-formal RESTORED bytes; no invented future PID."""
    need(type(binding) is dict and set(binding)=={'path','sha256','audit'},'exact predecessor binding')
    aref=binding['audit'];need(type(aref) is dict and set(aref)=={'path','sha256'},'exact predecessor audit binding')
    raw=[];paths=[]
    for item,name in ((binding,'restored-journal.json'),(aref,'restoration-audit.json')):
        p=Path(item['path'])
        need(p.is_absolute() and str(p.resolve())==str(p) and not p.is_symlink()
             and p.is_relative_to(ARCHIVE_ROOT.resolve()) and p.name==name,'immutable archived predecessor path')
        need(type(item['sha256']) is str and re.fullmatch('[0-9a-f]{64}',item['sha256']),'explicit predecessor SHA')
        data=p.read_bytes();need(hashlib.sha256(data).hexdigest()==item['sha256'],'predecessor bytes changed')
        paths.append(p);raw.append(data)
    need(paths[0].parent==paths[1].parent,'same closed predecessor archive')
    prior,audit=map(json.loads,raw)
    need(b.sha(PRIOR_PATH)==PRIOR_SHA and b.sha(PRIOR_AUDIT_PATH)==PRIOR_AUDIT_SHA,'reviewed predecessor sources changed')
    need(prior.get('schema')=='GDN-prework-http-formal-cold-journal-v1'
         and prior.get('controller_sha256')==PRIOR_SHA and prior.get('trials')==list(PRIOR_PHASES),
         'actual twelve-phase GDN-formal predecessor required')
    need(audit.get('schema')=='cold-service-restoration-audit-v1' and audit.get('status')=='PASS'
         and audit.get('qualified') is True and audit.get('audit_source_sha256')==PRIOR_AUDIT_SHA
         and audit.get('journal_sha256')==binding['sha256'] and audit.get('controller_sha256')==PRIOR_SHA,
         'independent restoration audit binding')
    need(prior.get('state')=='RESTORED' and prior.get('approved_contract')==CONTRACT
         and prior.get('plan_sha256')==PLAN_SHA and prior.get('support_sha256')==BASE_SHA
         and not prior.get('post_monitor_failed') and not any(e.get('event') in
         ('post_monitor_incomplete','post_worker_cleanup_failed') for e in prior.get('events',[])),
         'predecessor recovery incomplete')
    restored=audit.get('service_restore',{})
    need(restored.get('status')=='PERSISTED_VISIBLE_RECEIPTS_CONSISTENT'
         and restored.get('replacement')==prior.get('replacement')
         and restored.get('raw_log_sha256')==prior.get('log_window',{}).get('sha256')
         and restored.get('raw_log_bytes')==prior.get('log_window',{}).get('bytes')
         and restored.get('http_worker_returncode')==0
         and audit.get('owner_plan_sha256')==prior.get('owner_plan',{}).get('sha256'),'restoration receipt mismatch')
    fixed_identity(prior.get('replacement'))
    need(prior.get('final_status')==dict(version='0.7.0',default_model=b.TARGET,loaded=[b.TARGET],
         loading=0,target_loading=False,path=b.MODEL,pinned=False,engine_type='vlm',config_model_type='qwen4_exp'),
         'persisted service variant changed')
    need(prior.get('exact_runtime_signature_restored') is False and prior.get('warm_kv_restored') is False,
         'predecessor scope changed')
    return prior,raw

# Identical original code objects, explicitly injected actual local functions.
stop=_bound('stop',dict(authorize=authorize))
reaped=_bound('reaped',dict(registered_trials=registered_trials,phase_reaps=phase_reaps,no_popen_proof=no_popen_proof))
recover=_bound('recover',dict(authorize=authorize,reaped=reaped,monitored_load=monitored_load))

if __name__=='__main__':raise SystemExit(main())
