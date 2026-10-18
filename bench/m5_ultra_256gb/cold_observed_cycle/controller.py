"""Observed recovery bridge; frozen v3 lifecycle algorithms unchanged."""
import importlib.util as _iu
from pathlib import Path as _P
import hashlib as _hh
_rp=_P(__file__).resolve().parent/'_reuse.py'
if _hh.sha256(_rp.read_bytes()).hexdigest()!='b22ee6d9c505fefb3f111760096eb9d6542fc2370336ea30941ccae9baa82ecb':raise ValueError('reuse source changed')
_sp=_iu.spec_from_file_location('_observed_cold_reuse',_rp)
_reuse=_iu.module_from_spec(_sp);_sp.loader.exec_module(_reuse)
_original=_reuse.install(globals(),'controller.py')

_original_predecessor=predecessor
_original_source_pins=source_pins

def source_pins(plan,manifest):
    result=_original_source_pins(plan,manifest)
    result.update(_reuse.pins())
    return result

OBSERVED_ARCHIVE=ARCHIVE_ROOT/'GLM-flash-load-v1'
OBSERVED_BASELINE_SHA='80654a8a05c863b2c57eb07fc6adf2c629fc725ad0e033749d04e6cdf93b3c32'
OBSERVED_AUDIT_SHA='3af7fa650c0b1412ebd1c48a5ae0d9398d6942e1872fd633ff044ec7d6cfb8d6'
OBSERVED_TOOL_SHA='743d4d002244c081855cc126570be5b612dfc55f8960b0f696aeca3f99a6f2cb'

def predecessor(binding,lifecycle_ref):
    """Accept this observed service baseline, without changing historical failure.

    This branch verifies current-service observation provenance. It does not
    reinterpret LOAD_INTENT, the terminated HTTP client, or recovery monitor as
    PASS. prepare/stop still refresh actual service identity/config/idle state.
    Future successful cycles use the unchanged normal RESTORED predecessor gate.
    """
    if binding.get('sha256')!=OBSERVED_BASELINE_SHA:
        return _original_predecessor(binding,lifecycle_ref)
    need(set(binding)=={'path','sha256','audit'},'observed predecessor fields')
    ar=binding['audit']
    need(ar==dict(path=str(OBSERVED_ARCHIVE/'operational-recovery.json'),sha256=OBSERVED_AUDIT_SHA)
         and binding['path']==str(OBSERVED_ARCHIVE/'service-baseline.json'),'exact observed archive pair')
    lifecycle(lifecycle_ref)
    raw=[reference(dict(path=binding['path'],sha256=binding['sha256'])).read_bytes(),reference(ar).read_bytes()]
    prior,audit=map(json.loads,raw)
    need(prior.get('schema')=='GLM-observed-service-baseline-v1'
         and prior.get('status')=='OPERATIONALLY_RESTORED_WITH_MONITOR_FAILURE'
         and 'state' not in prior and prior.get('original_journal_state')=='LOAD_INTENT'
         and prior.get('post_monitor_failed') is True
         and prior.get('original_http_worker_returncode')==-15
         and prior.get('original_post_swap_growth_bytes')==849347,'historical failure must remain explicit')
    need(audit.get('schema')=='GLM-independent-operational-restoration-v1'
         and audit.get('status')==prior['status']
         and audit.get('independent_current_observation_passed') is True
         and audit.get('original_evidence_unchanged') is True
         and audit.get('tool_sha256')==OBSERVED_TOOL_SHA,'independent observed recovery proof')
    false_fields=('original_monitored_recovery_passed','recovery_command_succeeded',
                  'exact_runtime_signature_restored','warm_kv_restored',
                  'observed_diagnostic_complete','full_model_correctness_qualified',
                  'performance_qualified','physical_resource_safety_qualified')
    need(all(prior.get(k) is False and audit.get(k) is False for k in false_fields),
         'observation cannot become historical or experiment qualification')
    for name in ('original_journal','operational_verification','verification_source',
                 'original_driver_final','original_worker_release'):
        item=prior[name];path=reference(dict(path=item['path'],sha256=item['sha256']))
        need(path.stat().st_size==item['bytes'],'retained original evidence size changed')
    need(prior['operational_verification']['sha256']==OBSERVED_AUDIT_SHA
         and prior['verification_source']['sha256']==OBSERVED_TOOL_SHA,'observed tool/result binding')
    journal=b.read_json(prior['original_journal']['path'])
    need(journal['state']=='LOAD_INTENT' and journal['post_monitor_failed'] is True
         and journal['replacement']==prior['replacement']==audit['replacement']
         and journal['global_settings_stat']==prior['global_settings_stat']
         and journal['model_metadata']==prior['model_metadata'],'original failure/config binding')
    fixed_identity(prior['replacement'])
    need(prior['final_status']==audit['observed_status_before']==audit['observed_status_after']
         ==dict(version='0.7.0',default_model=b.TARGET,loaded=[b.TARGET],loading=0,
                target_loading=False,path=b.MODEL,pinned=False,engine_type='vlm',config_model_type='qwen4_exp'),
         'observed persisted-visible target variant')
    need(len(audit['current_resource_samples'])==2 and len(audit['log_markers'])==5,
         'operational physical/log observations absent')
    return prior,raw

if __name__=='__main__':raise SystemExit(main())
