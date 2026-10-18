"""Retry recovery namespace: only the actual failed-v1 no-experiment predecessor.

Twelve-server stop/reap/recovery bodies remain frozen. Future archive refs are
explicit immutable plan inputs and absent refs refuse. No invented PREPARED,
RESTORED, PID, experiment success or factory copy is accepted.
"""
import hashlib
import importlib.util
from pathlib import Path
from types import FunctionType

ORIGINAL = Path(__file__).resolve().parent.parent / 'gdn_graph_combined_formal'
ORIGINAL_CONTROLLER_SHA = '3b8b3033269fcd326e0a52e20470ee27faf65d313be3e3bf0da8881f26f3ac4c'
_path = ORIGINAL / 'controller.py'
if hashlib.sha256(_path.read_bytes()).hexdigest() != ORIGINAL_CONTROLLER_SHA:
    raise ValueError('frozen combined formal controller changed')
_spec = importlib.util.spec_from_file_location('_combined_formal_v2_original_controller', _path)
_original = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_original)
for _name, _value in vars(_original).items():
    if not _name.startswith('__'): globals()[_name] = _value
HERE = Path(__file__).resolve().parent
OWNER_SHA = 'f7f44001d11533aa6106d3e5dc5c2635dfb736433f10e7ea680a5a83f2863d9e'
OWNER_SCHEMA = 'GDN-graph-combined-http-formal-owner-plan-v2'
JOURNAL_SCHEMA = 'GDN-graph-combined-http-formal-cold-journal-v2'
PREDECESSOR_ARCHIVE = ARCHIVE_ROOT / 'E032-gdn-graph-combined-formal-v1'
PREDECESSOR_FOLDER = 'gdn_graph_combined_formal'
PREDECESSOR_SCHEMA = 'GDN-graph-combined-http-formal-cold-journal-v1'
PREDECESSOR_CONTROLLER_SHA = ORIGINAL_CONTROLLER_SHA
PREDECESSOR_AUDITOR_SHA = '157d6470a9cb87223c419a3f4edd82e8556b3714a7157ea1ad505270c8ed7493'
for _name, _value in vars(_original).items():
    if isinstance(_value, FunctionType) and _value.__globals__ is vars(_original):
        _fn = FunctionType(_value.__code__, globals(), _name, _value.__defaults__, _value.__closure__)
        _fn.__kwdefaults__ = _value.__kwdefaults__
        globals()[_name] = _fn


def predecessor(binding):
    need(type(binding) is dict and set(binding) == {'path','sha256','audit'}, 'predecessor binding fields')
    need(type(binding['audit']) is dict and set(binding['audit']) == {'path','sha256'}, 'audit binding fields')
    raw = []
    for item, name in zip((binding,binding['audit']), ('restored-journal.json','restoration-audit.json')):
        path = Path(item['path'])
        need(path.is_absolute() and path == path.resolve() and path == PREDECESSOR_ARCHIVE / name
             and path.is_file() and not path.is_symlink(), 'exact immutable E032 predecessor sidecar required')
        digest = item['sha256']
        need(type(digest) is str and len(digest) == 64 and all(x in '0123456789abcdef' for x in digest), 'predecessor SHA syntax')
        data = path.read_bytes(); need(hashlib.sha256(data).hexdigest() == digest, 'predecessor/audit SHA changed'); raw.append(data)
    prior, audit = map(json.loads, raw)
    need(prior.get('schema') == PREDECESSOR_SCHEMA and prior.get('trials') == list(PHASES)
         and b.sha(ORIGINAL / 'controller.py') == PREDECESSOR_CONTROLLER_SHA
         and b.sha(ORIGINAL / 'audit_restore.py') == PREDECESSOR_AUDITOR_SHA
         and prior.get('controller_sha256') == PREDECESSOR_CONTROLLER_SHA
         and audit.get('audit_source_sha256') == PREDECESSOR_AUDITOR_SHA,
         'exact failed-v1 twelve-trial controller/auditor identity')
    need(audit.get('schema') == 'cold-service-restoration-audit-v1' and audit.get('status') == 'PASS'
         and audit.get('qualified') is True and audit.get('journal_sha256') == binding['sha256']
         and audit.get('controller_sha256') == PREDECESSOR_CONTROLLER_SHA,
         'exact independent restoration receipt')
    need(prior.get('state') == 'RESTORED' and prior.get('approved_contract') == CONTRACT
         and prior.get('plan_sha256') == PLAN_SHA and prior.get('support_sha256') == BASE_SHA
         and not prior.get('post_monitor_failed')
         and not any(e.get('event') in ('post_monitor_incomplete','post_worker_cleanup_failed') for e in prior.get('events',[])),
         'predecessor recovery incomplete')
    need(prior.get('reap') == {'no_experiment': True}
         and audit.get('owner_reap') == {'status':'NO_EXPERIMENT','children':[]}
         and prior.get('owner_source') == dict(path=str(ORIGINAL / 'owner.py'),sha256='511b48fbe722ea031a7b4586439eaa67f25c9c9514af12b614de74fb8c53c103')
         and Path(prior['owner_directory']).is_absolute() and not Path(prior['owner_directory']).exists(),
         'only actual v1 no-experiment/absent-owner recovery is this retry predecessor')
    restored = audit.get('service_restore', {})
    need(restored.get('status') == 'PERSISTED_VISIBLE_RECEIPTS_CONSISTENT'
         and restored.get('replacement') == prior.get('replacement')
         and restored.get('raw_log_sha256') == prior.get('log_window',{}).get('sha256')
         and restored.get('raw_log_bytes') == prior.get('log_window',{}).get('bytes')
         and restored.get('http_worker_returncode') == 0
         and audit.get('owner_plan_sha256') == prior.get('owner_plan',{}).get('sha256'), 'restoration receipt mismatch')
    fixed_identity(prior.get('replacement'))
    need(prior.get('final_status') == dict(version='0.7.0',default_model=b.TARGET,loaded=[b.TARGET],
         loading=0,target_loading=False,path=b.MODEL,pinned=False,engine_type='vlm',config_model_type='qwen4_exp'),
         'predecessor persisted-visible variant changed')
    need(prior.get('exact_runtime_signature_restored') is False and prior.get('warm_kv_restored') is False,
         'unsupported restoration claim')
    return prior, raw


# Rebuild only the four explicit dependency-bound lifecycle functions. Their
# code objects are original and their imported donor dictionary is untouched.
stop = _bound('stop', dict(authorize=authorize))
phase_reaps = _bound('phase_reaps', dict(registered_trials=registered_trials))
reaped = _bound('reaped', dict(registered_trials=registered_trials, phase_reaps=phase_reaps))
recover = _bound('recover', dict(authorize=authorize, reaped=reaped, monitored_load=monitored_load))


if __name__ == '__main__': raise SystemExit(main())
