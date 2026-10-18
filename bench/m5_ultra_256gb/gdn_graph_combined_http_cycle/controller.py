"""Combined GDN+route four-phase HTTP lifecycle, fixed direct-child recovery semantics.

Frozen route-cycle functions use this module's explicit dependency namespace;
no donor globals, paths, code, or evidence are mutated. The actual worker CLI
and source identities are this module. Source-only by default; stop/recovery
require the existing authorized contract and a bound immutable journal.
"""
import argparse
import ast
import contextlib
import hashlib
import http.client
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid
from types import FunctionType, SimpleNamespace

HERE=Path(__file__).resolve().parent
DONOR_PATH=HERE.parent/'route_http_cycle/controller.py'
DONOR_SHA='e6bc72007043ed0376e98ea7f28ee152587294dec200953e93a5c7c736a90694'
OWNER_SHA='58c35e3a8f8e758bca0792e7c1d2819a96bc7c0aac0cd5dd9169fab4c7e6de9d'
JOURNAL_SCHEMA='GDN-graph-combined-http-cold-journal-v1'
OWNER_SCHEMA='GDN-graph-combined-http-owner-plan-v1'
PHASES=('c1-capture','c1-replay','c8-capture','c8-replay')
ARCHIVE_ROOT=HERE.parents[2]/'docs/experiments/m5-ultra-256gb'
PREDECESSOR_SCHEMA='cold-single-child-journal-v3'
PREDECESSOR_FOLDER='cold_observed_cycle'
PREDECESSOR_CONTROLLER_SHA='5ef9cf1c576eac223a6e82831b443296758f6a45a9fdd8a3486c47ac5e1e8196'
PREDECESSOR_AUDITOR_SHA='f7e5e7e751b52520a47facfdb65c33db40fb4c103e8ee272b5d154607087ce56'
PREDECESSOR_ARCHIVE=ARCHIVE_ROOT/'E030-gdn-graph-combined-serving-v1'

def load(path,digest,name):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('frozen lifecycle support changed: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

_donor=load(DONOR_PATH,DONOR_SHA,'gdn_http_frozen_route_lifecycle')
# Explicit aliases retain their original module identity and argument contracts.
b=_donor.b
stable,formal=_donor.stable,_donor.formal
STABLE_PATH,STABLE_SHA=_donor.STABLE_PATH,_donor.STABLE_SHA
FORMAL_PATH,FORMAL_SHA=_donor.FORMAL_PATH,_donor.FORMAL_SHA
STABLE_LIFECYCLE=_donor.STABLE_LIFECYCLE
PLAN_SHA,CONTRACT,GIB,BASE_PATH,BASE_SHA=_donor.PLAN_SHA,_donor.CONTRACT,_donor.GIB,_donor.BASE_PATH,_donor.BASE_SHA
Refuse,NotReady,need=_donor.Refuse,_donor.NotReady,_donor.need
Journal,System=_donor.Journal,_donor.System
fixed_identity,original_identity=_donor.fixed_identity,_donor.original_identity
absent_service,replacement=_donor.absent_service,_donor.replacement
atomic_bytes,exclusive_json=_donor.atomic_bytes,_donor.exclusive_json
parse_resources,sample,check_resource,resource_gate=_donor.parse_resources,_donor.sample,_donor.check_resource,_donor.resource_gate


def owner_plan(path,digest,journal):
    owner=load(HERE/'owner.py',OWNER_SHA,'gdn_http_actual_owner_contract')
    _,plan,_,_,_=owner.validate_plan(SimpleNamespace(plan=Path(path),plan_sha256=digest))
    need(plan['schema']==OWNER_SCHEMA and Path(plan['cold_restart']['journal_path']).resolve()==Path(journal).resolve(),
         'actual GDN owner plan/journal required')
    need(plan['cold_restart']['controller']==dict(path=str(Path(__file__).resolve()),sha256=b.sha(__file__)),
         'actual GDN controller source required')
    source_pins(plan)
    return plan


def source_pins(plan):
    pins=_bound_source_pins(plan)
    pins[str(DONOR_PATH)]=DONOR_SHA
    need(b.sha(DONOR_PATH)==DONOR_SHA,'frozen route lifecycle donor changed')
    return pins


def predecessor(binding):
    """Only the exact independently restored formal-v2 archive; pending is refused.

    The owner plan freezes both bytes. A failed experiment is permissible; a
    missing/incomplete restoration, unknown source or live journal is not.
    """
    owned=load(HERE/'owner.py',OWNER_SHA,'combined_http_closed_baseline')
    closed=owned.baseline()
    need(binding==closed['predecessor'],'predecessor differs from pinned CLOSED manifest')
    need(type(binding) is dict and set(binding)=={'path','sha256','audit'},'predecessor binding fields')
    need(type(binding['audit']) is dict and set(binding['audit'])=={'path','sha256'},'audit binding fields')
    raw=[]
    for item in (binding,binding['audit']):
        path=Path(item['path'])
        need(path.is_absolute() and path==path.resolve() and path.is_relative_to(ARCHIVE_ROOT.resolve())
             and path.is_file() and not path.is_symlink(),'regular immutable archived predecessor')
        need(type(item['sha256']) is str and len(item['sha256'])==64
             and all(x in '0123456789abcdef' for x in item['sha256']),'predecessor SHA syntax')
        data=path.read_bytes();need(hashlib.sha256(data).hexdigest()==item['sha256'],'predecessor/audit SHA changed');raw.append(data)
    prior,audit=map(json.loads,raw)
    need(Path(binding['path'])==PREDECESSOR_ARCHIVE/'restored-journal.json'
         and Path(binding['audit']['path'])==PREDECESSOR_ARCHIVE/'restoration-audit.json'
         and binding==closed['predecessor'],
         'exact archived combined serving restoration predecessor required')
    controller_sha,auditor_sha=PREDECESSOR_CONTROLLER_SHA,PREDECESSOR_AUDITOR_SHA
    need(prior.get('schema')==PREDECESSOR_SCHEMA
         and b.sha(HERE.parent/PREDECESSOR_FOLDER/'controller.py')==controller_sha
         and b.sha(HERE.parent/PREDECESSOR_FOLDER/'audit_restore.py')==auditor_sha
         and prior.get('controller_sha256')==controller_sha and audit.get('audit_source_sha256')==auditor_sha,
         'exact reviewed predecessor controller/auditor required')
    lifecycle=prior.get('lifecycle',{})
    need(lifecycle==dict(path=str(HERE.parent/'cold_observed_cycle/lifecycle.json'),sha256='01a3befda92835ecc49ab58e3acd99c50f0077aeeed5fda96b2b95fdb99fdd88')
         and b.sha(lifecycle['path'])==lifecycle['sha256'],'actual observed-bridge lifecycle binding')
    need(audit.get('schema')=='cold-service-restoration-audit-v1' and audit.get('status')=='PASS'
         and audit.get('qualified') is True and audit.get('journal_sha256')==binding['sha256']
         and audit.get('controller_sha256')==controller_sha,'exact independent restoration receipt')
    need(prior.get('state')=='RESTORED' and prior.get('approved_contract')==CONTRACT
         and prior.get('plan_sha256')==PLAN_SHA and prior.get('support_sha256')==BASE_SHA
         and not prior.get('post_monitor_failed')
         and not any(e.get('event') in ('post_monitor_incomplete','post_worker_cleanup_failed')
                     for e in prior.get('events',[])),'predecessor recovery incomplete')
    restored=audit.get('service_restore',{})
    need(restored.get('status')=='PERSISTED_VISIBLE_RECEIPTS_CONSISTENT'
         and restored.get('replacement')==prior.get('replacement')
         and restored.get('raw_log_sha256')==prior.get('log_window',{}).get('sha256')
         and restored.get('raw_log_bytes')==prior.get('log_window',{}).get('bytes')
         and restored.get('http_worker_returncode')==0
         and audit.get('owner_plan_sha256')==prior.get('owner_plan',{}).get('sha256'),
         'verified restoration receipt mismatch')
    fixed_identity(prior.get('replacement'))
    need(prior.get('final_status')==dict(version='0.7.0',default_model=b.TARGET,loaded=[b.TARGET],
         loading=0,target_loading=False,path=b.MODEL,pinned=False,engine_type='vlm',config_model_type='qwen4_exp'),
         'predecessor persisted-visible variant changed')
    need(prior.get('exact_runtime_signature_restored') is False and prior.get('warm_kv_restored') is False,
         'unsupported restoration claim')
    return prior,raw


# Functions with no protocol literal changes are bound to the actual adapter
# namespace. __file__/HERE are our true paths, and donor.__dict__ is untouched.
# No function closes over a model, runtime, server, or mutable donor state.
_REUSED=('no_popen_proof','registered_trials','prepare','authorize',
         'stop','phase_reaps','reaped','load_worker','monitored_load','recover','main')
for _name in _REUSED:
    _function=getattr(_donor,_name)
    globals()[_name]=FunctionType(_function.__code__,globals(),_name,_function.__defaults__,_function.__closure__)
    globals()[_name].__kwdefaults__=_function.__kwdefaults__
_bound_source_pins=FunctionType(_donor.source_pins.__code__,globals(),'source_pins',_donor.source_pins.__defaults__)

if __name__=='__main__':raise SystemExit(main())
