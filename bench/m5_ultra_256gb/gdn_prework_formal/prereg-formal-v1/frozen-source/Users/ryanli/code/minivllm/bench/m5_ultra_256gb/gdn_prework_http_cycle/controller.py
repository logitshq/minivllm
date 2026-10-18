"""GDN-only four-phase HTTP lifecycle, fixed direct-child recovery semantics.

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
OWNER_SHA='a4776f1125b9b8db82d83b0f2cebd78c57cf15894354d884d0e2cd07224a7ddd'
JOURNAL_SCHEMA='GDN-prework-http-cold-journal-v1'
OWNER_SCHEMA='GDN-prework-http-owner-plan-v1'
PHASES=('c1-capture','c1-replay','c8-capture','c8-replay')
ARCHIVE_ROOT=HERE.parents[2]/'docs/experiments/m5-ultra-256gb'
PREDECESSOR_SCHEMA='E024-route-http-formal-cold-journal-v1'
PREDECESSOR_FOLDER='route_http_formal'
PREDECESSOR_CONTROLLER_SHA='98d48eb4ad608be5d8d65f6a28622126483ee96ff9a312b3cbd16ee4c63817f3'
PREDECESSOR_AUDITOR_SHA='90a980d0dbe6a61fc1bab2ac6b2edaa9bf012bb60726cd0742801d11410eb0b1'
PREDECESSOR_JOURNAL_SHA='6ca290097aa14d1969a1282c43116fa8cd3ffd736edc9797299bca04064449eb'
PREDECESSOR_AUDIT_SHA='acb1207ebb8244299b3440d1b18a4d98f8f546d1821fbae90aa1528026af44f9'
PREDECESSOR_ARCHIVE=ARCHIVE_ROOT/'E024-route-http-formal-v2-reparse-v1'

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


# Functions with no protocol literal changes are bound to the actual adapter
# namespace. __file__/HERE are our true paths, and donor.__dict__ is untouched.
# No function closes over a model, runtime, server, or mutable donor state.
_REUSED=('no_popen_proof','predecessor','registered_trials','prepare','authorize',
         'stop','phase_reaps','reaped','load_worker','monitored_load','recover','main')
for _name in _REUSED:
    _function=getattr(_donor,_name)
    globals()[_name]=FunctionType(_function.__code__,globals(),_name,_function.__defaults__,_function.__closure__)
    globals()[_name].__kwdefaults__=_function.__kwdefaults__
_bound_source_pins=FunctionType(_donor.source_pins.__code__,globals(),'source_pins',_donor.source_pins.__defaults__)

if __name__=='__main__':raise SystemExit(main())
