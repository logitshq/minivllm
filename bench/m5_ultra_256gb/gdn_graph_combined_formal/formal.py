"""Prepared incremental GDN-on-graph timing factory; default description only.

Both arms use the same new combined HTTP entry, route ON, GDN OFF versus ON.
Observers are OFF. Native8warm/64formal, POST->DONE/count/EOS and cleanup code
remain frozen. The measured incremental gain must not be multiplied by prior
GDN/graph gains. Prerequisites cover recorded histories, not all future arrivals.
Cycle sources marked UNSEALED must be fixed before any real preparation.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import FunctionType, SimpleNamespace

HERE=Path(__file__).resolve().parent
NATIVE=HERE.parent/'pareto_realistic'
COMBINED=HERE.parent/'gdn_graph_combined'
CYCLE=HERE.parent/'gdn_graph_combined_http_cycle'
DONOR=NATIVE/'native_formal.py'
ROUTE_FORMAL=HERE.parent/'route_http_formal/formal.py'
ROUTE_FORMAL_SHA='0e590191d32b12027bc8b5f4f76335ed42a633a4179d90431427aa42c6e18f30'
SERVER=COMBINED/'http_server.py';PRODUCER=COMBINED/'http_qualification.py';AUDITOR=COMBINED/'http_audit.py'
CYCLE_AUDITOR=CYCLE/'audit_cycle.py'
BASE_SHA='21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb'


def fixed_module(path,digest):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('fixed factory support changed: '+str(path))
    spec=importlib.util.spec_from_file_location('_combined_formal_'+path.parent.name+'_'+path.stem,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


route=fixed_module(ROUTE_FORMAL,ROUTE_FORMAL_SHA)
need,same,sha,read,pinned,write=route.need,route.same,route.sha,route.read,route.pinned,route.write
TRIALS,TERMINALS=route.TRIALS,route.TERMINALS
FIXED={**route.FIXED,ROUTE_FORMAL:ROUTE_FORMAL_SHA,
    SERVER:'287f34561c83437823923cf45b67f0aa6811f7eba911ee375f62519ba508ef64',
    PRODUCER:'33b86cb57bf3c4616762c1bd1cfea0abb8f7f063325e9dfe815642e8f591bc7d',
    AUDITOR:'e41abc8e5073ca812ffdcfc398471a028a6bbd0e12c03a39ec75b0c0e4498717',
    CYCLE_AUDITOR:'45c98bf8d8eacbb07c060b05310c686320fd9cce3cd9bbaece0cd6fc887c492f',CYCLE/'owner.py':'58c35e3a8f8e758bca0792e7c1d2819a96bc7c0aac0cd5dd9169fab4c7e6de9d',
    CYCLE/'controller.py':'0f48afb64d769bbc1e24bcd0bbfaa9303d8727526c1f9ebfa1ef27903d865ad8',
    CYCLE/'audit_restore.py':'0946be64082a5a9264b04606b1a865bcaca1bf266aa3fb42888d2d21ad4bd184'}


def module(path):
    need(path in FIXED and FIXED[path]!='UNSEALED','actual cycle source must be sealed')
    return fixed_module(path,FIXED[path])
def donor():return module(DONOR)
def load(name):return donor().load(name)
def count_contract(spec,phase):return donor().count_contract(spec,phase)
def check_group(group,spec,phase,decode_text):return donor().check_group(group,spec,phase,decode_text)
class System:
    def __new__(cls):return donor().System()


def source_pins():
    need(all(h!='UNSEALED' for h in FIXED.values()),'cycle support not yet sealed')
    pins={str(p):h for p,h in FIXED.items()}
    for extra in (route.source_pins(),module(SERVER).source_pins(),module(PRODUCER).source_pins(),
                  {str(p):h for p,h in module(AUDITOR).SUPPORT.items()}):
        for p,h in extra.items():
            need(p not in pins or pins[p]==h,'source pin conflict');pins[p]=h
    pins[str(Path(__file__).resolve())]=sha(__file__)
    need(all(sha(p)==h for p,h in pins.items()),'factory source changed');return pins


def bind_combined(binding, concurrency, base_plan):
    """Read only closed small receipts; the recorded steps ledger is not opened.

    The source-pinned independent auditor already checked that ledger and raw
    dispatch/wire evidence. This binder verifies their terminal/source references;
    it is neither a rerun of the numerical audit nor a performance certificate.
    """
    need(type(binding) is dict and set(binding) == {'audit', 'replay_plan', 'child_directory', 'cycle_audit'},
         'explicit combined audit/replay/child/closed-cycle binding')
    audit = read(pinned(binding['audit'])); cycle = read(pinned(binding['cycle_audit']))
    need(cycle.get('schema') == 'GDN-graph-combined-http-cycle-independent-v1'
         and cycle.get('status') == 'PASS' and cycle.get('qualified') is True
         and cycle.get('http_history_qualified') is True and cycle.get('independent_resource_verified') is True
         and cycle.get('restoration_verified') is True and cycle.get('performance_qualified') is False
         and cycle.get('audit_source_sha256') == FIXED[CYCLE_AUDITOR], 'closed independently qualified combined cycle required')
    need(audit.get('schema') == 'GDN-graph-combined-recorded-http-independent-v1'
         and audit.get('status') == 'PASS' and audit.get('qualified') is True
         and audit.get('http_history_qualified') is True and audit.get('concurrency') == concurrency
         and audit.get('audit_source_sha256') == FIXED[AUDITOR]
         and audit.get('performance_qualified') is False
         and audit.get('owner_resources_restore_qualified') is False, 'actual combined numerical qualification required')
    child = Path(binding['child_directory'])
    need(child.is_absolute() and child.resolve() == child and child.is_dir() and not child.is_symlink(), 'canonical actual replay directory')
    need(audit['child_directory'] == str(child), 'audit/actual replay directory')
    replay_path = pinned(binding['replay_plan'])
    need(base_plan['sha256'] == BASE_SHA == audit['base_plan_sha256'], 'same original numeric workload')
    need(sha(child/'plan.json') == binding['replay_plan']['sha256'] == audit['plan_sha256'], 'actual combined replay plan')
    same(audit['terminal_sha256'], {name: sha(child/name) for name in TERMINALS}, 'closed numerical terminal bytes')
    producer = module(PRODUCER)
    plan, numeric, capture, _ = producer.validate(replay_path, binding['replay_plan']['sha256'])
    same(plan['qualification'], base_plan, 'same source/model/config numeric plan')
    need(plan['concurrency'] == concurrency, 'same qualified concurrency')
    for left, right in (('sources', 'sources'), ('raw_inputs', 'inputs'), ('capture', 'capture'), ('serving_prerequisite', 'serving')):
        same(audit[left], plan[right], 'actual audit binding: ' + left)
    top = read(child/'complete.json'); base = read(child/'base/complete.json')
    need(top['schema'] == 'GDN-graph-combined-recorded-http-result-v1' and top['status'] == base['status'] == 'PASS'
         and top['prepare_only'] is False and top['replay_numerical_checks_passed'] is True
         and top['input_source_pins_rechecked'] is True and top['independent_audit_required'] is True
         and top['numerical_qualified'] is top['http_qualified'] is top['performance_qualified'] is top['owner_resources_restore_qualified'] is False,
         'actual replay terminal scope, not an invented producer qualification')
    count = 8 if concurrency == 1 else 16
    need(type(top['total_requests']) is int and top['total_requests'] == count == audit['ledger']['requests']
         and top['concurrency'] == concurrency and audit['http']['max_http_outstanding'] == concurrency
         and audit['http']['fully_drained'] is True, 'all actual HTTP requests/drained concurrency')
    for key in ('sources', 'inputs', 'capture', 'serving'): same(top[key], plan[key], 'producer plan binding: ' + key)
    same(top['base_result'], dict(path='base/complete.json', sha256=sha(child/'base/complete.json'), returncode=0, status='PASS'), 'base terminal binding')
    need(set(cycle['numerical_audits']) == {'1', '8'}, 'both actual concurrency audits required')
    for C, ref in cycle['numerical_audits'].items():
        item = {key: ref[key] for key in ('path', 'sha256')}; other = read(pinned(item))
        need(other['status'] == 'PASS' and other['qualified'] is other['http_history_qualified'] is True
             and other['audit_source_sha256'] == FIXED[AUDITOR] and other['concurrency'] == int(C)
             and other['ledger']['requests'] == (8 if C == '1' else 16), 'closed two-C numerical audit set')
        if C == str(concurrency): same(item, binding['audit'], 'cycle audited this exact C receipt')
    restoration = cycle['restoration']
    need(restoration['status'] == 'PASS' and restoration['qualified'] is True
         and restoration['journal_sha256'] == cycle['journal_sha256']
         and restoration['owner_plan_sha256'] == cycle['owner_plan_sha256'], 'cycle restoration binding')
    owner = Path(cycle['owner_directory']); owner_plan = read(owner/'owner-plan.json')
    need(owner.is_absolute() and owner.resolve() == owner
         and sha(owner/'owner-plan.json') == cycle['owner_plan_sha256']
         and sha(owner/'complete.json') == cycle['owner_complete_sha256'], 'actual closed owner files')
    journal = Path(owner_plan['cold_restart']['journal_path'])/'journal.json'
    need(read(journal)['state'] == 'RESTORED' and sha(journal) == cycle['journal_sha256'], 'actual immutable restored journal')
    for path in (CYCLE_AUDITOR, AUDITOR, PRODUCER, CYCLE/'controller.py', CYCLE/'audit_restore.py', CYCLE/'owner.py'):
        need(cycle['source_pins'].get(str(path)) == FIXED[path], 'cycle audited actual adapter source')
    return plan, capture, audit


def select_entry(original,arm,server,pins):
    need(arm in ('off','on'),'GDN arm exactly off/on')
    command,identity=copy.deepcopy(original['command']),copy.deepcopy(original['identity'])
    need(command[3]==str(NATIVE/'native_server.py') and not any(flag in command for flag in
         ('--qualification-trace','--route-events','--dispatch-events','--route-decode-graph','--gdn-prework','--phase')),
         'unchanged observer-OFF native transport factory')
    need(identity['step_observer'] is identity['qualification_trace_enabled'] is False,'observers OFF')
    command[3]=str(SERVER)
    command+=['--phase','formal','--route-decode-graph','on','--gdn-prework',arm]
    identity.update(server.identity_fields(SimpleNamespace(phase='formal',route_decode_graph='on',gdn_prework=arm),
        pins,installed=True,route_observer=False))
    return command,identity


def prepare(base_plan,concurrency,repeat,*,arm,baseline_qualification,route_qualification,combined_qualification,
            environment,transport,deadlines):
    # Original native qualification supplies original job/body/count construction.
    original=donor().prepare(base_plan,concurrency,repeat,qualification=baseline_qualification,
        environment=environment,transport=transport,deadlines=deadlines)
    _,old_capture,_=route.bind_route(route_qualification,concurrency,base_plan)
    _,new_capture,_=bind_combined(combined_qualification,concurrency,base_plan)
    server=module(SERVER);pins=server.source_pins()
    native_identity=copy.deepcopy(original['identity'])
    native_identity.update(step_observer=True,qualification_trace_enabled=True)
    expected=copy.deepcopy(native_identity)
    expected.update(server.identity_fields(SimpleNamespace(phase='qualification',gdn_prework='on',route_decode_graph='on'),
        pins,installed=True,route_observer=True))
    same(new_capture['identity'],expected,'qualified combined source/model/config')
    old_server=route.module(route.SERVER);old_expected=copy.deepcopy(native_identity)
    old_expected.update(old_server.identity_fields(SimpleNamespace(phase='qualification',route_decode_graph='on'),
        old_server.source_pins(),installed=True,route_observer=True))
    same(old_capture['identity'],old_expected,'qualified route-only source/model/config')
    for captured in (old_capture,new_capture):
        same(captured['environment'],environment,'same qualified MLX environment')
        same({k:captured['transport'][k] for k in ('tcp_nodelay','io_timeout_seconds')},
            {k:transport[k] for k in ('tcp_nodelay','io_timeout_seconds')},'qualified socket contract')
    command,identity=select_entry(original,arm,server,pins)
    return {**original,'schema':'GDN-graph-combined-http-formal-spec-v1','arm':arm,
        'phase':f'r{repeat}-c{concurrency}-{arm}','command':command,'identity':identity,
        'qualification':route_qualification if arm=='off' else combined_qualification,
        'baseline_qualification':baseline_qualification,'route_qualification':route_qualification,
        'combined_qualification':combined_qualification,'observer_enabled':False,'sources':source_pins(),
        'scope':'Same new entry; route always ON, incremental GDN OFF/ON. Captured-history prerequisites only; no formal graph-hit observation or future-schedule proof.'}


def validate(spec):
    need(spec.get('schema')=='GDN-graph-combined-http-formal-spec-v1','combined incremental formal spec')
    same(spec['sources'],source_pins(),'fixed source closure')
    expected=prepare(spec['base_plan'],spec['concurrency'],spec['repeat'],arm=spec['arm'],
        baseline_qualification=spec['baseline_qualification'],route_qualification=spec['route_qualification'],
        combined_qualification=spec['combined_qualification'],environment=spec['environment'],
        transport=spec['transport'],deadlines=spec['deadlines'])
    same(spec,expected,'changed formal specification')


async def run(spec,out,*,checkpoint,backend=None):
    need(not any(flag in spec['command'] for flag in ('--qualification-trace','--route-events','--dispatch-events')),
         'formal observers OFF')
    native=donor();fn=native.run
    # Frozen transport/count/timing/cleanup code; private validate dependency only.
    namespace=dict(fn.__globals__,validate=validate)
    call=FunctionType(fn.__code__,namespace,fn.__name__,fn.__defaults__)
    call.__kwdefaults__=dict(fn.__kwdefaults__ or {})
    result=await call(spec,out,checkpoint=checkpoint,backend=backend)
    result.update(schema='GDN-graph-combined-http-formal-result-v1',arm=spec['arm'],phase=spec['phase'],
        observer_enabled=False,full_state_qualified=False,
        factory_source=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),
        timing_lifecycle_source=dict(path=str(DONOR),sha256=FIXED[DONOR]),
        scope='One incremental GDN-on-graph trial; outer resource/reap/restoration and independent raw timing audit still required.')
    write(Path(out)/'complete.json',result);return result


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.parse_args(argv)
    print(json.dumps(dict(schema='GDN-graph-combined-http-formal-description-v1',trials=TRIALS,
        route_always_on=True,gdn_arms=['off','on'],observers=False,
        warm_requests_per_process=8,formal_requests_per_process=64,device_run=False,
        performance_qualified=False,source_sealed=all(h!='UNSEALED' for h in FIXED.values())),indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
