"""Default-OFF actual candidate HTTP capture: original warm8 + screen8, MB50.

Reuse E047's drained groups and the original direct-child capture lifecycle.
E051 actual recorded-history PASS is a prerequisite, not qualification of this
new server's schedules. All mathematical/performance qualification flags stay OFF.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import FunctionType, SimpleNamespace

HERE = Path(__file__).resolve().parent; B = HERE.parent
OLD = B/'c8_schedule_diagnostic_v1/trial.py'
OLD_SHA = 'a140fa0b2cf4d6eb1bd58286696d4cc16ad3747bf396497817c91dcb98f5bad5'
FULL = B/'c8_batched_full_v1/qualify.py'
FULL_SHA = '7c2b5d6596ae23e8d7eb550a8121ab3c5fece370e803614cfaaa97f96b26eeb3'
SCHEMA = 'c8-batched-http-spec-v1'
SCOPE = 'Actual candidate warm8/screen8 HTTP capture only; new-history numerical replay and independent audit outstanding.'


def need(ok, message):
    if not ok: raise ValueError(message)
def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8 << 20): digest.update(block)
    return digest.hexdigest()
def read(path): return json.loads(Path(path).read_bytes())
def write(path, value): Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
def ref(path): return dict(path=str(Path(path).resolve()), sha256=sha(path))
def load(path, digest, name):
    need(sha(path) == digest, 'frozen capture support changed: '+str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

def old(): return load(OLD, OLD_SHA, '_batched_http_original_trial')
def server(): return load(HERE/'server.py', sha(HERE/'server.py'), '_batched_http_new_server')


def predecessor_ok(top):
    need(top['schema'] == 'c8-batched-gdn-graph-full-result-v1' and top['status'] == 'PASS'
         and top['source_input_pins_rechecked'] is True and top['full_history_qualified'] is True
         and top['http_qualified'] is False and top['performance_qualified'] is False,
         'actual full E051 PASS required, with recorded-history-only scope')
    need(top['steps'] == 1030 and top['total_requests'] == 16
         and top['base_result']['status'] == 'PASS' and top['base_result']['returncode'] == 0
         and top['route_comparison']['calls'] == top['route_comparison']['actual_compiled_invocations'] == 1028
         and top['route_comparison']['fallbacks'] == 0, 'actual E051 full result and complete graph route')


def source_pins():
    return {**old().source_pins(), **server().source_pins(), str(OLD):OLD_SHA,
            str(FULL):FULL_SHA, str(Path(__file__).resolve()):sha(__file__)}


def prepare(base_spec_ref, predecessor_ref):
    original = old(); base = original.prepare(base_spec_ref)
    full = load(FULL, FULL_SHA, '_batched_http_full_prerequisite')
    predecessor = full.pinned(predecessor_ref); top = read(predecessor); predecessor_ok(top)
    child = predecessor.parent; plan = read(child/'plan.json'); full.validate(plan)
    need(sha(child/'plan.json') == top['plan_sha256'] and plan['qualification'] == base['base_plan'], 'same actual E051 plan/model workload')
    inner_path = child/'base/complete.json'
    need(top['base_result']['path'] == str(inner_path) and top['base_result']['sha256'] == sha(inner_path), 'actual E051 base result')
    full.full_result_ok(read(inner_path), read(full.pinned(plan['recording'])), top['coverage'])
    full.composition_ok(top['composition'])
    for key, name in (('composition_events','composition.jsonl'), ('replay_route_events','replay-route-events.jsonl')):
        need(full.pinned(top[key]) == child/name, 'actual E051 emitted sidecar')
    s = server(); command = list(base['command'])
    need(command[3] == str(original.HERE/'server.py'), 'original E047 server command')
    command[3] = str(HERE/'server.py'); command.insert(4, '--run')
    command[command.index('--gdn-prework')+1] = 'on'
    identity = copy.deepcopy(base['identity'])
    identity.update(s.identity_fields(SimpleNamespace(phase='qualification',gdn_prework='on',route_decode_graph='on'),
        s.source_pins(), installed=True, route_observer=True))
    names = ('plan.json','base/complete.json','base/steps.jsonl','composition.jsonl','replay-route-events.jsonl')
    return {**base, 'schema':SCHEMA, 'label':'c8-batched-mb50-two-waves', 'sources':source_pins(),
        'command':command, 'identity':identity, 'predecessor':copy.deepcopy(predecessor_ref),
        'predecessor_inputs':{str(child/n):sha(child/n) for n in names},
        'numerical_qualified':False, 'full_state_qualified':False, 'performance_qualified':False, 'scope':SCOPE}


def validate(spec):
    need(spec['schema'] == SCHEMA and old().same(spec, prepare(spec['base_spec'], spec['predecessor'])),
         'changed candidate HTTP capture specification')


def capture_function(capture, system_type):
    # The only lifecycle change is the schema of its closed route startup check.
    route_namespace = dict(vars(capture), ROUTE_SCHEMA=server().ROUTE_SCHEMA)
    startup = FunctionType(capture.route_startup.__code__, route_namespace, 'route_startup', capture.route_startup.__defaults__)
    namespace = dict(vars(capture), validate=validate, System=system_type, route_startup=startup)
    function = FunctionType(capture.capture.__code__, namespace, 'capture', capture.capture.__defaults__)
    function.__kwdefaults__ = dict(capture.capture.__kwdefaults__ or {})
    return function


async def run(spec, out, *, checkpoint, backend=None):
    original = old(); capture = original.module(original.CAPTURE)
    call = capture_function(capture, original.System)
    result = await call(spec, out, checkpoint=checkpoint,
                        backend=original.System(backend) if backend is not None else None)
    result.update(schema='c8-batched-http-result-v1', label=spec['label'], diagnostic_only=True,
        numerical_qualified=False, full_state_qualified=False, performance_qualified=False,
        replay_input_checked=False, scope=SCOPE)
    write(Path(out)/'complete.json', result)
    return result


def join_child(child, spec_sha256):
    original = old()
    namespace = dict(vars(original), validate=validate, SCOPE=SCOPE)
    call = FunctionType(original.join_child.__code__, namespace, 'join_child', original.join_child.__defaults__)
    output = call(child, spec_sha256)
    receipt = read(output/'receipt.json'); inherited_schema = receipt['schema']
    receipt.update(schema='c8-batched-http-joined-input-v1', inherited_join_schema=inherited_schema)
    write(output/'receipt.json', receipt)
    return output


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare-base', type=Path); p.add_argument('--base-sha256')
    p.add_argument('--predecessor', type=Path); p.add_argument('--predecessor-sha256'); p.add_argument('--out', type=Path)
    p.add_argument('--join-child', type=Path); p.add_argument('--spec-sha256'); a = p.parse_args(argv)
    if a.prepare_base:
        need(a.base_sha256 and a.predecessor and a.predecessor_sha256 and a.out and not a.join_child,
             'exact base and actual full E051 predecessor SHA/fresh output required')
        spec = prepare(dict(path=str(a.prepare_base.resolve()), sha256=a.base_sha256),
                       dict(path=str(a.predecessor.resolve()), sha256=a.predecessor_sha256))
        a.out.mkdir(parents=True, exist_ok=False); write(a.out/'spec.json', spec)
        print(json.dumps(dict(status='PREPARED_NOT_RUN', spec=str(a.out/'spec.json'), sha256=sha(a.out/'spec.json'))))
    elif a.join_child:
        need(a.spec_sha256 and not any((a.out,a.predecessor,a.base_sha256,a.predecessor_sha256)), 'join child/spec SHA only')
        print(str(join_child(a.join_child,a.spec_sha256)))
    else:
        need(not any((a.out,a.base_sha256,a.predecessor,a.predecessor_sha256,a.spec_sha256)), 'incomplete action arguments')
        print(json.dumps(dict(schema=SCHEMA, default_action='describe only', waves=[8,8], scope=SCOPE)))
    return 0


if __name__ == '__main__': raise SystemExit(main())
