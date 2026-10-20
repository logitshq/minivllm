"""Default-OFF E050: combined batched GDN+graph, unchanged E049 warm49 replay."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import traceback
from types import FunctionType

HERE = Path(__file__).resolve().parent; B = HERE.parent
BASE = B/'c8_prefix_replay_v1/prefix.py'
BASE_SHA = 'a71b2e2fc3ff2b063efe234c5a75f9f85588f6b285e4daf3c82cee2f35d3f0b7'
COMBINED = B/'gdn_batched_graph/integration.py'
COMBINED_SHA = '6e2e4d4350c36476292f2d0ff2bc7e095561f76a2c233b1a99575027378d50b7'
SCOPE = ('E050 batched GDN+graph warm49 same-history prefix only; incomplete by design. '
         'No full-history, HTTP throughput, performance or Pareto qualification.')
SCHEMA = 'c8-batched-gdn-graph-prefix-result-v1'


def need(ok, message):
    if not ok: raise ValueError(message)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_bytes())
def write(path, value): Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
def ref(path): return dict(path=str(Path(path).resolve()), sha256=sha(path))
def pinned(item):
    path = Path(item['path'])
    need(set(item) == {'path', 'sha256'} and path.is_absolute() and path.resolve() == path
         and path.is_file() and not path.is_symlink() and sha(path) == item['sha256'], 'pinned input changed')
    return path

def load(path, digest, name):
    need(sha(path) == digest, 'fixed source changed: '+str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def predecessor_ok(top, inner, plan_sha):
    need(top['schema'] == 'c8-native-graph-prefix-result-v1' and top['status'] == 'DIAGNOSTIC_PREFIX_COMPLETE'
         and top['plan_sha256'] == plan_sha and top['source_input_pins_rechecked'] is True,
         'actual successful E049 prefix required')
    need(top['completion_state'] == 'INCOMPLETE_BY_DESIGN' and top['full_history_qualified'] is False
         and top['performance_qualified'] is False and top['base_result']['returncode'] == 0
         and top['base_result']['status'] == inner['status'] == 'PASS'
         and inner['source_model_pins_rechecked'] is True, 'E049 numeric and scope proof')
    case = top['prefix']
    need(case['status'] == 'DIAGNOSTIC_PREFIX_COMPLETE' and case['steps'] == 49 and case['completed'] == []
         and len(case['active_request_ids']) == len(case['future_unadmitted_request_ids']) == 8
         and inner['cases'] == [case] and inner['total_requests'] == 0,
         'actual unfinished warm49 boundary')
    need(top['coverage']['installs'] == 1 and top['coverage']['restores'] == 2
         and top['route_prefix']['full_route_stream_consumed'] is False, 'E049 pair release and partial route evidence')


def prepare(base_plan, predecessor):
    base_plan, predecessor = Path(base_plan).resolve(), Path(predecessor).resolve()
    original = load(BASE, BASE_SHA, '_e050_original_prefix_contract')
    old = read(base_plan); original.validate(old)
    top = read(predecessor); directory = predecessor.parent
    inner_path = directory/'base/complete.json'; inner = read(inner_path)
    need(top['base_result']['sha256'] == sha(inner_path) and sha(directory/'plan.json') == sha(base_plan),
         'E049 actual plan and inner result binding')
    predecessor_ok(top, inner, sha(base_plan))
    combined = load(COMBINED, COMBINED_SHA, '_e050_combined_contract')
    sources = {**old['sources'], **combined.source_pins(), str(BASE): BASE_SHA,
               str(HERE/'bridge.py'): sha(HERE/'bridge.py'), str(Path(__file__).resolve()): sha(__file__)}
    need(all(sha(p) == h for p, h in sources.items()), 'source closure changed')
    inputs = [directory/n for n in ('plan.json', 'base/complete.json', 'base/steps.jsonl', 'replay-route-events.jsonl')]
    return dict(schema='c8-batched-gdn-graph-prefix-plan-v1', sources=sources,
        base_prefix_plan=ref(base_plan), predecessor=ref(predecessor),
        predecessor_inputs={str(p): sha(p) for p in inputs}, switches=dict(gdn_prework=True, route_graph=True),
        qualification=old['qualification'], recording=old['recording'], recipe=old['recipe'], environment=old['environment'],
        stop_after_step=49, completion_state='INCOMPLETE_BY_DESIGN', full_history_qualified=False,
        performance_qualified=False, scope=SCOPE)


def validate(plan):
    expected = prepare(pinned(plan['base_prefix_plan']), pinned(plan['predecessor']))
    need(json.dumps(plan, sort_keys=True) == json.dumps(expected, sort_keys=True), 'changed E050 prefix specification')


def inherited_run(original, bridge, validator):
    """The frozen code executes with isolated substitutions; old globals stay intact."""
    namespace = dict(vars(original), integration_module=bridge.integration_module, validate=validator, SCOPE=SCOPE)
    fn = FunctionType(original.run.__code__, namespace, 'run', original.run.__defaults__, original.run.__closure__)
    fn.__kwdefaults__ = original.run.__kwdefaults__
    return fn


def run(plan, out):
    validate(plan)
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=False); write(out/'plan.json', plan)
    result = dict(schema=SCHEMA, status='INCOMPLETE', pid=os.getpid(), ppid=os.getppid(),
        plan_sha256=sha(out/'plan.json'), completion_state='INCOMPLETE_BY_DESIGN',
        full_history_qualified=False, performance_qualified=False, scope=SCOPE)
    write(out/'complete.json', result)
    bridge = None
    try:
        original = load(BASE, BASE_SHA, '_e050_original_prefix_execution')
        combined = load(COMBINED, COMBINED_SHA, '_e050_combined_execution')
        module = load(HERE/'bridge.py', plan['sources'][str(HERE/'bridge.py')], '_e050_engine_bridge')
        old = read(pinned(plan['base_prefix_plan']))
        def validator(actual):
            need(actual == old, 'unmodified inherited prefix plan required')
            validate(plan)
        with (out/'composition.jsonl').open('x') as events:
            def emit(event):
                events.write(json.dumps(event, sort_keys=True, allow_nan=False)+'\n'); events.flush()
            bridge = module.Bridge(combined, emit)
            try:
                code = inherited_run(original, bridge, validator)(old, out/'inherited')
            finally:
                bridge.close_private_hook()
            inner_path = out/'inherited/complete.json'; inner = read(inner_path)
            result['inherited_result'] = dict(path=str(inner_path), sha256=sha(inner_path),
                                             returncode=code, status=inner['status'])
            # These remain the actual original replay outputs, with their provenance intact.
            for name in ('base_result', 'prefix', 'coverage', 'route_prefix', 'peak_mlx_bytes', 'elapsed_base_seconds'):
                if name in inner: result[name] = inner[name]
            need(code == 0 and inner['status'] == 'DIAGNOSTIC_PREFIX_COMPLETE'
                 and inner['pid'] == os.getpid() and inner['ppid'] == os.getppid()
                 and inner['plan_sha256'] == plan['base_prefix_plan']['sha256']
                 and inner['source_input_pins_rechecked'] is True, 'inherited numeric checks and cleanup must pass')
            result['composition'] = bridge.checked()
        validate(plan)
        result.update(status='DIAGNOSTIC_PREFIX_COMPLETE', source_input_pins_rechecked=True)
    except BaseException as exc:
        result.update(status='FAIL', error=dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
    finally:
        if bridge is not None: result['composition'] = bridge.snapshot()
        if (out/'composition.jsonl').is_file(): result['composition_events'] = ref(out/'composition.jsonl')
        write(out/'complete.json', result)
    print(json.dumps(dict(status=result['status'], out=str(out))))
    return 0 if result['status'] == 'DIAGNOSTIC_PREFIX_COMPLETE' else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); modes = p.add_mutually_exclusive_group()
    modes.add_argument('--prepare', action='store_true'); modes.add_argument('--run', action='store_true')
    p.add_argument('--base-plan', type=Path); p.add_argument('--predecessor', type=Path)
    p.add_argument('--plan', type=Path); p.add_argument('--plan-sha256'); p.add_argument('--out', type=Path); a = p.parse_args()
    if a.prepare:
        need(a.base_plan and a.predecessor and a.out, 'prepare needs E049 plan, actual successful prefix, fresh output')
        plan = prepare(a.base_plan, a.predecessor); a.out.mkdir(parents=True, exist_ok=False); write(a.out/'plan.json', plan)
        print(json.dumps(dict(status='PREPARED_NOT_RUN', plan=str(a.out/'plan.json'), sha256=sha(a.out/'plan.json'))))
    elif a.run:
        need(a.plan and a.plan_sha256 and a.out and sha(a.plan) == a.plan_sha256, 'explicit plan SHA/new output required')
        raise SystemExit(run(read(a.plan), a.out))
    else: print('DESCRIPTION ONLY: combined candidate; original warm49 incomplete prefix; explicit prepare/run required.')
