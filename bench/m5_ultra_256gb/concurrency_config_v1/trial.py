"""Eight-warm/eight-screen MB-threshold diagnostic; never a Pareto certificate.

Frozen E033 sources, original 50/50 specification and wire/count receipt are
verified before deriving the one-variable environment. The old numerical audit
DOES NOT qualify the new environment. Outer ownership, resources, restoration,
and cross-configuration token comparison remain mandatory external checks.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from types import FunctionType

HERE = Path(__file__).resolve().parent
FORMAL = HERE.parent / 'gdn_graph_combined_formal/formal.py'
NATIVE = HERE.parent / 'pareto_realistic/native_formal.py'
FIXED = {FORMAL: '4b567f8ed6090175f3639aa109584ca26c1426165c03de3167c3ebde74891a89',
         NATIVE: 'ec9cdc924f7987ad46eac6820b6264495c95fb182c726545a6e3ce283df7e654'}
MB_VALUES = (16, 50, 200, 800)
MLX_BASE = {'MLX_MAX_OPS_PER_BUFFER': '50', 'MLX_MAX_MB_PER_BUFFER': '50',
            'MLX_BFS_MAX_WIDTH': '20', 'MLX_METAL_FAST_SYNCH': '0'}
SCHEMA = 'concurrency-config-screen-spec-v1'
SCOPE = ('Diagnostic eight-request screen only. The inherited qualification receipt '
         'supplies the frozen wire/count contract, not numerical qualification for '
         'the changed MLX environment. Exact token comparison and all outer checks '
         'remain outstanding; no full-state or performance qualification.')


def need(ok, message):
    if not ok: raise ValueError(message)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_bytes())
def same(a, b): return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def module(path):
    need(path in FIXED and sha(path) == FIXED[path], 'frozen support changed: ' + str(path))
    spec = importlib.util.spec_from_file_location('_concurrency_config_' + path.stem, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def source_pins():
    pins = module(FORMAL).source_pins()
    for path, digest in FIXED.items():
        need(sha(path) == digest, 'frozen support changed')
        need(str(path) not in pins or pins[str(path)] == digest, 'source conflict')
        pins[str(path)] = digest
    pins[str(Path(__file__).resolve())] = sha(__file__)
    return pins


def prepare(base_spec_ref, mb, label):
    """CPU-only preparation. Keep E033 original bytes and validator untouched."""
    need(type(mb) is int and mb in MB_VALUES, 'registered integer MB threshold')
    need(type(label) is str and re.fullmatch(r'[a-z0-9][a-z0-9-]{0,63}', label), 'bounded ASCII trial label')
    formal = module(FORMAL)
    base = read(formal.pinned(base_spec_ref)); formal.validate(base)
    need((base['concurrency'], base['arm']) in ((1, 'on'), (8, 'off')) and base['repeat'] == 1,
         'registered E033 C1/on or C8/off first-repeat base')
    need({k: v for k, v in base['environment'].items() if k.startswith('MLX_')} == MLX_BASE,
         'base exact four-variable MLX environment')
    need(len(base['jobs']['warm']) == 8 and len(base['jobs']['formal']) == 64,
         'original eight-warm/64-formal queue')
    result = copy.deepcopy(base)
    result.update(schema=SCHEMA, base_spec=copy.deepcopy(base_spec_ref), label=label,
                  mb_threshold=mb, sources=source_pins(), numerical_qualified=False,
                  full_state_qualified=False, performance_qualified=False,
                  qualification_scope='inherited wire/count contract only', scope=SCOPE)
    result['environment']['MLX_MAX_MB_PER_BUFFER'] = str(mb)
    result['jobs']['formal'] = result['jobs']['formal'][:8]
    result['deadlines']['resource_interval'] = 3
    return result


def validate(spec):
    need(spec.get('schema') == SCHEMA, 'diagnostic specification required')
    expected = prepare(spec['base_spec'], spec['mb_threshold'], spec['label'])
    need(same(spec, expected), 'diagnostic specification differs from allowed derivation')


CLIENT_PHASE = {'warm': 'warm', 'formal': 'admission_probe'}


def check_group(group, spec, phase, decode_text):
    """Keep diagnostic raw bytes intact; select the frozen eight-row warm gate."""
    need(phase in CLIENT_PHASE and group['phase'] == CLIENT_PHASE[phase]
         and group['summary']['formal_point_eligible'] is False, 'original nonformal client phase')
    view = {**group, 'phase': 'warm'}
    spec_view = {**spec, 'jobs': {'warm': spec['jobs'][phase]}}
    checked = module(NATIVE).check_group(view, spec_view, 'warm', decode_text)
    checked.update(numerical_qualified=False, full_state_qualified=False,
                   performance_qualified=False, diagnostic_only=True, scope=SCOPE,
                   raw_client_phase=group['phase'], validation_view_phase='warm')
    return checked


class System:
    def __init__(self): self.base = module(NATIVE).System()
    def __getattr__(self, name): return getattr(self.base, name)
    async def group(self, spec, phase, output):
        native = module(NATIVE); client = native.load('native_cleanup'); d = spec['deadlines']
        return await client.run_group(f"http://127.0.0.1:{spec['transport']['port']}",
            spec['jobs'][phase], spec['concurrency'], phase=CLIENT_PHASE[phase],
            qualification=native.count_contract(spec, phase), request_timeout_s=d['request'],
            group_timeout_s=d[phase], cleanup_timeout_s=d['close'], output_dir=output)


async def run(spec, out, *, checkpoint, backend=None):
    """Reuse native direct-child ownership/cleanup; caller owns the service lease."""
    native = module(NATIVE); function = native.run
    namespace = dict(function.__globals__, validate=validate, check_group=check_group, System=System)
    call = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__)
    call.__kwdefaults__ = dict(function.__kwdefaults__ or {})
    result = await call(spec, out, checkpoint=checkpoint, backend=backend)
    result.update(schema='concurrency-config-screen-result-v1', label=spec['label'],
                  mb_threshold=spec['mb_threshold'], diagnostic_only=True,
                  numerical_qualified=False, full_state_qualified=False, performance_qualified=False,
                  token_equivalence_checked=False, source_pins=spec['sources'],
                  base_spec=spec['base_spec'], scope=SCOPE)
    native.write(Path(out) / 'complete.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); parser.parse_args(argv)
    print(json.dumps(dict(schema='concurrency-config-screen-description-v1',
        default_action='describe only; outer owner calls prepare/validate/run',
        mb_thresholds=MB_VALUES, warm_requests=8, screen_requests=8,
        fixed_mlx={k: v for k, v in MLX_BASE.items() if k != 'MLX_MAX_MB_PER_BUFFER'},
        performance_qualified=False, numerical_qualified=False), indent=2))
    return 0


if __name__ == '__main__': raise SystemExit(main())
