"""E047 two eight-request waves, observed schedules, fixed C8 graph-only MB50.

Reuse the frozen direct-child capture lifecycle. Each wave is an original client
qualification group with no count receipt. group.json is explicitly a derived
join view of those two raw groups, not a single HTTP barrier or timing sample.
Default CLI describes only. The outer owner supplies resources/service recovery.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import FunctionType, SimpleNamespace

HERE = Path(__file__).resolve().parent
FORMAL = HERE.parent / 'gdn_graph_combined_formal/formal.py'
CAPTURE = HERE.parent / 'route_decode_graph/http_capture.py'
JOIN = HERE.parent / 'pareto_realistic/http_join.py'
REPLAY = HERE.parent / 'pareto_realistic/http_replay.py'
FIXED = {FORMAL: '4b567f8ed6090175f3639aa109584ca26c1426165c03de3167c3ebde74891a89',
         CAPTURE: 'b331a02988413356e113b3a1283ab0eb8d40de17796b5623a17df312c214151f',
         JOIN: 'a362c321c2329d7ab53ab6a1a5a96981bb95b44733739fdbb56dd3d38bef7b4e',
         REPLAY: '1304c58c66e56b3aeaa582ef9f7d61317280cf3107f8f600485426119970fb65'}
MLX = {'MLX_MAX_OPS_PER_BUFFER': '50', 'MLX_MAX_MB_PER_BUFFER': '50',
       'MLX_BFS_MAX_WIDTH': '20', 'MLX_METAL_FAST_SYNCH': '0'}
SCHEMA = 'E047-c8-schedule-spec-v1'
SCOPE = 'Observed schedule diagnosis only; no inherited numerical or performance qualification.'


def need(ok, message):
    if not ok: raise ValueError(message)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_bytes())
def same(a, b): return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)
def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def module(path):
    need(path == HERE / 'server.py' or path in FIXED and sha(path) == FIXED[path], 'fixed diagnostic support changed')
    spec = importlib.util.spec_from_file_location('_e047_' + path.parent.name + '_' + path.stem, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def source_pins():
    pins = module(FORMAL).source_pins()
    pins.update(module(HERE / 'server.py').source_pins())
    for path, digest in FIXED.items():
        need(sha(path) == digest, 'frozen diagnostic support changed')
        pins[str(path)] = digest
    pins[str(Path(__file__).resolve())] = sha(__file__)
    return pins


def prepare(base_spec_ref):
    formal = module(FORMAL)
    base = read(formal.pinned(base_spec_ref)); formal.validate(base)
    need((base['concurrency'], base['repeat'], base['arm']) == (8, 1, 'off'), 'original E033 C8/off/r1 only')
    need({k: v for k, v in base['environment'].items() if k.startswith('MLX_')} == MLX, 'fixed four MLX settings')
    need(len(base['jobs']['warm']) == 8 and len(base['jobs']['formal']) == 64, 'original E033 queues')
    waves = [copy.deepcopy(base['jobs']['warm']), copy.deepcopy(base['jobs']['formal'][:8])]
    need(all(a['fixture_id'] == b['fixture_id'] and a['body'] == b['body'] and a['path'] == b['path']
             for a, b in zip(*waves)), 'two waves must repeat exact request bodies')
    server = module(HERE / 'server.py'); pins = server.source_pins()
    command = list(base['command']); need(command[3] == str(server.COMBINED), 'actual combined entry base')
    command[3] = str(HERE / 'server.py'); command[command.index('--phase') + 1] = 'qualification'
    identity = copy.deepcopy(base['identity'])
    identity.update(step_observer=True, qualification_trace_enabled=True)
    identity.update(server.identity_fields(SimpleNamespace(phase='qualification', gdn_prework='off',
        route_decode_graph='on'), pins, installed=True, route_observer=True))
    deadlines = dict(ready=120, request=600, group=720, close=10, term=15, kill=15,
                     overall=1620, resource_interval=3)
    return dict(schema=SCHEMA, label='c8-mb50-two-waves', base_spec=copy.deepcopy(base_spec_ref),
        base_plan=base['base_plan'], sources=source_pins(), command=command,
        python=base['command'][0], environment=base['environment'], transport=base['transport'],
        deadlines=deadlines, concurrency=8, jobs=waves[0] + waves[1], wave_names=['warm', 'screen'],
        identity=identity, fixtures=base['fixtures'], tokenizer=base['tokenizer'],
        numerical_qualified=False, full_state_qualified=False, performance_qualified=False, scope=SCOPE)


def validate(spec):
    need(spec.get('schema') == SCHEMA and same(spec, prepare(spec['base_spec'])), 'changed diagnostic specification')


def group_ok(group):
    cleanup = group.get('cleanup', {})
    return (group.get('schema') == 'E023-http-group-v1' and group.get('phase') == 'qualification'
        and group.get('qualification') is None and group.get('concurrency') == 8
        and group.get('group_error') is None and cleanup.get('complete') is True
        and cleanup.get('deadline_exceeded') is False and cleanup.get('errors') == []
        and cleanup.get('pending_tasks') == [] and cleanup.get('cancelled_again') is False
        and cleanup.get('adapter') == 'native-close-cancellation-v1'
        and group.get('summary', {}).get('transport_complete') is True
        and group.get('summary', {}).get('formal_point_eligible') is False)


def join_view(groups, refs):
    """Retain exact request rows; report conjunctions, never invented wave clocks."""
    complete = len(groups) == 2 and all(group_ok(g) for g in groups)
    closed = bool(groups) and all(g.get('cleanup', {}).get('complete') is True for g in groups)
    return dict(schema='E023-http-group-v1', phase='qualification', concurrency=8, qualification=None,
        derived_view=True, derivation='two sequential drained eight-request groups; no shared barrier',
        source_groups=refs, requests=[r for g in groups for r in g['requests']],
        group_error=None if complete else {'type': 'IncompleteOrFailedWave'},
        cleanup=dict(complete=closed, deadline_exceeded=any(g['cleanup'].get('deadline_exceeded') for g in groups),
            errors=[e for g in groups for e in g['cleanup'].get('errors', [])],
            pending_tasks=[p for g in groups for p in g['cleanup'].get('pending_tasks', [])],
            cancelled_again=any(g['cleanup'].get('cancelled_again') for g in groups),
            adapter='native-close-cancellation-v1', scope='conjunction of retained per-wave cleanup receipts'),
        summary=dict(transport_complete=complete, formal_point_eligible=False), performance_qualified=False)


class System:
    def __init__(self, base=None): self.base = base or module(CAPTURE).System()
    def __getattr__(self, name): return getattr(self.base, name)
    async def group(self, spec, output):
        groups, refs = [], []
        for i, phase in enumerate(spec['wave_names']):
            target = Path(output) / phase; target.mkdir(parents=True, exist_ok=False)
            view = {**spec, 'jobs': spec['jobs'][8*i:8*i+8]}
            try:
                group = await self.base.group(view, target / 'raw')
            except BaseException as exc:
                if hasattr(exc, 'result'):
                    group = exc.result; write(target / 'group.json', group); groups.append(group)
                    refs.append(dict(phase=phase, path=str(target / 'group.json'), sha256=sha(target / 'group.json')))
                exc.result = join_view(groups, refs)
                raise
            write(target / 'group.json', group); groups.append(group)
            refs.append(dict(phase=phase, path=str(target / 'group.json'), sha256=sha(target / 'group.json')))
            if not group_ok(group): break
        return join_view(groups, refs)


async def run(spec, out, *, checkpoint, backend=None):
    original = module(CAPTURE).capture
    scope = dict(original.__globals__, validate=validate, System=System)
    call = FunctionType(original.__code__, scope, original.__name__, original.__defaults__)
    call.__kwdefaults__ = dict(original.__kwdefaults__ or {})
    result = await call(spec, out, checkpoint=checkpoint,
                        backend=System(backend) if backend is not None else None)
    result.update(schema='E047-c8-schedule-result-v1', label=spec['label'], diagnostic_only=True,
        numerical_qualified=False, full_state_qualified=False, performance_qualified=False,
        replay_input_checked=False, scope=SCOPE)
    module(CAPTURE).write(Path(out) / 'complete.json', result)
    return result


def join_child(child, spec_sha256):
    """Post-close CPU raw join. Retained raw sources remain the authority."""
    child = Path(child).resolve(); spec = read(child / 'spec.json')
    need(sha(child / 'spec.json') == spec_sha256, 'actual spec SHA required'); validate(spec)
    top = read(child / 'complete.json')
    need(top['status'] == 'PASS' and top['server_reaped'] is True and top['cleanup_complete'] is True,
         'closed captured child required before joining')
    groups, refs = [], []
    for phase in spec['wave_names']:
        path = child / 'client' / phase / 'group.json'; groups.append(read(path))
        refs.append(dict(phase=phase, path=str(path), sha256=sha(path)))
    expected = join_view(groups, refs)
    need(same(expected, read(child / 'group.json')), 'derived view differs from original groups')
    need(sha(spec['tokenizer']['path']) == spec['tokenizer']['sha256'], 'tokenizer source changed')
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(spec['tokenizer']['path'])
    with (child / 'trace.jsonl').open('rb') as trace:
        recording = module(JOIN).join(expected, trace, spec['jobs'],
            {f['fixture_id']: f for f in spec['fixtures']}, spec['identity'],
            lambda ids: tokenizer.decode(ids, skip_special_tokens=False))
    out = child / 'joined-input'; out.mkdir(exist_ok=False)
    write(out / 'recording.json', recording)
    write(out / 'receipt.json', dict(schema='E047-joined-input-v1', scope=SCOPE,
        spec_sha256=spec_sha256, source_groups=refs, trace_sha256=sha(child / 'trace.jsonl'),
        recording_sha256=sha(out / 'recording.json'), sources=spec['sources'],
        replay_numerical_checked=False, performance_qualified=False))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare-base', type=Path); p.add_argument('--base-sha256'); p.add_argument('--out', type=Path)
    p.add_argument('--join-child', type=Path); p.add_argument('--spec-sha256'); a = p.parse_args(argv)
    if a.prepare_base:
        need(a.base_sha256 and a.out and not a.join_child and not a.spec_sha256, 'prepare requires exact base SHA/output only')
        spec = prepare(dict(path=str(a.prepare_base.resolve()), sha256=a.base_sha256))
        a.out.mkdir(parents=True, exist_ok=False); write(a.out / 'spec.json', spec)
        print(json.dumps(dict(spec=str(a.out / 'spec.json'), sha256=sha(a.out / 'spec.json'))))
    elif a.join_child:
        need(a.spec_sha256 and not a.out and not a.base_sha256, 'join requires child/spec SHA only')
        print(str(join_child(a.join_child, a.spec_sha256)))
    else:
        need(not any((a.out, a.base_sha256, a.spec_sha256)), 'incomplete action arguments')
        print(json.dumps(dict(schema=SCHEMA, default_action='describe only', waves=[8, 8],
            mlx=MLX, launch='outer owner calls run(spec,out,checkpoint=...)', scope=SCOPE)))
    return 0


if __name__ == '__main__': raise SystemExit(main())
