"""Combined capture factory; frozen direct-child lifecycle, no service control.

Only prepare/validate/startup are new. The frozen capture code runs in its own
explicit dependency namespace; its module globals and source identity stay
unchanged. Final metadata identifies both the reused lifecycle and this adapter.
No numerical or performance qualification is granted by a successful capture.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import FunctionType, SimpleNamespace

HERE = Path(__file__).resolve().parent
DONOR = HERE.parent / 'route_decode_graph/http_capture.py'
DONOR_SHA = 'b331a02988413356e113b3a1283ab0eb8d40de17796b5623a17df312c214151f'
SERVER = HERE / 'http_server.py'
SERVER_SHA = '287f34561c83437823923cf45b67f0aa6811f7eba911ee375f62519ba508ef64'
ROUTE_SCHEMA = 'GDN-graph-combined-http-events-v1'


def need(ok, message):
    if not ok: raise ValueError(message)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def same(a, b): return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)
def load_fixed(path, digest, name):
    need(sha(path) == digest, 'fixed capture dependency changed: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


donor = load_fixed(DONOR, DONOR_SHA, '_combined_frozen_route_capture')
System, no_process, owned_listener = donor.System, donor.no_process, donor.owned_listener
write = donor.write


def prepare(base_plan_path, base_plan_sha, concurrency, *, python, environment, transport, deadlines):
    server = load_fixed(SERVER, SERVER_SHA, '_combined_capture_server')
    original = donor.donor.prepare(base_plan_path, base_plan_sha, concurrency,
        python=python, environment=environment, transport=transport, deadlines=deadlines)
    need(type(concurrency) is int and concurrency in (1, 8), 'fixed C1/C8 domain')
    cfg = original['identity']['configuration']
    need(cfg['max_len'] == 2048 and cfg['max_seqs'] == concurrency
         and cfg['compile_decode'] == cfg['fuse_decode'] == cfg['gates_compile'] == 'on'
         and cfg['norm_reuse'] == 'off', 'unchanged native flags/context')
    pins = server.source_pins()
    sources = dict(original['sources'])
    for path, digest in {**pins, str(DONOR): DONOR_SHA, str(Path(__file__).resolve()): sha(__file__)}.items():
        need(path not in sources or sources[path] == digest, 'conflicting source pin')
        sources[path] = digest
    command = list(original['command'])
    need(command[3] == str(donor.donor.HERE / 'native_server.py'), 'original native entry')
    command[3] = str(SERVER)
    command += ['--phase', 'qualification', '--gdn-prework', 'on', '--route-decode-graph', 'on']
    args = SimpleNamespace(phase='qualification', gdn_prework='on', route_decode_graph='on')
    identity = dict(original['identity'])
    identity.update(server.identity_fields(args, pins, installed=True, route_observer=True))
    return {**original, 'schema': 'GDN-graph-combined-http-capture-spec-v1',
        'sources': sources, 'command': command, 'identity': identity,
        'route_events_schema': ROUTE_SCHEMA,
        'qualification_scope': 'Observed combined capture only; independent numerical replay and outer resources/recovery required.'}


def validate(spec):
    need(spec.get('schema') == 'GDN-graph-combined-http-capture-spec-v1', 'combined capture spec required')
    expected = prepare(spec['base_plan']['path'], spec['base_plan']['sha256'], spec['concurrency'],
        python=spec['python'], environment=spec['environment'], transport=spec['transport'], deadlines=spec['deadlines'])
    need(same(spec, expected), 'combined capture spec was changed')
    need(all(sha(path) == digest for path, digest in spec['sources'].items()), 'capture source changed')


def route_startup(out, spec, pid):
    with (out/'route-events.jsonl').open('rb') as stream: first = stream.readline()
    expected = dict(event='route-startup', schema=ROUTE_SCHEMA, pid=pid,
        phase=f"HTTP-C{spec['concurrency']}", identity=spec['identity'])
    need(type(pid) is int and pid > 1 and first.endswith(b'\n') and same(json.loads(first), expected),
         'closed combined sidecar must bind actual child/source/phase')


async def capture(spec, out, *, checkpoint, backend=None):
    # No lifecycle fork and no donor module-global writes. These are the only
    # variable contracts referenced by the frozen coroutine.
    namespace = dict(donor.capture.__globals__, validate=validate, route_startup=route_startup)
    run = FunctionType(donor.capture.__code__, namespace, donor.capture.__name__, donor.capture.__defaults__)
    run.__kwdefaults__ = dict(donor.capture.__kwdefaults__ or {})
    result = await run(spec, out, checkpoint=checkpoint, backend=backend)
    result.update(schema='GDN-graph-combined-http-capture-result-v1',
        lifecycle_source=dict(path=str(DONOR), sha256=DONOR_SHA),
        capture_adapter_source=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)),
        scope='Instrumented combined capture; independent same-history numerical/route audit and resources/recovery remain required.')
    write(Path(out)/'complete.json', result)
    return result
