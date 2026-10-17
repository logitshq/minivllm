"""E022 CPU constants, source reuse, launch construction and fixed gate."""
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / '.artifacts/m5-ultra-256gb'
DEFAULT_E020 = BASE / 'external-spec-http-v1-inputs/plan.json'
CONFIGS = [{'pair': pair, 'arm': arm} for pair, arms in
           ((1, ('off', 'on')), (2, ('on', 'off')), (3, ('off', 'on'))) for arm in arms]
GATE = {'all_correctness_resource_reap_checks': True, 'paired_wall_geomean_min': 1.05,
        'every_paired_wall_speedup_strictly_above': 1.0, 'every_paired_TTFT_speedup_min': .97}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def frozen_e020(e020_plan, original):
    return Path(e020_plan).parent / 'frozen-sources' / str(original).lstrip('/')


def load_lifecycle(plan):
    """Import the frozen E020 controller, not its CLI main or model runtimes."""
    directory = Path(plan['reused_sources']['run']).parent
    sys.path.insert(0, str(directory.parent))
    sys.path.insert(0, str(directory))
    # Generic imports inside the original E020 code must resolve to its own
    # frozen files, never a preloaded unrelated contract/client helper.
    for name in ('contract', 'launch', 'client', 'resources_v2', 'resources', 'resource_base', 'suspended_service'):
        expected = directory / (name + '.py') if name in ('contract', 'launch', 'client') else directory.parent / (name + '.py')
        if name in sys.modules:
            assert Path(sys.modules[name].__file__).resolve() == expected.resolve(), name
    spec = importlib.util.spec_from_file_location('e022_frozen_e020_lifecycle', plan['reused_sources']['run'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def launch_spec(plan, e020, arm, out, port):
    assert arm in ('off', 'on')
    out = Path(out)
    command = [e020['runtime_versions']['controller-native']['python'], '-B', str(HERE / 'serve.py'),
        '--server-source', plan['reused_sources']['native_server'],
        '--server-source-sha256', plan['native_server_sha256'], '--tcp-nodelay', arm,
        '--socket-receipt', str(out / 'socket.json'), '--socket-failure', str(out / 'socket-failure.json'),
        '--model', e020['workload']['model_dir'], '--port', str(port),
        '--package-root', e020['native_source_root'], '--source-manifest', e020['native_source_manifest']]
    return {'command': command, 'cwd': str(ROOT),
        'environment': dict(e020['environment'], HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false'),
        'files': {}, 'symlinks': {}}


def summarize(rows):
    keys = [(r['pair'], r['arm']) for r in rows]
    assert len(keys) == len(set(keys)), 'duplicate fresh arm'
    assert all({'pair': pair, 'arm': arm} in CONFIGS for pair, arm in keys)
    pairs = []
    for pair in (1, 2, 3):
        found = {r['arm']: r for r in rows if r['pair'] == pair}
        off, on = found.get('off'), found.get('on')
        valid = bool(off and on and off['valid'] and on['valid'])
        pairs.append({'pair': pair, 'valid': valid,
            'wall_speedup': off['median_wall_s'] / on['median_wall_s'] if valid else None,
            'TTFT_speedup': off['median_TTFT_s'] / on['median_TTFT_s'] if valid else None,
            'throughput_speedup': on['median_output_tok_s'] / off['median_output_tok_s'] if valid else None})
    complete = len(rows) == 6
    qualified = complete and all(r['valid'] for r in rows)
    gm = statistics.geometric_mean(p['wall_speedup'] for p in pairs) if qualified else None
    advance = bool(qualified and gm >= GATE['paired_wall_geomean_min']
        and all(p['wall_speedup'] > GATE['every_paired_wall_speedup_strictly_above'] for p in pairs)
        and all(p['TTFT_speedup'] >= GATE['every_paired_TTFT_speedup_min'] for p in pairs))
    return {'rows': rows, 'pairs': pairs, 'matrix_complete': complete,
        'all_correctness_resource_reap_checks': qualified, 'paired_wall_geomean': gm,
        'advance': advance, 'gate': GATE,
        'scope': 'Transport-only C1 19->32: same E020 payload/core/client, no model speedup or historical-gap causal claim.'}
