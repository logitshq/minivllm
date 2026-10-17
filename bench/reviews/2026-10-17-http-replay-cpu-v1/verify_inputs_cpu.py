"""Reproduce real D-body/parser/tokenizer checks; never starts HTTP or a model."""
import argparse
import hashlib
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'bench/m5_ultra_256gb/pareto_realistic'


class NoBackend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mlx', 'torch', 'transformers'}:
            raise AssertionError('CPU-only input check forbids ' + fullname)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists() or not args.out.parent.is_dir():
        raise ValueError('new output file with existing parent required')
    sys.meta_path.insert(0, NoBackend())
    m = load(BASE / 'http_qualification.py', '_http_input_cpu_adapter')
    server = load(BASE / 'native_server.py', '_http_input_cpu_native')
    if m.sha(BASE / 'native_server.py') != m.FIXED[BASE / 'native_server.py']:
        raise ValueError('frozen server changed')
    plan_path = BASE / 'prereg-qualification-v1/plan.json'
    if m.sha(plan_path) != m.BASE_PLAN_SHA:
        raise ValueError('fixed numerical plan changed')
    p = m.read(plan_path)
    fixtures = m.read(m.pinned(p['fixtures']))['fixtures']
    source, model = [m.read(m.pinned(p[k])) for k in ('source_manifest', 'model_manifest')]
    tokenizer_path = Path(p['model']) / 'tokenizer.json'
    if m.sha(tokenizer_path) != model['files']['tokenizer.json']:
        raise ValueError('fixed tokenizer changed')
    import tokenizers
    tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))
    contracts = []
    for c in (1, 8):
        rows = []
        for i, job in enumerate(m.jobs_for(fixtures, c)):
            ids, cap = server.parse_request(job['body'], tokenizer, p['configuration']['max_len'])
            if ids != fixtures[i % 8]['prompt_ids'] or cap != 512:
                raise ValueError('actual parser input differs from prepared fixture')
            raw = json.dumps(job['body'], sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
            rows.append({'request_id': job['request_id'], 'fixture_id': job['fixture_id'], 'path': job['path'],
                         'body_sha256': hashlib.sha256(raw).hexdigest(), 'prompt_tokens': len(ids), 'cap': cap,
                         'prompt_ids_sha256': hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()})
        contracts.append({'concurrency': c, 'phase': 'qualification', 'requests': rows,
            'expected_startup_identity': m.expected_identity(p, fixtures, source, model, c,
                {'tcp_nodelay': False, 'io_timeout_seconds': 300})})
    if {'mlx', 'torch', 'transformers'}.intersection(sys.modules):
        raise AssertionError('model backend imported')
    m.write(args.out, {'schema': 'E023-native-recorded-http-input-cpu-v2', 'status': 'CPU_INPUTS_MATCH',
        'scope': '24 real D bodies through frozen parser/pinned tokenizer only; no HTTP/model/GPU or numerical/performance qualification.',
        'generator_sha256': m.sha(__file__), 'parser_sha256': m.sha(BASE / 'native_server.py'),
        'adapter_sha256': m.sha(BASE / 'http_qualification.py'), 'fixture_sha256': p['fixtures']['sha256'],
        'tokenizer_json_sha256': m.sha(tokenizer_path), 'tokenizers_version': tokenizers.__version__,
        'backend_imports': [], 'contracts': contracts})


if __name__ == '__main__':
    main()
