"""Replay the fixed D fixtures' Lithos prefix probes without loading a model.

Run with lithos-metal/.venv/bin/python -I -B. Only protocol, Jinja and tokenizer
code executes; serve/generate/runtime are source-pinned, not imported. This is
CPU input/source evidence, not device qualification or a runtime cache receipt.
An existing --out is never overwritten. Without --out, print the full receipt.
"""
import argparse
import hashlib
import importlib.abc
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys


REPO = Path('/Users/ryanli/code/minivllm')
LITHOS = Path('/Users/ryanli/code/lithos-metal')
COMMIT = '223658090ae549e59220bdec459957cbdad7926a'
FIXTURE = 'bench/m5_ultra_256gb/pareto_realistic/fixtures/prepared-D-v1/fixtures.json'
FIXTURE_SHA = '6a4f7ce1578a0c6fead596e8b3d646033d90a2e2b843263747b89a3c6b3c88db'
LITHOS_PINS = {
    'monolith/__init__.py': 'c3ac80466fc80a68646b069ac4374582961b223dfb3a7ddd1d863c61a413921a',
    'monolith/serving/__init__.py': '33ebf796e2573fbb73025bd1231e93b44f1a7348266dec63caf11e2162125f7e',
    'monolith/serve.py': '9759ee14c4860ed43218118d6a3eafbb680e9e722e3ddad72bae52473579f419',
    'monolith/generate.py': '434ac1398c6010b37fa44f138505d5ca1757f4e7b747bc6b582d386e2e3ba3b4',
    'monolith/serving/protocol.py': '5f015e2c55037bd0ae73f8b33e7eb64adf8fcb67f19f0f1dc2d25157b1f83ada',
    'monolith/serving/events.py': 'aaa11acb7cb058f55916365e94a0b45b160f286356ffd1d610f08fbaf67b316d',
    'monolith/runtime/prefix_cache.py': '2ffa0f9bfed792765d27af811f8df3d08f9c6b5709c05d6b1e996ba0721297ad',
}
FORBIDDEN = ('mlx', 'torch', 'transformers', 'monolith.runtime',
             'monolith.generate', 'monolith.serve')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def body_sha(body):
    # Same canonical JSON bytes as the frozen E023 client's body contract.
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


class NoModelImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in FORBIDDEN):
            raise ImportError('CPU prefix replay forbids ' + fullname)
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    require(args.out is None or not args.out.exists(), '--out must be a new file')

    fixture_path = REPO / FIXTURE
    require(sha(fixture_path) == FIXTURE_SHA, 'fixed D fixture changed')
    fixtures = json.loads(fixture_path.read_text())
    require(fixtures['schema'] == 'E023-D-inputs-v1' and len(fixtures['fixtures']) == 8,
            'expected the original eight D tasks')
    expected = {str(LITHOS / name): value for name, value in LITHOS_PINS.items()}
    expected[str(fixture_path)] = FIXTURE_SHA
    expected.update(fixtures['pins'])
    for path, digest in expected.items():
        require(sha(path) == digest, 'source/input pin mismatch: ' + path)
    git_head = subprocess.check_output(['git', '-C', str(LITHOS), 'rev-parse', 'HEAD'], text=True).strip()
    git_status = subprocess.check_output(['git', '-C', str(LITHOS), 'status', '--porcelain'], text=True)
    require(git_head == COMMIT and not git_status, 'Lithos checkout must be the pinned clean commit')
    expected[str(Path(__file__).resolve())] = sha(__file__)

    sys.meta_path.insert(0, NoModelImports())
    sys.path.insert(0, str(LITHOS))
    import monolith.serving.protocol as protocol
    import jinja2
    import tokenizers
    require(Path(protocol.__file__).resolve() == (LITHOS / 'monolith/serving/protocol.py').resolve(),
            'wrong actual protocol module')
    env = jinja2.Environment()

    def template_error(message):
        raise ValueError(message)

    env.globals['raise_exception'] = template_error
    model = Path(fixtures['model'])
    template = env.from_string((model / 'chat_template.jinja').read_text())
    tokenizer = tokenizers.Tokenizer.from_file(str(model / 'tokenizer.json'))

    def render_ids(messages, generation_prompt):
        rendered = template.render(messages=messages, enable_thinking=False,
                                   add_generation_prompt=generation_prompt)
        return rendered, tokenizer.encode(rendered, add_special_tokens=False).ids

    rows = []
    served = 'Qwen3.8-27B-nvfp4'
    for item in fixtures['fixtures']:
        require(len(item['messages']) == 1 and item['messages'][0]['role'] == 'user'
                and type(item['messages'][0]['content']) is str,
                'proof only covers one string user message, without tools/system content')
        require(item['max_output_tokens'] == 512 and item['temperature'] == 0
                and item['normal_eos'] is True
                and item['template_options'] == {'add_generation_prompt': True, 'enable_thinking': False},
                'D semantic contract changed')
        chat_body = {'model': served, 'messages': item['messages'], 'stream': True,
                     'stream_options': {'include_usage': True}, 'max_tokens': 512,
                     'temperature': 0, 'top_p': 1, 'top_k': 0, 'seed': 0}
        ollama_body = {'model': served, 'messages': item['messages'], 'stream': True,
                       'options': {'num_predict': 512, 'temperature': 0, 'top_p': 1, 'top_k': 0, 'seed': 0}}
        requests = [('/v1/chat/completions', chat_body, protocol.ChatRequest(**chat_body)),
                    ('/api/chat', ollama_body, protocol.ollama_request(ollama_body, served))]
        for path, body, request in requests:
            messages, tools = request.template_inputs()
            require(not tools and request.token_limit == 512 and request.stop is None,
                    'unexpected tools/output/EOS setting')
            rendered, ids = render_ids(messages, True)
            require(rendered == item['prompt_text'] and ids == item['prompt_ids'],
                    'actual protocol conversion differs from frozen prompt')
            stable, probes = set(), []
            # Exact serve.py probe control flow, with real template_inputs().
            seen = []
            for mode in ('message', True):
                prefix_messages, _ = request.template_inputs(cache_prefix=mode)
                if prefix_messages in seen:
                    continue
                seen.append(prefix_messages)
                prefix_text, prefix_ids = render_ids(prefix_messages, False)
                count = 0
                for a, b in zip(ids, prefix_ids):
                    if a != b:
                        break
                    count += 1
                if count:
                    stable.add(count)
                probes.append({'text': prefix_text, 'token_ids': prefix_ids, 'matching_tokens': count})
            cache_prefix_tokens = sorted(stable) if len(stable) > 1 else next(iter(stable), 0)
            # Exact generate.py checkpoint selection at fresh offset=0. Crucial:
            # a valid candidate below min_tokens suppresses the p-1 fallback.
            candidates = [cache_prefix_tokens] if isinstance(cache_prefix_tokens, int) else cache_prefix_tokens
            candidates = [n for n in candidates if 0 < n < len(ids)] or [len(ids) - 1]
            checkpoints = [n for n in candidates if 0 < n < len(ids) and n >= 128]
            require(stable == {3} and candidates == [3] and checkpoints == [],
                    'fixture no longer has the proved empty-checkpoint property')
            rows.append({'fixture_id': item['fixture_id'], 'path': path,
                         'body_sha256': body_sha(body), 'prompt_tokens': len(ids),
                         'prompt_text_and_ids_equal_frozen_fixture': True,
                         'probes': probes, 'stable_prefix_tokens': sorted(stable),
                         'candidate_checkpoints': candidates, 'saved_checkpoints': checkpoints})

    imported_forbidden = [name for name in sys.modules if any(
        name == blocked or name.startswith(blocked + '.') for blocked in FORBIDDEN)]
    require(not imported_forbidden, 'unexpected model/runtime module import')
    for path, digest in expected.items():
        require(sha(path) == digest, 'source/input changed during replay: ' + path)
    result = {
        'schema': 'Lithos-fixed-D-prefix-CPU-v1', 'status': 'CPU_SOURCE_REPLAY_PASS',
        'device_qualified': False, 'performance_measured': False,
        'lithos_commit': git_head, 'lithos_clean': True,
        'python': {'executable': sys.executable, 'version': sys.version},
        'versions': {name: importlib.metadata.version(name)
                     for name in ('lithos-metal', 'jinja2', 'tokenizers', 'pydantic')},
        'actual_protocol_module': protocol.__file__,
        'sources_sha256': expected, 'source_pins_rechecked': True,
        'rows': rows, 'fixture_count': 8, 'protocol_fixture_checks': 16,
        'forbidden_modules_imported': imported_forbidden,
        'server_contract': {'prefill_chunk_size': '128-exact', 'prefix_cache_min_tokens': 128,
            'fresh_process': True, 'prefix_cache_dir': None, 'keep_warm': 0, 'no_warmup': True,
            'only_registered_D_requests': True, 'initial_cache_entries': 0},
        'proof_boundary': 'The pinned eight single-user string fixtures yield no checkpoint candidates after the 128-token threshold; a fresh empty memory cache therefore stays empty under their repetition. No general cache-disabled claim.',
        'still_required': 'Device qualification and complete warm/formal Generation logs: exactly one per completed registered request, every cached_prompt_tokens=0, source/config/model pins unchanged. Any nonzero/missing record invalidates the zero-hit claim.',
        'rerun': f'{LITHOS}/.venv/bin/python -I -B {Path(__file__).resolve()} --out NEW.json',
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.out is not None:
        with args.out.open('x') as handle:
            handle.write(encoded)
        print(json.dumps({'out': str(args.out.resolve()), 'sha256': sha(args.out),
                          'status': result['status'], 'protocol_fixture_checks': len(rows)}))
    else:
        print(encoded, end='')


if __name__ == '__main__':
    main()
