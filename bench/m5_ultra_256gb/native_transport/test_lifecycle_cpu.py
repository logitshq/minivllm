"""Use the actual frozen E020 lifecycle with only fake process/socket/client IO.

This local provenance smoke test needs the E020 prepared source snapshot. It
opens no listener, starts no subprocess, and imports no MLX or real tokenizer.
"""
import importlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from common import DEFAULT_E020, ROOT, frozen_e020, launch_spec, load_lifecycle, read


@unittest.skipUnless(DEFAULT_E020.exists(), 'local frozen E020 plan unavailable')
class FrozenLifecycleTests(unittest.TestCase):
    def test_frozen_lifecycle_launch_and_six_requests_without_listener(self):
        e020 = read(DEFAULT_E020)
        original = ROOT / 'bench/m5_ultra_256gb/external_spec_http'
        reused = {name: str(frozen_e020(DEFAULT_E020, original / (name + '.py')))
                  for name in ('run', 'client', 'contract', 'launch', 'native_server')}
        plan = {'reused_sources': reused, 'native_server_sha256': e020['pins'][str(original / 'native_server.py')]}
        names = ('contract', 'launch', 'client')
        previous = {name: sys.modules.get(name) for name in names}
        previous_path = sys.path[:]
        try:
            lifecycle = load_lifecycle(plan)
            client = importlib.import_module('client')
            for name in names:
                self.assertEqual(Path(sys.modules[name].__file__).resolve(), Path(reused[name]).resolve())
            self.assertEqual(Path(lifecycle.__file__).resolve(), Path(reused['run']).resolve())
            files = read(e020['native_source_manifest'])['files']
            identity = {'manifest_all_sources_match': True, 'step_observer': False,
                'cache_limit_gib': 2, 'compile_decode': True, 'fuse_decode': True,
                'drafter': None, 'normal_eos': True, 'source_files': files}
            class HTTP:
                def __init__(self, **kwargs): pass
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def get(self, path):
                    return types.SimpleNamespace(status_code=200, json=lambda: {
                        'data': [{'id': 'native-current-target-only'}], 'benchmark_identity': identity})
            class Socket:
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def bind(self, address): pass
                def getsockname(self): return ('127.0.0.1', 12345)
            class Child:
                pid = 8675309
                def poll(self): return None
            launched, sampled, request_ids = [], [], []
            def popen(command, **kwargs):
                launched.append((command, kwargs))
                return Child()
            def request(base, headers, path, body, tok, request_id):
                request_ids.append(request_id)
                return {'request_id': request_id, 'http_status': 200, 'done': True,
                    'finish_reason': 'length', 'usage': {'prompt_tokens': 19, 'completion_tokens': 32},
                    'visible_roundtrip_ids': e020['oracle_ids']['native'],
                    'output_ids_when_available': e020['oracle_ids']['native'],
                    'first_text_perf_counter_s': 1, 'first_chunk_prefix_verified': True,
                    'cached_prompt_tokens_when_available': 0, 'error': None,
                    'client_total_s': 1, 'client_ttft_text_s': .1, 'output_tok_s': 32,
                    'client_estimated_interactivity_tok_s': 31/.9}
            def listeners(port):
                return {'pids': [Child.pid] if launched else []}
            def close(child, port):
                return {'valid': True, 'pid': child.pid, 'waited_reaped': True}
            fake_modules = {'httpx': types.SimpleNamespace(Client=HTTP, HTTPError=OSError),
                'tokenizers': types.SimpleNamespace(Tokenizer=types.SimpleNamespace(from_file=lambda path: object()))}
            with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, fake_modules), \
                 patch.object(lifecycle.socket, 'socket', Socket), \
                 patch.object(lifecycle.subprocess, 'Popen', side_effect=popen), \
                 patch.object(lifecycle, 'listeners', side_effect=listeners), \
                 patch.object(lifecycle, 'ps', return_value={'returncode': 0, 'stdout': 'mock', 'stderr': ''}), \
                 patch.object(lifecycle, 'lease_check'), \
                 patch.object(lifecycle, 'sample_resources', side_effect=lambda *args: sampled.append(args[2])), \
                 patch.object(lifecycle, 'close_server', side_effect=close), \
                 patch.object(lifecycle.time, 'sleep'), patch.object(client, 'request', side_effect=request):
                for arm in ('off', 'on'):
                    launched.clear()
                    output = Path(tmp) / arm
                    output.mkdir()
                    lifecycle.specification = lambda parent, legacy_arm, out, port, key: launch_spec(plan, e020, arm, out, port)
                    result = lifecycle.run_one(e020, {'pair': 1, 'arm': 'native-none'}, output, Path('unused'))
                    self.assertTrue(result['valid'], result)
                    self.assertEqual(result['request_count'], 6)
                    self.assertEqual(result['formal_count'], 3)
                    command, kwargs = launched[0]
                    self.assertEqual(command[command.index('--tcp-nodelay') + 1], arm)
                    self.assertEqual(command[command.index('--server-source') + 1], reused['native_server'])
                    self.assertTrue(kwargs['start_new_session'])
                self.assertEqual(len(request_ids), 12)
                self.assertEqual(sampled.count('after-unload'), 2)
        finally:
            sys.path[:] = previous_path
            for name, module in previous.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module


if __name__ == '__main__':
    unittest.main()
