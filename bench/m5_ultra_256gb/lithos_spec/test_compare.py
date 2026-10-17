"""CPU-only corruption checks; never imports the Lithos runtime."""
import builtins
import copy
import hashlib
import json
from pathlib import Path
import runpy
import sys
import struct
import tempfile
import unittest
from unittest.mock import patch

from compare import compare


def receipts(root):
    paths = [root / mode for mode in ('none', 'dspark')]
    for pid, (mode, path) in enumerate(zip(('none', 'dspark'), paths), 1):
        path.mkdir()
        spec = mode == 'dspark'
        plan = {key: {} for key in ('target_identity', 'drafter_identity', 'source_sha256', 'harness_sha256', 'environment')}
        plan.update(mode=mode, pid=pid, prompt_ids=list(range(19)), workload_sha256='workload', lithos_head='same')
        generation = {'tokens': list(range(100, 132)), 'cached_prompt_tokens': 0, 'decode_tokens': 31,
            'prefill_timings': [{'tokens': 19}], 'steps': 4 if spec else 31,
            'accepted': [7, 7, 7, 6] if spec else None, 'committed': [8, 8, 8, 7] if spec else None,
            'verify_len': [7] * 4 if spec else None}
        words = [(c << 16) | (7 << 8) | a for a, c in zip(generation['accepted'], generation['committed'])] if spec else None
        state = {'error': 0, 'done': 1, 'stop_at': 32, 'ring_head': 32, 'step': 5 if spec else 32}
        route = {'device': {'name': 'Apple M5 Ultra', 'gpu_cores': 80}, 'backend': 'm5_ultra_80c',
            'recipes': {}, 'recipe_key': None, 'capacity': 2054 if spec else 2048,
            'prefix_cache': False, 'prefill_chunk_size': 128, 'requested_prefill_exact': True,
            'effective_prefill_exact': False, 'gamma': 7 if spec else 0, 'has_drafter': spec,
            'verify': 'fixed' if spec else 'cost', 'verify_length': 7 if spec else None,
            'drafter_class': 'monolith.spec.dspark.model.DSparkDrafter' if spec else None}
        program_data = b'{"fixture":true}'
        (path / 'program-0.json').write_bytes(program_data)
        program = {'index': 0, 'prefill': False, 'backend': 'm5_ultra_80c', 'bound': 8 if spec else 1, 'dynamic': spec,
            'kernel_functions': ['accept_scan', 'verify_select'] if spec else ['advance'],
            'draft_ops': 1 if spec else 0, 'op_count': 10, 'program_sha256': hashlib.sha256(program_data).hexdigest()}
        result = {'status': 'PASS_ARM_REQUIRES_FRESH_CONTROL', 'mode': mode, 'route': route,
            'packs': {'target': {'quantize': None}, 'draft': {'quantize': None} if spec else None},
            'programs': [program], 'pins_after_match': True,
            'requests': [{'index': i, 'prompt_ids': list(range(19)), 'generation': copy.deepcopy(generation),
                          'step_state': copy.deepcopy(state), 'eos_in_returned_prefix': False,
                          'accept_log_rows': copy.deepcopy(words)} for i in range(3)]}
        (path / 'plan.json').write_text(json.dumps(plan))
        (path / 'result.json').write_text(json.dumps(result))
        (path / 'cleanup.json').write_text(json.dumps({'released_engines': True, 'buffers_cleared': True, 'last_engine_cleared': True}))
        if spec:
            for i in range(3):
                (path / f'accept-log-{i}.bin').write_bytes(struct.pack('<5I', 1 << 16, *words))
    return paths


class ReceiptContract(unittest.TestCase):
    def run_case(self, mutation=None):
        with tempfile.TemporaryDirectory() as temp:
            control, candidate = receipts(Path(temp))
            if mutation:
                path = candidate / 'result.json'
                value = json.loads(path.read_text())
                mutation(value)
                path.write_text(json.dumps(value))
                with self.assertRaises(AssertionError):
                    compare(control, candidate)
            else:
                self.assertEqual(compare(control, candidate)['raw_ids_compared'], 192)

    def test_valid(self):
        self.run_case()

    def test_wrong_ids(self):
        self.run_case(lambda value: value['requests'][0]['generation']['tokens'].__setitem__(0, 999))

    def test_soft_fallback(self):
        self.run_case(lambda value: value['route'].__setitem__('has_drafter', False))

    def test_wrong_accept_log(self):
        self.run_case(lambda value: value['requests'][0]['accept_log_rows'].__setitem__(0, 0))

    def test_counter_overclaim(self):
        self.run_case(lambda value: value['requests'][0]['step_state'].__setitem__('ring_head', 40))

    def test_changed_verification_length(self):
        self.run_case(lambda value: value['requests'][0]['generation']['verify_len'].__setitem__(0, 3))

    def test_borrowed_backend(self):
        self.run_case(lambda value: value['route'].__setitem__('backend', 'm5_max_40c'))

    def test_wrong_prompt(self):
        self.run_case(lambda value: value['requests'][0]['prompt_ids'].__setitem__(0, 999))

    def test_help_and_unknown_flags_do_not_import_runtime(self):
        original_import = builtins.__import__
        def no_runtime(name, *args, **kwargs):
            if name.startswith(('monolith', 'numpy', 'mlx')):
                raise AssertionError('CLI imported GPU/runtime dependency: ' + name)
            return original_import(name, *args, **kwargs)
        script = Path(__file__).with_name('qualify.py')
        required = ['--mode', 'none', '--workload', 'x', '--drafter', 'x', '--download-receipt', 'x',
                    '--source-root', 'x', '--pack-root', 'x', '--out', 'x']
        for arguments, code in ((['--help'], 0), (required + ['--not-a-real-flag'], 2)):
            with patch.object(sys, 'argv', [str(script)] + arguments), patch('builtins.__import__', no_runtime):
                with self.assertRaises(SystemExit) as result:
                    runpy.run_path(str(script), run_name='__main__')
                self.assertEqual(result.exception.code, code)


if __name__ == '__main__':
    unittest.main()
