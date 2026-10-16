"""CPU doubles/source checks only; no MLX/NumPy/GPU/resource probes."""
import ast
import importlib.util
from pathlib import Path
import sys
import unittest

D = Path(__file__).resolve().parent

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

m = load('v23_micro_cpu', D / 'micro.py')
t = load('v23_timing_cpu', D / 'timing_block.py')

class Tests(unittest.TestCase):
    def test_numeric_pass_and_sources_bound(self):
        value = m.identity(m.NUMERIC_RECEIPT)
        self.assertEqual(value['numeric_eligibility']['sum_y'], 4)
        self.assertEqual(value['numeric_eligibility']['cases'], 6)
        self.assertFalse(value['numeric_eligibility']['real_preconv_qualified'])
        self.assertFalse(any(n == 'mlx' or n.startswith('mlx.') for n in sys.modules))

    def test_equal_prime_repeat_and_all_call_indices(self):
        for variant in ('A', 'B'):
            events = []
            class MX:
                def synchronize(self): events.append('sync')
            def invoke(v, j):
                self.assertEqual(v, variant)
                events.append(j)
                return ('act', 'new_tail')
            record, output = t.block(MX(), invoke, variant)
            self.assertEqual(events, list(range(128)) + ['sync'] + list(range(4096)) + ['sync'])
            self.assertEqual(output, ('act', 'new_tail'))
            self.assertEqual(record['prime_repeats'], 128)
            self.assertEqual(record['repeats'], 4096)
            self.assertEqual(record['segments_per_call'], 2)
            self.assertLessEqual(record['prime_end_ns'], record['start_ns'])
            self.assertEqual([j % 8 for j in range(m.REPEATS)].count(7), 512)

    def test_symmetric_schedule_and_strict_controls(self):
        self.assertEqual((m.PRIME, m.REPEATS), (128, 4096))
        self.assertEqual(len(m.BANK_NAMES), 8)
        self.assertEqual(m.SCHEDULE.count(('A', 'B')), 3)
        self.assertEqual(m.SCHEDULE.count(('B', 'A')), 3)
        self.assertEqual(m.SCHEDULE.count(('A', 'A')), 2)
        self.assertEqual(m.SCHEDULE.count(('B', 'B')), 2)
        self.assertTrue(m.control_passes({'A': [.98, 1.02], 'B': [1., 1.]}))
        self.assertFalse(m.control_passes({'A': [.97, .97], 'B': [1., 1.]}))
        self.assertFalse(m.control_passes({'A': [.949, 1.051], 'B': [1., 1.]}))

    def test_timed_blocks_adjacent_and_both_outputs_consumed(self):
        tree = ast.parse((D / 'micro.py').read_text())
        gpu = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'gpu')
        loop = next(n for n in ast.walk(gpu) if isinstance(n, ast.For) and ast.unparse(n.iter) == 'enumerate(SCHEDULE)')
        positions = [i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == 'timing.block']
        self.assertEqual(len(positions), 2)
        self.assertEqual(positions[1], positions[0] + 1)
        invoke = next(n for n in ast.walk(gpu) if isinstance(n, ast.FunctionDef) and n.name == 'invoke')
        self.assertIn('mx.eval(*outputs)', ast.unparse(invoke))
        self.assertNotIn('hash', ast.unparse(invoke))
        self.assertNotIn('probe(', ast.unparse(invoke))
        self.assertNotIn('clear_cache(', (D / 'micro.py').read_text())

if __name__ == '__main__':
    unittest.main(verbosity=2)
