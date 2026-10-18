"""Tiny CPU mechanism checks; no real MLX, model, service or numerical claim."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('e024_qualify_cpu', HERE / 'qualify.py')
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)
SUPPORT = HERE.parent / 'pareto_realistic/qualify_native.py'
s = q.load_pinned({'path': str(SUPPORT), 'sha256': q.SUPPORT_SHA}, 'e024_test_comparator', q.SUPPORT_SHA)


class CPU:
    bfloat16 = np.float16
    def __getattr__(self, name):
        return getattr(np, name)
    def eval(self, *args):
        pass
    def save_safetensors(self, path, arrays):
        Path(path).write_bytes(b'CPU MOCK ONLY')
    def array_equal(self, a, b):
        return np.bool_(np.array_equal(a, b))


class Graph:
    def __init__(self):
        self._functions = {}
        self.eligible, self.fallback, self.raise_inside = True, False, False
    def _eligible(self, *args):
        return self.eligible
    def __call__(self, tokens, ctx):
        if self.fallback:
            return 'fallback'
        if 'key' not in self._functions:
            def compiled(ids, context):
                if self.raise_inside:
                    raise RuntimeError('compiled failure')
                return ids + 1
            self._functions['key'] = compiled
        return self._functions['key'](tokens, ctx)


def context(backing=1536, used=1024, B=1):
    return NS(positions=np.array([used - 1] * B), lens=np.array([used] * B), max_len=used,
        caches=[NS(K=NS(shape=(B, 2, backing, 4)), V=NS(shape=(B, 2, backing, 4)))])


class QualificationMechanisms(unittest.TestCase):
    def test_actual_callable_count_and_batch_reuse(self):
        events, graph = [], Graph()
        witness = q.GraphWitness(graph, events.append)
        for B in (1, 8, 1):
            self.assertEqual(witness(np.zeros(B, np.int32), context(B=B)).tolist(), [1] * B)
        self.assertEqual(witness.functions.invocations, 3)
        self.assertEqual(len(witness.functions.compiled_objects), 1)
        self.assertEqual([r['batch'] for r in events], [1, 8, 1])
        self.assertTrue(all(r['status'] == 'SUBMITTED' for r in events))

    def test_underallocated_or_ineligible_or_silent_fallback_fail(self):
        for reason in ('backing', 'eligible', 'fallback'):
            graph = Graph()
            graph.eligible = reason != 'eligible'
            graph.fallback = reason == 'fallback'
            witness = q.GraphWitness(graph, lambda row: None)
            with self.subTest(reason=reason), self.assertRaises(RuntimeError):
                witness(np.array([1]), context(backing=1280 if reason == 'backing' else 1536))
            self.assertEqual(witness.functions.invocations, 0)

    def test_wrong_positions_lens_and_used_reject(self):
        for field, value in (('lens', np.array([1023])), ('max_len', 1025), ('positions', np.array([-1]))):
            ctx = context()
            setattr(ctx, field, value)
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                q.GraphWitness(Graph(), lambda row: None)(np.array([1]), ctx)

    def test_compiler_failure_retains_running_event_not_success(self):
        graph, events = Graph(), []
        graph.raise_inside = True
        witness = q.GraphWitness(graph, events.append)
        with self.assertRaisesRegex(RuntimeError, 'compiled failure'):
            witness(np.array([1]), context())
        self.assertEqual(events[0]['status'], 'RUNNING')
        self.assertEqual(witness.rows, [])

    def test_diagnostic_inputs_are_exact_repeat_and_explicitly_labeled(self):
        fixtures = [{'fixture_id': str(i), 'prompt_ids': [i, i + 1]} for i in range(8)]
        result = q.diagnostic_fixtures(fixtures, 5)
        self.assertEqual(result[0]['prompt_ids'], [0, 1, 0, 1, 0])
        self.assertTrue(all(f['diagnostic_only'] for f in result))
        self.assertEqual(fixtures[0]['prompt_ids'], [0, 1])
        self.assertEqual(sum([1535, 1533]) * 8, 24544)
        with self.assertRaises(ValueError):
            q.diagnostic_fixtures(fixtures[:7], 5)

    def test_normal_eos_does_not_waive_actual_ahead_coverage(self):
        rows = [{'phase': 'lifecycle', 'batch': 8, 'used': x} for x in (1534, 1535, 1536)]
        steps = [{'case': 'E024-diagnostic-ahead', 'ahead': {'candidate': {'consumed_prior_future': ['key'], 'carried_future': ['next']}}}]
        q.check_lifecycle_coverage(rows, steps)
        for bad_rows, bad_steps in ((rows[:2], steps), (rows, []),
                ([{**r, 'batch': 1} for r in rows], steps)):
            with self.subTest(rows=bad_rows), self.assertRaises(RuntimeError):
                q.check_lifecycle_coverage(bad_rows, bad_steps)

    def test_changed_parameter_encodings_and_exception_restore(self):
        mx = CPU()
        originals = (np.array([0, .5], np.float32), np.array([[0, 0x12345678], [1, 2]], np.uint32),
                     np.array([[16, 32], [64, 80]], np.uint8))
        with tempfile.TemporaryDirectory() as out:
            cmp = s.Comparator(mx, out)
            for index, value in enumerate(originals):
                models = [NS(parameter=value.copy()), NS(parameter=value.copy())]
                saved = [m.parameter for m in models]
                events = []
                with self.subTest(index=index), self.assertRaisesRegex(RuntimeError, 'injected'):
                    with q.parameter_replacement(mx, models, 'parameter', index, cmp, s.snapshot, events.append):
                        self.assertIsNot(models[0].parameter, models[1].parameter)
                        self.assertFalse(np.array_equal(models[0].parameter, saved[0]))
                        np.testing.assert_array_equal(models[0].parameter, models[1].parameter)
                        if index == 2:
                            self.assertEqual(sorted(models[0].parameter.flat), sorted(saved[0].flat))
                        raise RuntimeError('injected')
                self.assertTrue(all(m.parameter is old for m, old in zip(models, saved)))
                self.assertTrue(events[-1]['both'])

    def test_noop_scales_and_shared_parameters_reject(self):
        with tempfile.TemporaryDirectory() as out:
            cmp = s.Comparator(CPU(), out)
            shared = np.array([[1], [2]], np.uint8)
            for models in ([NS(p=shared), NS(p=shared)], [NS(p=np.ones((2, 1), np.uint8)) for _ in range(2)]):
                with self.assertRaises(RuntimeError):
                    with q.parameter_replacement(CPU(), models, 'p', 2, cmp, s.snapshot, lambda row: None):
                        self.fail('invalid replacement entered')

    def test_retained_prefill_handles_survive_later_cache_replacement(self):
        caches = [[[NS(keys=np.array([arm, row]), offset=1023)] for row in range(8)] for arm in range(2)]
        retained = q.retain_prefill_caches(caches, [None, None], lambda model: [NS(keys=None, offset=0)])
        self.assertEqual([len(arm) for arm in retained], [8, 8])
        for a in range(2):
            for row in range(8):
                self.assertIsNot(retained[a][row][0], caches[a][row][0])
                self.assertIsNot(retained[a][row][0].keys, caches[a][row][0].keys)
                caches[a][row][0].keys = np.array([99])
                caches[a][row][0].offset = 1535
                np.testing.assert_array_equal(retained[a][row][0].keys, [a, row])
                self.assertEqual(retained[a][row][0].offset, 1023)

    def test_pinned_e023_comparator_still_rejects_bit_mismatch(self):
        with tempfile.TemporaryDirectory() as out:
            cmp = s.Comparator(CPU(), out)
            with self.assertRaises(s.QualificationFailure):
                cmp.compare('signedzero', s.snapshot(np.array([0.], np.float32)), s.snapshot(np.array([-0.], np.float32)))
            self.assertTrue((Path(out) / 'failure.json').is_file())

    def test_fixed_source_corruption_and_deadline_bounds(self):
        with tempfile.TemporaryDirectory() as out:
            path = Path(out) / 'source.py'
            path.write_text('raise AssertionError("should never execute")\n')
            with self.assertRaises(ValueError):
                q.load_pinned({'path': str(path), 'sha256': q.SUPPORT_SHA}, 'bad', q.SUPPORT_SHA)
        plan = dict(schema='E024-diagnostic-qualification-v1', qualifier_sha256=q.digest(q.__file__),
                    e023_plan={}, e023_qualifier={}, candidate={}, deadline_s=1800)
        self.assertEqual(q.validate_plan(plan), 1800)
        for bad in (0, -1, True, float('nan'), 1801):
            with self.subTest(budget=bad), self.assertRaises(ValueError):
                q.validate_plan({**plan, 'deadline_s': bad})

    def test_help_and_unknown_arguments_never_import_gpu_or_create_socket(self):
        code = '''
import importlib.abc, runpy, socket, sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, name, *args):
  if name.split('.')[0] in {'mlx', 'tutorial_code', 'numpy'}: raise AssertionError(name)
sys.meta_path.insert(0, Block())
socket.socket = lambda *a, **k: (_ for _ in ()).throw(AssertionError('socket'))
sys.argv = [sys.argv[1]] + sys.argv[2:]
runpy.run_path(sys.argv[0], run_name='__main__')
'''
        for args, expected in ((['--help'], 0), (['--unknown'], 2)):
            result = subprocess.run([sys.executable, '-B', '-c', code, str(HERE / 'qualify.py'), *args], capture_output=True, text=True)
            self.assertEqual(result.returncode, expected, result.stderr)
            self.assertNotIn('AssertionError', result.stderr)


if __name__ == '__main__':
    unittest.main()
