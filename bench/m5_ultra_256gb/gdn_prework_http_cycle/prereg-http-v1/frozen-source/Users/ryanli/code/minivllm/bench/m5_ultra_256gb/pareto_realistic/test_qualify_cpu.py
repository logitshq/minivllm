"""CPU-only corruption/lifecycle tests; never imports mlx or loads a model."""
import io
import json
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest

import numpy as np

from bench.m5_ultra_256gb.pareto_realistic.qualify_native import (
    Comparator, Observer, QualificationFailure, ahead_semantic_key, decode_key,
    pair_calls, run_case, snapshot, workloads,
)
from tutorial_code.miniengine.engine.request import Request
from tutorial_code.miniengine.engine.scheduler import Scheduler
from tutorial_code.miniengine.sampling_params import SamplingParams


class CPU:
    bfloat16 = np.float16  # CPU stand-in only; actual CLI compares MLX original dtype.
    def __getattr__(self, name):
        return getattr(np, name)
    def eval(self, *unused):
        pass
    def save_safetensors(self, path, arrays):
        # Test artifact is clearly a mock, never used by the real qualification CLI.
        Path(path).write_bytes(b'CPU MOCK\n' + b''.join(x.tobytes() for x in arrays.values()))


def record(key, value=1):
    return {'key': key, 'context': {'mode': 'decode', 'max_len': 12, 'lens': [12]},
            'logits': snapshot(np.array([value, 2], dtype=np.float32)),
            'state': snapshot([NS(conv=np.array([value], dtype=np.float32), S=np.array([2], dtype=np.float32))])}


class Model:
    layers = [NS(is_attn=False)]
    def __call__(self, tokens, ctx, **kwargs):
        for cache in ctx.caches:
            cache.conv = cache.conv + 1
            cache.S = cache.S + 2
            if ctx.mode == 'prefill':
                cache.offset += len(tokens)
        batch = len(tokens) if ctx.mode == 'decode' else 1
        logits = np.zeros((batch, 100), dtype=np.float32)
        logits[:, 7] = 1
        return logits


class Batch:
    def __init__(self):
        self.req_ids, self.lens, self.layers = [], [], [None]
    @property
    def size(self):
        return len(self.req_ids)


class Runner:
    def __init__(self):
        self.model = self.decode_model = Model()
        self.batch, self.staging, self.feats, self._ahead = Batch(), {}, {}, None
    def drop_many(self, rids):
        for rid in rids:
            self.staging.pop(rid, None)
        keep = [i for i, rid in enumerate(self.batch.req_ids) if rid not in rids]
        if not keep:
            self.batch.req_ids, self.batch.lens, self.batch.layers = [], [], [None]
        else:
            self.batch.req_ids = [self.batch.req_ids[i] for i in keep]
            self.batch.lens = [self.batch.lens[i] for i in keep]
            self.batch.layers = [NS(conv=x.conv[keep].copy(), S=x.S[keep].copy()) for x in self.batch.layers]
    def run(self, out):
        result = {}
        if out.decode:
            indices = {rid: i for i, rid in enumerate(self.batch.req_ids)}
            conv, S = [], []
            for req in out.decode:
                if req.req_id in indices:
                    i = indices[req.req_id]
                    conv.append(self.batch.layers[0].conv[i])
                    S.append(self.batch.layers[0].S[i])
                else:
                    cache = self.staging.pop(req.req_id)[0]
                    conv.append(cache.conv[0]); S.append(cache.S[0])
            self.batch.req_ids = [r.req_id for r in out.decode]
            self.batch.layers = [NS(conv=np.array(conv, dtype=np.float32), S=np.array(S, dtype=np.float32))]
            positions = np.array([r.num_computed_tokens for r in out.decode], dtype=np.int32)
            context = NS(mode='decode', caches=self.batch.layers, positions=positions, lens=positions + 1,
                         max_len=int(max(positions)) + 1)
            self.decode_model(np.array([r.output_token_ids[-1] for r in out.decode], dtype=np.int32), context)
            self.batch.lens = list(map(int, positions + 1))
            result.update({r.req_id: 7 for r in out.decode})
        for req, n in out.prefill:
            cache = self.staging.setdefault(req.req_id, [NS(conv=np.zeros(1, dtype=np.float32), S=np.zeros(1, dtype=np.float32), offset=0)])
            self.model(np.array(req.all_token_ids[req.num_computed_tokens:req.num_computed_tokens + n]),
                       NS(mode='prefill', caches=cache, positions=None, lens=None, max_len=0), last_only=True)
            if req.num_computed_tokens + n >= req.num_prompt_tokens:
                result[req.req_id] = 7
        return result


class Engine:
    def __init__(self, max_seqs=8):
        self.runner, self.scheduler, self.requests, self.eos_ids = Runner(), Scheduler(max_seqs, 2048, 64, {99}), {}, {99}
    def has_work(self):
        return self.scheduler.has_work()
    def add_request(self, prompt, sampling, req_id):
        req = Request(req_id, list(prompt), sampling)
        self.requests[req_id] = req
        self.scheduler.add(req)
        return req_id
    def step(self):
        out = self.scheduler.schedule()
        self.runner.drop_many(out.finished_ids)
        sampled = self.runner.run(out)
        self.scheduler.update(out, sampled)
        result, finished = [], []
        for rid, token in sampled.items():
            req = self.requests[rid]
            result.append((rid, token, req.is_finished))
            if req.is_finished:
                finished.append(rid)
                self.requests.pop(rid)
        self.runner.drop_many(finished)
        return result


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.cmp = Comparator(CPU(), self.temp.name)
    def tearDown(self):
        self.temp.cleanup()

    def test_raw_bits_distinguish_signed_zero_and_preserve_failure_tensor(self):
        with self.assertRaises(QualificationFailure):
            self.cmp.compare('bits', snapshot(np.array([0.], np.float32)), snapshot(np.array([-0.], np.float32)))
        artifact = json.loads((Path(self.temp.name) / 'failure.json').read_text())
        self.assertFalse(artifact['details']['bit_equal'])
        self.assertIn('array_artifact', artifact)

    def test_shape_dtype_and_nonfinite_fail(self):
        for left, right in ((np.zeros(2, np.float32), np.zeros(3, np.float32)),
                            (np.zeros(2, np.float16), np.zeros(2, np.float32)),
                            (np.array([np.inf], np.float32), np.array([np.inf], np.float32))):
            with self.subTest(left=str(left.dtype), right=str(right.shape)):
                with self.assertRaises(QualificationFailure):
                    self.cmp.compare('corruption', snapshot(left), snapshot(right))

    def test_semantic_ahead_pairs_when_call_counts_differ(self):
        current = decode_key(['r0'], [11], [7])
        future = decode_key(['r0'], [12], [8])
        ref = NS(calls={current: record(current), future: record(future)}, consumed=None)
        cand = NS(calls={future: record(future)}, consumed=current, previous_future=record(current))
        keys = pair_calls(ref, cand, self.cmp)
        self.assertEqual(len(keys), 2)
        self.assertTrue(all(x['bit_equal'] for x in self.cmp.rows))

    def test_missing_or_wrong_input_ahead_is_not_paired_by_call_order(self):
        key = decode_key(['r0'], [11], [7])
        ref = NS(calls={key: record(key)})
        for carried in (None, record(decode_key(['r0'], [11], [8])), record(decode_key(['other-rid'], [11], [7]))):
            with self.subTest(carried=carried and carried['key']):
                with self.assertRaises(QualificationFailure):
                    pair_calls(ref, NS(calls={}, consumed=key, previous_future=carried), self.cmp)

    def test_schedule_reordering_and_wrong_delivered_rid_rejected(self):
        with self.assertRaises(QualificationFailure):
            self.cmp.meta('schedule', {'decode': ['a', 'b']}, {'decode': ['b', 'a']})
        with self.assertRaises(QualificationFailure):
            self.cmp.meta('outputs', [('a', 7, False)], [('b', 7, False)])

    def test_ahead_key_requires_actual_finalized_input_ids(self):
        self.assertEqual(ahead_semantic_key(((['a'], [4], (), (7,)), None, [])), decode_key(['a'], [4], [7]))
        with self.assertRaises(QualificationFailure):
            ahead_semantic_key(((['a'], [4], (), None), None, []))

    def test_fixed_workload_counts_and_caps(self):
        fixtures = [{'fixture_id': f'D{i}', 'prompt_ids': [1, 2]} for i in range(8)]
        cases = workloads(fixtures)
        self.assertEqual([len(c['jobs']) for c in cases], [8, 16, 13])
        self.assertEqual(sum(j['cap'] for c in cases for j in c['jobs']), 12411)
        self.assertEqual(cases[2]['jobs'][0]['rid'], cases[2]['jobs'][-1]['rid'])
        self.assertNotEqual(cases[2]['jobs'][0]['fixture'], cases[2]['jobs'][-1]['fixture'])

    def test_cpu_lifecycle_late_arrivals_retirement_and_rid_reuse(self):
        fixtures = [{'fixture_id': f'D{i}', 'prompt_ids': [1] * (2 + i)} for i in range(8)]
        case = workloads(fixtures)[2]
        engines = [Engine(), Engine()]
        observers = [Observer(e, self.cmp, arm) for e, arm in zip(engines, ('reference', 'candidate'))]
        runner_module = NS(prefill_batch_backend_supported=lambda: True)
        ledger = io.StringIO()
        result = run_case(case, engines, observers, SamplingParams, runner_module, self.cmp, ledger, lambda: None)
        self.assertEqual(len(result['completed']), 13)
        self.assertEqual(result['raw_output_count'], 123)
        self.assertEqual([x['before_step'] for x in result['admissions'][8:12]], [2] * 4)
        self.assertGreater(result['admissions'][12]['before_step'], 2)
        self.assertTrue(all(json.loads(line)['status'] == 'PASS' for line in ledger.getvalue().splitlines()))
        self.assertTrue(runner_module.prefill_batch_backend_supported())
        self.assertTrue(engines[0].scheduler._finished_since_last_step)  # Legal terminal buffer.
        for observer in observers:
            observer.restore()

    def test_corrupt_schedule_keeps_partial_ledger(self):
        fixtures = [{'fixture_id': f'D{i}', 'prompt_ids': [1] * (2 + i)} for i in range(8)]
        engines = [Engine(), Engine()]
        original = engines[1].scheduler.schedule
        def wrong_order():
            out = original()
            out.prefill.reverse()
            return out
        engines[1].scheduler.schedule = wrong_order
        observers = [Observer(e, self.cmp, arm) for e, arm in zip(engines, ('reference', 'candidate'))]
        ledger = io.StringIO()
        with self.assertRaises(QualificationFailure):
            run_case(workloads(fixtures)[2], engines, observers, SamplingParams,
                     NS(prefill_batch_backend_supported=lambda: True), self.cmp, ledger, lambda: None)
        partial = json.loads(ledger.getvalue().splitlines()[0])
        self.assertEqual(partial['status'], 'RUNNING')
        self.assertIn('schedule', partial['failure']['label'])

    def test_c1_and_c8_fixed_fifo_refill_and_drain(self):
        fixtures = [{'fixture_id': f'D{i}', 'prompt_ids': [1] * (2 + i)} for i in range(8)]
        for case in workloads(fixtures)[:2]:
            # CPU mechanism test caps are deliberately tiny, not a D512 qualification.
            for job in case['jobs']:
                job['cap'] = 3
            engines = [Engine(case['concurrency']), Engine(case['concurrency'])]
            observers = [Observer(e, self.cmp, arm) for e, arm in zip(engines, ('reference', 'candidate'))]
            result = run_case(case, engines, observers, SamplingParams,
                             NS(prefill_batch_backend_supported=lambda: True), self.cmp, io.StringIO(), lambda: None)
            self.assertEqual([x['rid'] for x in result['admissions']], [j['rid'] for j in case['jobs']])
            self.assertEqual(len(result['completed']), len(case['jobs']))
            self.assertEqual(max(result['decode_batch_histogram']), case['concurrency'])
            for observer in observers:
                observer.restore()

    def test_absolute_deadline_keeps_successful_prefix_without_pass(self):
        fixtures = [{'fixture_id': f'D{i}', 'prompt_ids': [1] * (2 + i)} for i in range(8)]
        engines = [Engine(), Engine()]
        observers = [Observer(e, self.cmp, arm) for e, arm in zip(engines, ('reference', 'candidate'))]
        calls = 0
        def deadline():
            nonlocal calls
            calls += 1
            if calls == 2:
                raise TimeoutError('mock absolute deadline')
        ledger = io.StringIO()
        with self.assertRaises(TimeoutError):
            run_case(workloads(fixtures)[2], engines, observers, SamplingParams,
                     NS(prefill_batch_backend_supported=lambda: True), self.cmp, ledger, deadline)
        self.assertEqual(len(ledger.getvalue().splitlines()), 1)
        self.assertEqual(json.loads(ledger.getvalue())['status'], 'PASS')


if __name__ == '__main__':
    unittest.main()
