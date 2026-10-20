"""CPU-only binding, real MockTransport SSE and inherited cleanup tests."""
import asyncio
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bench.m5_ultra_256gb.concurrency_config_v1 import trial as m
from bench.m5_ultra_256gb.pareto_realistic import test_native_formal_cpu as old


def fixture(C=1):
    return dict(schema='GDN-graph-combined-http-formal-spec-v1', concurrency=C,
        arm='on' if C == 1 else 'off', repeat=1, phase=f'r1-c{C}',
        environment={**m.MLX_BASE, 'PATH': '/cpu/fixture'},
        command=['/cpu/python', '--max-seqs', str(C), '--max-batched-tokens', '2048'],
        identity={'cpu': 'fixture'}, sources={}, qualification={'cpu': 'wire-contract'},
        jobs={p: [dict(request_id=f'{p}-{i}', body={'prompt': 'fixture'}) for i in range(n)]
              for p, n in (('warm', 8), ('formal', 64))}, deadlines={'resource_interval': 5, 'overall': 3300})


async def group(spec, phase, output=None):
    client_phase = m.CLIENT_PHASE[phase]
    view = {**spec, 'jobs': {**spec['jobs'], client_phase: spec['jobs'][phase]}}
    return await old.group(view, client_phase, output)


class Backend(old.Backend):
    async def group(self, spec, phase, output):
        self.phases.append(phase); self.group_started.set()
        if self.block:
            try: await asyncio.Future()
            except asyncio.CancelledError:
                raise old.m.load('native_cleanup').GroupCancelled(dict(
                    group_error={'type': 'CancelledError'}, cleanup={'complete': False}))
        return await group(spec, phase, output)


class Preparation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'base.json'
        self.base = fixture(); self.save()
        native = m.module(m.NATIVE)
        self.originals = []
        formal = SimpleNamespace(pinned=native.pinned, validate=lambda spec: self.originals.append(copy.deepcopy(spec)))
        self.module_patch = patch.object(m, 'module', return_value=formal)
        self.module_patch.start(); self.addCleanup(self.module_patch.stop)
        pins = patch.object(m, 'source_pins', return_value={'/cpu/source': 'fake-only'})
        pins.start(); self.addCleanup(pins.stop)
    def save(self):
        self.path.write_text(json.dumps(self.base)); self.ref = {'path': str(self.path), 'sha256': m.sha(self.path)}
    def test_one_variable_only_and_original_validation(self):
        for C in (1, 8):
            self.base = fixture(C); self.save()
            for mb in m.MB_VALUES:
                result = m.prepare(self.ref, mb, f'c{C}-mb{mb}')
                self.assertEqual(result['command'], self.base['command'])
                self.assertEqual(result['identity'], self.base['identity'])
                self.assertEqual(result['jobs']['warm'], self.base['jobs']['warm'])
                self.assertEqual(result['jobs']['formal'], self.base['jobs']['formal'][:8])
                env = {**result['environment'], 'MLX_MAX_MB_PER_BUFFER': '50'}
                self.assertEqual(env, self.base['environment'])
                self.assertEqual(result['deadlines'], {'resource_interval': 3, 'overall': 3300})
                self.assertFalse(result['performance_qualified']); self.assertFalse(result['full_state_qualified'])
                self.assertEqual(self.originals[-1], self.base)
                m.validate(result)
        self.assertEqual(m.read(self.path), self.base)
    def test_mutation_rejected_even_with_rebuilt_spec_container(self):
        spec = m.prepare(self.ref, 200, 'c1-mb200')
        mutations = [lambda s: s['command'].append('--other'),
            lambda s: s['environment'].__setitem__('MLX_MAX_OPS_PER_BUFFER', '200'),
            lambda s: s['jobs']['formal'].pop(), lambda s: s.update(performance_qualified=True),
            lambda s: s['sources'].clear(), lambda s: s['deadlines'].update(overall=99999)]
        for mutate in mutations:
            bad = copy.deepcopy(spec); mutate(bad)
            with self.assertRaises(ValueError): m.validate(bad)
    def test_unregistered_values_wrong_base_and_changed_bytes_rejected(self):
        for mb in (True, 0, 1000, '200'):
            with self.assertRaises(ValueError): m.prepare(self.ref, mb, 'valid')
        for label in ('', '../escape', 'UPPER', 'a' * 65):
            with self.assertRaises(ValueError): m.prepare(self.ref, 50, label)
        for mutate in (lambda d: d.update(arm='off'), lambda d: d.update(repeat=2),
                       lambda d: d['environment'].update(MLX_UNLISTED='1')):
            self.base = fixture(); mutate(self.base); self.save()
            with self.assertRaises(ValueError): m.prepare(self.ref, 50, 'base')
        self.path.write_text('{}')
        with self.assertRaises(ValueError): m.prepare(self.ref, 50, 'base')
    def test_description_is_inert(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream), patch.object(m, 'prepare', side_effect=AssertionError('no preparation')):
            self.assertEqual(m.main([]), 0)
        result = json.loads(stream.getvalue()); self.assertEqual(result['screen_requests'], 8)
        self.assertFalse(result['performance_qualified'])


class Transport(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
    def spec(self, C):
        spec = old.prepare(self.root / f'metadata-{C}', C=C)
        spec['jobs']['formal'] = spec['jobs']['formal'][:8]
        spec.update(label=f'c{C}-mb200', mb_threshold=200, base_spec={'cpu': 'fixture'})
        spec['environment']['MLX_MAX_MB_PER_BUFFER'] = '200'
        return spec
    async def test_actual_system_maps_only_screen_to_nonformal_client_phase(self):
        native = m.module(m.NATIVE)
        async def record(*args, **kwargs): return {'args': args, 'kwargs': kwargs}
        fake = SimpleNamespace(load=lambda name: SimpleNamespace(run_group=record),
                               count_contract=native.count_contract)
        for C in (1, 8):
            spec = self.spec(C)
            for phase in ('warm', 'formal'):
                with patch.object(m, 'module', return_value=fake):
                    result = await m.System.group(None, spec, phase, self.root / 'output')
                args, kw = result['args'], result['kwargs']
                self.assertEqual(args[1], spec['jobs'][phase]); self.assertEqual(args[2], C)
                self.assertEqual(kw['phase'], 'warm' if phase == 'warm' else 'admission_probe')
                self.assertEqual(kw['qualification'], native.count_contract(spec, phase))
                self.assertEqual(kw['group_timeout_s'], spec['deadlines'][phase])

    async def test_real_raw_eight_queue_and_frozen_checker_unchanged(self):
        native = m.module(m.NATIVE); original_code = native.check_group.__code__
        for C in (1, 8):
            spec = self.spec(C); raw = await group(spec, 'formal')
            saved = copy.deepcopy(raw)
            checked = m.check_group(raw, spec, 'formal', old.decode)
            self.assertEqual(checked['raw_requests_checked'], 8)
            self.assertEqual(checked['point']['committed_output_tokens'], 3 + 2 * 512)
            self.assertEqual(raw, saved)
            self.assertFalse(checked['performance_qualified'])
            with self.assertRaises(ValueError):
                native.check_group(raw, spec, 'formal', old.decode)
            self.assertEqual(raw['phase'], 'admission_probe')
            self.assertFalse(raw['summary']['formal_point_eligible'])
        self.assertIs(native.check_group.__code__, original_code)
    async def test_raw_prompt_eos_count_and_cleanup_fail_closed(self):
        spec = self.spec(8); raw = await group(spec, 'formal')
        for key, value in (('benchmark_prompt_ids', [777]), ('benchmark_terminal_eos_id', None),
                           ('usage', {'completion_tokens': 512})):
            bad = copy.deepcopy(raw); old.rewrite_terminal(bad['requests'][0], lambda row: row.__setitem__(key, value))
            with self.assertRaises((ValueError, KeyError)): m.check_group(bad, spec, 'formal', old.decode)
        for change in (lambda d: d['requests'].pop(), lambda d: d['cleanup'].update(complete=False)):
            bad = copy.deepcopy(raw); change(bad)
            with self.assertRaises(ValueError): m.check_group(bad, spec, 'formal', old.decode)
    async def test_real_queue_both_C_private_namespace_and_reap(self):
        native = m.module(m.NATIVE); saved = dict(native.run.__globals__)
        actual_module = m.module
        for C in (1, 8):
            spec = self.spec(C); backend = Backend(spec); checkpoints = []
            with patch.object(m, 'validate', return_value=None), patch.object(m, 'module', side_effect=lambda p: native if p == m.NATIVE else actual_module(p)):
                result = await m.run(spec, self.root / f'trial-{C}',
                    checkpoint=lambda *args: checkpoints.append(args), backend=backend)
            self.assertEqual(result['status'], 'PASS', result['errors'])
            self.assertEqual(result['point']['requests'], 8)
            self.assertEqual(backend.phases, ['warm', 'formal']); self.assertEqual(backend.child.actions, ['TERM'])
            self.assertTrue(result['server_reaped']); self.assertFalse(result['token_equivalence_checked'])
            self.assertFalse(result['full_state_qualified']); self.assertEqual(checkpoints[-1], ('after-reap', None))
            release = m.read(self.root / f'trial-{C}/release.json')
            self.assertTrue(release['waited_reaped']); self.assertTrue(release['strict_absence'])
        self.assertEqual(native.run.__globals__, saved)
    async def test_invalid_warm_stops_screen_and_preserves_raw(self):
        spec = self.spec(1); backend = Backend(spec); real = backend.group
        async def wrong(spec, phase, out):
            group = await real(spec, phase, out); group['requests'][0]['answer'] = 'wrong'; return group
        backend.group = wrong
        with patch.object(m, 'validate', return_value=None):
            result = await m.run(spec, self.root / 'trial', checkpoint=lambda *args: None, backend=backend)
        self.assertEqual(result['status'], 'FAIL'); self.assertEqual(backend.phases, ['warm'])
        self.assertTrue(result['server_reaped']); self.assertTrue((self.root / 'trial/warm-group.json').exists())
        self.assertFalse((self.root / 'trial/formal-group.json').exists())
    async def test_unknown_reap_is_failure(self):
        spec = self.spec(1); backend = Backend(spec, bad_reap=True)
        with patch.object(m, 'validate', return_value=None):
            result = await m.run(spec, self.root / 'trial', checkpoint=lambda *args: None, backend=backend)
        self.assertEqual(result['status'], 'FAIL'); self.assertFalse(result['server_reaped'])
    async def test_cancelled_queue_still_TERM_KILL_waits_and_retains_failure(self):
        spec = self.spec(1); backend = Backend(spec, ignore_term=True, block=True)
        with patch.object(m, 'validate', return_value=None):
            task = asyncio.create_task(m.run(spec, self.root / 'trial', checkpoint=lambda *args: None, backend=backend))
            await backend.group_started.wait(); task.cancel(); result = await task
        self.assertEqual(result['status'], 'FAIL'); self.assertTrue(result['server_reaped'])
        self.assertEqual(backend.child.actions, ['TERM', 'KILL']); self.assertTrue(backend.waited)
        self.assertTrue((self.root / 'trial/warm-group.json').exists())


if __name__ == '__main__': unittest.main()
