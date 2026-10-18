"""CPU transport/process fakes, not oMLX execution or a quality qualification.

The actual fixed SSE client and raw wire checker run over MockTransport. Only
qualification binding and reconstructive validation are mocked for lifecycle
tests. A separate preparation test consumes the real CLOSED small metadata.
No sockets, actual Popen, signals, official runtime or model imports occur.
"""
import asyncio
import base64
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import httpx

from bench.m5_ultra_256gb.omlx_d512_formal_retry import formal as m
from bench.m5_ultra_256gb.omlx_d512.test_v2_cpu import fixture


def fake_spec(out, C=8):
    out = Path(out).resolve(); fs = list(fixture(C)[3].values())
    binding = m.qualification_binding()
    bound = dict(base_plan={'path': '/CPU/base.json', 'sha256': 'b'*64}, fixtures_ref={'path': '/CPU/fixtures.json', 'sha256': 'f'*64},
        fixtures=fs, model_directory=str(Path('/tmp/CPU-no-model').resolve()), served_model='Qwen3.8-27B-nvfp4',
        qualification_identity={'scope': 'explicit CPU fake qualification'})
    jobs = m.jobs_for(fs, C, 1)
    spec = dict(**bound, schema='E023-omlx-D512-formal-spec-v1', output_directory=str(out), concurrency=C, repeat=1,
        port=18882, api_key='CPU-key', command=[m.OFFICIAL, 'serve', '--CPU-no-actual-command'],
        official_args=['serve', '--CPU-no-actual-command'], cwd='/CPU-no-cwd', environment={'CPU': 'only'},
        observer_enabled=False, qualification=binding, qualification_binding=copy.deepcopy(binding), jobs=jobs,
        deadlines=dict(ready=.2, request=2, warm=3, formal=3, close=.1, term=15, kill=15, overall=7, resource_interval=.01),
        files={str(out/'state/settings.json'): {'cache': {'enabled': False}}, str(out/'state/model_settings.json'): {}},
        symlinks={str(out/'models/Qwen3.8-27B-nvfp4'): bound['model_directory']},
        contracts={phase: m._contract(rows, fs, binding, C) for phase, rows in jobs.items()})
    return spec, bound


class Process:
    pid = 79123
    def __init__(self, kill_only=False): self.returncode = None; self.actions = []; self.kill_only = kill_only
    def poll(self): return self.returncode
    def terminate(self):
        self.actions.append('TERM')
        if not self.kill_only: self.returncode = -15
    def kill(self): self.actions.append('KILL'); self.returncode = -9


class Bytes(httpx.AsyncByteStream):
    def __init__(self, raw): self.raw = raw
    async def __aiter__(self):
        await asyncio.sleep(0)
        yield self.raw
    async def aclose(self): pass


class Backend:
    def __init__(self):
        self.child = Process(); self.phases = []; self.headers = []; self.counts = {}; self.waited = 0
        self.flag = None; self.formal_began = asyncio.Event(); self.cleanup_started = asyncio.Event(); self.allow_reap = None
    def spawn(self, command, environment, stdout, stderr, *, cwd):
        self.spawned = dict(command=command, environment=environment, cwd=cwd)
        if self.flag == 'unknown': raise RuntimeError('unknown Popen outcome')
        stdout.write(b'CPU official log placeholder\n'); stdout.flush(); stderr.write(b'CPU stderr placeholder\n'); stderr.flush()
        return self.child
    async def port(self, port, pid=None):
        if pid is None and (not hasattr(self, 'spawned') or self.child.poll() is not None):
            return dict(returncode=1, stdout='', stderr='')
        return dict(returncode=0, stdout=f'p{self.child.pid}\nf4\nn127.0.0.1:{port}\n', stderr='')
    async def identity(self, pid):
        if self.flag == 'identity': raise RuntimeError('CPU identity read failed after known Popen')
        return dict(returncode=0, stdout=f'{pid} {os.getpid()} Sat Oct 17 00:00:00 2026 omlx\n', stderr='')
    async def readiness(self, spec):
        data = {'object': 'list', 'data': [dict(id=spec['served_model'], object='model', owned_by='omlx')]}
        if self.flag == 'wrong-ready': data['data'][0]['id'] = 'another-model'
        return dict(status=200, body_base64=base64.b64encode(json.dumps(data).encode()).decode())
    async def absent(self, pid):
        return dict(returncode=0 if self.flag == 'bad-absence' else 1, stdout=str(pid) if self.flag == 'bad-absence' else '', stderr='')
    async def wait(self, child, seconds):
        self.waited += 1; self.cleanup_started.set()
        if self.allow_reap is not None: await self.allow_reap.wait()
        if child.poll() is None: raise subprocess.TimeoutExpired('CPU-fake', seconds)
        return child.returncode
    async def group(self, spec, phase, output):
        self.phases.append(phase)
        if phase == 'formal': self.formal_began.set()
        client = m.load('../pareto_realistic/native_cleanup.py'); submitted = []
        async def handle(request):
            self.headers.append(request.headers['authorization'])
            if self.flag == 'block' and phase == 'formal': await asyncio.Future()
            body = json.loads(request.content); i = len(submitted); submitted.append(body)
            f = next(f for f in spec['fixtures'] if f['prompt_text'] == body['prompt'])
            expected = spec['jobs'][phase][i]['body']
            if body != expected: raise AssertionError('actual sent original body changed')
            n = (0, 1, 512)[i % 3]; finish = 'length' if n == 512 else 'stop'
            rid = f'cmpl-{(100000 if phase == "warm" else 200000)+i:08x}'
            if self.flag == 'reused-id' and phase == 'formal': rid = 'cmpl-00000000'
            details = {'cached_tokens': 1 if self.flag == 'cached' and phase == 'formal' else 0}
            usage = dict(prompt_tokens=len(f['prompt_ids']), completion_tokens=n,
                total_tokens=len(f['prompt_ids'])+n, prompt_tokens_details=details,
                total_time=.25, time_to_first_token=.01, prompt_eval_duration=.01,
                generation_duration=.24, prompt_tokens_per_second=100.0, generation_tokens_per_second=float(n))
            if self.flag == 'bad-timing' and phase == 'formal': usage['total_time'] = -1.0
            if self.flag == 'missing-cache' and phase == 'formal': usage.pop('prompt_tokens_details')
            header = dict(id=rid, object='text_completion', model=spec['served_model'], created=1)
            rows = [dict(**header, choices=[dict(index=0, text='a'*n, finish_reason=None)]),
                    dict(**header, choices=[dict(index=0, text='', finish_reason=finish)]),
                    dict(**header, choices=[], usage=usage)]
            raw = ''.join('data: '+json.dumps(row)+'\n\n' for row in rows).encode()+b'data: [DONE]\n\n'
            if self.flag == 'wire-tail' and phase == 'formal': raw += b'late bytes'
            return httpx.Response(200, stream=Bytes(raw), headers={'content-type': 'text/event-stream'})
        value = await client.run_group(f"http://127.0.0.1:{spec['port']}", spec['jobs'][phase], spec['concurrency'], phase=phase,
            qualification=m.count_contract(spec, phase), headers={'Authorization': 'Bearer '+spec['api_key']},
            transport=httpx.MockTransport(handle), request_timeout_s=spec['deadlines']['request'],
            group_timeout_s=spec['deadlines'][phase], cleanup_timeout_s=spec['deadlines']['close'], output_dir=output)
        self.counts[phase] = len(submitted)
        if self.flag == 'unclosed-warm' and phase == 'warm': value['cleanup']['complete'] = False
        return value


class Tests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); self.root = Path(self.tmp.name).resolve()
        self.out = self.root/'trial'; self.spec, self.bound = fake_spec(self.out); self.backend = Backend(); self.checkpoints = []
        self.validator = patch.object(m, 'validate', return_value=None); self.validator.start(); self.addCleanup(self.validator.stop)
        self.binder = patch.object(m, 'bind_qualification', side_effect=lambda binding, C: fake_spec(self.root/'unused', C)[1])
        self.binder.start(); self.addCleanup(self.binder.stop)
    async def checkpoint(self, label, pid): self.checkpoints.append((label, pid))
    async def run_trial(self): return await m.run(self.spec, self.out, self.checkpoint, backend=self.backend)

    async def test_original_client_two_drained_phases_c1_c8_usage_identity_and_direct_pid(self):
        for C in (1, 8):
            out = self.root/f'c{C}'; spec, _ = fake_spec(out, C); backend = Backend()
            result = await m.run(spec, out, self.checkpoint, backend=backend)
            self.assertEqual(result['status'], 'PASS', result['errors'])
            self.assertEqual(backend.counts, {'warm': 8, 'formal': 64}); self.assertEqual(backend.phases, ['warm', 'formal'])
            self.assertEqual(backend.spawned['cwd'], spec['cwd']); self.assertEqual(backend.child.actions, ['TERM'])
            self.assertEqual(set(backend.headers), {'Bearer CPU-key'}); self.assertGreaterEqual(backend.waited, 1)
            self.assertTrue(result['server_reaped'] and result['cleanup_task_resolved'] and result['client_task_resolved'])
            self.assertFalse(result['performance_qualified'] or result['full_state_qualified'])
            self.assertFalse((out/'probe').exists()); self.assertFalse((out/'request-seal.json').exists())
            warm = m.read(out/'warm-group.json'); formal = m.read(out/'formal-group.json')
            self.assertLessEqual(warm['cleanup']['end_perf_counter_s'], min(r['start_perf_counter_s'] for r in formal['requests']))
            checked = m.read(out/'formal-raw-check.json'); self.assertEqual([r['committed_output_tokens'] for r in checked['requests'][:3]], [0, 1, 512])
            point = m.read(out/'point.json'); self.assertEqual(point['requests'], 64); self.assertEqual(point['max_http_outstanding'], C)
            self.assertEqual(point['committed_output_tokens'], sum((0, 1, 512)[i % 3] for i in range(64)))
            self.assertEqual(point, dict(**result['point'], count_eligible=True, performance_qualified=False))
            self.assertEqual(result['artifacts']['formal-group.json']['sha256'], m.sha(out/'formal/group.json'))
            release = m.read(out/'release.json'); self.assertEqual(release['returncode'], -15); self.assertTrue(release['term_sent'])

    async def test_raw_cache_missing_cache_uuid_and_tail_fail_without_point(self):
        for flag in ('cached', 'missing-cache', 'reused-id', 'wire-tail', 'bad-timing'):
            out = self.root/flag; spec, _ = fake_spec(out); backend = Backend(); backend.flag = flag
            result = await m.run(spec, out, self.checkpoint, backend=backend)
            self.assertEqual(result['status'], 'FAIL', flag); self.assertFalse(result['count_eligible'])
            self.assertTrue(result['server_reaped']); self.assertTrue((out/'formal-group.json').is_file())
            self.assertFalse((out/'point.json').exists()); self.assertTrue(result['errors'])

    async def test_warm_close_failure_never_submits_formal(self):
        self.backend.flag = 'unclosed-warm'; result = await self.run_trial()
        self.assertEqual(result['status'], 'FAIL'); self.assertEqual(self.backend.phases, ['warm'])
        self.assertTrue(result['server_reaped']); self.assertNotIn('formal-start', m.read(self.out/'phase-boundaries.json'))

    async def test_actual_bad_model_readiness_rejected_before_requests(self):
        self.backend.flag = 'wrong-ready'; result = await self.run_trial()
        self.assertEqual(result['status'], 'FAIL'); self.assertEqual(self.backend.phases, [])
        self.assertTrue(result['server_reaped']); self.assertFalse(result['count_eligible'])

    async def test_known_identity_failure_reaped_unknown_popen_retained(self):
        for flag in ('identity', 'unknown'):
            out = self.root/flag; spec, _ = fake_spec(out); backend = Backend(); backend.flag = flag
            result = await m.run(spec, out, self.checkpoint, backend=backend)
            self.assertEqual(result['status'], 'FAIL'); self.assertTrue(result['spawn_attempted'])
            self.assertEqual(result['popen_returned'], flag == 'identity'); self.assertEqual(result['server_reaped'], flag == 'identity')
            self.assertTrue((out/'spawn-intent.json').is_file()); self.assertEqual((out/'release.json').exists(), flag == 'identity')

    async def test_kill_is_reaped_but_never_a_successful_trial(self):
        self.backend.child.kill_only = True; result = await self.run_trial()
        self.assertEqual(result['status'], 'FAIL'); self.assertTrue(result['server_reaped'])
        self.assertEqual(self.backend.child.actions, ['TERM', 'KILL']); self.assertEqual(m.read(self.out/'release.json')['returncode'], -9)
        self.assertFalse((self.out/'point.json').exists())

    async def test_resource_failure_preserves_client_partial_before_reap(self):
        self.backend.flag = 'block'
        async def reject(label, pid):
            if label == 'formal': raise RuntimeError('CPU resource rejection')
        result = await m.run(self.spec, self.out, reject, backend=self.backend)
        self.assertEqual(result['status'], 'FAIL'); self.assertTrue(result['server_reaped'] and result['client_task_resolved'])
        self.assertEqual(m.read(self.out/'formal-group.json')['group_error']['type'], 'CancelledError')
        self.assertFalse((self.out/'point.json').exists())

    async def test_repeated_cancellation_waits_for_owned_reap(self):
        self.backend.flag = 'block'; self.backend.allow_reap = asyncio.Event()
        task = asyncio.create_task(self.run_trial()); await asyncio.wait_for(self.backend.formal_began.wait(), 3); task.cancel()
        await asyncio.wait_for(self.backend.cleanup_started.wait(), 3); task.cancel(); await asyncio.sleep(0)
        self.assertFalse(task.done()); self.backend.allow_reap.set(); result = await task
        self.assertEqual(result['status'], 'FAIL'); self.assertTrue(result['server_reaped'] and result['cleanup_task_resolved'])
        self.assertTrue(any(row['phase'] == 'cancel-during-cleanup' for row in result['errors']))

    async def test_changed_prompt_and_body_never_reuse_qualification(self):
        for field in ('prompt', 'temperature'):
            spec = copy.deepcopy(self.spec); spec['jobs']['formal'][0]['body'][field] = 'other' if field == 'prompt' else .1
            with self.assertRaisesRegex(ValueError, 'original-prompt queues'): m.count_contract(spec, 'formal')


class PreparationTests(unittest.TestCase):
    def test_real_closed_binding_reconstructs_observer_off_original_cli(self):
        budgets = dict(ready=180, request=600, warm=240, formal=1200, close=10, term=15, kill=15, overall=1680, resource_interval=5)
        with tempfile.TemporaryDirectory() as directory, patch.object(m, 'System', side_effect=AssertionError('no backend during prepare')):
            for C in (1, 8):
                out = Path(directory)/f'c{C}'; probe = m.load('../omlx_d512/serve_probe_v2.py')
                env = probe.specification(out, C, 18882, 'CPU-prepare-only', observed=False)['environment']
                spec = m.prepare(out, C, 1, 18882, 'CPU-prepare-only', qualification=m.qualification_binding(), deadlines=budgets, environment=env)
                m.validate(spec); self.assertFalse(out.exists()); self.assertFalse(spec['observer_enabled'])
                self.assertEqual(spec['command'], [m.OFFICIAL, *spec['official_args']])
                self.assertEqual({phase: len(rows) for phase, rows in spec['jobs'].items()}, {'warm': 8, 'formal': 64})
                self.assertEqual(spec['jobs']['formal'][0]['body']['prompt'], spec['fixtures'][0]['prompt_text'])
                for phase in ('warm', 'formal'):
                    self.assertEqual(m.count_contract(spec, phase), spec['contracts'][phase])
                    self.assertEqual(spec['contracts'][phase]['raw_ids_count_rule'], 'none')
                    self.assertEqual(spec['contracts'][phase]['usage_count_rule'], 'identity')
                bad = copy.deepcopy(spec); bad['observer_enabled'] = True
                with self.assertRaisesRegex(ValueError, 'formal spec changed'): m.validate(bad)

    def test_nonactual_qualification_digest_is_rejected_before_backend(self):
        binding = m.qualification_binding(); binding['audit']['sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'actual fixed archived qualification'): m.bind_qualification(binding, 1)

    def test_default_cli_is_description_only(self):
        with patch.object(m, 'prepare', side_effect=AssertionError('no prepare')), patch.object(m, 'System', side_effect=AssertionError('no backend')):
            self.assertEqual(m.main([]), 0)


class ActualWarmRegressionTests(unittest.TestCase):
    """Use retained v1 bytes; v1 failed before any formal request or point."""
    def actual(self):
        root = m.ROOT/'.artifacts/m5-ultra-256gb/omlx-d512-formal-v1-owner/phases/r1-c1/trial'
        self.assertEqual(m.sha(root/'warm-group.json'), '0d1f42f0b276db1d05c42824591a060ee9cf336f76b71751e2a470e4c9ee1d4c')
        self.assertEqual(m.sha(root/'spec.json'), 'dde6fbc9bd9ee3cf5fee138891bc2e200d9daad9eab3a0aa1c8062b2feb59b7b')
        return m.read(root/'warm-group.json'), m.read(root/'spec.json')

    def test_all_eight_actual_retained_usage_schemas_reparse_without_eos_subtraction(self):
        group, spec = self.actual()
        result = m.check_group(group, spec, 'warm')
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['point']['requests'], 8)
        self.assertEqual(result['point']['committed_output_tokens'], 4096)
        self.assertFalse(result['performance_qualified'])
        for row in group['requests']:
            usage = m.load('../omlx_d512/join_v2.py').wire(row)['usage']
            self.assertEqual(type(usage['input_tokens']), int)
            self.assertEqual(type(usage['output_tokens']), int)
            self.assertEqual(usage['input_tokens'], usage['prompt_tokens'])
            self.assertEqual(usage['output_tokens'], usage['completion_tokens'])

    @staticmethod
    def change_usage(row, name, value, remove=False):
        # Keep the actual frame order/text/UUID. This synthetic corruption uses
        # one received chunk at the original DONE clock, then updates the raw
        # byte receipt and parsed usage so only the alias contract can reject it.
        raw = b''.join(base64.b64decode(c['body_bytes_base64']) for c in row['raw_sse_chunks'])
        lines = []
        for line in raw.decode().splitlines(keepends=True):
            if line.startswith('data: {'):
                frame = json.loads(line[6:])
                if 'usage' in frame:
                    if remove: frame['usage'].pop(name, None)
                    else: frame['usage'][name] = value
                    row['usage'] = copy.deepcopy(frame['usage'])
                    line = 'data: '+json.dumps(frame, ensure_ascii=False)+'\n'
            lines.append(line)
        changed = ''.join(lines).encode()
        row['raw_sse_chunks'] = [dict(client_perf_counter_s=row['done_perf_counter_s'],
                                     body_bytes_base64=base64.b64encode(changed).decode())]
        row['raw_body_bytes'] = len(changed)

    def test_actual_wire_alias_mismatch_bool_float_reject_and_omitted_alias_allowed(self):
        group, spec = self.actual()
        for name in ('input_tokens', 'output_tokens'):
            expected = group['requests'][0]['usage'][name]
            for value in (expected+1, True, float(expected)):
                changed = copy.deepcopy(group)
                self.change_usage(changed['requests'][0], name, value)
                with self.subTest(name=name,value=value), self.assertRaisesRegex(ValueError, 'usage alias count/type'):
                    m.check_group(changed, spec, 'warm')
        changed = copy.deepcopy(group)
        for name in ('input_tokens', 'output_tokens'):
            self.change_usage(changed['requests'][0], name, None, remove=True)
        self.assertEqual(m.check_group(changed, spec, 'warm')['point']['committed_output_tokens'], 4096)


if __name__ == '__main__': unittest.main()
