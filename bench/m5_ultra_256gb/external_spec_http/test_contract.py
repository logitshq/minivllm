"""Pure CPU fixtures. No socket listener, server, model runtime or GPU import."""
import json
import signal
import types
import unittest
from unittest.mock import patch
from unittest.mock import call

from client import body_for, request
from contract import CONFIGS, MODEL_ID, official_log_evidence, summarize, validate_request
from launch import specification
import run


class Tokenizer:
    def encode(self, text, **unused):
        return types.SimpleNamespace(ids=[int(x) for x in text.split()])


def client_fixture(events):
    class Response:
        status_code = 200
        def __enter__(self): return self
        def __exit__(self, *unused): return None
        def raise_for_status(self): pass
        def iter_lines(self): return iter(events)
    class Client:
        def __init__(self, **unused): pass
        def __enter__(self): return self
        def __exit__(self, *unused): return None
        def stream(self, *unused, **kwargs): return Response()
    with patch.dict('sys.modules', {'httpx': types.SimpleNamespace(Client=Client)}):
        return request('unused', {}, '/v1/completions', {}, Tokenizer(), 'fixture')


def sse(data):
    return 'data: ' + json.dumps(data)


def good_row():
    return client_fixture([
        sse({'choices': [{'text': '0 1 ', 'finish_reason': None}]}),
        sse({'choices': [{'text': ' '.join(map(str, range(2, 32))), 'finish_reason': None}]}),
        sse({'choices': [{'text': '', 'finish_reason': 'length'}],
             'usage': {'prompt_tokens': 19, 'completion_tokens': 32},
             'benchmark_output_ids': list(range(32))}),
        'data: [DONE]'])


class ContractTests(unittest.TestCase):
    def test_sse_retains_events_and_actual_count(self):
        row = good_row()
        self.assertEqual(validate_request(row, list(range(32))), [])
        self.assertEqual(row['actual_visible_canonical_count'], 32)
        self.assertEqual(row['aggregate_output_tok_s'], 32 / row['client_total_s'])
        self.assertEqual(len(row['sse_events']), 4)
        self.assertIsNone(row['cached_prompt_tokens_when_available'])
        self.assertFalse(row['cache_usage_field_present'])
    def test_native_requires_live_raw_ids_and_explicit_cache(self):
        row = good_row()
        row['cached_prompt_tokens_when_available'] = 0
        self.assertEqual(validate_request(row, list(range(32)), native=True), [])
        row['output_ids_when_available'] = None
        self.assertTrue(validate_request(row, list(range(32)), native=True))
    def test_early_eos_and_wrong_oracle_remain_failures(self):
        row = good_row()
        row['finish_reason'] = 'stop'
        row['usage']['completion_tokens'] = 4
        failures = validate_request(row, [9] * 32)
        self.assertGreaterEqual(len(failures), 4)
        self.assertEqual(row['actual_visible_canonical_count'], 32)
    def test_partial_stream_retained_on_parse_error(self):
        row = client_fixture([sse({'choices': [{'text': '0 1 '}]}), 'data: {broken'])
        self.assertFalse(row['done'])
        self.assertIsNotNone(row['error'])
        self.assertEqual(row['visible_roundtrip_ids'], [0, 1])
        self.assertEqual(len(row['sse_events']), 2)
    def test_missing_done_is_not_a_success(self):
        row = client_fixture([sse({'choices': [{'text': '0 1 '}], 'usage': {'completion_tokens': 2}})])
        self.assertIsNone(row['client_total_s'])
        self.assertTrue(validate_request(row, [0, 1]))
    def test_trickling_sse_does_not_rearm_absolute_budget(self):
        installed = {}
        def handler(signum, callback):
            old = installed.get(signum, signal.SIG_DFL)
            installed[signum] = callback
            return old
        class Response:
            status_code = 200
            def __enter__(self): return self
            def __exit__(self, *unused): pass
            def raise_for_status(self): pass
            def iter_lines(self):
                # Every simulated gap is < 300 seconds; only the absolute alarm
                # ends this stream. No real sleeping/network/server is involved.
                for token in range(3):
                    yield sse({'choices': [{'text': str(token) + ' '}]})
                installed[signal.SIGALRM](signal.SIGALRM, None)
                yield 'data: [DONE]'
        class Client:
            def __init__(self, **unused): pass
            def __enter__(self): return self
            def __exit__(self, *unused): pass
            def stream(self, *unused, **kwargs): return Response()
        with patch.dict('sys.modules', {'httpx': types.SimpleNamespace(Client=Client)}), \
             patch('client.signal.getitimer', return_value=(0.0, 0.0)), \
             patch('client.signal.signal', side_effect=handler), \
             patch('client.signal.setitimer') as timer, \
             patch('client.time.perf_counter', side_effect=[100, 110, 200, 399, 400.01]):
            row = request('unused', {}, '/v1/completions', {}, Tokenizer(), 'deadline')
        self.assertEqual(timer.call_args_list, [call(signal.ITIMER_REAL, 300), call(signal.ITIMER_REAL, 0)])
        self.assertEqual(row['error']['type'], 'TimeoutError')
        self.assertFalse(row['done'])
        self.assertEqual(row['visible_roundtrip_ids'], [0, 1, 2])
        self.assertEqual(len(row['sse_events']), 3)
    def test_normal_eos_not_overridden_in_request(self):
        for arm in ('native-none', 'omlx-mtp', 'lithos-dspark'):
            path, body = body_for(arm, 'model', 'prompt', 'text')
            self.assertNotIn('ignore_eos', body)
            self.assertNotIn('stop', body)
            self.assertEqual(body['temperature'], 0)
            self.assertEqual(body.get('max_tokens', body.get('max_completion_tokens')), 32)
            self.assertEqual(path, '/v1/chat/completions' if arm.startswith('lithos') else '/v1/completions')
    def test_mtp_logs_capacity_denominator_not_actual_proposals(self):
        text = f'VLM MTP enabled for {MODEL_ID}, drafter=/mtp\n'
        for i in range(6):
            text += f'vlm_mtp decode started: request=r{i} uid=-{i + 1} block_size=None\n'
            text += (f'vlm_mtp stats: request=r{i} finish=length rounds=16 accepted=15/32 (46.9%) '
                     'tokens_per_round=1.94 emitted=32 block_size=3\n')
        row = official_log_evidence('omlx-mtp', text, target='/target', draft='/mtp', packs={})
        self.assertTrue(row['valid'], row)
        self.assertEqual(row['stats'][0]['nominal_capacity_denominator'], 32)
        self.assertNotIn('drafted', row['stats'][0])
        failed = official_log_evidence('omlx-mtp', text + 'VLM MTP load failed; toggle ignored',
                                       target='/target', draft='/mtp', packs={})
        self.assertFalse(failed['valid'])
        self.assertFalse(official_log_evidence('omlx-none', text, target='/target', draft='/mtp', packs={})['valid'])
    def test_lithos_logs_reject_hit_and_wrong_recipe(self):
        text = ('Using existing target pack: /pack\nUsing existing draft pack: /draftpack\n'
                'Serving backend=m5_ultra_80c, target=/target, draft=/draft, verify=8, recipe contexts=[]\n')
        rows = [{'cached_prompt_tokens': 0, 'verify_tokens': 8, 'setup_ms': 50 if i == 0 else 0}
                for i in range(6)]
        def check(prefix, values):
            log = prefix + '\n'.join('Generation: ' + repr(r) for r in values)
            return official_log_evidence('lithos-dspark', log, target='/target', draft='/draft',
                packs={'target': {'path': '/pack'}, 'draft': {'path': '/draftpack'}})
        self.assertTrue(check(text, rows)['valid'])
        rows[4]['cached_prompt_tokens'] = 19
        self.assertFalse(check(text, rows)['valid'])
        rows[4]['cached_prompt_tokens'] = 0
        self.assertFalse(check(text.replace('contexts=[]', 'contexts=[128]'), rows)['valid'])
    def test_matrix_preserves_negative_pair(self):
        self.assertEqual(len(CONFIGS), 15)
        self.assertEqual([c['arm'] for c in CONFIGS].count('native-none'), 3)
        rows = []
        for c in CONFIGS:
            row = dict(c, valid=True, median_output_tok_s=100, median_TTFT_s=1, median_wall_s=1)
            if c == {'pair': 2, 'arm': 'omlx-mtp'}:
                row['median_output_tok_s'] = 80
            rows.append(row)
        result = summarize(rows)
        self.assertFalse(result['descriptive_aggregates']['omlx']['every_pair_faster'])
        self.assertEqual(result['descriptive_aggregates']['omlx']['all_paired_points'], [1, .8, 1])
        self.assertEqual(len(result['rows']), 15)
        rows[0]['valid'] = False
        self.assertFalse(summarize(rows)['rows'][0]['valid'])
    def test_failed_pair_not_silently_pooled(self):
        rows = [dict(c, valid=c != {'pair': 2, 'arm': 'omlx-mtp'},
                     median_output_tok_s=100, median_TTFT_s=1, median_wall_s=1) for c in CONFIGS]
        result = summarize(rows)['descriptive_aggregates']['omlx']
        self.assertFalse(result['three_pairs_complete'])
        self.assertIsNone(result['paired_throughput_geomean'])
    def test_direct_cli_and_exact_pack_launch(self):
        plan = {'workload': {'model_dir': '/target'}, 'native_source_root': '/source',
                'native_source_manifest': '/manifest', 'qualified_model_settings': {'omlx-mtp': {'vlm_mtp_enabled': True}},
                'packs': {'lithos-dspark': {'target': {'path': '/pack'}, 'draft': {'path': '/draftpack'}}},
                'drafters': {'lithos': '/draft'}}
        omlx = specification(plan, 'omlx-mtp', '/new', 1234, 'key')
        self.assertTrue(omlx['command'][0].endswith('/bin/omlx'))
        lithos = specification(plan, 'lithos-dspark', '/new', 1234, 'key')
        self.assertEqual(lithos['command'][2:4], ['-m', 'monolith.serve'])
        self.assertEqual(lithos['command'][lithos['command'].index('--pack') + 1], '/pack')
        self.assertEqual(lithos['command'][lithos['command'].index('--draft-quantization') + 1], 'none')
        self.assertNotIn('--kernel-config', lithos['command'])
    def test_reap_and_port_absence_both_required(self):
        child = types.SimpleNamespace(pid=123, returncode=-15, poll=lambda: -15, wait=lambda: -15)
        class Socket:
            def __enter__(self): return self
            def __exit__(self, *unused): pass
            def settimeout(self, unused): pass
            def connect_ex(self, unused): return 61
        with patch.object(run, 'ps', return_value={'returncode': 1, 'stdout': '', 'stderr': ''}), \
             patch.object(run, 'listeners', return_value={'pids': []}), \
             patch.object(run.socket, 'socket', return_value=Socket()):
            self.assertTrue(run.close_server(child, 1234)['valid'])
        with patch.object(run, 'ps', return_value={'returncode': 1, 'stdout': '', 'stderr': ''}), \
             patch.object(run, 'listeners', return_value={'pids': [456]}), \
             patch.object(run.socket, 'socket', return_value=Socket()):
            self.assertFalse(run.close_server(child, 1234)['valid'])


if __name__ == '__main__':
    unittest.main()
