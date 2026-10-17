"""E022 mock accepted sockets only: no listener, real socket, MLX, or model."""
import io
import socket
import json
from pathlib import Path
from socketserver import StreamRequestHandler, ThreadingMixIn
import tempfile
import unittest
from unittest.mock import patch

from candidate import transport_handler
from common import CONFIGS, GATE, launch_spec, summarize
from run import transport_result
from serve import fatal_socket


class FakeSocket:
    def __init__(self, initial=0, apply=True):
        self.option, self.apply, self.options, self.sent = initial, apply, [], []

    def setsockopt(self, level, name, value):
        self.options.append((level, name, value))
        if self.apply:
            self.option = int(value)

    def getsockopt(self, level, name):
        assert (level, name) == (socket.IPPROTO_TCP, socket.TCP_NODELAY)
        return self.option

    def makefile(self, mode, size):
        assert mode == 'rb' and size == -1
        return io.BytesIO()

    def sendall(self, payload):
        self.sent.append(bytes(payload))


def setup(enabled, sock=None):
    observed = []
    handler = transport_handler(StreamRequestHandler, tcp_nodelay=enabled, receipt=observed.append)
    instance = handler.__new__(handler)
    instance.request = sock or FakeSocket()
    instance.setup()
    return instance, observed


class TransportTests(unittest.TestCase):
    def test_off_keeps_stdlib_default(self):
        instance, observed = setup(False)
        self.assertEqual(instance.request.options, [])
        self.assertEqual(observed[0]['tcp_nodelay_actual'], 0)
        instance.finish()

    def test_on_sets_accepted_socket_option(self):
        instance, observed = setup(True)
        self.assertEqual(instance.request.options, [(socket.IPPROTO_TCP, socket.TCP_NODELAY, True)])
        self.assertEqual(observed[0]['tcp_nodelay_actual'], 1)
        self.assertEqual(observed[0]['wbufsize'], 0)
        instance.finish()

    def test_both_arms_send_identical_bytes_without_extra_buffer(self):
        payloads = [b'HTTP/1.1 200 OK\r\n\r\n', b'data: {"choices":[]}\n\n', b'data: [DONE]\n\n']
        for enabled in (False, True):
            instance, _ = setup(enabled)
            for payload in payloads:
                self.assertEqual(instance.wfile.write(payload), len(payload))
                instance.wfile.flush()
            self.assertEqual(instance.request.sent, payloads)
            instance.finish()

    def test_failed_set_and_unexpected_reference_option_stop(self):
        with self.assertRaises(RuntimeError):
            setup(True, FakeSocket(apply=False))
        with self.assertRaises(RuntimeError):
            setup(False, FakeSocket(initial=1))

    def test_configuration_fails_early(self):
        for value in (0, 1, 'on', None):
            with self.assertRaises(TypeError):
                transport_handler(StreamRequestHandler, tcp_nodelay=value)
        with self.assertRaises(TypeError):
            transport_handler(StreamRequestHandler, receipt='file')

    def test_no_shared_class_mutation(self):
        off = transport_handler(StreamRequestHandler, tcp_nodelay=False)
        on = transport_handler(StreamRequestHandler, tcp_nodelay=True)
        self.assertFalse(off.disable_nagle_algorithm)
        self.assertTrue(on.disable_nagle_algorithm)
        self.assertFalse(StreamRequestHandler.disable_nagle_algorithm)

    def test_threading_mixin_cannot_hide_fatal_callback(self):
        failures, swallowed, shut = [], [], []
        handler = transport_handler(StreamRequestHandler, tcp_nodelay=True, fatal=failures.append)
        class Server(ThreadingMixIn):
            def finish_request(self, request, address):
                instance = handler.__new__(handler)
                instance.request = request
                instance.setup()
            def handle_error(self, request, address):
                swallowed.append(True)
            def shutdown_request(self, request):
                shut.append(True)
        # Execute the real stdlib exception-swallowing method without starting
        # a thread, opening a socket, or listening on any network interface.
        Server().process_request_thread(FakeSocket(apply=False), ('mock', 0))
        self.assertEqual(swallowed, [True])
        self.assertEqual(shut, [True])
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]['tcp_nodelay_actual'], 0)

    def test_receipt_error_also_reaches_fatal_callback(self):
        failures = []
        def broken(row):
            raise OSError('receipt write failed')
        handler = transport_handler(StreamRequestHandler, tcp_nodelay=False,
                                    receipt=broken, fatal=failures.append)
        instance = handler.__new__(handler)
        instance.request = FakeSocket()
        with self.assertRaises(OSError):
            instance.setup()
        self.assertEqual(failures[0]['error_type'], 'OSError')

    def test_fatal_exits_whole_process_even_when_receipt_cannot_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'fatal.json'
            with patch('serve.os._exit', side_effect=SystemExit(70)) as exit_process:
                with self.assertRaises(SystemExit):
                    fatal_socket(path, {'error': 'failed'})
                exit_process.assert_called_once_with(70)
            self.assertEqual(json.loads(path.read_text())['error'], 'failed')
            with patch('serve.os._exit', side_effect=SystemExit(70)) as exit_process:
                with self.assertRaises(SystemExit):
                    fatal_socket(Path(tmp) / 'missing' / 'fatal.json', {})
                exit_process.assert_called_once_with(70)


class PlanTests(unittest.TestCase):
    def rows(self):
        return [dict(c, valid=True, median_wall_s=1 if c['arm'] == 'off' else .9,
                     median_TTFT_s=.1, median_output_tok_s=32 if c['arm'] == 'off' else 32/.9)
                for c in CONFIGS]

    def test_fixed_schedule_and_gate(self):
        self.assertEqual([(c['pair'], c['arm']) for c in CONFIGS],
                         [(1, 'off'), (1, 'on'), (2, 'on'), (2, 'off'), (3, 'off'), (3, 'on')])
        result = summarize(self.rows())
        self.assertTrue(result['advance'])
        self.assertEqual(result['gate'], GATE)
        self.assertAlmostEqual(result['paired_wall_geomean'], 1/.9)

    def test_single_negative_wall_pair_is_not_pooled_away(self):
        rows = self.rows()
        rows[2]['median_wall_s'] = 1.01
        result = summarize(rows)
        self.assertGreater(result['paired_wall_geomean'], 1.05)
        self.assertFalse(result['advance'])
        self.assertEqual(len(result['rows']), 6)

    def test_ttft_floor_and_invalid_process_reject(self):
        rows = self.rows()
        rows[1]['median_TTFT_s'] = .104
        self.assertFalse(summarize(rows)['advance'])
        rows = self.rows()
        rows[1]['valid'] = False
        result = summarize(rows)
        self.assertFalse(result['advance'])
        self.assertIsNone(result['paired_wall_geomean'])
        self.assertIsNone(result['pairs'][0]['wall_speedup'])

    def test_duplicates_rejected_and_partial_not_qualified(self):
        rows = self.rows()
        with self.assertRaises(AssertionError):
            summarize(rows + [rows[0]])
        self.assertFalse(summarize(rows[:4])['advance'])

    def test_late_socket_failure_overrides_initial_success(self):
        plan = {'reused_sources': {'native_server': '/frozen/server.py'}, 'native_server_sha256': 'sha'}
        receipt = {'schema': 'E022-socket-v1', 'pid': 123, 'tcp_nodelay_requested': True,
            'tcp_nodelay_actual': 1, 'wbufsize': 0, 'rbufsize': -1,
            'server_source': '/frozen/server.py', 'server_source_sha256': 'sha'}
        result = {'valid': True, 'release': {'pid': 123}, 'contract_failures': [],
                  'diagnostic_medians_including_invalid_samples': {}, 'median_wall_s': 1,
                  'median_TTFT_s': .1, 'median_output_tok_s': 32}
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / 'socket.json').write_text(json.dumps(receipt))
            config = {'pair': 1, 'arm': 'on'}
            self.assertTrue(transport_result(result, plan, config, directory)['valid'])
            (directory / 'socket-failure.json').write_text('{}')
            rejected = transport_result(result, plan, config, directory)
            self.assertFalse(rejected['valid'])
            self.assertIsNone(rejected['median_wall_s'])

    def test_launch_has_same_model_and_explicit_socket_arm(self):
        plan = {'reused_sources': {'native_server': '/frozen/server.py'}, 'native_server_sha256': 'sha'}
        e020 = {'runtime_versions': {'controller-native': {'python': '/venv/python'}},
                'workload': {'model_dir': '/model'}, 'native_source_root': '/core',
                'native_source_manifest': '/manifest', 'environment': {'MLX_MAX_OPS_PER_BUFFER': '50'}}
        off, on = [launch_spec(plan, e020, arm, '/out', 1234)['command'] for arm in ('off', 'on')]
        index = off.index('--tcp-nodelay') + 1
        self.assertEqual((off[index], on[index]), ('off', 'on'))
        off[index] = 'on'
        self.assertEqual(off, on)


if __name__ == '__main__':
    unittest.main()
