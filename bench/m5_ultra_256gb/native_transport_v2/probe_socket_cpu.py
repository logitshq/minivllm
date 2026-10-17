"""Bounded localhost TCP_NODELAY semantics probe; no HTTP/model/MLX service.

Two short-lived 127.0.0.1 ephemeral listeners, no subprocesses or threads.
The output is exclusive-created before any socket, preserving partial failure.
"""
import argparse
from datetime import datetime, timezone
import errno
import hashlib
import json
from pathlib import Path
import platform
import socket
from socketserver import StreamRequestHandler
import sys
import traceback

from candidate import transport_handler


def probe(enabled, row):
    listener, client, accepted, handler = None, None, None, None
    try:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.settimeout(3)
        listener.bind(('127.0.0.1', 0))
        listener.listen(1)
        address = listener.getsockname()
        row['listener_address'] = list(address)
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(3)
        client.connect(address)
        accepted, peer = listener.accept()
        accepted.settimeout(3)
        row['accepted_peer'] = list(peer)
        row['before_setup_raw'] = accepted.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY)
        row['receipts'] = []
        handler_type = transport_handler(StreamRequestHandler, tcp_nodelay=enabled,
                                         receipt=row['receipts'].append)
        handler = handler_type.__new__(handler_type)
        handler.request = accepted
        handler.setup()
        row['after_setup_raw'] = accepted.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY)
        assert bool(row['after_setup_raw']) is enabled
        payload = b'data: {"choices":[]}\n\ndata: [DONE]\n\n'
        handler.wfile.write(payload)
        handler.wfile.flush()
        received = bytearray()
        while len(received) < len(payload):
            chunk = client.recv(len(payload) - len(received))
            if not chunk:
                raise RuntimeError('premature socket EOF')
            received.extend(chunk)
        row.update(payload_hex=payload.hex(), received_hex=received.hex())
        assert bytes(received) == payload
        row['direct_setsockopt_trials'] = []
        for requested_raw in (0, 1, 4):
            accepted.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, requested_raw)
            actual = accepted.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY)
            row['direct_setsockopt_trials'].append({'set_raw': requested_raw,
                'get_raw': actual, 'get_enabled': bool(actual)})
            assert bool(actual) == bool(requested_raw)
    finally:
        # finish() closes rfile/wfile; the socket descriptors are separately
        # owned here. Always close all three even if setup/finish failed.
        try:
            if handler is not None and hasattr(handler, 'wfile'):
                handler.finish()
        finally:
            sockets = {'accepted': accepted, 'client': client, 'listener': listener}
            for connection in sockets.values():
                if connection is not None:
                    connection.close()
            row['closed_filenos'] = {name: connection.fileno()
                for name, connection in sockets.items() if connection is not None}
            if 'listener_address' in row:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as check:
                    check.settimeout(3)
                    code = check.connect_ex(tuple(row['listener_address']))
                row['post_close_connect_errno'] = code
                row['post_close_connect_error_name'] = errno.errorcode.get(code)
                assert code == errno.ECONNREFUSED, ('port still accepts or cannot prove closed', code)
            assert all(fd == -1 for fd in row['closed_filenos'].values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    files = [Path(__file__).resolve(), Path(__file__).resolve().with_name('candidate.py')]
    result = {'schema': 'E022-v2-localhost-CPU-probe',
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.executable, 'python_version': sys.version,
        'platform': platform.platform(), 'uname': platform.uname()._asdict(),
        'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        'scope': 'TCP boolean semantics and bytes only; no HTTP latency/model/GPU claim', 'arms': []}
    with args.out.open('x') as output:
        try:
            for enabled in (False, True):
                row = {'tcp_nodelay_requested': enabled}
                result['arms'].append(row)
                probe(enabled, row)
            result['status'] = 'PASS'
        except BaseException as error:
            result.update(status='FAIL', error_type=type(error).__name__, error=str(error),
                          traceback=traceback.format_exc())
            raise
        finally:
            result['finished_utc'] = datetime.now(timezone.utc).isoformat()
            json.dump(result, output, indent=2)
            output.write('\n')
    print(json.dumps({'status': result['status'], 'raw': [r['after_setup_raw'] for r in result['arms']]}))


if __name__ == '__main__':
    main()
