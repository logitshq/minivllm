"""E022 isolated OFF/ON entrypoint over a pinned copy of the E020 native shim.

--help imports no MLX or model. Only main() loads the requested shim, and its
own main() owns the actual HTTP listener in this same process. Does not edit
the source shim, core, E020 result, client, or SSE payload.
"""
import argparse
import hashlib
from http.server import ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading

from candidate import transport_handler


def fatal_socket(path, row):
    """Every accepted-socket failure is terminal, even if its receipt cannot write."""
    try:
        with Path(path).open('x') as stream:
            json.dump({'schema': 'E022-fatal-socket-v2', 'pid': os.getpid(), **row}, stream, indent=2)
            stream.write('\n')
    finally:
        # Exit the whole directly owned process, not just the handler thread.
        # The controller observes this code, reaps the child and stops the run.
        os._exit(70)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server-source', type=Path, required=True)
    parser.add_argument('--server-source-sha256', required=True)
    parser.add_argument('--tcp-nodelay', choices=('off', 'on'), required=True)
    parser.add_argument('--socket-receipt', type=Path, required=True)
    parser.add_argument('--socket-failure', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--package-root', type=Path, required=True)
    parser.add_argument('--source-manifest', type=Path, required=True)
    args = parser.parse_args()
    source = args.server_source.resolve()
    actual_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    assert actual_sha == args.server_source_sha256, 'frozen native shim SHA mismatch'
    assert not args.socket_receipt.exists(), 'refuse to overwrite a previous socket receipt'
    assert not args.socket_failure.exists(), 'refuse a previous fatal socket receipt'
    assert args.socket_receipt.parent.is_dir(), 'receipt parent must already exist'
    assert args.socket_failure.parent.is_dir(), 'failure receipt parent must already exist'
    spec = importlib.util.spec_from_file_location('e022_frozen_native_server', source)
    if spec is None or spec.loader is None:
        raise RuntimeError('cannot load the frozen native shim')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.ThreadingHTTPServer is ThreadingHTTPServer
    written, lock = threading.Event(), threading.Lock()

    def first_receipt(row):
        # First readiness connection only, before any warm/formal request.
        # Every later accepted socket is still checked by candidate.setup().
        if written.is_set():
            return
        with lock:
            if written.is_set():
                return
            payload = {'schema': 'E022-socket-v2', 'pid': os.getpid(),
                       'server_source': str(source), 'server_source_sha256': actual_sha,
                       'scope': 'first accepted connection; every connection readback asserted', **row}
            with args.socket_receipt.open('x') as stream:
                json.dump(payload, stream, indent=2)
                stream.write('\n')
            written.set()

    def server_factory(address, handler):
        return ThreadingHTTPServer(address, transport_handler(handler,
            tcp_nodelay=args.tcp_nodelay == 'on', receipt=first_receipt,
            fatal=lambda row: fatal_socket(args.socket_failure, row)))

    # This injection is local to the loaded experiment module. The only
    # handler behavior added is the existing socket flag + readback above.
    module.ThreadingHTTPServer = server_factory
    sys.argv = [str(source), '--model', args.model, '--port', str(args.port),
                '--package-root', str(args.package_root),
                '--source-manifest', str(args.source_manifest)]
    module.main()


if __name__ == '__main__':
    main()
