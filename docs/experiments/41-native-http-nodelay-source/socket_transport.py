"""V121 accepted-socket transport only. No runtime/model/client imports."""
import json
import os
from pathlib import Path
import socket
import threading


def fatal_socket(path, row):
    try:
        with Path(path).open('x') as f:
            json.dump({'status': 'FATAL_V121_SOCKET', 'pid': os.getpid(), **row}, f, indent=2)
    finally:
        # ThreadingMixIn otherwise swallows handler errors and readiness could retry.
        os._exit(70)


def first_socket_receipt(path, failure_path):
    path, failure_path = Path(path), Path(failure_path)
    if path.exists() or failure_path.exists():
        raise FileExistsError('refuse previous socket success/failure evidence')
    written = threading.Event()
    lock = threading.Lock()
    def record(row):
        if written.is_set(): return
        with lock:
            if written.is_set(): return
            with path.open('x') as f:
                json.dump({'status': 'V121_FIRST_ACCEPTED_SOCKET', 'pid': os.getpid(),
                           'scope': 'first accepted connection; every connection is read back', **row}, f, indent=2)
            written.set()
    return record


def handler_with_nodelay(base, *, tcp_nodelay=False, receipt=None, fatal):
    if type(tcp_nodelay) is not bool or not callable(fatal) or (receipt is not None and not callable(receipt)):
        raise TypeError('strict bool and callable failure handler required')
    class Handler(base):
        disable_nagle_algorithm = tcp_nodelay
        def setup(self):
            actual = None
            try:
                super().setup()
                actual = self.connection.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY)
                # Darwin returns raw4 on this machine, not normalized1.
                if bool(actual) != tcp_nodelay:
                    raise RuntimeError(f'TCP_NODELAY readback {actual}, expected {int(tcp_nodelay)}')
                if receipt is not None:
                    receipt({'tcp_nodelay_requested': tcp_nodelay, 'tcp_nodelay_actual': actual,
                             'tcp_nodelay_enabled': bool(actual),
                             'wbufsize': self.wbufsize, 'rbufsize': self.rbufsize})
            except BaseException as error:
                fatal({'tcp_nodelay_requested': tcp_nodelay, 'tcp_nodelay_actual': actual,
                       'error_type': type(error).__name__, 'error': str(error)})
                raise
    return Handler
