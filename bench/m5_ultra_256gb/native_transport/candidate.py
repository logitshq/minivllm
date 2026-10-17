"""E022: use StreamRequestHandler's existing per-connection Nagle switch.

No model import, socket creation, listener, SSE rewriting, or buffering change.
Both arms read back the actual accepted socket option and fail closed on a
mismatch. The optional receipt callback runs during connection setup only.
"""
import socket


def transport_handler(base, *, tcp_nodelay=False, receipt=None, fatal=None):
    if type(tcp_nodelay) is not bool:
        raise TypeError('tcp_nodelay must be bool')
    if receipt is not None and not callable(receipt):
        raise TypeError('receipt must be callable or None')
    if fatal is not None and not callable(fatal):
        raise TypeError('fatal must be callable or None')

    class TransportHandler(base):
        # CPython StreamRequestHandler.setup applies this to the accepted
        # socket before creating rfile/wfile. OFF retains the original default.
        disable_nagle_algorithm = tcp_nodelay

        def setup(self):
            actual = None
            try:
                super().setup()
                actual = self.connection.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY)
                if actual != int(tcp_nodelay):
                    raise RuntimeError(f'accepted socket TCP_NODELAY={actual}, expected {int(tcp_nodelay)}')
                if receipt is not None:
                    receipt({'tcp_nodelay_requested': tcp_nodelay, 'tcp_nodelay_actual': actual,
                             'wbufsize': self.wbufsize, 'rbufsize': self.rbufsize})
            except BaseException as error:
                # ThreadingMixIn catches ordinary Exceptions and otherwise
                # could let a readiness retry hide the broken connection.
                if fatal is not None:
                    fatal({'tcp_nodelay_requested': tcp_nodelay, 'tcp_nodelay_actual': actual,
                           'error_type': type(error).__name__, 'error': str(error)})
                raise

    return TransportHandler
