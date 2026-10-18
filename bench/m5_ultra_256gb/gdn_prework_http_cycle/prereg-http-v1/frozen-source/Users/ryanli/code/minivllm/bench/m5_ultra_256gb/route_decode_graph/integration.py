"""Default-OFF bench installation at fresh target-only ModelRunner construction.

Only this instance's decode_model changes. Its original callable still handles
short/out-of-domain decode; the frozen long candidate owns its own guard and
eager fallback. Both normal decode and ahead already use this entry point.
Install before requests/observers; no hot toggle, cache migration, eval or patch
of native methods. Restore observers before the returned cleanup scope. Weights
must remain fixed while work/ahead exists; this does not implement hot reload.
"""
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import FunctionType
import weakref

HERE = Path(__file__).resolve().parent
CANDIDATE_SHA256 = '6531be25acb53713d6ddcd930d2c778f6640a3b1f53304fd5375d20b044525c3'
RUNNER_SHA256 = '14fd964c57814846887585a9cbbdf1e11d2500ab80c23b10c945d2aa9316b7b1'
CACHE_SHA256 = '056a3bf3bc42163e93f55eb914fa35d5c081f803da9b5faa740588a67057a791'
SHORT_SHA256 = '22111101109b9d9c69e8fd114aa114533fe708e462303d5e72bedd6d217da1bf'
_installed = weakref.WeakSet()


def need(ok, message):
    if not ok: raise ValueError(message)


def verify(module, path, digest, cls, names):
    need(Path(module.__file__).resolve() == path and
         hashlib.sha256(path.read_bytes()).hexdigest() == digest, 'native source mismatch: ' + str(path))
    for name in names:
        fn = getattr(cls, name)
        need(type(fn) is FunctionType and fn.__globals__ is vars(module)
             and Path(fn.__code__.co_filename).resolve() == path, 'unrecognized loaded native method: ' + name)


def _candidate():
    path = HERE / 'candidate.py'
    need(hashlib.sha256(path.read_bytes()).hexdigest() == CANDIDATE_SHA256, 'route candidate source mismatch')
    spec = importlib.util.spec_from_file_location('_e024_route_install_candidate', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.RoutePreservingDecode


class _Dispatch:
    def __init__(self, original, long):
        self.original, self.long = original, long

    def __call__(self, tokens, ctx):
        used = getattr(ctx, 'max_len', None)
        if type(used) is int and 257 <= used <= 1536:
            return self.long(tokens, ctx)
        return self.original(tokens, ctx)


class Installation:
    """Cleanup handle, not a runtime enable/disable API; single installation only.

    Observer wrappers must be restored first. Refuse to overwrite a foreign
    decoder; a failed restore can be retried after that wrapper is removed.
    Closing releases this handle's strong runner/model references. Any requests
    must already be drained, or the caller must be tearing down the engine.
    """
    def __init__(self, runner, previous, decoder):
        self.runner, self.previous, self.decoder = runner, previous, decoder
        self.closed = False

    def __enter__(self):
        need(not self.closed, 'installation is already closed')
        return self

    def restore(self):
        if self.closed: return
        need(self.runner.decode_model is self.decoder, 'restore observer/foreign decoder before route installation')
        self.runner.decode_model = self.previous
        self.runner = self.previous = self.decoder = None
        self.closed = True

    def __exit__(self, exc_type, exc, tb):
        self.restore()
        return False


def install(runner, *, enabled=False):
    """OFF returns None without touching runner; ON returns a cleanup handle."""
    if type(enabled) is not bool: raise TypeError('enabled must be bool')
    if not enabled: return None
    need(runner not in _installed, 'route installation already attempted successfully on this runner')
    module = sys.modules.get(type(runner).__module__)
    need(module is not None and type(runner) is getattr(module, 'ModelRunner', None), 'actual ModelRunner required')
    cls = module.DecodeBatch; cache = sys.modules.get(cls.__module__)
    need(cache is not None and cls is getattr(cache, 'DecodeBatch', None), 'actual DecodeBatch source required')
    path = Path(module.__file__).resolve(); cache_path = Path(cache.__file__).resolve()
    need(path.name == 'model_runner.py' and path.parent.name == 'engine'
         and cache_path == path.parent.parent / 'cache.py', 'mixed native source roots')
    verify(module, path, RUNNER_SHA256, type(runner), ('__init__', 'run', '_sync_batch', 'drop_many'))
    verify(cache, cache_path, CACHE_SHA256, cls, ('__init__', 'rebuild'))
    need(module.mx is cache.mx and cache.mx is sys.modules.get('mlx.core'), 'mixed loaded MLX backend')
    batch = runner.batch
    need(isinstance(batch, cls) and batch.model is runner.model and runner.spec is None
         and type(batch.req_ids) is list and not batch.req_ids and type(batch.lens) is list and not batch.lens
         and type(batch.layers) is list and len(batch.layers) == len(runner.model.layers)
         and all(x is None for x in batch.layers) and type(runner.staging) is dict and not runner.staging
         and type(runner.feats) is dict and not runner.feats and runner._ahead is None,
         'fresh empty target-only runner required; no state migration')
    previous = runner.decode_model
    if previous is not runner.model:
        short = sys.modules.get(type(previous).__module__)
        need(short is not None and type(previous) is getattr(short, 'CompiledDecode', None)
             and previous.model is runner.model, 'unwrapped original short decoder required')
        verify(short, path.parent/'compiled_decode.py', SHORT_SHA256, type(previous), ('__init__', '__call__'))
        need(short.mx is module.mx, 'short decoder uses another MLX backend')
    long = _candidate()(runner.model, enabled=True)
    decoder = _Dispatch(previous, long)
    handle = Installation(runner, previous, decoder)
    _installed.add(runner)
    runner.decode_model = decoder
    return handle
