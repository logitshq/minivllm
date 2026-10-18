"""Default-OFF instance composition; importing this module is stdlib-only.

Install before observers or any model call; the caller owns queue quiescence.
Restore only after the worker stops, with observers already restored. This is
not hot switching, parameter reload, a server, or a numerical qualification.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import weakref


HERE = Path(__file__).resolve().parent
GRAPH = HERE.parent / 'route_decode_graph/integration.py'
GDN = HERE.parent / 'gdn_prework_unbound/candidate.py'
FIXED = {
    GRAPH: '4a8dc5518f2ea0a913ad0a5d753a5032324ccb69bacede416abe8ae126bfa800',
    GRAPH.with_name('candidate.py'): '6531be25acb53713d6ddcd930d2c778f6640a3b1f53304fd5375d20b044525c3',
    GDN: '1bf5a837c6515ca505bc5d1beaa1718b88e1ccd52609f6814bb995afbe6c1260',
    HERE.parent / 'long_decode_graph/candidate.py': 'eb14b671a66764fe677219762090acb495b89d384cf078f3f63b918a40aa2b84',
    HERE.parent / 'long_decode_graph/model-source-pins.json': '7143538a168bfcd8b01292c9687472c8647db393c6d2a510d628f71c670260c5',
}
_installed = weakref.WeakSet()


def need(ok, message):
    if not ok: raise ValueError(message)


def source_pins():
    for path, digest in FIXED.items():
        need(hashlib.sha256(path.read_bytes()).hexdigest() == digest,
             'frozen dependency changed: ' + str(path))
    return {**{str(p): s for p, s in FIXED.items()},
            str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def _dependencies():
    source_pins()
    modules = []
    for label, path in (('graph', GRAPH), ('gdn', GDN)):
        spec = importlib.util.spec_from_file_location('_combined_frozen_' + label, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return tuple(modules)


def _fresh(engine, graph):
    """Reuse pinned native method checks, including the unchanged E028-OFF cache."""
    runner = engine.runner
    need(runner not in _installed, 'one installation lifetime per runner')
    module = sys.modules.get(type(runner).__module__)
    need(module is not None and type(runner) is getattr(module, 'ModelRunner', None),
         'actual unwrapped ModelRunner required')
    path = Path(module.__file__).resolve()
    cache = sys.modules.get(module.DecodeBatch.__module__)
    need(cache is not None and Path(cache.__file__).resolve() == path.parent.parent / 'cache.py',
         'mixed native source roots')
    graph.verify(module, path, graph.RUNNER_SHA256, type(runner),
                 ('__init__', 'run', '_sync_batch', 'drop_many'))
    graph.verify(cache, Path(cache.__file__).resolve(), graph.CACHE_SHA256,
                 module.DecodeBatch, ('__init__', 'rebuild'))
    need(module.mx is cache.mx is sys.modules.get('mlx.core'), 'mixed MLX backend')
    batch = runner.batch
    need(engine.model is runner.model and not engine.requests and runner.spec is None
         and type(batch) is module.DecodeBatch and batch.model is runner.model
         and type(batch.req_ids) is list and not batch.req_ids
         and type(batch.lens) is list and not batch.lens
         and type(batch.layers) is list and len(batch.layers) == 64
         and all(value is None for value in batch.layers)
         and type(runner.staging) is dict and not runner.staging
         and type(runner.feats) is dict and not runner.feats and runner._ahead is None
         and 'run' not in vars(runner) and runner.max_len == 2048,
         'fresh empty context2048 target-only runner; E028/observers OFF required')
    short = runner.decode_model
    short_module = sys.modules.get(type(short).__module__)
    need(short_module is not None and type(short) is getattr(short_module, 'CompiledDecode', None)
         and short.model is runner.model and short_module.mx is module.mx,
         'original compiled short decoder required')
    graph.verify(short_module, path.parent / 'compiled_decode.py', graph.SHORT_SHA256,
                 type(short), ('__init__', '__call__'))
    need(short._fuse_decode is True and type(short._functions) is dict
         and not short._functions and short._packed is None, 'unused original fused short decoder required')
    layers = [layer.linear_attn for layer in runner.model.layers if not layer.is_attn]
    need(len(runner.model.layers) == 64 and len(layers) == len({id(x) for x in layers}) == 48,
         'actual distinct64/48 layer layout required')
    need(all(layer._gates_compile_enabled is True and layer._norm_reuse_enabled is False
             and layer._compiled_decode_gates is None and '_decode' not in vars(layer)
             for layer in layers), 'fresh gates ON / norm OFF / unmodified GDN layers required')
    return runner, short, layers


class Installation:
    """One construction/teardown scope; retained compiled graphs are forbidden."""
    def __init__(self, runner, short):
        self.runner, self.model, self.short = runner, runner.model, short
        self.gdn_handles, self.graph_handle = [], None
        self.closed = False

    def restore(self):
        if self.closed: return
        expected = self.graph_handle.decoder if self.graph_handle is not None else self.short
        # Check all ownership before any partial rollback. Never remove GDN
        # while a foreign observer could retain a graph traced with its hooks.
        need(self.runner.model is self.model and self.runner.decode_model is expected
             and 'run' not in vars(self.runner), 'restore all observers before combined installation')
        need(all(vars(h.layer).get('_decode') is h.wrapper for h in self.gdn_handles),
             'restore foreign GDN observers before combined installation')
        if self.graph_handle is not None:
            self.graph_handle.decoder.long._functions.clear()
            self.graph_handle.restore()
            self.graph_handle = None
        # These caches started empty. Discard graphs built inside this scope,
        # including the unfused fallback trace if native eligibility changed.
        self.short._functions.clear()
        self.short._packed = None
        first = None
        remaining = []
        for handle in reversed(self.gdn_handles):
            try: handle.restore()
            except BaseException as exc:
                remaining.append(handle)
                if first is None: first = exc
        self.gdn_handles = list(reversed(remaining))
        if first is not None: raise first
        self.runner = self.model = self.short = None
        self.closed = True


def install(engine, *, gdn_prework=False, route_graph=False):
    """Two independent switches; OFF/OFF touches neither engine nor sources.

    Current native flags stay short/fuse/gates ON, norm OFF; E028 stays OFF.
    Runtime eligibility and arithmetic remain in the two frozen candidates.
    """
    if type(gdn_prework) is not bool or type(route_graph) is not bool:
        raise TypeError('switches must be explicit bools')
    if not (gdn_prework or route_graph): return None
    graph, gdn = _dependencies()
    runner, short, layers = _fresh(engine, graph)
    scope = Installation(runner, short)
    try:
        if gdn_prework:
            for layer in layers:
                handle = gdn.install(layer, sys.modules[type(layer).__module__], enabled=True)
                need(handle is not None, 'GDN installation unexpectedly OFF')
                scope.gdn_handles.append(handle)
        # No graph exists before all hooks have their final meaning.
        if route_graph:
            scope.graph_handle = graph.install(runner, enabled=True)
            need(scope.graph_handle is not None, 'graph installation unexpectedly OFF')
    except BaseException:
        scope.restore()
        raise
    _installed.add(runner)
    return scope


if __name__ == '__main__':
    print(json.dumps(dict(schema='GDN-graph-combined-description-v1',
        gdn_prework=False, route_graph=False, model_loaded=False,
        numerical_qualified=False, http_qualified=False, performance_qualified=False,
        entry='install(engine, gdn_prework=False, route_graph=False)',
        scope='Fresh instance composition only; no device CLI or live switching')))
