"""Stdlib-only observer seam; actual engine enters before the frozen install call."""
from contextlib import contextmanager

COUNTER_SCOPE = ('Per-layer Python candidate/fallback branches, including graph tracing; '
                 'not GPU dispatches, completed kernels, or per-token device calls.')


def need(ok, message):
    if not ok: raise ValueError(message)


class Facade:
    """Expose the frozen graph handle interface; delegate all teardown to combined."""
    def __init__(self, scope, engine, emit):
        self.scope, self.engine, self.emit = scope, engine, emit
        self.decoder = scope.graph_handle.decoder
        self.runner, self.model, self.short = scope.runner, scope.model, scope.short
        self.handles = list(scope.gdn_handles)
        self.layers = [(i, layer.linear_attn) for i, layer in enumerate(self.model.layers) if not layer.is_attn]
        need(len(self.layers) == len(self.handles) == 48
             and len({id(h) for h in self.handles}) == 48, '48 actual distinct handles required')
        need(all(h.layer is layer and vars(layer).get('_decode') is h.wrapper
                 and h.calls == {'candidate': 0, 'fallback': 0}
                 for (_, layer), h in zip(self.layers, self.handles)), 'actual installed layers and initial counters')
        self.record = dict(status='INSTALLED', engine_id=id(engine), runner_id=id(self.runner),
            model_id=id(self.model), decoder_id=id(self.decoder), graph_handle_id=id(scope.graph_handle),
            graph_functions_before=len(self.decoder.long._functions),
            short_functions_before=len(self.short._functions), short_packed_before=self.short._packed is not None,
            counter_scope=COUNTER_SCOPE, layers=[dict(layer=i, layer_id=id(layer), handle_id=id(h),
                wrapper_id=id(h.wrapper), initial_counts=dict(h.calls)) for (i, layer), h in zip(self.layers, self.handles)])
        need(self.record['graph_functions_before'] == self.record['short_functions_before'] == 0
             and self.record['short_packed_before'] is False, 'install before any trace')
        self.closed = False
        self.emit(dict(event='combined-installed', **self.record))

    def restore(self):
        if self.closed: return
        try:
            # Read Python counters while handles still own their original layers.
            for row, h in zip(self.record['layers'], self.handles): row['final_counts'] = dict(h.calls)
            self.record['graph_functions_before_restore'] = len(self.decoder.long._functions)
            self.record['short_functions_before_restore'] = len(self.short._functions)
            self.scope.restore()  # Frozen ownership, graph/cache clearing and reverse handle restore.
            need(self.scope.closed and self.runner.model is self.model and self.runner.decode_model is self.short
                 and all(h.closed and h.layer is None and h.wrapper is None for h in self.handles)
                 and all('_decode' not in vars(layer) for _, layer in self.layers)
                 and not self.short._functions and self.short._packed is None, 'combined release proof')
            self.record.update(status='RESTORED', handles_closed=48, native_decode_restored=True,
                               caches_cleared=True)
            self.closed = True
            self.emit(dict(event='combined-restored', **self.record))
        except BaseException as exc:
            self.record.update(status='RESTORE_FAILED', error=dict(type=type(exc).__name__, message=str(exc)))
            self.emit(dict(event='combined-restore-failed', **self.record))
            raise
        finally:
            if self.closed:
                self.scope = self.engine = self.runner = self.model = self.short = self.decoder = None
                self.handles.clear(); self.layers.clear()


class Bridge:
    def __init__(self, combined, emit):
        self.combined, self.emit = combined, emit
        self._engine = self._q = self._original_adapted = self._wrapper = None
        self.facades = []

    def integration_module(self, q):
        need(self._q is None, 'one private qualification module per bridge')
        self._q, self._original_adapted = q, q.adapted
        original = q.adapted

        @contextmanager
        def adapted(base, replay, integration, witness, counter, recording, fixtures, routes):
            need(integration is self, 'bridge integration identity')
            with original(base, replay, integration, witness, counter, recording, fixtures, routes) as coverage:
                route_observer = base.Observer
                bridge = self

                class EngineAwareObserver(route_observer):
                    def __init__(self, engine, comparator, label):
                        if label != 'candidate':
                            super().__init__(engine, comparator, label)
                            return
                        need(bridge._engine is None, 'candidate observer reentry')
                        bridge._engine = engine
                        try: super().__init__(engine, comparator, label)
                        finally: bridge._engine = None

                base.Observer = EngineAwareObserver
                try: yield coverage
                finally:
                    need(base.Observer is EngineAwareObserver, 'foreign private observer replacement')
                    base.Observer = route_observer

        self._wrapper = adapted; q.adapted = adapted
        return self

    def install(self, runner, *, enabled=False):
        if type(enabled) is not bool: raise TypeError('enabled must be bool')
        if not enabled: return None
        engine = self._engine
        need(engine is not None and engine.runner is runner, 'actual currently constructing candidate engine required')
        need(not self.facades, 'one combined candidate installation')
        scope = self.combined.install(engine, gdn_prework=True, route_graph=True)
        try: facade = Facade(scope, engine, self.emit)
        except BaseException:
            scope.restore()
            raise
        self.facades.append(facade)
        return facade

    def close_private_hook(self):
        need(self._engine is None, 'observer construction still active')
        if self._q is not None:
            need(self._q.adapted is self._wrapper, 'foreign private adapted replacement')
            self._q.adapted = self._original_adapted
            self._q = self._original_adapted = self._wrapper = None

    def snapshot(self):
        records = [f.record for f in self.facades]
        return dict(installs=len(records), restores=sum(f.closed for f in self.facades),
                    handle_count=sum(len(r['layers']) for r in records), records=records,
                    trace_counter_scope=COUNTER_SCOPE)

    def checked(self):
        result = self.snapshot()
        need(result['installs'] == result['restores'] == 1 and result['handle_count'] == 48,
             'exact combined lifetime must close')
        record = result['records'][0]
        need(record['status'] == 'RESTORED' and record['handles_closed'] == 48,
             'all actual handles restored')
        need(all(type(r['final_counts']['candidate']) is int and r['final_counts']['candidate'] > 0
                 and type(r['final_counts']['fallback']) is int and r['final_counts']['fallback'] == 0
                 for r in record['layers']), 'every GDN layer must trace the candidate without fallback')
        return result
