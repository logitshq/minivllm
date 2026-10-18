"""Optional frozen-native worker/handler taps, separate from runner diagnostics.

No replacement worker loop, transport, tokenizer algorithm or extra flush. The
caller must stop/drain worker and HTTP handlers before restoring and writing.
"""
import hashlib
from pathlib import Path
from types import FunctionType, MethodType

DONOR_SHA = '9bcb37eff8bdecd79800afdedc82deacf9f7a0a9a7a1e0d62dc8ba49e70fb68f'


def need(ok, message):
    if not ok: raise ValueError(message)


class _Queue:
    def __init__(self, queue, recorder, rid):
        self.queue, self.recorder, self.rid = queue, recorder, rid
        self.put_count = self.get_count = 0
    def __getattr__(self, name): return getattr(self.queue, name)
    def put(self, item, *args, **kwargs):
        self.put_count += 1
        return self.recorder.timed('delivery_queue_put', self.queue.put, item, *args,
            fields=dict(rid=self.rid, ordinal=self.put_count, done=item.get('done'),
                        is_error='error' in item), **kwargs)
    def get(self, *args, **kwargs):
        self.get_count += 1
        result = self.recorder.timed('delivery_queue_get', self.queue.get, *args,
            fields=dict(rid=self.rid, ordinal=self.get_count), **kwargs)
        self.recorder.local.request = dict(rid=self.rid, ordinal=self.get_count)
        return result


class _Tokenizer:
    def __init__(self, tokenizer, recorder): self.tokenizer, self.recorder = tokenizer, recorder
    def __getattr__(self, name): return getattr(self.tokenizer, name)
    def decode(self, *args, **kwargs):
        return self.recorder.timed('text_decode', self.tokenizer.decode, *args,
            fields=dict(getattr(self.recorder.local, 'request', {})), **kwargs)


class _Writer:
    def __init__(self, writer, recorder): self.writer, self.recorder = writer, recorder
    def __getattr__(self, name): return getattr(self.writer, name)
    def write(self, value):
        return self.recorder.timed('http_write', self.writer.write, value,
            fields=dict(bytes=len(value), **getattr(self.recorder.local, 'request', {})))
    def flush(self):
        return self.recorder.timed('http_flush', self.writer.flush,
            fields=dict(getattr(self.recorder.local, 'request', {})))


class Delivery:
    def __init__(self, engine, worker, donor, tokenizer, recorder):
        self.engine, self.worker, self.donor = None, worker, donor
        self.tokenizer, self.recorder = tokenizer, recorder
        self.active_handlers = 0; self.active_step = False; self.closed = False
        self.saved_step = self.step = None
        self.saved_submit = ('submit' in vars(worker), vars(worker).get('submit'))
        original = type(worker).submit
        private = dict(original.__globals__); job = private['Job']
        def new_job(*args, **kwargs):
            result = job(*args, **kwargs)
            result.events = _Queue(result.events, recorder, result.rid)
            return result
        private['Job'] = new_job
        clone = FunctionType(original.__code__, private, original.__name__, original.__defaults__, original.__closure__)
        self.submit = MethodType(clone, worker)
        worker.submit = self.submit
        if engine is not None: self.attach_engine(engine)

    def attach_engine(self, engine):
        """Worker-thread hook after model load, before worker.initialize()."""
        need(not self.closed and self.engine is None and self.worker.engine is None, 'engine hook must precede initialize')
        previous = engine.step
        def step():
            need(not self.closed and not self.active_step, 'nested/closed engine step')
            self.active_step = True
            try: return self.recorder.timed('engine_step', previous)
            finally: self.active_step = False
        self.engine = engine; self.saved_step = ('step' in vars(engine), vars(engine).get('step'))
        self.step = step; engine.step = step

    def handler_class(self, **kwargs):
        scope = self
        base = self.donor.handler_class(self.worker, _Tokenizer(self.tokenizer, self.recorder), **kwargs)
        class Handler(base):
            def setup(self):
                super().setup()
                self.wfile = _Writer(self.wfile, scope.recorder)
            def do_POST(self):
                with scope.recorder.lock:
                    need(not scope.closed, 'handler entered after profile restored')
                    scope.active_handlers += 1
                scope.recorder.local.request = {}
                try: return scope.recorder.timed('http_handler', super().do_POST)
                finally:
                    scope.recorder.local.request = {}
                    with scope.recorder.lock: scope.active_handlers -= 1
        return Handler

    def restore(self):
        if self.closed: return
        need(not self.active_step and not self.active_handlers, 'delivery still active')
        need((self.engine is None or self.engine.step is self.step) and self.worker.submit is self.submit, 'foreign delivery observer')
        hooks = [(self.worker, 'submit', self.saved_submit)]
        if self.engine is not None: hooks.insert(0, (self.engine, 'step', self.saved_step))
        for obj, name, saved in hooks:
            if saved[0]: setattr(obj, name, saved[1])
            else: vars(obj).pop(name, None)
        self.closed = True
        self.engine = self.worker = self.tokenizer = self.donor = None
        self.step = self.submit = self.saved_step = self.saved_submit = None


def install(engine, worker, donor, tokenizer, recorder, *, enabled=False):
    if type(enabled) is not bool: raise TypeError('enabled must be bool')
    if not enabled: return None
    path = Path(donor.__file__).resolve()
    need(path.name == 'native_server.py' and hashlib.sha256(path.read_bytes()).hexdigest() == DONOR_SHA,
         'frozen actual native HTTP donor required')
    need(type(worker) is donor.BatchWorker and worker.engine is None and not worker.ready.is_set()
         and 'submit' not in vars(worker) and worker.trace is None,
         'attach before worker initialization; scheduling observer must be OFF')
    need(type(worker).submit.__globals__ is vars(donor)
         and type(worker).submit.__code__.co_filename == str(path), 'foreign worker method')
    need(donor.handler_class.__globals__ is vars(donor)
         and Path(donor.handler_class.__code__.co_filename).resolve() == path, 'foreign handler factory')
    return Delivery(engine, worker, donor, tokenizer, recorder)
