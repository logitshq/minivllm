"""Default-OFF single-runner suppression of one already submitted ahead root set.

Source prepared, not CPU/device qualified. This skips a redundant API submission,
not a model forward or proven GPU computation. The original run code object is
unchanged. Its private mx proxy forwards every eval and all other submissions.
No backend import, global patch, added evaluation or host token read occurs here.
"""
import hashlib
from pathlib import Path
import sys
from types import FunctionType, MethodType
import weakref

RUNNER_SHA = '14fd964c57814846887585a9cbbdf1e11d2500ab80c23b10c945d2aa9316b7b1'
CACHE_SHA = '056a3bf3bc42163e93f55eb914fa35d5c081f803da9b5faa740588a67057a791'
PROFILE_SHA = '58873f2eb395a0ca7d8ab1bac35f75b67581332187ada9be3de74fbdf77ef1bf'


def need(ok, message):
    if not ok: raise ValueError(message)


def references(values):
    # No extra strong reference to a retired token, cache or model survives a run.
    # Unsupported weakrefs make the following step ordinary; never infer by id().
    try: return tuple(weakref.ref(value) for value in values)
    except TypeError: return None


def identical(refs, values):
    return refs is not None and len(refs) == len(values) and all(ref() is value for ref,value in zip(refs,values))


class _Mx:
    def __init__(self, original, scope): self.original, self.scope = original, scope
    def __getattr__(self, name): return getattr(self.original, name)
    def async_eval(self, *args, **kwargs):
        scope = self.scope
        scope.ordinal += 1
        if (scope.active and scope.ordinal == 1 and scope.may_skip
                and not kwargs and identical(scope.expected, args)):
            scope.counts['omitted_current_submits'] += 1
            scope.may_skip = False
            return None  # The only omitted call site is native run's first async_eval.
        scope.counts['forwarded_async_submits'] += 1
        result = self.original.async_eval(*args, **kwargs)
        if scope.ordinal == 2:
            scope.last_success = references(args) if not kwargs else None
        return result  # Only a normally returned call can certify future roots.


class Scope:
    """Internal injectable CPU seam. Real callers must use source-verifying install.

    target is either this runner's run or the explicit frozen profile's cloned
    method. Ineligible calls execute the same original run code with all backend
    calls forwarded. A failed original call is propagated, never retried.
    """
    def __init__(self, runner, target, attribute, original, terminal_ids=()):
        self.runner, self.target, self.attribute, self.original = runner, target, attribute, original
        self.saved = (attribute in vars(target), vars(target).get(attribute))
        self.terminal_ids = frozenset(terminal_ids)
        self.closed = self.active = self.may_skip = False
        self.certificate = self.expected = self.last_success = None
        self.ordinal = 0
        self.counts = dict(runs=0, exact_hit_eligible=0, omitted_current_submits=0,
                           forwarded_async_submits=0, fallback_runs=0, errors=0)
        fn = original.__func__; private = dict(fn.__globals__)
        self.BatchKV = private['BatchKV']
        private['mx'] = _Mx(private['mx'], self)
        clone = FunctionType(fn.__code__, private, fn.__name__, fn.__defaults__, fn.__closure__)
        clone.__kwdefaults__ = fn.__kwdefaults__
        self.cloned = MethodType(clone, runner)
        self.wrapper = self.run
        setattr(target, attribute, self.wrapper)

    def roots(self, ahead):
        if ahead is None: return ()
        _, token, layers = ahead
        return (token, *(array for layer in layers for array in
            ((layer.K,layer.V) if isinstance(layer,self.BatchKV) else (layer.conv,layer.S))))

    def eligible(self, out):
        runner = self.runner
        if runner.spec is not None or out.prefill or out.finished_ids or len(out.decode) != 1:
            return False
        req = out.decode[0];sampling = req.sampling
        if (req.is_finished or not req.output_token_ids or sampling.temperature != 0
                or req.output_token_ids[-1] in self.terminal_ids
                or req.output_token_ids[-1] in sampling.stop_token_ids
                or len(req.output_token_ids)+1 >= sampling.max_tokens
                or req.num_tokens+1 >= runner.max_len
                or runner.batch.req_ids != [req.req_id] or runner._ahead is None):
            return False
        key = ([req.req_id],[req.num_computed_tokens],
               ((sampling.temperature,sampling.top_p,sampling.top_k,sampling.max_tokens,
                 sampling.ignore_eos,tuple(sampling.stop_token_ids)),),(req.output_token_ids[-1],))
        return runner._ahead[0] == key and identical(self.certificate,self.roots(runner._ahead))

    def run(self, out):
        need(not self.closed and not self.active, 'closed or nested ahead-submit scope')
        self.active = True; self.ordinal = 0; self.last_success = None
        self.counts['runs'] += 1
        try:
            self.may_skip = self.eligible(out)
            self.counts['exact_hit_eligible' if self.may_skip else 'fallback_runs'] += 1
            self.expected = self.certificate if self.may_skip else None
            result = self.cloned(out)
            # Native run may have queued a successor or may have cleared ahead.
            # Only its exact future roots, previously submitted by this scope,
            # become a weak certificate. No token values or GPU status are read.
            roots = self.roots(self.runner._ahead)
            self.certificate = self.last_success if roots and identical(self.last_success,roots) else None
            return result
        except BaseException:
            self.counts['errors'] += 1; self.certificate = None
            raise
        finally:
            self.expected = self.last_success = None
            self.active = self.may_skip = False

    def restore(self):
        if self.closed: return
        need(not self.active, 'cannot restore an active runner')
        need(getattr(self.target,self.attribute) is self.wrapper, 'restore outer observer before this scope')
        if self.saved[0]: setattr(self.target,self.attribute,self.saved[1])
        else: vars(self.target).pop(self.attribute,None)
        self.closed = True
        self.runner = self.target = self.original = self.wrapper = self.cloned = None
        self.saved = self.certificate = self.expected = self.last_success = self.BatchKV = None

    def __enter__(self): return self
    def __exit__(self,*unused): self.restore(); return False


def _source(path, digest):
    path = Path(path).resolve()
    need(hashlib.sha256(path.read_bytes()).hexdigest() == digest, 'frozen source changed: '+str(path))
    return path


def install(runner, *, enabled=False, terminal_ids=(), profile_session=None):
    """Fresh idle runner only. No args/OFF touch no runtime or file.

    Supported order: optional frozen profile -> this scope -> qualification
    observer. Restore in reverse after worker/handlers drain. Unknown wrappers
    refuse. With profile, omitted calls have no eval event because no API call
    occurred; all real forwarded calls retain its original mx proxy/events.
    Current device EOS is not inspected; original EOS/retire semantics remain.
    """
    if type(enabled) is not bool: raise TypeError('enabled must be bool')
    if not enabled: return None
    cls = type(runner);fn = cls.run;namespace = fn.__globals__
    need(type(fn) is FunctionType and namespace.get('ModelRunner') is cls,
         'exact native runner class required')
    path = _source(namespace['__file__'],RUNNER_SHA)
    need(path.name == 'model_runner.py' and Path(fn.__code__.co_filename).resolve() == path,
         'actual native run function required')
    _source(path.parent.parent/'cache.py',CACHE_SHA)
    need(namespace['mx'] is sys.modules.get('mlx.core')
         and type(runner.batch) is namespace['DecodeBatch']
         and runner._sync_batch.__func__ is cls._sync_batch and '_sync_batch' not in vars(runner),
         'unmodified runtime/batch/sync required')
    need(runner.spec is None and runner._ahead is None and not runner.batch.req_ids
         and not runner.staging and not runner.feats, 'attach to a fresh idle target-only runner')
    terminal_ids = tuple(terminal_ids)
    need(all(type(token) is int and token >= 0 for token in terminal_ids), 'integer terminal IDs required')
    if profile_session is None:
        need('run' not in vars(runner) and runner.run.__func__ is fn, 'unknown existing runner wrapper')
        target,attribute,original = runner,'run',runner.run
    else:
        pcls = type(profile_session);init = pcls.__init__;pns = init.__globals__
        ppath = _source(pns['__file__'],PROFILE_SHA)
        need(pns.get('Session') is pcls and Path(init.__code__.co_filename).resolve() == ppath
             and ppath.name == 'instrumentation.py' and not profile_session.closed
             and profile_session.runner is runner and profile_session.current is None
             and profile_session.step == 0 and runner.run is profile_session.observed_run,
             'explicit fresh frozen C1 profile session required')
        original = profile_session.cloned
        need(original.__self__ is runner and original.__func__.__code__ is fn.__code__
             and type(original.__func__.__globals__['mx']) is pns['_Mx']
             and original.__func__.__globals__['mx'].session is profile_session,
             'profile private run/backend chain changed')
        target,attribute = profile_session,'cloned'
    return Scope(runner,target,attribute,original,terminal_ids)
