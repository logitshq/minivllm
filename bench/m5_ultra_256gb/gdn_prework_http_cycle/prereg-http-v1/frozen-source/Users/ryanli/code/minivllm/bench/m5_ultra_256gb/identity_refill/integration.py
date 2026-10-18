"""E028 bench-only installation at fresh ModelRunner construction; default OFF.

Call install(engine.runner, enabled=True) before adding any requests or attaching
observers. OFF neither inspects the runner nor loads the candidate. ON replaces
only this empty runner's batch; no classes/globals are patched and no arrays are
created or evaluated. The caller owns runtime/model pins and resource admission.

Emptiness is observable, prior use and scheduler queues are not: the caller must
install immediately after construction. No hot migration or speculative route.
This prepares a full-model experiment; it confers no model or HTTP qualification.
"""
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import FunctionType


HERE = Path(__file__).resolve().parent
CANDIDATE_SHA256 = 'bb9f8a461625e97c80e75140c9d9b7b8718a633595ab4f3a422267e752ded8f5'
CACHE_SHA256 = '056a3bf3bc42163e93f55eb914fa35d5c081f803da9b5faa740588a67057a791'
RUNNER_SHA256 = '14fd964c57814846887585a9cbbdf1e11d2500ab80c23b10c945d2aa9316b7b1'


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _candidate():
    path = HERE / 'candidate.py'
    _need(hashlib.sha256(path.read_bytes()).hexdigest() == CANDIDATE_SHA256,
          'E028 candidate source mismatch')
    spec = importlib.util.spec_from_file_location('_e028_install_candidate', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _need(module.REFERENCE_CACHE_SHA256 == CACHE_SHA256, 'candidate cache contract mismatch')
    return module


def install(runner, *, enabled=False):
    """Return the new batch when ON, or None without touching runner when OFF.

    Fresh instances only. A second ON call is rejected, even before any work.
    Counters on the returned batch are branch witnesses, not device qualification.
    All checks and construction finish before the sole assignment runner.batch.
    """
    if type(enabled) is not bool:
        raise TypeError('enabled must be bool')
    if not enabled:
        return None

    runner_module = sys.modules.get(type(runner).__module__)
    _need(runner_module is not None and type(runner) is getattr(runner_module, 'ModelRunner', None),
          'expected the actual ModelRunner class')
    original_class = runner_module.DecodeBatch
    cache = sys.modules.get(original_class.__module__)
    _need(cache is not None and original_class is getattr(cache, 'DecodeBatch', None),
          'runner and cache classes differ')
    runner_path = Path(runner_module.__file__).resolve()
    cache_path = Path(cache.__file__).resolve()
    _need(runner_path.name == 'model_runner.py' and runner_path.parent.name == 'engine'
          and cache_path == runner_path.parent.parent / 'cache.py', 'mixed native source roots')
    for module, path, expected, cls, names in (
            (runner_module, runner_path, RUNNER_SHA256, type(runner), ('__init__', 'run', '_sync_batch')),
            (cache, cache_path, CACHE_SHA256, original_class, ('__init__', 'rebuild'))):
        _need(hashlib.sha256(path.read_bytes()).hexdigest() == expected, 'native source mismatch: ' + str(path))
        for name in names:
            fn = getattr(cls, name)
            _need(type(fn) is FunctionType and fn.__globals__ is vars(module)
                  and Path(fn.__code__.co_filename).resolve() == path,
                  'unrecognized loaded native method: ' + name)
    _need(cache.mx is runner_module.mx and cache.mx is sys.modules.get('mlx.core'),
          'runner and cache must share the loaded MLX backend')
    old = runner.batch
    _need(type(old) is original_class, 'batch already installed or is not original DecodeBatch')
    _need(runner.spec is None, 'speculative runners are outside this experiment')
    _need(old.model is runner.model and type(old.req_ids) is list and not old.req_ids
          and type(old.lens) is list and not old.lens and type(old.layers) is list
          and len(old.layers) == len(runner.model.layers) and all(x is None for x in old.layers)
          and type(runner.staging) is dict and not runner.staging
          and type(runner.feats) is dict and not runner.feats and runner._ahead is None,
          'installation requires a fresh empty runner; no state migration')
    helper = _candidate()
    replacement = helper.make_batch_class(cache, cache.mx)(runner.model, enabled=True)
    runner.batch = replacement
    return replacement
