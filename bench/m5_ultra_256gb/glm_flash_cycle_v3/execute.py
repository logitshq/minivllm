"""Explicit v3 driver. Default describe; real source-sealed PREPARED proof required.

The original stop/direct-owner/finally-recover and unknown-launch algorithms
are unchanged. The two admission schema literals identify this actual v3
registration/journal; no existing journal or source is modified.
"""
import ast as _ast
import hashlib as _hashlib
import importlib.util as _importlib
import inspect as _inspect
from pathlib import Path as _Path

_reuse_path = _Path(__file__).resolve().parents[1]/'glm_flash_cycle_v2/_reuse.py'
if _hashlib.sha256(_reuse_path.read_bytes()).hexdigest() != 'fbf523c586629ce6e5e2030d629b00c9f1cd99dfec9edef6c35ac8e054468f97':
    raise ValueError('fixed original execution reuse source changed')
_spec = _importlib.spec_from_file_location('_glm_v3_execute_reuse', _reuse_path)
_reuse = _importlib.module_from_spec(_spec); _spec.loader.exec_module(_reuse)
_original = _reuse.install(globals(), 'execute.py', dict(REGISTRAR_SHA='319df1f18efee2ec5587d353c213a79aa8e7373ece904c13e1fd0e06529f68af'))

_preflight_source = _inspect.getsource(_original.preflight)
for _before, _after in (
    ("'GLM-cold-load-prepared-receipt-v1'", "'GLM-cold-load-prepared-receipt-v3'"),
    ("'GLM-cold-load-journal-v1'", "'GLM-cold-load-journal-v3'")):
    need(_preflight_source.count(_before) == 1, 'fixed execution schema anchor changed')
    _preflight_source = _preflight_source.replace(_before, _after, 1)
exec(compile(_ast.parse(_preflight_source), __file__, 'exec'), globals())

if __name__ == '__main__': raise SystemExit(main())
