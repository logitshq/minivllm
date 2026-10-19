"""Frozen raw restoration audit, new owner schema and dual-monitor reap chain.

Pure offline evidence only. Failed observations may still restore when every
model/read-command handle is settled. This is not the v1 1:1 resource audit.
Controller pin binds the actual sealed E033 predecessor and recovery lifecycle.
"""
import importlib.util as _importlib
from pathlib import Path as _Path
import hashlib as _hashlib
import inspect as _inspect
import ast as _ast

_reuse_path = _Path(__file__).resolve().parents[1]/'glm_flash_cycle_v2/_reuse.py'
if _hashlib.sha256(_reuse_path.read_bytes()).hexdigest() != 'fbf523c586629ce6e5e2030d629b00c9f1cd99dfec9edef6c35ac8e054468f97':
    raise ValueError('fixed namespace reuse source changed')
_spec = _importlib.spec_from_file_location('_glm_v3_restore_reuse', _reuse_path)
_reuse = _importlib.module_from_spec(_spec); _spec.loader.exec_module(_reuse)
_original = _reuse.install(globals(), 'audit_restore.py', dict(
    OWNER_SHA='9d69c5a64dc50b7d65330ad130b8af9f757b91e884aa197d1e828775b27c88a8',
    CONTROLLER_SHA='ca35f97e5ee39d3ca81a59ab851249423d3317fd8176f569d8580260328d5731',
    SCOPE='Independent original raw restoration and model/read-command reap receipts. '
          'Two monitoring lanes are not the old checkpoint-per-sample resource contract; '
          'no experiment, physical-safety or performance qualification.'))

_owner_reap_source = _inspect.getsource(_original.owner_reap)
_before = "complete['schema']=='GLM-cold-load-owner-result-v1'"
_after = "complete['schema']=='GLM-cold-load-owner-result-v3'"
need(_owner_reap_source.count(_before) == 1, 'frozen restoration schema anchor changed')
_owner_reap_source = _owner_reap_source.replace(_before, _after, 1)
exec(compile(_ast.parse(_owner_reap_source), __file__, 'exec'), globals())

if __name__ == '__main__': raise SystemExit(main())
