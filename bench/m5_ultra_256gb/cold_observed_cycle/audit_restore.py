"""Observed recovery bridge; frozen v3 lifecycle algorithms unchanged."""
import importlib.util as _iu
from pathlib import Path as _P
import hashlib as _hh
_rp=_P(__file__).resolve().parent/'_reuse.py'
if _hh.sha256(_rp.read_bytes()).hexdigest()!='b22ee6d9c505fefb3f111760096eb9d6542fc2370336ea30941ccae9baa82ecb':raise ValueError('reuse source changed')
_sp=_iu.spec_from_file_location('_observed_cold_reuse',_rp)
_reuse=_iu.module_from_spec(_sp);_sp.loader.exec_module(_reuse)
_original=_reuse.install(globals(),'audit_restore.py')

if __name__=='__main__':raise SystemExit(main())
