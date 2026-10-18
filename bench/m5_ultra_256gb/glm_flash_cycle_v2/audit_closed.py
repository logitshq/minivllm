"""Original physical/count/reset/cleanup audit with actual v2 owner/restoration entry. No performance or full-state qualification."""
import importlib.util as _importlib
from pathlib import Path as _Path
import hashlib as _hashlib

_adapter_path=_Path(__file__).resolve().parent/'_reuse.py'
if _hashlib.sha256(_adapter_path.read_bytes()).hexdigest()!='fbf523c586629ce6e5e2030d629b00c9f1cd99dfec9edef6c35ac8e054468f97':
    raise ValueError('fixed GLM v2 reuse source changed')
_adapter_spec=_importlib.spec_from_file_location('_glm_v2_code_reuse',_adapter_path)
_adapter=_importlib.module_from_spec(_adapter_spec);_adapter_spec.loader.exec_module(_adapter)

_original=_adapter.install(globals(),'audit_closed.py',{'OWNER_SHA': 'f8e32b6532ed0d22b21b27ef83231ae7a1d7423a4269ea32b5e4aa20ff2a0657', 'RESTORE_SHA': 'de1674780f5d3de7cdcf0cd4d54c6b9136f086120bb827d7bcd2f25377084996'})

if __name__=='__main__':raise SystemExit(main())
