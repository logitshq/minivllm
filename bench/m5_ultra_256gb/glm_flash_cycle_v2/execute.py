"""Actual prepared GLM v2 stop / direct owner / finally restore driver. Default describe; explicit --run required."""
import importlib.util as _importlib
from pathlib import Path as _Path
import hashlib as _hashlib

_adapter_path=_Path(__file__).resolve().parent/'_reuse.py'
if _hashlib.sha256(_adapter_path.read_bytes()).hexdigest()!='fbf523c586629ce6e5e2030d629b00c9f1cd99dfec9edef6c35ac8e054468f97':
    raise ValueError('fixed GLM v2 reuse source changed')
_adapter_spec=_importlib.spec_from_file_location('_glm_v2_code_reuse',_adapter_path)
_adapter=_importlib.module_from_spec(_adapter_spec);_adapter_spec.loader.exec_module(_adapter)

_original=_adapter.install(globals(),'execute.py',{'REGISTRAR_SHA': 'b6eeeda7489028ef1d28db1cb1c291b66941b621c1c554c24ba30e49e41b610f'})

if __name__=='__main__':raise SystemExit(main())
