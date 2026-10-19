"""Frozen v3 driver: stop/direct owner/finally recover, same unknown-launch guard."""
import importlib.util as _importlib
from pathlib import Path as _Path
_path=_Path(__file__).resolve().with_name('_reuse.py')
_spec=_importlib.spec_from_file_location('_glm_stacked_reuse',_path)
_adapter=_importlib.module_from_spec(_spec);_spec.loader.exec_module(_adapter)

_v3=_adapter.install(globals(),'glm_flash_cycle_v3/execute.py')
REGISTRAR_SHA='4339cdd341134507ca523a570e9a09641e5193a4c10a63ba2356c31b8ef05512'
if __name__=='__main__':raise SystemExit(main())
