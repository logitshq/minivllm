"""CPU registration; same v3 lifecycle with exact current E035 and E036 inputs."""
import importlib.util as _importlib
from pathlib import Path as _Path
_path=_Path(__file__).resolve().with_name('_reuse.py')
_spec=_importlib.spec_from_file_location('_glm_stacked_reuse',_path)
_adapter=_importlib.module_from_spec(_spec);_spec.loader.exec_module(_adapter)

_v3=_adapter.install(globals(),'glm_flash_cycle_v3/preregister.py')
OWNER_SHA='bfcba6459dc167996833e2582e6caf04753765efd1fc0cf2f5e47f83a9059386'
REQUIRED_PREDECESSOR_ARCHIVE=HERE.parents[2]/'docs/experiments/m5-ultra-256gb/E035-c1-pipeline-profile'
def require_seals():
    c=module(HERE/'controller.py',sha(HERE/'controller.py'),'stacked_registration_controller')
    need(c.PRIOR_ARCHIVE==REQUIRED_PREDECESSOR_ARCHIVE,'actual E035 predecessor');c.predecessor(c.predecessor_manifest()['binding'])
    r=module(HERE/'audit_restore.py',sha(HERE/'audit_restore.py'),'stacked_registration_restore')
    a=module(HERE/'audit_closed.py',sha(HERE/'audit_closed.py'),'stacked_registration_closed')
    d=module(HERE/'execute.py',sha(HERE/'execute.py'),'stacked_registration_execute')
    need(r.CONTROLLER_SHA==sha(HERE/'controller.py') and a.RESTORE_SHA==sha(HERE/'audit_restore.py') and
         d.REGISTRAR_SHA==sha(__file__),'reviewed source seals')
# The v3 builder's input-builder reference was historical v1; bind the actual
# local metadata builder. Child/monitor policy and plan validation remain v3.
_adapter.variant(globals(),_v3._prepare_source,[("_original.HERE/'prepare_inputs.py'","HERE/'prepare_inputs.py'")])
if __name__=='__main__':raise SystemExit(main())
