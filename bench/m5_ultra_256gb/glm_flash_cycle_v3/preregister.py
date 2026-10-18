"""CPU-only v3 registration, inactive until actual E033 restoration seals exist.

No arguments describe. --run writes a fresh plan using existing GLM inputs;
--seal-prepared only accepts the real controller's PREPARED journal. This script
never seals future facts, prepares a service, loads a model or executes a child.
"""
import ast as _ast
import hashlib as _hashlib
import importlib.util as _importlib
import inspect as _inspect
from pathlib import Path as _Path

_reuse_path = _Path(__file__).resolve().parents[1]/'glm_flash_cycle_v2/_reuse.py'
if _hashlib.sha256(_reuse_path.read_bytes()).hexdigest() != 'fbf523c586629ce6e5e2030d629b00c9f1cd99dfec9edef6c35ac8e054468f97':
    raise ValueError('fixed original registration reuse source changed')
_spec = _importlib.spec_from_file_location('_glm_v3_registration_reuse', _reuse_path)
_reuse = _importlib.module_from_spec(_spec); _spec.loader.exec_module(_reuse)
_original = _reuse.install(globals(), 'preregister.py', dict(
    OWNER_SHA='9d69c5a64dc50b7d65330ad130b8af9f757b91e884aa197d1e828775b27c88a8'))


REQUIRED_PREDECESSOR_ARCHIVE = HERE.parents[2]/'docs/experiments/m5-ultra-256gb/E033-gdn-graph-combined-formal-v2'


def require_seals():
    """All source seals precede plan construction; no writes to these sources."""
    controller = module(HERE/'controller.py', sha(HERE/'controller.py'), '_glm_v3_registration_controller')
    need(controller.PRIOR_ARCHIVE == REQUIRED_PREDECESSOR_ARCHIVE,
         'GLM is deferred until the actual E033 retry-v2 predecessor; E032 is not the next cycle')
    manifest = controller.predecessor_manifest()
    controller.predecessor(manifest['binding'])
    restoration = module(HERE/'audit_restore.py', sha(HERE/'audit_restore.py'), '_glm_v3_registration_restore')
    closed = module(HERE/'audit_closed.py', sha(HERE/'audit_closed.py'), '_glm_v3_registration_closed')
    driver = module(HERE/'execute.py', sha(HERE/'execute.py'), '_glm_v3_registration_driver')
    need(restoration.CONTROLLER_SHA == sha(HERE/'controller.py')
         and closed.RESTORE_SHA == sha(HERE/'audit_restore.py')
         and driver.REGISTRAR_SHA == sha(__file__), 'actual reviewed v3 source seals required')


_original_build_commands = build_commands


def build_commands(plan_path, plan_digest, plan, analysis, python):
    need(Path(analysis).name != 'restoration-audit.json', 'distinct diagnostic/restoration audit outputs')
    commands = _original_build_commands(plan_path, plan_digest, plan, analysis, python)
    commands['restoration_audit'] = [python,'-I','-B',str(HERE/'audit_restore.py'),'--journal',
        plan['cold_restart']['journal_path'],'--out',str(Path(analysis).with_name('restoration-audit.json'))]
    return commands


def _variant(name, replacements):
    source = _inspect.getsource(getattr(_original, name))
    for before, after in replacements:
        need(source.count(before) == 1, 'fixed registration source anchor changed: '+name)
        source = source.replace(before, after, 1)
    tree = _ast.parse(source)
    need(len(tree.body) == 1 and isinstance(tree.body[0], _ast.FunctionDef), 'one registration function')
    exec(compile(tree, __file__, 'exec'), globals())
    return source


# Preparation/source checks remain original; no supplied inventory claim is widened.
_prepare_source = _variant('prepare', [
    ('    out=Path(args.out).resolve();', '    require_seals()\n    out=Path(args.out).resolve();'),
    ("    plan['execution']=dict(owner=ref(HERE/'owner.py'),pins=owner.source_pins(plan))",
     "    plan['monitor_policy']=dict(owner.MONITOR_POLICY)\n"
     "    plan['execution']=dict(owner=ref(HERE/'owner.py'),pins=owner.source_pins(plan))"),
    ("HERE/'prepare_inputs.py'", "_original.HERE/'prepare_inputs.py'"),
    ("schema='GLM-cold-load-plan-build-v1'", "schema='GLM-cold-load-plan-build-v3'")])
_seal_source = _variant('seal_prepared', [
    ('    folder=Path(folder).resolve();', '    require_seals()\n    folder=Path(folder).resolve();'),
    ("schema='GLM-cold-load-prepared-receipt-v1'", "schema='GLM-cold-load-prepared-receipt-v3'")])

if __name__ == '__main__': raise SystemExit(main())
