"""Narrow E055 audit v2: one hash-bound HF tokenizer link; original math unchanged."""
import hashlib
import importlib.util
from pathlib import Path
from types import FunctionType,SimpleNamespace

PREVIOUS=Path(__file__).resolve().parent.parent/'c8_batched_formal_v1/audit.py'
PREVIOUS_SHA='7b054fbf35772730b8e9f4522f46622819622001064bfb81537ce031ac6e4fca'
TOKENIZER=Path('/Users/ryanli/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-nvfp4/snapshots/5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6/tokenizer.json')
TARGET=Path('/Users/ryanli/.cache/huggingface/hub/blobs/77/777bcaa63794fa47b8f53680be9d6d176f1fcbd7ba03cdc6c3bae2b3d76b323f')
TOKENIZER_SHA='06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523'
TOKENIZER_BYTES=19989325
if hashlib.sha256(PREVIOUS.read_bytes()).hexdigest()!=PREVIOUS_SHA:raise ValueError('frozen v1 audit source changed')
_spec=importlib.util.spec_from_file_location('_e055_audit_frozen_v1',PREVIOUS)
base=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(base)


def pinned_tokenizer(ref,fallback):
    base.need(type(ref) is dict and set(ref)=={'path','sha256'},'exact path/SHA reference')
    if ref['path']!=str(TOKENIZER):return fallback(ref)
    base.need(ref==dict(path=str(TOKENIZER),sha256=TOKENIZER_SHA),'literal tokenizer reference')
    base.need(TOKENIZER.is_symlink() and TOKENIZER.resolve(strict=True)==TARGET
        and TARGET.is_absolute() and TARGET.resolve(strict=True)==TARGET and not TARGET.is_symlink()
        and TARGET.is_file() and TARGET.stat().st_size==TOKENIZER_BYTES
        and base.sha(TARGET)==TOKENIZER_SHA,'exact hash-verified tokenizer blob')
    return TARGET


def adapted_load(path,digest):
    module=base.load(path,digest)
    if path!=base.TRIAL:return module
    def load_module(requested):
        value=module.module(requested)
        if requested!=module.FORMAL:return value
        return SimpleNamespace(**dict(vars(value),pinned=lambda ref:pinned_tokenizer(ref,value.pinned)))
    # Original trial validation/source binding is untouched. Only the formal
    # module returned to the audit's direct tokenizer loader gets this wrapper.
    return SimpleNamespace(**dict(vars(module),module=load_module))


_scope=dict(vars(base),load=adapted_load,__file__=__file__)
_original_audit=FunctionType(base.audit.__code__,_scope,'original_audit',base.audit.__defaults__,base.audit.__closure__)


def audit(screen,plan_path,plan_sha256):
    result=_original_audit(screen,plan_path,plan_sha256)
    result.update(schema='E055-six-process-metrics-ids-v2',previous_auditor=dict(path=str(PREVIOUS),sha256=PREVIOUS_SHA),
        correction='Only the literal original tokenizer reference may resolve to the exact hash-verified HF blob; all raw checks/math/gates unchanged.')
    if result['status']=='PASS_METRICS_AND_IDS':
        try:
            pinned_tokenizer(dict(path=str(TOKENIZER),sha256=TOKENIZER_SHA),lambda ref:base.need(False,'unexpected fallback'))
            result['tokenizer_resolution']=dict(original_path=str(TOKENIZER),resolved_path=str(TARGET),sha256=TOKENIZER_SHA,
                bytes=TOKENIZER_BYTES,rechecked_after_audit=True,other_symlinks_allowed=False)
        except Exception as exc:
            result.update(status='FAIL',timing_valid=False,same_schedule_numerical_prerequisite=False,
                error=dict(type=type(exc).__name__,message=str(exc)))
    return result


_main_scope=dict(vars(base),audit=audit)
main=FunctionType(base.main.__code__,_main_scope,'main',base.main.__defaults__,base.main.__closure__)

if __name__=='__main__':raise SystemExit(main())
