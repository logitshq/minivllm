"""Default-OFF GLM projection materialization; injected backend, no import-time IO.

The owned loader must source/model-pin `original_load` in `admit` and install this
returned callable only for its one cold child. This module installs nothing.
"""
from pathlib import Path
import traceback

ORIGINAL=Path('/Users/ryanli/code/minivllm/.artifacts/models/huggingface/models--mlx-community--GLM-5.3-Flash-4bit/snapshots/fa04cbc3622b16aa7475404060b05e506d81bd4e')
PROJECTIONS=('gate_proj','down_proj','up_proj');SUFFIXES=('weight','scales','biases')
MAX_GROUP_BYTES=1660944384  # 288 down experts including scales+biases; NOT a RAM cap.


def need(ok,message):
    if not ok:raise ValueError(message)


def schedule(model,mx,tree_flatten):
    """Validate all final parameters before any candidate eval; keys only in groups."""
    args=model.language_model.args
    need((args.num_hidden_layers,args.n_routed_experts,args.first_k_dense_replace,
          args.num_nextn_predict_layers,args.hidden_size,args.moe_intermediate_size)==(45,288,3,1,4096,2048),'exact GLM architecture')
    pairs=tree_flatten(model.parameters());flat=dict(pairs)
    need(len(flat)==len(pairs) and all(isinstance(v,mx.array) for v in flat.values()),'unique parameter keys/arrays')
    groups=[];used=set()
    prefixes=[f'language_model.model.layers.{i}.mlp.switch_mlp' for i in range(3,45)]
    prefixes.append('language_model.mtp.0.block.mlp.switch_mlp')
    for prefix in prefixes:
        for projection in PROJECTIONS:
            keys=tuple(prefix+'.'+projection+'.'+suffix for suffix in SUFFIXES)
            down=projection=='down_proj';width=4096 if down else 2048
            shapes=((288,width,320 if down else 512),(288,width,32 if down else 64),(288,width,32 if down else 64))
            for key,shape,dtype in zip(keys,shapes,(mx.uint32,mx.float16,mx.float16)):
                need(key in flat and tuple(flat[key].shape)==shape and flat[key].dtype==dtype,'original packed projection '+key)
            groups.append(keys);used.update(keys)
    need(len(groups)==129 and len(used)==387,'42 backbone plus1 MTP, three projections each')
    need(not any('.experts.' in k or ('.switch_mlp.' in k and k not in used) for k in flat),'unexpected expert layout')
    groups.extend((key,) for key in sorted(set(flat)-used))
    for keys in groups:
        need(0<sum(flat[k].nbytes for k in keys)<=MAX_GROUP_BYTES,'bounded logical payload group')
    need(sum(map(len,groups))==len(flat),'complete exactly-once parameter-key coverage')
    return flat,groups


def materialize(model,mx,tree_flatten,emit):
    """Final model handles stay owned by model; temporary evaluation lists are dropped."""
    flat={};groups=[];arrays=[]
    try:
        flat,groups=schedule(model,mx,tree_flatten)
        count=len(flat)
        for index,keys in enumerate(groups):
            arrays=[flat.pop(key) for key in keys];payload=sum(a.nbytes for a in arrays)
            mx.eval(*arrays);mx.synchronize()
            arrays.clear()  # No candidate list owns a consumed lazy input graph.
            mx.clear_cache()  # Only after synchronization; no allocator/wired-limit changes.
            emit(dict(event='projection-materialized',index=index,keys=list(keys),payload_bytes=payload,
                      active_bytes=mx.get_active_memory(),cache_bytes=mx.get_cache_memory(),peak_bytes=mx.get_peak_memory()))
        need(not flat,'all final parameters evaluated')
        return dict(parameter_keys=count,groups=len(groups),expert_projection_groups=129,
                    logical_group_bound_bytes=MAX_GROUP_BYTES,physical_memory_bound=False)
    finally:
        arrays.clear();flat.clear();groups.clear()


def build(original_load,mx,tree_flatten,*,enabled=False,admit=None,emit=None):
    """Return a one-call loader wrapper, or None while OFF; never patch global state.

    admit(path, original_load) must check the owned plan, original62-shard model,
    source/runtime pins, config/quantization and disabled custom-quant/offload.
    It runs before every load. No actual cold-run admission is included here.
    """
    if not enabled:return None
    need(all(callable(v) for v in (original_load,tree_flatten,admit,emit)),'explicit loader/admission/scalar sink required')
    def load(path,*,adapter_path=None,lazy=False,strict=True,**kwargs):
        need(adapter_path is None and lazy is False and strict is True,'no adapter, deferred-return or non-strict route')
        need(set(kwargs)<= {'trust_remote_code'} and kwargs.get('trust_remote_code',False) is False,'no remote/custom loader options')
        path=Path(path);need(path.is_absolute() and path.resolve()==ORIGINAL and path.is_dir(),'exact local original checkpoint')
        need(not (path/'offload_index.json').exists(),'offload would silently disable strict loading')
        admit(path,original_load)
        model=processor=None
        try:
            emit(dict(event='before-original-lazy-load',active_bytes=mx.get_active_memory(),cache_bytes=mx.get_cache_memory(),
                      peak_bytes=mx.get_peak_memory(),memory_guideline_bytes=mx.get_memory_limit()))
            # Unchanged official sanitizers, numeric expert order and mx.stack.
            # original_load and load_model frames/weight dictionaries end here.
            model,processor=original_load(str(path),adapter_path=None,lazy=True,strict=True,trust_remote_code=False)
            emit(dict(event='original-lazy-load-returned',active_bytes=mx.get_active_memory(),cache_bytes=mx.get_cache_memory(),
                      peak_bytes=mx.get_peak_memory()))
            receipt=materialize(model,mx,tree_flatten,emit)
            emit(dict(event='projection-materialization-complete',**receipt))
            return model,processor
        except BaseException as exc:
            # Retain neither completed donor frames nor model refs in a failure traceback.
            traceback.clear_frames(exc.__traceback__)
            raise
        finally:model=processor=None
    return load
