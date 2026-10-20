"""Default-OFF fixed Qwen C1 shared-MLP verifier experiment; not an engine.

The explicit shared branch copies the source-sealed original admission, prefix
ownership and head logic, changing only the layer loop. No backend import,
weight packing, eval, scheduler, accept/rollback, resource owner or retry.
"""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib.util
from pathlib import Path
import sys

ORIGINAL=Path(__file__).resolve().parent.parent/'rowwise_native_verify/candidate.py'
ORIGINAL_SHA='4ce1b1720dc256b01d30a0f9b19b109bde9949050e98b50a31d308f9e71c8f15'
PROJECTOR=Path(__file__).resolve().parent.parent/'nvfp4_multirow_shared/candidate.py'
PROJECTOR_SHA='d1826b3c4d07d44adbae071ef2da43197348c09b2d8be1035eb1a0c2788d132e'


def need(ok,message):
    if not ok: raise ValueError(message)


@lru_cache(maxsize=1)
def frozen():
    need(hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()==ORIGINAL_SHA,'frozen original verifier source')
    spec=importlib.util.spec_from_file_location('_rowwise_mlp_frozen_original',ORIGINAL)
    module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module)
    return module


@dataclass
class Output:
    logits: object
    prefixes: list


def verify(model, tokens, context, runtime, *, enabled=False, head_project=None, mlp_input_project=None):
    """Optional shared MLP gate/up; runtime.silu is the original mlx.nn.silu.

    No projector delegates to the pinned original verifier. Explicit ON admits
    all64 MLP metadata before private state construction; errors never retry.
    """
    if type(enabled) is not bool: raise TypeError('enabled must be bool')
    if not enabled: return None
    if mlp_input_project is None:
        return frozen().verify(model,tokens,context,runtime,enabled=True,head_project=head_project)
    mx = runtime.mx
    if not (type(context) is runtime.ForwardContext and context.mode == 'decode'
            and set(vars(context)) == {'mode','caches','positions','lens','max_len','prefill_shape'}
            and context.prefill_shape is None and tokens.ndim == 1
            and tokens.shape[0] in (2,4,8) and tokens.dtype in (mx.int32,mx.int64)
            and type(context.max_len) is int and 257 <= context.max_len
            and context.max_len + tokens.shape[0] - 1 <= 1536
            and context.positions is not None and context.lens is not None
            and context.positions.shape == context.lens.shape == (1,)
            and context.positions.dtype in (mx.int32,mx.int64)
            and context.lens.dtype in (mx.int32,mx.int64)
            and len(model.layers) == len(context.caches) == 64
            and model.cfg.hidden_size == 5120
            and model.cfg.quantization == {'group_size':16,'bits':4,'mode':'nvfp4'}
            and model.embed_tokens._output_dtype == mx.bfloat16):
        return None
    for layer, cache in zip(model.layers,context.caches):
        if layer.is_attn:
            if not (type(cache) is runtime.BatchKV and set(vars(cache)) == {'K','V'}
                    and cache.K.ndim == cache.V.ndim == 4 and cache.K.shape == cache.V.shape
                    and cache.K.shape[0] == 1 and cache.K.shape[2] >= 1536
                    and cache.K.dtype == cache.V.dtype == mx.bfloat16):
                return None
        elif not (type(cache) is runtime.BatchState and set(vars(cache)) == {'conv','S'}
                  and cache.conv.shape == (1,3,10240) and cache.conv.dtype == mx.bfloat16
                  and cache.S.shape == (1,48,128,128) and cache.S.dtype == mx.float32):
            return None
    need(callable(mlp_input_project) and callable(getattr(runtime,'silu',None)),
         'explicit projector and original runtime.silu required')
    frozen()  # Bind the copied admission/ownership contract before any private state.
    need(getattr(model.cfg,'intermediate_size',None)==17408,'fixed MLP intermediate size')
    for layer in model.layers:
        for name,K,N in (('gate_proj',5120,17408),('up_proj',5120,17408),('down_proj',17408,5120)):
            op=getattr(layer.mlp,name)
            need(getattr(op,'q',None)=={'group_size':16,'bits':4,'mode':'nvfp4'}
                 and getattr(op,'biases',None) is None
                 and getattr(op,'in_features',None)==K and getattr(op,'out_features',None)==N
                 and op.weight.shape==(N,K//8) and op.weight.dtype==mx.uint32
                 and op.scales.shape==(N,K//16) and op.scales.dtype==mx.uint8,
                 'fixed live NVFP4 MLP projection metadata: '+name)
    if head_project is not None:
        head = getattr(model, 'lm_head', None)
        need(callable(head_project) and model.cfg.tie_word_embeddings is False
             and getattr(model.cfg, 'vocab_size', None) == 248320
             and getattr(head, 'q', None) == {'group_size':16,'bits':4,'mode':'nvfp4'}
             and getattr(head, 'biases', None) is None
             and getattr(head, 'in_features', None) == 5120 and getattr(head, 'out_features', None) == 248320
             and head.weight.shape == (248320,640) and head.weight.dtype == mx.uint32
             and head.scales.shape == (248320,320) and head.scales.dtype == mx.uint8,
             'explicit head reuse requires the actual untied NVFP4 vocabulary head')
    T = tokens.shape[0]
    # Only the current layer's entry is read by the frozen DecoderLayer. All
    # entries are replaced with independent prefix-owned wrappers before return.
    prefixes = [list(context.caches) for _ in range(T)]
    contexts = [runtime.ForwardContext('decode',prefixes[t],
        positions=context.positions+t,lens=context.lens+t,max_len=context.max_len+t)
        for t in range(T)]
    # Preserve even the embedding's original one-row input shape.
    x = mx.concatenate([model.embed_tokens(tokens[t:t+1]) for t in range(T)],axis=0)
    for index,layer in enumerate(model.layers):
        mixed=[]; normalized=[]
        for t in range(T):
            previous = context.caches[index] if t == 0 else prefixes[t-1][index]
            cache = (runtime.BatchKV(previous.K[:],previous.V[:]) if layer.is_attn
                     else runtime.BatchState(previous.conv,previous.S))
            prefixes[t][index] = cache
            row=x[t:t+1]; h=layer.input_layernorm(row)
            # Do not pass GDN _projection: that selects a different fused body.
            y=(layer.self_attn(h,contexts[t],index) if layer.is_attn else
               layer.linear_attn(h,contexts[t],index,_norm_reuse_normal=True))
            residual=row+y; mixed.append(residual)
            normalized.append(layer.post_attention_layernorm(residual))
        h=mx.concatenate(normalized,axis=0)
        need(h.shape==(T,5120) and h.dtype==mx.bfloat16,'original M1 post-attention norms')
        projected=[]
        for op in (layer.mlp.gate_proj,layer.mlp.up_proj):
            value=mlp_input_project(h,op.weight,op.scales)
            need(value is not None and value.shape==(T,17408) and value.dtype==mx.bfloat16,
                 'explicit shared MLP projection rejected or returned wrong output')
            projected.append(value)
        gate,up=projected; rows=[]
        for t in range(T):
            activated=runtime.silu(gate[t:t+1])*up[t:t+1]
            rows.append(mixed[t]+layer.mlp.down_proj(activated))
        x=mx.concatenate(rows,axis=0)
    if head_project is not None:
        # Preserve original M1 norm rounding; only the final projection shares rows.
        normalized = mx.concatenate([model.norm(x[t:t+1]) for t in range(T)],axis=0)
        need(normalized.shape == (T,5120) and normalized.dtype == mx.bfloat16,'original row norms')
        logits = head_project(normalized,model.lm_head.weight,model.lm_head.scales)
        need(logits is not None and logits.shape == (T,248320) and logits.dtype == mx.bfloat16,
             'explicit head reuse rejected or returned an incomplete vocabulary')
        return Output(logits,prefixes)
    logits=[]
    for t in range(T):
        row=model.norm(x[t:t+1])
        logits.append(model.embed_tokens.as_linear(row) if model.cfg.tie_word_embeddings else model.lm_head(row))
    return Output(mx.concatenate(logits,axis=0),prefixes)
