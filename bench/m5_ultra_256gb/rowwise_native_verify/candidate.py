"""Default-inert, one-sequence layer-major native decode verifier prototype.

This is a fixed teacher-forced block, not an installed speculative decoder.
Each row uses the original B1 layer arithmetic and its own absolute position.
Each prefix owns distinct KV handles and receives complete recurrent states.
No runtime/backend import, eval, parameter pack, acceptance or rollback here.
"""
from dataclasses import dataclass


def need(ok, message):
    if not ok: raise ValueError(message)


@dataclass
class Output:
    logits: object
    prefixes: list


def verify(model, tokens, context, runtime, *, enabled=False, head_project=None):
    """Dependency-injected research seam; not a source-qualified production API.

    OFF returns None without inspecting inputs. ON is only for a source-verified
    external child owning a fresh native model and a canonical B1 decode state.
    All token IDs are fixed before execution; candidate logits never feed input.
    No cache/parameter object supplied by the caller is assigned or updated.
    Rejected domains return None; errors after admission propagate without retry.
    """
    if type(enabled) is not bool: raise TypeError('enabled must be bool')
    if not enabled: return None
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
        rows=[]
        for t in range(T):
            previous = context.caches[index] if t == 0 else prefixes[t-1][index]
            cache = (runtime.BatchKV(previous.K[:],previous.V[:]) if layer.is_attn
                     else runtime.BatchState(previous.conv,previous.S))
            prefixes[t][index] = cache
            rows.append(layer(x[t:t+1],contexts[t],index,_norm_reuse_normal=True))
        # A layer-wide dependency exposes nearby same-weight B1 operations.
        # MLX's eventual scheduling/cache behavior must be measured, not assumed.
        x = mx.concatenate(rows,axis=0)
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
