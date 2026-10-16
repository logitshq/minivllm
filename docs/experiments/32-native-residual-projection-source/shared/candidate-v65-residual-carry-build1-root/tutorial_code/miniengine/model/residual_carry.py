"""Opt-in normal B1 carried residual graph; eligibility precedes cache mutation.

No cross-token state or array parameters are retained here. All fallback selection
is complete before the first mixer. Runtime errors after eligibility propagate.
"""
import math

_FLAGS = ("record", "capture", "speculative", "verify")


def _runtime_types():
    import mlx.core as mx
    from ..context import ForwardContext
    from ..cache import BatchKV, BatchState
    from .layers import RMSNorm, Embedding
    from .qwen3_5 import Qwen3_5ForCausalLM, DecoderLayer, MLP
    from .attention import Attention
    from .gdn import GatedDeltaNet
    return mx, ForwardContext, BatchKV, BatchState, RMSNorm, Embedding, Qwen3_5ForCausalLM, DecoderLayer, MLP, Attention, GatedDeltaNet


def _unflagged(value):
    return all(getattr(value, key, False) is False for key in _FLAGS)


def _eligible_metadata(model, input_ids, ctx, last_only, capture):
    """Metadata only: no values, synchronization, tensor copies or cache writes."""
    mx, Context, KV, State, Norm, Embed, Model, Block, MLP, Attn, GDN = _runtime_types()
    if (type(model) is not Model or type(ctx) is not Context
            or ctx.mode != "decode" or not _unflagged(ctx)
            or ctx.prefill_shape is not None or last_only is not False
            or type(capture) is not tuple or capture
            or input_ids.shape != (1,) or input_ids.dtype != mx.int32
            or type(model.layers) is not list or len(model.layers) != 64
            or model.cfg.hidden_size != 5120 or type(model.embed_tokens) is not Embed
            or type(ctx.caches) is not list or len(ctx.caches) != 64
            or type(ctx.max_len) is not int or ctx.max_len <= 0
            or ctx.positions is None or ctx.lens is None
            or ctx.positions.shape != (1,) or ctx.lens.shape != (1,)
            or ctx.positions.dtype != mx.int32 or ctx.lens.dtype != mx.int32):
        return False
    norms = [model.norm]
    for layer, cache in zip(model.layers, ctx.caches):
        if (type(layer) is not Block or type(layer.is_attn) is not bool
                or type(layer.mlp) is not MLP or not _unflagged(cache)):
            return False
        norms.extend((layer.input_layernorm, layer.post_attention_layernorm))
        if layer.is_attn:
            if (type(layer.self_attn) is not Attn or type(cache) is not KV
                    or cache.K.shape != cache.V.shape or len(cache.K.shape) != 4
                    or cache.K.shape[0] != 1 or cache.K.shape[2] < ctx.max_len
                    or cache.K.dtype != mx.bfloat16 or cache.V.dtype != mx.bfloat16):
                return False
        elif (type(layer.linear_attn) is not GDN or type(cache) is not State
              or cache.conv.shape != (1, 3, 10240) or cache.S.shape != (1, 48, 128, 128)
              or cache.conv.dtype != mx.bfloat16 or cache.S.dtype != mx.float32):
            return False
    for norm in norms:
        if (type(norm) is not Norm or norm.zero_centered is not True
                or norm.weight.shape != (5120,)
                or norm.weight.dtype not in (mx.bfloat16, mx.float32)
                or type(norm.eps) is not float or not math.isfinite(norm.eps)
                or not 0.0 < norm.eps < 1.0):
            return False
    return True


def _eligible(model, input_ids, ctx, last_only, capture):
    # Unknown metadata selects the old path before any embedding/mixer/cache write.
    try:
        return _eligible_metadata(model, input_ids, ctx, last_only, capture)
    except (AttributeError, TypeError, IndexError):
        return False


def _carry_forward(model, residual, ctx):
    from ..kernels.residual_norm_metal import residual_norm
    # First norm is unchanged. Each later input norm consumes the previous MLP add.
    hidden = model.layers[0].input_layernorm(residual)
    for i, layer in enumerate(model.layers):
        delta = (layer.self_attn(hidden, ctx, i) if layer.is_attn
                 else layer.linear_attn(hidden, ctx, i))
        mid, post = residual_norm(residual, delta,
                                  layer.post_attention_layernorm.weight,
                                  layer.post_attention_layernorm.eps)
        mlp_delta = layer.mlp(post)
        norm = model.layers[i + 1].input_layernorm if i + 1 < len(model.layers) else model.norm
        residual, hidden = residual_norm(mid, mlp_delta, norm.weight, norm.eps)
    return (model.embed_tokens.as_linear(hidden) if model.cfg.tie_word_embeddings
            else model.lm_head(hidden))


def try_residual_carry(model, input_ids, ctx, last_only, capture):
    """Return None for upfront fallback; logits otherwise. No partial fallback."""
    if not _eligible(model, input_ids, ctx, last_only, capture):
        return None
    mx = _runtime_types()[0]
    residual = model.embed_tokens(input_ids)
    # The read-only embedding operation is the last check before any cache writes.
    if residual.shape != (1, 5120) or residual.dtype != mx.bfloat16:
        return None
    return _carry_forward(model, residual, ctx)
