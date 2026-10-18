"""Chapter 05: complete independent model, replacing HF model.forward in generation."""
import mlx.core as mx
import mlx.nn as nn

from ..config import ModelConfig
from ..context import ForwardContext
from .attention import Attention
from .gdn import GatedDeltaNet
from .layers import Embedding, Linear, RMSNorm, prefill_linear


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        q = cfg.quantization
        self.gate_proj = Linear(cfg.hidden_size, cfg.intermediate_size, q)
        self.up_proj = Linear(cfg.hidden_size, cfg.intermediate_size, q)
        self.down_proj = Linear(cfg.intermediate_size, cfg.hidden_size, q)

    def __call__(self, x, *, _projection=None):
        gate, up = _projection(x) if _projection is not None else (self.gate_proj(x), self.up_proj(x))
        activated = nn.silu(gate) * up
        return _projection.linear(self.down_proj, activated) if _projection is not None else self.down_proj(activated)

    def _prefill_batch(self, x, shape):
        gate = prefill_linear(self.gate_proj, x, shape)
        up = prefill_linear(self.up_proj, x, shape)
        return prefill_linear(self.down_proj, nn.silu(gate) * up, shape)


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_type: str):
        super().__init__()
        self.is_attn = layer_type == "full_attention"
        if self.is_attn:
            self.self_attn = Attention(cfg)
        else:
            self.linear_attn = GatedDeltaNet(cfg)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def __call__(self, x, ctx=None, layer_idx=0, *, _norm_reuse_normal=False, _decode_projections=None):
        mlp, mixer = _decode_projections or (None, None)
        h = self.input_layernorm(x)
        x = x + (self.self_attn(h, ctx, layer_idx, _projection=mixer) if self.is_attn else self.linear_attn(h, ctx, layer_idx, _norm_reuse_normal=_norm_reuse_normal, _projection=mixer))
        if ctx is not None and ctx.mode == "prefill_batch":
            return x + self.mlp._prefill_batch(self.post_attention_layernorm(x), ctx.prefill_shape)
        return x + self.mlp(self.post_attention_layernorm(x), _projection=mlp)


class Qwen3_5ForCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = Embedding(cfg.vocab_size, cfg.hidden_size, cfg.quantization)
        self.layers = [DecoderLayer(cfg, t) for t in cfg.layer_types]
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        if not cfg.tie_word_embeddings:
            self.lm_head = Linear(cfg.hidden_size, cfg.vocab_size, cfg.quantization)

    def __call__(self, input_ids: mx.array, ctx=None, last_only: bool = False, capture: tuple[int, ...] = (), *, _decode_projections=None):
        """input_ids: prefill 时 [T] 个 token，decode 时 [B] 个（每个序列一个）。ctx 为后续 cache/batch 接口（本章仅整句无 cache）。
        last_only=True 时只对最后一个位置算 lm_head（prefill 只需要最后一个位置的 logits，省 1 GB 的 2048×vocab）。
        capture=(5, 19, ...) 时额外返回这些层的输出拼起来的特征 [T, len(capture)·hidden]（DFlash2 drafter 的上下文）。"""
        grouped = ctx is not None and ctx.mode == "prefill_batch"
        if grouped:
            if capture:
                raise ValueError("grouped prefill does not support capture")
            B, T = ctx.prefill_shape
            if input_ids.shape != (B, T):
                raise ValueError("grouped prefill token shape mismatch")
            x = self.embed_tokens(input_ids.reshape(-1))
        else:
            x = self.embed_tokens(input_ids)
        # Explicit per-call proof; never store transient eligibility on a module.
        _norm_reuse_normal = (
            type(ctx) is ForwardContext and ctx.mode == "decode" and not capture
            and set(vars(ctx)) == {"mode", "caches", "positions", "lens", "max_len", "prefill_shape"}
            and ctx.prefill_shape is None
        )
        feats = []
        for i, layer in enumerate(self.layers):
            x = layer(x, ctx, i, _norm_reuse_normal=_norm_reuse_normal,
                      _decode_projections=None if _decode_projections is None else _decode_projections[i])
            if i in capture:
                feats.append(x)
        if grouped and last_only:
            x = x.reshape(B, T, -1)[:, -1]
        else:
            x = x[-1:] if last_only else x
        x = self.norm(x)
        if grouped:
            output = self.embed_tokens.as_linear if self.cfg.tie_word_embeddings else self.lm_head
            logits = prefill_linear(output, x, (B, 1 if last_only else T))
        else:
            if self.cfg.tie_word_embeddings:
                logits = self.embed_tokens.as_linear(x)
            elif _decode_projections is not None:
                logits = _decode_projections[0][0].linear(self.lm_head, x)
            else:
                logits = self.lm_head(x)
        if capture:
            return logits, mx.concatenate(feats, axis=-1)
        return logits.reshape(B, T, -1) if grouped and not last_only else logits

    def lm_head_fn(self, h: mx.array) -> mx.array:
        """给 drafter 用的输出头（drafter 自己没有 lm_head，借目标模型的）。"""
        return self.embed_tokens.as_linear(h) if self.cfg.tie_word_embeddings else self.lm_head(h)
