"""Qwen3.5 架构（Qwen3.8-27B 用的就是它）：embed → N 个 decoder 层 → norm → lm_head。
每层：x = x + mixer(norm1(x)); x = x + mlp(norm2(x))，mixer 是 GDN 或全注意力，由 layer_types 决定。
输入 [T] 个 token id，输出 [T, vocab] 的 logits。cache 见 minivllm/cache.py（第 2 步）。
"""
import mlx.core as mx
import mlx.nn as nn

from ..config import ModelConfig
from .attention import Attention
from .gdn import GatedDeltaNet
from .layers import Embedding, Linear, RMSNorm


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        q = cfg.quantization
        self.gate_proj = Linear(cfg.hidden_size, cfg.intermediate_size, q)
        self.up_proj = Linear(cfg.hidden_size, cfg.intermediate_size, q)
        self.down_proj = Linear(cfg.intermediate_size, cfg.hidden_size, q)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


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

    def __call__(self, x, ctx=None, layer_idx=0):
        h = self.input_layernorm(x)
        x = x + (self.self_attn(h, ctx, layer_idx) if self.is_attn else self.linear_attn(h, ctx, layer_idx))
        return x + self.mlp(self.post_attention_layernorm(x))


class Qwen3_5ForCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = Embedding(cfg.vocab_size, cfg.hidden_size, cfg.quantization)
        self.layers = [DecoderLayer(cfg, t) for t in cfg.layer_types]
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        if not cfg.tie_word_embeddings:
            self.lm_head = Linear(cfg.hidden_size, cfg.vocab_size, cfg.quantization)

    def __call__(self, input_ids: mx.array, ctx=None, last_only: bool = False, capture: tuple[int, ...] = ()):
        """input_ids: prefill 时 [T] 个 token，decode 时 [B] 个（每个序列一个）。ctx 见 forward_context.py。
        last_only=True 时只对最后一个位置算 lm_head（prefill 只需要最后一个位置的 logits，省 1 GB 的 2048×vocab）。
        capture=(5, 19, ...) 时额外返回这些层的输出拼起来的特征 [T, len(capture)·hidden]（DFlash2 drafter 的上下文）。"""
        x = self.embed_tokens(input_ids)
        feats = []
        for i, layer in enumerate(self.layers):
            x = layer(x, ctx, i)
            if i in capture:
                feats.append(x)
        x = self.norm(x[-1:] if last_only else x)
        logits = self.embed_tokens.as_linear(x) if self.cfg.tie_word_embeddings else self.lm_head(x)
        if capture:
            return logits, mx.concatenate(feats, axis=-1)
        return logits

    def lm_head_fn(self, h: mx.array) -> mx.array:
        """给 drafter 用的输出头（drafter 自己没有 lm_head，借目标模型的）。"""
        return self.embed_tokens.as_linear(h) if self.cfg.tie_word_embeddings else self.lm_head(h)
