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

    def __call__(self, x, cache=None):
        h = self.input_layernorm(x)
        x = x + (self.self_attn(h, cache) if self.is_attn else self.linear_attn(h, cache))
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

    def __call__(self, input_ids: mx.array, cache: list | None = None) -> mx.array:
        """input_ids: [T] 个新 token。cache=None：整句前向；否则 cache[i] 是第 i 层的 AttnCache / GDNCache，会被原地更新。"""
        x = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            x = layer(x, cache[i] if cache is not None else None)
        x = self.norm(x)
        return self.embed_tokens.as_linear(x) if self.cfg.tie_word_embeddings else self.lm_head(x)
