"""DFlash2 drafter（block diffusion 投机解码的草稿模型），MLX 实现。照 HF 的 dflash_model.py（ref/）移植。

一轮（block 8 = 1 个锚点 + 7 个草稿）：
  1. 上下文特征：目标模型第 [5, 19, 33, 47, 61] 层的输出拼起来 [n, 5·5120]，过 fc + hidden_norm 变成 drafter 的"上下文"。
     每层的 K/V 由同一份上下文算出来并缓存（drafter 自己的 KV cache 只存上下文，不存块）。
  2. 块：[锚点, mask × 7] 用目标模型的词嵌入，过 5 层：每层 = 动态卷积(prepare) → 注意力（q 来自块，K/V = 上下文 + 块，
     块内非因果，滑窗 2048）→ 动态卷积(finish) → 残差；MLP 同样夹一对动态卷积。
  3. 最后 7 个位置的 hidden 过目标模型的 lm_head 得 logits → 每个位置 top-16 候选 → 选路器按位置顺序挑一条连贯的路：
     score = logit + (predecessor_codebook[前一个 token] ⊙ proj(hidden)) · successor_codebook[候选]。
  4. 目标模型一次前向验证 8 个 token（kernels 都是现成的 prefill 路径），接受最长匹配前缀 + 1 个 bonus。无损：greedy 结果和不投机完全一样。
"""
import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from ..model.layers import Linear, RMSNorm, apply_rope


@dataclass
class DraftConfig:
    hidden_size: int; num_layers: int; num_heads: int; num_kv_heads: int; head_dim: int; intermediate_size: int
    rms_norm_eps: float; rope_theta: float; sliding_window: int; vocab_size: int
    block_size: int; conv_group_size: int; conv_kernel_size: int; mask_token_id: int
    selector_rank: int; selector_top_k: int; target_layer_ids: list[int]

    @classmethod
    def from_dir(cls, d):
        c = json.loads((Path(d) / "config.json").read_text()); dc = c["dflash_config"]
        return cls(c["hidden_size"], c["num_hidden_layers"], c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"],
                   c["intermediate_size"], c["rms_norm_eps"], float(c["rope_parameters"]["rope_theta"]), c["sliding_window"], c["vocab_size"],
                   dc["block_size"], dc["conv_group_size"], dc["conv_kernel_size"], dc["mask_token_id"], dc["selector_rank"], dc["selector_top_k"],
                   list(dc["target_layer_ids"]))


class DynamicConv(nn.Module):
    """分组动态因果卷积（kernel 2）：卷积核 = 固定 base_kernel + 由输入投影出来的动态部分。prepare 用第 0 套，finish 用第 1 套。"""

    def __init__(self, hidden: int, kernel: int, group: int, q=None):
        super().__init__()
        self.K, self.G = kernel, group
        self.groups = hidden // group
        self.base_kernel = mx.zeros((2, kernel, hidden), dtype=mx.bfloat16)
        self.kernel_projection = Linear(hidden, 2 * kernel * self.groups, q)

    def _conv(self, x, dyn, base):
        """x [L, H]，dyn [L, K, groups]，base [K, H]。y[t] = Σ_k (base[k] + dyn[t, k] 按组广播) ⊙ x[t−k]（块内，越界补 0）。"""
        L, H = x.shape
        xb = x.reshape(L, self.groups, self.G)
        out = mx.zeros_like(xb)
        for k in range(self.K):
            shifted = xb if k == 0 else mx.pad(xb[:-k], [(k, 0), (0, 0), (0, 0)])
            out = out + (base[k].reshape(1, self.groups, self.G) + dyn[:, k][:, :, None].astype(x.dtype)) * shifted
        return out.reshape(L, H)

    def prepare(self, x):
        dyn = self.kernel_projection(x).reshape(x.shape[0], 2, self.K, self.groups)
        return self._conv(x, dyn[:, 0], self.base_kernel[0]), dyn[:, 1]

    def finish(self, x, dyn):
        return self._conv(x, dyn, self.base_kernel[1])


class DraftAttention(nn.Module):
    def __init__(self, cfg: DraftConfig, q=None):
        super().__init__()
        self.H, self.Hkv, self.D, self.theta, self.window = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.rope_theta, cfg.sliding_window
        self.q_proj = Linear(cfg.hidden_size, self.H * self.D, q); self.k_proj = Linear(cfg.hidden_size, self.Hkv * self.D, q)
        self.v_proj = Linear(cfg.hidden_size, self.Hkv * self.D, q); self.o_proj = Linear(self.H * self.D, cfg.hidden_size, q)
        self.q_norm = RMSNorm(self.D, cfg.rms_norm_eps, zero_centered=False); self.k_norm = RMSNorm(self.D, cfg.rms_norm_eps, zero_centered=False)
        self.ctx_k = None; self.ctx_v = None; self.ctx_len = 0           # 上下文的 K/V cache [1, Hkv, N, D]

    def reset(self):
        self.ctx_k = self.ctx_v = None; self.ctx_len = 0

    def _kv(self, h, pos0):
        n = h.shape[0]
        k = self.k_norm(self.k_proj(h).reshape(n, self.Hkv, self.D)).transpose(1, 0, 2)[None]
        v = self.v_proj(h).reshape(n, self.Hkv, self.D).transpose(1, 0, 2)[None]
        return apply_rope(k, pos0, self.D, self.theta), v

    def __call__(self, x, ctx_new, start):
        """x: 块 [L, H]（位置 start..start+L-1）；ctx_new: 新增的上下文 [n, H]（位置 start-n..start-1），先追加进 cache。"""
        if ctx_new is not None and ctx_new.shape[0] > 0:
            k, v = self._kv(ctx_new, start - ctx_new.shape[0])
            self.ctx_k = k if self.ctx_k is None else mx.concatenate([self.ctx_k, k], axis=2)
            self.ctx_v = v if self.ctx_v is None else mx.concatenate([self.ctx_v, v], axis=2)
            self.ctx_len = self.ctx_k.shape[2]
        L = x.shape[0]
        q = apply_rope(self.q_norm(self.q_proj(x).reshape(L, self.H, self.D)).transpose(1, 0, 2)[None], start, self.D, self.theta)
        kb, vb = self._kv(x, start)
        lo = max(0, self.ctx_len - self.window + L)                      # 滑窗：块里最后一个位置往前 2048
        k = mx.concatenate([self.ctx_k[:, :, lo:], kb], axis=2); v = mx.concatenate([self.ctx_v[:, :, lo:], vb], axis=2)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.D ** -0.5, mask=None)   # 块内非因果，上下文全可见
        return self.o_proj(o[0].transpose(1, 0, 2).reshape(L, -1))


class MLP(nn.Module):
    def __init__(self, cfg, q=None):
        super().__init__()
        self.gate_proj = Linear(cfg.hidden_size, cfg.intermediate_size, q); self.up_proj = Linear(cfg.hidden_size, cfg.intermediate_size, q)
        self.down_proj = Linear(cfg.intermediate_size, cfg.hidden_size, q)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DraftLayer(nn.Module):
    def __init__(self, cfg, q=None):
        super().__init__()
        self.self_attn = DraftAttention(cfg, q); self.mlp = MLP(cfg, q)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, zero_centered=False)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, zero_centered=False)
        self.attention_conv = DynamicConv(cfg.hidden_size, cfg.conv_kernel_size, cfg.conv_group_size, q)
        self.mlp_conv = DynamicConv(cfg.hidden_size, cfg.conv_kernel_size, cfg.conv_group_size, q)

    def __call__(self, x, ctx_new, start):
        h, dyn = self.attention_conv.prepare(self.input_layernorm(x))
        x = x + self.attention_conv.finish(self.self_attn(h, ctx_new, start), dyn)
        h, dyn = self.mlp_conv.prepare(self.post_attention_layernorm(x))
        return x + self.mlp_conv.finish(self.mlp(h), dyn)


class DFlash2Drafter(nn.Module):
    def __init__(self, cfg: DraftConfig, q=None):
        super().__init__()
        self.cfg = cfg
        self.fc = Linear(len(cfg.target_layer_ids) * cfg.hidden_size, cfg.hidden_size, q)
        self.hidden_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, zero_centered=False)
        self.layers = [DraftLayer(cfg, q) for _ in range(cfg.num_layers)]
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, zero_centered=False)
        self.selector_proj = Linear(cfg.hidden_size, cfg.selector_rank, q)
        self.predecessor_codebook = mx.zeros((cfg.vocab_size, cfg.selector_rank), dtype=mx.bfloat16)
        self.successor_codebook = mx.zeros((cfg.vocab_size, cfg.selector_rank), dtype=mx.bfloat16)

    def reset(self):
        for l in self.layers:
            l.self_attn.reset()

    def draft(self, feats_new, anchor: int, start: int, embed_fn, lm_head_fn) -> list[int]:
        """feats_new: 还没喂给 drafter 的上下文特征 [n, 5·H]（位置 start-n..start-1）；anchor 在位置 start。返回 7 个草稿 token。"""
        ctx = self.hidden_norm(self.fc(feats_new)) if feats_new is not None and feats_new.shape[0] > 0 else None
        B = self.cfg.block_size
        ids = mx.array([anchor] + [self.cfg.mask_token_id] * (B - 1))
        x = embed_fn(ids)
        for layer in self.layers:
            x = layer(x, ctx, start)
        h = self.norm(x)[1:]                                             # 7 个草稿位置
        logits = lm_head_fn(h).astype(mx.float32)                       # [7, V]
        return self._select(h, logits, anchor)

    def _select(self, h, logits, anchor):
        k = self.cfg.selector_top_k
        cand = mx.argpartition(-logits, k - 1, axis=-1)[:, :k]           # [7, k] 每个位置的 top-k 候选
        unary = mx.take_along_axis(logits, cand, axis=-1)               # 它们的 logit
        proj = self.selector_proj(h).astype(mx.float32)                 # [7, r]
        succ = self.successor_codebook[cand].astype(mx.float32)         # [7, k, r]
        path, prev = [], anchor
        mx.eval(cand, unary, proj, succ)
        for p in range(h.shape[0]):
            pred = self.predecessor_codebook[prev].astype(mx.float32)   # [r]
            score = unary[p] + (succ[p] @ (pred * proj[p]))             # [k]
            j = int(mx.argmax(score).item())
            prev = int(cand[p, j].item())
            path.append(prev)
        return path


def load_drafter(model_dir: str, quantize: dict | None = {"group_size": 64, "bits": 4, "mode": "affine"}) -> DFlash2Drafter:
    """读 bf16 的 safetensors；quantize 不为 None 时把大的 Linear 在加载时量化（默认 affine 4bit，和 dflash-mlx 一样）。"""
    cfg = DraftConfig.from_dir(model_dir)
    model = DFlash2Drafter(cfg, quantize)
    weights = {}
    for k, v in mx.load(str(Path(model_dir) / "model.safetensors")).items():
        k = k.replace("candidate_selector.hidden_projection", "selector_proj").replace("candidate_selector.", "")
        weights[k] = v
    out = {}
    for k, v in weights.items():
        if quantize is not None and k.endswith(".weight") and v.ndim == 2 and ("proj" in k or k.startswith("fc.") or "mlp." in k):
            w, scales, biases = mx.quantize(v, group_size=quantize["group_size"], bits=quantize["bits"], mode=quantize["mode"])
            base = k[: -len(".weight")]
            out[k], out[base + ".scales"], out[base + ".biases"] = w, scales, biases
        else:
            out[k] = v
    model.load_weights(list(out.items()), strict=True)
    mx.eval(model.parameters())
    return model
