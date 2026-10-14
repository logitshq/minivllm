"""Chapter 05: load raw HF text weights, map names and reject incomplete/extra tensors strictly."""
import glob
from pathlib import Path

import mlx.core as mx

from ..config import ModelConfig
from .qwen3_5 import Qwen3_5ForCausalLM

PREFIXES = ("model.language_model.",)
MLX_PREFIX = "language_model.model."
# mlx-community 转换时把零中心 RMSNorm 的 "+1" 烧进了权重（存的是 1 + w），GDN 里的普通 norm 不动。
# 我们的 RMSNorm 和 HF 一样存 w，所以读 mlx 格式时要减回去。
ZERO_CENTERED_NORMS = ("input_layernorm.weight", "post_attention_layernorm.weight", "q_norm.weight", "k_norm.weight")


def _map_name(name: str) -> str | None:
    if name.startswith(("model.visual.", "visual.", "vision_tower.", "mtp.")):
        return None                                                 # 视觉塔和 MTP 头永远不加载
    for p in PREFIXES:                       # 27B 的 mlx 版把 lm_head 放在 language_model.lm_head
        if name.startswith(p):
            name = name[len(p):]
            break
    return name.replace("conv1d.weight", "conv1d_weight")


def load_model(model_dir: str | Path, dtype=mx.bfloat16) -> tuple[Qwen3_5ForCausalLM, ModelConfig]:
    cfg = ModelConfig.from_dir(model_dir)
    if cfg.quantization is not None:
        raise ValueError("Use raw HF checkpoint until chapter 07")
    model = Qwen3_5ForCausalLM(cfg)
    weights = {}
    for f in sorted(glob.glob(str(Path(model_dir) / "*.safetensors"))):
        for raw_k, v in mx.load(f).items():
            k = _map_name(raw_k)
            if k is None:
                continue
            if k.endswith("conv1d_weight"):
                v = v.reshape(v.shape[0], -1)                       # [C,1,K] 或 [C,K,1] → [C,K]
            if k.endswith("A_log") or k.endswith("dt_bias"):
                v = v.astype(mx.float32)                            # 门控参数永远 fp32
            elif v.dtype in (mx.bfloat16, mx.float16, mx.float32):
                v = v.astype(dtype)
            weights[k] = v
    if cfg.tie_word_embeddings:
        weights.pop("lm_head.weight", None)
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    return model, cfg
