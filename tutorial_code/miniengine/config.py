"""模型配置：只从 HF 的 config.json 读字段，不依赖 transformers 的模型类。

Qwen3.8-27B 和 Qwen3.5-0.8B 的 config 里 model_type 都是 qwen3_5（同一架构），
HF 原版和 mlx-community 的量化版都把文本配置放在 text_config 里，量化版多一个 quantization 块。
"""
from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class ModelConfig:
    hidden_size: int
    num_hidden_layers: int
    layer_types: list[str]            # "linear_attention" | "full_attention"
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    tie_word_embeddings: bool
    rope_theta: float
    rotary_dim: int                   # head_dim * partial_rotary_factor
    quantization: dict | None = None  # {"group_size", "bits", "mode"}，None = bf16
    model_dir: str = ""

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> "ModelConfig":
        model_dir = Path(model_dir)
        raw = json.loads((model_dir / "config.json").read_text())
        tc = raw.get("text_config", raw)
        rp = tc.get("rope_parameters") or {}
        head_dim = tc["head_dim"]
        n = tc["num_hidden_layers"]
        layer_types = tc.get("layer_types") or [
            "full_attention" if (i + 1) % tc["full_attention_interval"] == 0 else "linear_attention" for i in range(n)
        ]
        return cls(
            hidden_size=tc["hidden_size"], num_hidden_layers=n, layer_types=layer_types,
            num_attention_heads=tc["num_attention_heads"], num_key_value_heads=tc["num_key_value_heads"], head_dim=head_dim,
            linear_num_key_heads=tc["linear_num_key_heads"], linear_num_value_heads=tc["linear_num_value_heads"],
            linear_key_head_dim=tc["linear_key_head_dim"], linear_value_head_dim=tc["linear_value_head_dim"],
            linear_conv_kernel_dim=tc["linear_conv_kernel_dim"], intermediate_size=tc["intermediate_size"],
            vocab_size=tc["vocab_size"], rms_norm_eps=tc["rms_norm_eps"],
            tie_word_embeddings=raw.get("tie_word_embeddings", tc.get("tie_word_embeddings", False)),
            rope_theta=float(rp.get("rope_theta", tc.get("rope_theta", 1e7))),
            rotary_dim=int(head_dim * rp.get("partial_rotary_factor", 1.0)),
            quantization=raw.get("quantization"), model_dir=str(model_dir),
        )
