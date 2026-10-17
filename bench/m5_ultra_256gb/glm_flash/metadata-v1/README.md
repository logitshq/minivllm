---
license: mit
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
library_name: mlx
pipeline_tag: image-text-to-text
tags:
  - mlx
---

# mlx-community/GLM-5.3-Flash-4bit

[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) in MLX affine format, mixed 4/5/6-bit, group size 64.

The weights are the 4-bit variant of [orcarouter/GLM-5.3-Flash-MLX](https://huggingface.co/orcarouter/GLM-5.3-Flash-MLX)
(revision `c80f6810b1a95b5be9042761becc6aa78d189782`, the `4-bit/` build, which is also the repo root there). The quantisation is orcarouter's.
The 62 safetensors shards are byte-identical to that build (sha256 checked).

Change from the source: `config.json` now lists the per-module widths of the MTP layer (layer 45) in
`quantization` and `quantization_config`. The source config named them for layers 3-44 only. The tensors are not changed.

Size: 62 shards, 203,992,076,296 bytes (190 GiB). It needs a Mac with 256 GB of memory or more.

## Widths

Read from each tensor's shapes (`bits = 32 * weight_columns / (group_size * scales_columns)`), group size 64 throughout.

| Modules | Layers | Bits |
|---|---|---:|
| Routed experts `gate_proj`, `up_proj` | 3-45 | 4 |
| Routed experts `down_proj` | 3-45 | 5 |
| Shared expert `gate_proj`, `up_proj`, `down_proj` | 3-45 | 6 |
| Dense MLP `gate_proj`, `up_proj` | 0-2 | 4 |
| Dense MLP `down_proj` | 0-2 | 5 |
| Sparse attention `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `o_proj` | 3, 7, 11, ..., 43, 45 (12 layers) | 4 |
| Linear attention layers, `kv_b_proj`, indexer, router, mHC, norms, `embed_tokens`, `lm_head`, MTP `eh_proj`, vision tower | all | bf16 |

Layer 45 is the MTP (next-token prediction) layer.

## Usage

mlx-lm does not support the `glm5_next` architecture. [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) has a `glm5_next` model;
see the orcarouter model card for its use.

## License

MIT, the same as the base model.
