---
language:
- en
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
library_name: mlx
pipeline_tag: text-generation
tags:
- mlx
- mlx-vlm
- qwen3_5_mtp
- qwen3.8
- qwen3.8-27b
- qwen
- mtp
- speculative-decoding
- draft-model
- nvfp4
inference: false
---

# Qwen3.8-27B-MTP-nvfp4

This repository contains the Multi-Token Prediction (MTP) drafter weights split from [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) and quantized with `mlx_vlm.convert` for `mlx-vlm` speculative decoding.

This is not a standalone model. Use it as the draft model alongside a compatible Qwen3.8 27B target checkpoint.

## Use with mlx-vlm

```bash
mlx_vlm generate \
  --model mlx-community/Qwen3.8-27B-nvfp4 \
  --draft-model mlx-community/Qwen3.8-27B-MTP-nvfp4 \
  --prompt "Write a quicksort in Python." \
  --max-tokens 256 \
  --enable-thinking
```

`--draft-kind mtp` is detected automatically from `model_type`.

## Model details

- Model type: `qwen3_5_mtp`
- MTP block size: 3
- Target architecture: Qwen3.8 27B
- Precision: MLX NVFP4, group size 16
- Runtime: MLX / `mlx-vlm`
- Source revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`

The adapter holds only the MTP drafter weights; the target supplies the token embeddings and language-model head at runtime. Use an adapter and target derived from the same Qwen3.8 27B checkpoint. The upstream Apache 2.0 license and model limitations continue to apply.
