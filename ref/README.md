# ref/

只读的参考实现，不被 `minivllm/` import，只用来对照数学和权重格式：

- `modeling_qwen3_5.py`、`configuration_qwen3_5.py`：来自 [Hugging Face transformers](https://github.com/huggingface/transformers)（Apache-2.0），Qwen3.5 的官方实现，`torch_recurrent_gated_delta_rule` / `torch_chunk_gated_delta_rule` 是分块 GDN kernel 的参考答案。
- `dflash_model.py`：来自 [z-lab/dflash](https://github.com/z-lab/dflash)（Apache-2.0），DFlash2 drafter 和 draft/verify 循环的参考实现。
- `nvfp4_safetensors_header.json`：`unsloth/Qwen3.8-27B-NVFP4` 的 safetensors 头，记录 compressed-tensors 的权重名和形状。
