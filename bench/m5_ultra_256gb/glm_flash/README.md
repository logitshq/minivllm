# GLM-5.3-Flash 环境准备

2026-10-17。固定 `mlx-community/GLM-5.3-Flash-4bit` revision
`fa04cbc3622b16aa7475404060b05e506d81bd4e`；62 shards / 203,992,076,296 bytes
（189.982 GiB）。名称为4bit，实际依模型配置混合4/5/6bit；保持原权重，不转换量化。

`metadata-v1/` 保存公开HF API文件清单、固定版本config/README/index和每shard发布方
LFS SHA256。下载器只下载和逐shard核验，不导入模型、不占GPU、不声称能加载或已资格。
权重进入忽略的 `.artifacts/models/huggingface`，不进Git；小源码与manifest进Git。
库断线重试/缓存由huggingface_hub管理，整个attempt失败写failure并保留缓存；重新启动须新out。

```sh
.venv/bin/python -B bench/m5_ultra_256gb/glm_flash/download.py \
  --out .artifacts/m5-ultra-256gb/glm-flash-download-v1
```

**避开性能计时**运行下载、SHA和压缩。启动前要求权重总量加64GiB的磁盘余量。
2026-10-17检查时磁盘约1.1TiB可用；已有Jundot oQ4e/oQ5e缓存只有metadata，未见权重shard。
它们是不同量化，不能替代此固定版本而沿用结果。

后续加载先检查256GiB系统实际压力、其他模型驻留与峰值，C1/context2048；已有外部服务暂停
并不会释放其内存。权重文件能放下不等于运行峰值已验证。先建立官方oMLX/MLX-VLM无投机
raw-token/state对照，再做可开关的最小MoE/KDA/DSA/mHC实现；正确性通过后才进入
独立GLM throughput–latency图，不与Qwen点拼成一条前沿。
