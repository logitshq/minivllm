# GLM-5.3-Flash：固定权重已下载与校验

`mlx-community/GLM-5.3-Flash-4bit`，revision
`fa04cbc3622b16aa7475404060b05e506d81bd4e`。
62 shards 共 **203,992,076,296 bytes（189.982 GiB）**，全部下载文件 SHA256 与固定 publisher LFS 清单一致。
下载结束凭据见 [complete.json](complete.json)，原清单见
[metadata-v1](../../../../bench/m5_ultra_256gb/glm_flash/metadata-v1/manifest.json)。

CPU 只读 safetensors header 检查共 113,446 tensors，payload 为 203,976,457,080 bytes。
其中 next-token/MTP layer 占 4,573,936,256 bytes，去掉该层后的逻辑 payload 约 185.708 GiB；
单层合并 gate/up 的新增 payload 上界为 2,717,908,992 bytes。详细分类见
[header-inventory.json](header-inventory.json)。这些数值不能代替加载峰值和运行内存实测。

本次没有加载模型、执行 GPU、验证生成或测量速度，**没有 GLM Pareto 点**。
根线程按固定清单复核全部 62 份下载 hash 凭据，并核对本目录文件 hash；归档步骤没有再读一遍 190 GiB 权重。
范围和 SHA 见 [archive.json](archive.json)。

加载前还需释放另一个服务的驻留模型并保证恢复：该服务实测 physical footprint 约 142.82 GiB，
暂停进程不会释放这部分内存。当前尚未卸载；预算和真实服务恢复路线见
[加载审查](../../../../bench/reviews/glm-flash-load-budget.md) 与
[临时卸载审查](../../../../bench/reviews/omlx-temporary-unload.md)。
