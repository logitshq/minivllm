# M5 Ultra 256 GiB 模型路线

2026-10-17。双重点为 **Qwen3.8-27B + GLM-5.3-Flash**。Qwen 已有性能实验；GLM 固定权重的 62 shards 已下载、全部 SHA 校验和 CPU header 清点通过，尚未加载或取得性能点。

| 模型 | 参数与架构 | 已核实权重大小 | 本机路线 |
| --- | --- | --- | --- |
| [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | 27B dense；64 层，48 GDN + 16 full attention；原生多模态 | 原 BF16 51.747 GiB；当前 [MLX NVFP4](https://huggingface.co/mlx-community/Qwen3.8-27B-nvfp4) 14.952 GiB | 第一重点，继续已有正确性与 throughput–latency 实验 |
| [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | 官方 320B total / 18B active；45 层、288 routed experts/top-8；线性 attention + DSA + mHC；原生多模态 | 原 FP8 305.788 GiB；[MLX mixed 4/5/6-bit](https://huggingface.co/mlx-community/GLM-5.3-Flash-4bit) **189.982 GiB** | 第二重点；先短上下文 C1 加载及资格，再优化 |
| [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) | `glm_moe_dsa`，78 层；checkpoint 张量约 753.33B，[约 40B active](https://docs.api.nvidia.com/nim/reference/z-ai-glm-5-3) | 原 FP8 703.737 GiB；[MLX 4-bit](https://huggingface.co/mlx-community/GLM-5.3-4bit) 389.592 GiB | 256 GiB 不能全驻留；不列入当前双重点 |
| [Qwen/Qwen3.8-2.4T-A95B](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B) | 2.4T / 95B active；92 层 hybrid MoE；512 experts、10 routed + 1 shared | 原 BF16 4556.371 GiB；仅按 4 bit/参数估算也超过 1 TiB，尚未核实可用 MLX quant | 不适合本机全驻留 |

大小是 HF 文件清单中 `.safetensors` 的 byte sum，包含 checkpoint 附属张量，不等于运行峰值。MoE 的 active 参数决定部分算力需求，不能当作只需加载这些权重。GLM-5.3 有 744B 与 753B 的不同公开统计口径；此表明确使用官方 checkpoint 的张量统计，不猜测差额来源。不要把量化文件中打包 U32 的 HF 参数自动统计当作模型架构参数量。

[Qwen 官方发布表](https://github.com/QwenLM/Qwen3.8/blob/main/README.md)当前列出的 3.8 开放型号是 27B 与 2.4T-A95B 及其 FP8。35B-A3B 属于 Qwen3.6，397B-A17B 属于 Qwen3.5；未核实到名为 Qwen3.8-35B/397B 的官方权重。

## GLM Flash 的支持与下载前置条件

下载阶段已完成，原始计划、全部 hash 凭据和 header 清点见
[GLM 模型归档](../experiments/m5-ultra-256gb/GLM-flash-model-v1/README.md)。
当前另一个服务驻留模型 physical footprint 实测约 142.82 GiB；只暂停不会腾出内存。
下一步先完成可恢复的临时卸载，再独立测 GLM 加载峰值和数值资格。
以下大小与支持路线保留原核实口径，不把下载成功算作运行成功。

- 目标应为 `mlx-community/GLM-5.3-Flash-4bit`，固定 revision `fa04cbc3622b16aa7475404060b05e506d81bd4e`。62 shards 共 **203,992,076,296 bytes**；名称虽是 4bit，模型作者明确它实际混合 4/5/6 bit，部分组件保留 BF16。先确认磁盘空间并冻结文件清单/SHA，不自动改用不同 quant。
- 256 GiB 减权重文件约剩 66 GiB 是理论差额。OS、加载峰值、KV/DSA/KDA 状态和临时数组仍需实测，不能据此承诺 1M context 或多并发。首轮建议 context 2048、C1、普通 greedy、正常 EOS。
- 模型卡明确走 **mlx-vlm 的 `glm5_next`**。本机 oMLX 0.7.1.dev1 环境已有 `mlx_vlm.models.glm5_next`、兼容补丁和 `dflash_glm5.py` 的 KDA/DSA/mHC target adapter；这是源码支持证据，尚无本机加载/质量/性能结果。不要把 HF 自动生成的 mlx-lm 示例当作资格依据。
- [Lithos 官方模型适配文档](https://github.com/lithos-ai/lithos-metal/blob/main/docs/design/models.md)与本机注册表未见 GLM adapter；目前不能把它列成 GLM Flash 的可运行对照。自研当前 Qwen dense 核心也不直接支持 `glm5_next`，需要独立 MoE/KDA/DSA/mHC 模块，不能只改模型名。
- 先建立官方 oMLX 无投机 reference，再做自研 raw token/状态资格；通过后才进入同 revision、quant、prompt、EOS 与输出长度的 **throughput vs latency** 图。主 latency 用请求起点到最终输出的 E2E 时间；TTFT/ITL 单列。此页没有任何 GLM 速度或领先结论。

本次只读核实的精确文件总量与 revision：Qwen27B NVFP4 `5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6`，16,054,475,404 B；GLM Flash FP8 `46a050bec13002ad29ee795c16e61cf4fa3c4a9c`，328,337,455,672 B；GLM5.3 MLX4bit `81ad9b73c6188f808737dbb52458f082d249dcbe`，418,321,321,530 B。数据最初通过 HF 官方 `api/models/<id>?blobs=true` 取得；GLM Flash 后续下载完成情况见上述独立归档。
