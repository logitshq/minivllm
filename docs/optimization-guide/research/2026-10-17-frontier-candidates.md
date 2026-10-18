# Qwen27B / GLM Flash：下一批可验证候选

访问日期：**2026-10-17（America/Los_Angeles）**。本次只读本地证据和一手网站，没有安装、下载权重、启动服务或增加性能点。外部 `main/master` 链接是访问时资料，尚不是本机注册的 commit / runtime / checkpoint pins；开始实验前必须重新冻结。

先沿用[模型路线](../model-roadmap.md)和 [v14 实测口径](../progress/2026-10-17-v14/README.md)：目标是 **Qwen3.8-27B dense、MLX NVFP4 revision `5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6`**，以及 **GLM-5.3-Flash、`glm5_next`、mixed 4/5/6-bit revision `fa04cbc3622b16aa7475404060b05e506d81bd4e`**。Qwen3.8-Flash-Next 是另一个模型；GLM-5.3 非 Flash 也不能替换本目标。

v14 的联合非支配端点仍是 oMLX C1 和 Native ON C8，并非自研全线超过。26 个进程观察汇总成 10 组配置/并发点；主要三引擎各 C1/C8 有 3 个新进程。横轴完整 POST→DONE p50/p95，纵轴 committed tokens / 共同请求窗口；进程 min–max 不是置信区间，历史跨引擎结果不是配对实验。**外部项目未在本机同合同运行，不能算领先或补进这张图。**

## 竞争路线：支持声明和可比性分开

“未确认”表示本次读到的一手资料不足以确认该精确路径，不断言项目不支持。

| 项目 / 一手来源 | Qwen3.8-27B / NVFP4 | GLM-5.3-Flash / `glm5_next` | 对本机下一步的意义 |
| --- | --- | --- | --- |
| [vLLM Metal 支持表](https://github.com/vllm-project/vllm-metal/blob/main/docs/supported_models.md) | 明列 Qwen3.5/3.6/3.8 hybrid SDPA+GDN；27B 示例为 **8-bit**。同表其他模型的 NVFP4 示例不足以证明我们的精确 NVFP4 snapshot。 | 当前表列 GLM4.5 / 4.7 等，未确认本目标。CUDA vLLM 的 GLM/NVFP4 支持不能代替 Metal 证据。 | 优先做精确 checkpoint 的加载/量化路径核验，再同 D512 请求资格。prefix cache 默认开启的路径必须显式关闭并取证。 |
| [Splash / incoai](https://github.com/incoai/splash/blob/main/README.md) | 明列 dense 27B，MLX affine 2–8-bit、mixed、MXFP4/NVFP4，以及 NVIDIA/unsloth NVFP4；是新增同格式对照的强候选。 | 本次模型表未确认。 | 默认 DFlash2、INT8 KV、缓存及 27B ANE 路线需逐项记录；不能将这些组合默认为当前 target-only/BF16-KV 条件。先验证冻结 MLX 权重能否原值加载。 |
| [Sushi](https://github.com/beamivalice/sushi/blob/main/README.md) | 当前列 Qwen3.8-Flash-Next 自有 packs，不能替代 dense 27B NVFP4。 | 明列 GLM Flash **Sushi 2 / 2.4 bpw**、DFlash2，默认 8-bit KV；不同于本机 mixed 4/5/6-bit。 | 可开独立的质量–内存–速度分支；不能并入固定 quant 曲线。其可选 expert-pick-tolerance 明示有损，不能静默启用。 |
| [MLTF](https://github.com/DT-HJ/my-little-trie-forge/blob/main/README.md) | 精确 27B，但准备包为 **MLTF-q8c / 8-bit**，DFlash2、INT8 KV；不是当前 NVFP4。 | 未确认。 | M5 Max 128 GiB 报告可提示专用调度/验证路径；其文档还明确自然 prefill 与 all-ready decode 计时不同。公开 C1/C4 decode 数不能当本机完整 HTTP 指标。 |
| [Slipstream / npanj](https://github.com/npanj/slipstream/blob/main/README.md) | 主路线是 **Qwen3.8-Flash-Next V3 / Swift GGUF**，SSD expert streaming、PLD/MTP；不是 dense 27B。 | 未确认。 | 它解决超内存 MoE 驻留问题。我们的 dense NVFP4 权重约 14.95 GiB 已可驻留，SSD streaming 不是已证实的 C1 提速方向。GLM 可借鉴内存策略，不能据此宣称模型已支持。 |
| [llama.cpp Qwen 模型](https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen35.cpp)、[GLM 模型](https://github.com/ggml-org/llama.cpp/blob/master/src/models/glm5-next.cpp) | 有 Qwen hybrid/GDN 与 GGUF 路径；[Metal NVFP4 实现记录](https://github.com/ggml-org/llama.cpp/actions/runs/22614228818)不等于我们的 MLX checkpoint 可直接读入。精确 27B 转换、Metal 路由和输出仍待资格。 | 源码明确 GLM5-Next = GLM-5.3-Flash、KDA+MLA/DSA+mHC；[GLM5Next MTP PR](https://github.com/ggml-org/llama.cpp/pull/29928)已合并。 | 值得作独立 GGUF runtime 对照；先冻结转换和每 tensor 格式，确认全 Metal op 覆盖/真实 fallback。不能把 CPU/CUDA 支持或 GGUF 名称视为本机同权重资格。 |
| [oMLX releases](https://github.com/jundot/omlx/releases)、[GLM DFlash 适配](https://github.com/jundot/omlx/blob/main/docs/experimental/dflash_mlx_integration.md) | 本机已有同冻结 MLX NVFP4 的 D512 协议/计数资格与正式点；是当前 C1 强对照。 | 官方 `glm5_next` 通过 mlx-vlm 及兼容适配，有 KDA replay / DSA rollback；模型卡明确 [mixed 4/5/6-bit](https://huggingface.co/mlx-community/GLM-5.3-Flash-4bit)。 | GLM 首选无投机 reference。189.982 GiB 权重文件不是加载峰值；先实际冷加载/短 C1，再计数、状态、长输出。官方其他 quant 或 M3 Ultra 数据不是本机结果。 |
| [Lithos 模型合同](https://github.com/lithos-ai/lithos-metal/blob/main/docs/design/models.md) | 明列 dense 27B、48 GDN+16 attention、NVIDIA NVFP4 混合存储及 FP32 recurrent state；本机已有其独立官方 pack/投机路线的 D512 正式点。 | 当前适配表未确认。 | 保留已有对照，但协议/计数合格不意味着与 MLX runtime 全状态逐位等价。不能以“NVFP4”名称省略 pack、activation、draft 和状态格式边界。 |

Splash 的[当前开发说明](https://github.com/incoai/splash/blob/main/DEVELOPMENT.md)还提醒旧预打包格式与当前 loader 不兼容；不要直接复用旧帖子里的下载命令。Slipstream 也有同名仓库；本表专指 `npanj/slipstream`，不混合不同作者、模型或硬件的速度声明。任何新外部候选至少先保留完整 prompt IDs、原始响应、EOS/计数/缓存证据，再用 observer OFF 的 8 warm + 64 formal、C1/C8、3 fresh processes；若切换 quant/KV/draft/ANE，则单独命名并披露，不能把改变质量条件的数字写成同条件收益。

## MSInfer：runtime 边界为什么会改变优化方向

[MSInfer 技术报告 2607.16831v1](https://arxiv.org/html/2607.16831v1)研究的是 **NVIDIA B200 GDN**。报告区分 runtime/FFI dispatch、shape 专门化与 prefill 算法重构：更强参考基线和计时边界会改变候选排序；直接 FFI 的收益与 kernel 内部收益不能混写。它还用 profiler 反驳“状态大，所以必然带宽饱和”的简单判断。本文不移植其 CUDA 数字、warp 配置或容差门到 Apple。

下面是**我们的可测试推断**，不是该论文已经证明的 M5 收益：

| 推断 / 默认 OFF 实验 | 最小可证伪测试 | 明确停止条件 |
| --- | --- | --- |
| 长 decode 的 GDN 重复 prework/提交成本值得先消除，保留五个 native Linear 和原状态递推。 | 复用 E029 单开关，真实 D 前缀逐位 logits/全部状态；再实际 HTTP capture/replay，最后 paired OFF/ON 完整请求计时。E024/E028 单独开关，组合另做资格。 | 任一 finite/原 dtype 位比较/参数或输入不变性失败；只有微 kernel 变快但完整请求无收益也不采纳。 |
| C1 图构造节省可能被已有 ahead 重叠隐藏，剩余差距也可能在既有 eval 等待或交付。 | [c1_pipeline_profile](../../../bench/m5_ultra_256gb/c1_pipeline_profile/README.md)只在原边界记内存时间戳，不加 eval/sync；先测 OFF/ON 观察开销，再区分构图、等待、已有 ahead 消费和交付区间。 | 不把 Python span 当 GPU 独占时间；若观察开销显著或区间重叠不可分，不据此声称某 kernel 占比。 |
| B1 与 B2–8 的提交/并行形状不同，B1 prework 结果不能外推 C8。 | 新 batched 默认 OFF 路线保留原 gate、五投影、状态 commit；真实 B1..8、异长/续写/退场逐位比较，记录实际模型 B 分布后才计时。 | HTTP C 不等于实际 B；未过新数值域不扩 B16/32，也不把 C8 尾段 B1 命中当整批优化。 |
| GLM 的首要约束可能是驻留/状态生命周期而非 Qwen 的同一 kernel。 | 原 mixed quant、官方无投机 `glm5_next`，先受资源门控制的短 C1 冷加载；记录 physical/MLX 峰值、paging、KDA/DSA/mHC 实际路由，再决定专家/缓存方向。 | 不能用 18B active 估算只需驻留 18B；不能把改变 expert 选择或降低 state/KV 精度当原模型精确优化。 |

优先级：**先完成已经具备本机输入/审计链的 GDN 与 C1 分段诊断；新增外部筛选先 Splash 精确 NVFP4，再 vLLM Metal 精确 snapshot；GLM 先 oMLX 官方原 quant reference，Sushi/llama.cpp 另立转换或质量分支。** 这比同时引入新模型、quant、draft、缓存和调度更容易解释一次收益或失败。

保留负结果约束：[E004 与 packing 证据](../../../bench/reviews/long-d-gdn-projection-packing.md)没有证明任意 batch 精确；M4 GDN packing 的微小变化未通过性能门，旧 +21.9% 受次序混淆仍被拒绝。新 runtime 的宣传数字不能撤销这些证据。本文没有新的速度预测、部署许可变更或 Pareto 点。
