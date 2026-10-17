# 近期研究：先测 verification 的盈亏，再增加 speculation

2026-10-17。本页是阅读与实现方向，不是已执行实验。查询/打开的是作者论文原文；没有把论文图或速度数字并入 M5 Ultra Pareto。当前 Lithos D512 六进程正式测量持续运行，期间只做轻量源码/文献工作。

## 新找到的十月论文

[AgSpec v2](https://arxiv.org/html/2610.01108v2) 于10月1日提交、10月6日修订。其可借鉴部分是按来源保留代码文本、按实际输出格式索引，以及用验证反馈控制草稿长度。作者在 RTX PRO 6000 Blackwell 96 GB、vLLM 0.17.1 上测试；报告的 generation throughput 排除了 prefill，不能当成本机 POST→DONE 收益。论文使用离线选择的长度上限，并明确区分构造语料与评测实例。[版本和日期](https://arxiv.org/abs/2610.01108v2)。

**本项目推论与候选，不是论文已证明的 M5 结论：** 可以先做默认 OFF 的 request-local lookup：只索引当前请求已经收到的 prompt 与已经提交的输出，返回最多1/2/4个草稿；目标模型仍逐位置验证。这样可去掉额外 draft 模型、训练、跨请求语料管理和特征提取，先回答最小机制是否有收益。没有合格 match 就回原路径。先不加入全局语料、离线 profile 或在线自适应，避免把两个未知量混在一起。

本机当前 [SpecRunner](../../../tutorial_code/miniengine/engine/spec_runner.py) 使用 DFlash2 的 `capture`、`history`、drafter context 和单序列 cache rollback。lookup 不是替换 `drafter.draft()` 后就完整等价：移除 feature capture、检查 target verify 的计算路径、EOS/cap裁剪以及每一层 recurrent state 回滚都要独立证明。当前 target-only 的 decode 与 prefill verification 也不能仅凭数学等价推定浮点逐位相等。先做相同输入 state 下 T1/2/4/5 的完整 logits/state 检查与成本筛选，再考虑整模型/HTTP。

## 一个必须可证伪的收益条件

以下是本项目的简单成本模型，所有量都要在同一模型/量化、相同实际 batch 与长度下测量：

- `t1`：普通目标模型一次 decode 的成本。
- `K`：提出的草稿数；verification 输入为 anchor 加 K 个草稿。
- `A`：实际接受的草稿数；正常一步最多提交 `A+1`，EOS/cap可能进一步减少。
- `t_lookup + t_verify(K+1) + t_rollback + t_dispatch`：一次投机循环完整成本。

非末尾情况下，至少要满足

```
t_lookup + t_verify(K+1) + t_rollback + t_dispatch < (1 + E[A]) * t1
```

才能有局部吞吐收益；它仍不保证整体请求延迟改善。较高接受率不抵消过贵的 verify；较低 lookup CPU 成本也不消除 target 的矩阵形状变化。最后按实际提交 token 重算整请求吞吐、POST→DONE p50/p95与失败率，保留排队与prefill。不能把 `drafts + bonus` 的理论数量替代 EOS/cap后的实际输出。

M4 与 M5 先交换相同 T 矩阵的成本/数值证据，分别做设备选择。M4 的最佳 K 不能直接拷到 M5；B1可获益也不能推断B8或B64获益。当前D512 fixture只有8个独立coding prompt，不是多轮仓库agent任务；在这组数据上重复64次时**禁止把上一重复的完整答案加入下次lookup语料**。资格用逐步已见历史，formal每请求独立；若将来评估跨轮reuse，另冻结多轮工作负载、语料来源与生命周期，并给所有引擎同等输入。

## recurrent state 的大方向：减少真实读写，但保留语义

[KVBuffer](https://arxiv.org/html/2605.19049v1)（5月18日）缓冲近期K/V、延后状态更新，并在投机验证中减少临时状态。它针对Qwen3-Next/SGLang，说明固定大小状态仍可能有每token IO瓶颈。它不是新发表的十月论文，也没有证明本机GDN核的逐位相等。

**本项目推论：** Qwen目标的48层FP32 S每请求144 MiB；批量增大时，S的读写与重排值得单独计量。但改变递推与归约顺序可能改变位模式，因此不能直接把延迟更新接入当前精确Pareto。近期先落地保持计算顺序的图复用和identity refill，并采样真正的state搬运/生存期；若后续尝试buffered-state，用独立OFF开关、单层→64层→真实调度验证，不改变现有容差。

[DAMP](https://arxiv.org/html/2608.27513v1)（8月27日）采用混合精度 recurrent state，强调统一低位量化会伤害推理准确率。其“接近FP32平均准确率”属于不同精度的质量权衡，不能证明逐位相同。因此**不纳入当前保持正确性的精确Pareto优化**。可借鉴其固定布局和减少搬运的分析方法，不采用其状态量化结果冒充无损加速。

[When Quantization Breaks Memory](https://arxiv.org/abs/2609.04490)（9月3日）在GRU/LSTM应用中展示反复低精度write-back的影响。它不是Qwen或GLM的实验；本项目只据此提醒自己区分一次性权重量化和逐步状态舍入，不据此推算本机误差。

## 执行顺序

1. 关闭当前 Lithos 正式周期、验证原服务恢复、发布真实 D512 比较点。
2. E024 图适配的真实ahead/retire/refill资格与HTTP，E028独立开关保持可归因；未覆盖的discard路径明确报告。
3. oMLX D512相同输入/计数资格，补齐三引擎公平比较；C16/32/64按资源与实际batch递进。
4. T1/2/4/5 verify成本及精确状态诊断。只有存在可测收益空间，再实现最小request-local lookup与真实目标验证。

本页没有执行新lookup、buffered-state或state量化实验，没有读取未来答案生成草稿，没有改变模型/运行时/当前正式配置。论文负责提供可检验的假设，实际加速结论由本机独立完整测量给出。
