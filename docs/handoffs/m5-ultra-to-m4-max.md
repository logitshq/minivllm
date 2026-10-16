# M5 Ultra → M4 Max 同步说明

2026-10-16。共享仓库 `logitshq/minivllm`；M5 分支为
`experiment/m5-ultra-256gb`。本分支已合入 M4 分支
`tutorial/18-gdn-projection-packing` 的 `1363d9f`（截至2026-10-16再次fetch），
包含第22–36章原生执行、投影/归一化、CAP64及失败记录。此次合并没有改变
M5 已测量的引擎源码；M4 的原型仍是未接入状态。

## 先看这些提交和结果

- `dc745bc`：一次清理同一步结束的全部请求。只改两个核心文件，净增11行。
  `ModelRunner.drop_many` 最多压缩一次剩余 KV/GDN 行；整批结束直接释放。
  这是最适合先在 M4 独立评估的通用改动，不依赖 M5 kernel。
- `044a263`：B1–B8 的有界 decode 图、投影/GDN 融合扩展。
- `f44bc5a`：单行 NVFP4 QMV 及编译器参数别名处理。
- `b0d83c3`：完整 Pareto 报告归档。
- [最终报告](../experiments/m5-ultra-256gb/pareto-batch-v1/m5-ultra-pareto-report.txt)、
  [图](../experiments/m5-ultra-256gb/pareto-batch-v1/m5-ultra-pareto.png)、
  [结果 JSON](../experiments/m5-ultra-256gb/pareto-batch-v1/m5-ultra-pareto-results.json)、
  [设计取舍](../experiments/m5-ultra-256gb/pareto-batch-v1/m5-ultra-design-notes.txt)。

改动在 `tutorial_code/miniengine`，不是根目录的 `minivllm` 包。
源码检查用 `git show dc745bc -- tutorial_code/miniengine`。在 M4 自己的分支
先 `git fetch origin`，再选择合并 M5 分支或单独移植需要的改动；M5 继续只推送
自己的实验分支，不更新 M4 的工作分支或 main。

## 已测范围与跨机器边界

M5 Ultra 80 GPU cores / 256 GiB / MLX0.32.3；同一 Qwen3.8-27B NVFP4，
19输入/32输出、greedy、正常EOS、无草稿模型和前缀缓存。69个独立进程，
3,843条正式测量请求，9个外部配置中位数均被实测 native 配置覆盖。
Lithos 官方接口仅测 C1，使用本机 M5 配置；oMLX C1含 balanced/aggressive。
这不是长上下文、其他模型或投机解码的胜出结论。

批量清理的 C64 吞吐中位数为175.83 tok/s，旧版本119.69；三轮范围
155.09–176.38。C32约+2.95%，C8约-4.35%，全部保留。新旧版本未逐对配对。
该改动通过24个真实引擎步骤的输出及全部剩余缓存字节签名比对，覆盖
B2/8/32/64、交错退出、迟到请求和stop token，以及4项CPU生命周期检查。
在 M4 上仍需按本机内存和负载独立验证，不照搬 M5 的 C64 队列或性能结论。

`compile_decode/fuse_decode` 默认OFF。M5的融合路径有明确硬件/版本/
模型/形状检查，范围B1..8、used length≤256；其他情况回到原路径。
派生权重额外约8.714 GiB。不要为了在 M4 命中而删除硬件 guard；M4上
需要单独证明数值、资源和收益。M5上的结果不推翻M4 v20/v21的STOP。

## 可复用与留在 M5 的文件

Git包含引擎源码、基准脚本、汇总JSON、图、校验清单、失败/拒绝候选说明。
`.artifacts/`、`.venv/`、模型和约196MB本地原始证据不在Git；清单中的绝对
路径只是M5证据索引。归档的启动脚本也绑定M5的路径和冻结源码目录，不能
直接在M4执行。跨机复现需在本机构建路径、冻结环境和模型后重跑。

CPU生命周期检查（仓库根目录、已安装MLX的Python环境）：

```sh
python -B bench/m5_ultra_256gb/retire_batch/test_retirement.py
```

本次同步还核对了安装来源：M5的oMLX是官方0.7.1.dev1 wheel，无法证明其
VCS提交；本地源码checkout的SHA不代表被测wheel。已纠正报告中的这一
元数据归属，原始计时和性能数字未变。

## 系统优化攻略与后续实验（2026-10-16）

新增[攻略入口](../optimization-guide/README.md)、[论文笔记](../optimization-guide/papers.md)
和[实验账本](../optimization-guide/experiments.md)。后续每项优化先落盘计划，
再追加正确性、全部计时、资源、失败和取舍；生产热路径保持小。

M5新实验E007（独立大batch GDN融合）、E008（allocator 2/16 GiB配对）、
E009（线程内状态读取复用）均未过预声明收益门，保留原型和证据，未改生产引擎。
E008有12个新HTTP进程、1,170条正式请求，C64配对吞吐几何平均仅+0.131%；
更大池多占约14 GiB空闲buffer，不采用。下一项E010计划检查M5成组prefill/NAX，
不能直接删除旧backend guard。

M4的`17bd7be`[卷积融合报告](../experiments/21-gdn-conv-fusion.md)已合入：局部micro下降12.21%是继续做模型资格的
依据，不是HTTP收益；与这里大batch的E007/E009不是同一个实验，不能互相替代。

## E010进行中：M5保留M=T的NAX分组

29组模型正确性、448层级检查和16步真实调度通过。核心候选只改layers及runner，
用广播packed weight保留单序列的NAX tile；旧split-K投影仍逐行。M4继续使用原legacy
路径，不能移除硬件/版本guard套用M5结论。

初筛完整prefill约1.26–1.42×，但HTTP发现另一活跃`omlx-server`被旧matcher漏识别，
计时受干扰。因此本提交是**资格中的候选**，不是新HTTP/Pareto胜出；受干扰数据完整
保留，正在用可执行名探针、用户授权暂停/恢复服务和最终核心源码重跑。
请以[实验账本E010](../optimization-guide/experiments.md)的后续采用结论为准。
