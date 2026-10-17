# E016：draft backbone 图复用，拒绝采用

固定 shape 完整 draft 几何平均 **1.0400×**，最低 **1.0066×**，未达到预注册 **1.10×** 门槛。单进程初筛结束，未合入生产，也未进入 bucket 或长生成资格。

| Prompt | Block | Reference ms | Graph ms | Speedup |
|---|---:|---:|---:|---:|
| math | 3 | 3.588 | 3.529 | 1.0168× |
| math | 5 | 3.883 | 3.732 | 1.0405× |
| math | 8 | 5.696 | 5.344 | 1.0658× |
| code | 3 | 3.625 | 3.481 | 1.0413× |
| code | 5 | 3.948 | 3.922 | 1.0066× |
| code | 8 | 5.468 | 5.249 | 1.0416× |
| chinese | 3 | 3.601 | 3.430 | 1.0498× |
| chinese | 5 | 4.173 | 3.864 | 1.0799× |
| chinese | 8 | 5.568 | 5.459 | 1.0199× |

24 组检查的 312 个数组比较全部严格通过，草稿 ID 相同。检查包括 backbone hidden、normalized hidden、完整 logits 和每层 KV；math/code 两个 seed 的十个 KV 实际都不同，再交错推进各三步，未串请求。位置变化与同 shape 切换请求均复用已有图。

首次图调用 **205.702 ms**；其余新 shape 首调 **4.530–6.797 ms**。共 9 次 trace；三个增长 KV 步骤每步新增 shape。所有首调和 warm 样本保留。这里的首调包含编译及实际运算，不能把差值称为纯编译时间。

峰值 MLX active **18.620 GiB**。5 次资源快照 pressure=1、thermal=0，swap 数值与 Swapouts 未增加；已知外部推理进程只有获授权暂停的服务。子进程 exit 0、wait/reap 与严格不存在检查通过，服务 PID 74384 身份匹配且从 T 恢复至 S。

`summary.json` 是重算结论，`audit.json` 是独立 CPU 审计，`evidence.tar.gz` 保存所有原始 run/owner/lease/jobs、逐 hash 核验的被测源、workload 与模型 config。`manifest.json` 和 `tar.verify.json` 核验全部 62 个成员。模型权重 SHA 保存在原始 plan，权重本体未复制进 Git。

这只是该模型、短上下文、固定 shape 的 draft 模块初筛。没有 target verification、rollback、acceptance、生成或 HTTP 性能结论。
