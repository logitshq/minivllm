# E016：把五层 draft backbone 的请求状态放到图的输入和输出

状态：初筛完成，**拒绝采用**；完整 draft 几何平均 1.0400×，未过预注册 1.10× 门槛。本实验没有生产开关，入口独立位于
`bench/m5_ultra_256gb/draft_graph/`。生产 E012 selector 在两臂都开启。

## 为什么先测这个模块

E013 分解显示五层 backbone 的同步成本约 2.8–3.8 ms，是 draft 的最大组件；
lm_head 约 0.9–1.6 ms，低于原先约定的 40% 优先门槛。组件计时包含同步开销，
不能直接相加得到完整 draft 的时间。这里用完整 draft 的成对计时作决策。

一次 draft 的计算序列固定，但 Python 每次重新构造五层的很多小算子。
第一性问题是：固定 shape 下，把同一组运算复用为图，能否减少主机构图和提交成本？
本轮不改变 attention、量化、卷积、选路公式、候选数或草稿长度。

## 让图不拥有请求

`BackboneGraph(context, embedded, position, cache)` 的输入是已投影上下文、
块的 embedding、GPU 标量位置，以及五层全部十个 KV 数组；输出是 backbone hidden
和新的十个 KV 数组。图没有内部的“上一请求 KV”。每次 tracing 只浅拷贝现有
`DraftLayer` 和 `DraftAttention` 外壳，把显式输入放入复制后的 attention，然后
调用现有 `DraftLayer.__call__`。这避免复制一套层数学，也不会写回原层的 cache。

`draft()` 是 benchmark adapter：继续使用原来的上下文投影、embedding、final norm、
lm_head 和生产 E012 selector；它把图返回的 KV 写回当前 drafter，使完整调用的公共
行为可与原方法比较。每次测试前两臂都恢复同一 seed，恢复成本都计入时间。

本 prototype 只捕获冻结的层权重。每次调用检查参数对象 identity，换参数会报错；
这个检查包含在计时里。它不是生产通用权重管理方案，不支持运行中改模块结构或
Python 配置。请求 KV 在 identity 检查中明确排除，换请求不应被误判成换权重。

## 编译不能把增长的 KV 变成免费

`mx.compile` 使用正常 shape specialization，没有 `shapeless=True`，也不填充 KV。
每个新 shape 的首次调用时间和 Python trace 次数都保留在 `first-calls.json`。
数学资格检查本身触发的首次编译也必须保留，不能只报告资格检查之后的快调用。
同 shape 的 `position + 7` 测试使用 array scalar 输入，检查 position 没有被冻结；
这是数值检查，不把错位位置当成真实生成轨迹。

真实生成中接受长度变化会改变 KV shape；即使固定 shape 热身后收益很大，也仅是
复用上限。本轮达标后才决定是否值得做有界 bucket，并单独验证 mask、窗口边界、
首次编译摊销和逐步增长。固定 shape 通过绝不是长生成可采用资格。

## 预先规定的检查与门槛

1. 用本地同一目标 checkpoint 提取 math、code、中文三个真实 prompt 的历史特征。
   先用已有历史建立 drafter KV，然后每次追加最后四个历史 token。
2. 每个 prompt 的 block 3、5、8 共九组，与原始 `DFlash2Drafter.draft` 比较
   backbone hidden、normalized hidden、完整 logits、所有十个 KV 的位模式，以及选出的全部 ID。
   任何不等立即 STOP，保存失败两臂数组，不把容差等价改成通过。
3. 九组再做同 shape 的位置变化；另外用 math/code 的真实历史，以较短 prompt 长度
   减七作为共同 seed 长度，并确认两个 seed 的 KV 实际不同，避免共享模板前缀让
   串请求检查失去作用。以交错顺序分别推进三步、每步两个历史 token。每步都用各自的 reference
   和 candidate 状态继续，检查共享图没有串请求或忽略新 KV。
4. 每组保留首次调用，两个 warm round，三个 ABBA block 的完整 draft 样本。
   所有严格检查通过，九组完整 draft speedup 几何平均至少 **1.10×**，
   每点至少 **0.97×**，才进入形状增长实验；不直接合入生产。

检查结果逐组写盘，未完成也保留 `failure.json`、首调时间、trace history 和已经
完成的样本。plan pin 全部生产 Python、该实验源码、workload、两模型 config 和
safetensors SHA256。目标权重的完整 hash 在加载和计时前计算。

## 运行

由主线程的唯一 GPU owner 通过已有 `guarded_v2.py` / `suspended_service.py` 串行运行：

```bash
.venv/bin/python -B bench/m5_ultra_256gb/draft_graph/screen.py \
  --workload .artifacts/m5-ultra-256gb/fullqual-build2/workload.json \
  --drafter /absolute/path/to/Qwen3.8-27B-DFlash2/snapshot \
  --out .artifacts/m5-ultra-256gb/draft-graph-screen-v1
```

输出目录必须是新目录。冻结参数的 CPU 合同检查可独立执行：

```bash
.venv/bin/python -B -m unittest discover \
  -s bench/m5_ultra_256gb/draft_graph -p 'test_*.py'
```

本轮不经过 target verification 或 GDN rollback，不声称 acceptance、generation、
HTTP 或对外推理吞吐提升。

## 实测：算对了，但不值得继续扩成 bucket

2026-10-17，M5 Ultra / MLX 0.32.3，所有数学检查通过；24 组共 **312 个数组**逐位相同，
草稿 ID 相同。这里包含最后一层输出与 normalized hidden，防止只看 norm 后的输出而
忽略更早差异。math/code 的共同 seed 长度为 15，十个 KV 实际全部不同；交错推进的六组
检查全部通过。位置 `+7` 与同 shape 切换请求都没有触发额外 trace。

| Prompt | Block | 原始完整 draft ms | 图复用 ms | Speedup |
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

完整 draft 几何平均 **1.040008×**，最低 **1.006598×**。虽没有回退点，但收益不足以
进入下一阶段。为这个实现继续增加 bucket、padding、窗口 mask 和生命周期管理，
没有通过事先约定的收益门槛。这个结论只否定当前 prototype；没有证明所有编译方案
都无用，也不能从单进程初筛声称稳定 4% 的生产收益。

编译成本也不能丢掉：第一个图调用 **205.702 ms**；其余新 shape 的首次调用为
**4.530–6.797 ms**。共 **9 次 trace**，交错历史推进时每个新的 KV 长度都新增一次。
这些是完整首调时间，含编译和实际运算，未测纯编译分量。固定 shape 的热身收益尚且
不足，真实增长轨迹的编译摊销没有理由被乐观忽略。

本次峰值 MLX active **18.620 GiB**。5 次资源快照 pressure=1、thermal=0，swap 使用量
与 Swapouts 均稳定。已知外部推理进程只有获授权暂停的 oMLX；owned child exit 0、
wait/reap 和严格不存在检查通过，服务 PID 74384 身份匹配后从 T 恢复至 S。

独立 CPU 审计重新计算了全部 108 个 ABBA 样本的中位数与比值，核对全部严格检查、
trace 历史、被测源码 SHA、资源记录和恢复凭证。完整证据与 62 个 archive 成员的
hash 校验保存在 [`E016-draft-graph-v1`](../../experiments/m5-ultra-256gb/E016-draft-graph-v1/README.md)。
没有修改生产代码，所有初筛数据保留。
