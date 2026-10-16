# 21：把 GDN 真卷积、SiLU 和原始 tail 合成一核

V23 在本机 M4 Max 的局部实验中通过数值门与 micro 门：同样消费 act 和 new_tail，局部 host-wall 中位从 **115.489939µs 降到101.392166µs，减少12.2069275%**，6/6 对获胜。这是继续做模型资格的依据；本章没有模型、HTTP 或 Pareto 胜利，也没有把原型接入26文件引擎。

这一步承接[第20章的实际源码阅读](20-donor-implementation-map.md)。实际 oMLX Qwen3.5 路径会在读取 raw projection 和旧 tail 时做真 conv4＋SiLU，并输出原始新 tail。V23借鉴这个融合边界，按我们自己的 BF16 算术手写；没有照搬 donor 的 FP32 conv 累加。

## 先把 before 和 after 说清

数值参考来自冻结的 accepted15 包，见[原 GDN](21-gdn-conv-source/reference/accepted15/model/gdn.py)的 `_decode`。当前 worktree 另含默认 OFF 的 V19 guard，这轮没有拿它替换历史参考。

```python
# Before：原 decode 的卷积链
w = self.conv1d_weight.astype(qkv.dtype)
xp = mx.concatenate([st.conv.astype(qkv.dtype), qkv[:, None]], axis=1)
qkv = (xp * w.T[None]).sum(axis=1)
q, k, v = self._split_heads(nn.silu(qkv), B)
# ... 原递推保持独立 ...
st.conv, st.S = xp[:, 1:], S
```

`xp[:,1:]` 原本就是 view，不能虚报为省掉一次 tail-copy kernel。此前 V18 后卷积实验仍保留 concat/product/sum；V23才把这段真卷积一起处理。

```python
# After：目前仅为独立原型 API，尚未成为 engine 分支
act, new_tail = conv_silu_raw_tail(
    raw, old_tail, w, sum_y=4, backend=mx,
)
# 四个投影、head split/repeat、gates、递推和 norm 都保持在融合边界之外。
```

[Python API](21-gdn-conv-source/prototype/conv_silu_raw_tail.py)只收 BF16 `raw[1,10240]`、`old_tail[1,3,10240]`、`w[10240,4]`，返回独立 BF16 `act[1,10240]` 与 `new_tail[1,3,10240]`。[Metal body](21-gdn-conv-source/prototype/conv_silu_raw_tail.metal)让每个 channel 只有一个 owner：四个 tap 读取 old0/old1/old2/raw，新 tail 写 old1/old2/raw，不原地写输入。

每个 tap product 都先返回 BF16，每次 add 也返回 BF16；不能改成 FP32 连加后只 cast 一次。SiLU保留[原封不动的662字节 typed helper](21-gdn-conv-source/prototype/native_activation.metal)：BF16 sigmoid 中间值，再 BF16 multiply。它来自已有 V18 activation-only 证据与 MLX0.32.3表达式，完整[Apple MIT notice](21-gdn-conv-source/prototype/LICENSE-MLX-MIT)随存档保留；V18通过不自动赋予V23资格。

## 先确定规约顺序，再看数值

prototype 的 `sum_y` 是必填参数。四种分支只表达原 reduction 的 y-local fold 与升序合并，不是性能搜索空间，更不能用“哪个输出刚好相等”选树。[原规约契约](21-gdn-conv-source/reference/conv-reduction-contract.md)把 Y 绑定到原生 pipeline 的最大线程数及实际 product 布局。

root 对安装 MLX0.32.3 的 `col_reduce_small_1_reduce_sumbfloat16` 做了独立[原生 PSO 查询](21-gdn-conv-source/pipeline/results/native-pipeline.json)：M=1024、execution width=32，推导 Y=4。该工具初始化原生 pipeline，却提交了0个 dispatch；它本身没有观察 product 布局或实际 reduction dispatch。

[数值 runner](21-gdn-conv-source/numeric/source/runner.py)随后用 `ensure_row_contiguous=False` 的[metadata probe](21-gdn-conv-source/numeric/source/layout_probe.metal)读取原 product 的 GPU descriptor；14组都实际得到 shape `[1,4,10240]`、element strides `[40960,10240,1]`，与查询的 selector 前提一致。内部 C++ flags 没有直接读取；由实测 strides 推导的 row/column flags 明确标为 derived，未把它们写成 native flags 观测。

这轮只执行先验确定的 Y4。对每个 channel，它等价于四个独立 BF16 `product + +0`，再从 partial0 开始按1、2、3升序 BF16 合并，之后才做 typed SiLU。

PSO归档特别保留[executed-v1源码与build recipe](21-gdn-conv-source/pipeline/executed-v1/README.md)。它是补回 pre-dladdr 源码并 CPU 重建、binary SHA 与原 root receipt 一致的版本，不是找回了最初可执行文件；这个区别在[RESTORATION](21-gdn-conv-source/pipeline/executed-v1/RESTORATION.json)中保留。可执行文件不进仓库。

## 数值门目前覆盖什么

[原数值 receipt](21-gdn-conv-source/numeric/results/child-output/receipt.json)记录6个预声明 synthetic case：零 tail/零 raw、零 tail/非零 raw、非零 tail、BF16 cancellation、product rounding、head boundaries；另外执行8步 synthetic chain。

每组严格比较 conv、act、tail、probe-tail、原递推 o 与 S_new 的 shape、dtype 和 finite raw bytes，并检查全部输入 before/after SHA。合计 **84项 strict checks、46,678,016字节，全部相等**；chain 的 tail/state 输入输出链接也经[独立 CPU 复核](21-gdn-conv-source/numeric/review/INDEPENDENT-RESULTS-REVIEW.md)检查。

只读取真实 checkpoint 的 L0 conv 权重81,920字节，加上历史保存的真实 g/beta/S。旧 V18 banks 保存的是卷积后的 u，没有真实 preconv raw/old_tail/w；V19也没有补齐。复用真实权重和递推状态，不等于这些 synthetic raw/tail 就成了真实模型激活。结果因此叫 `PASS_CONV_NUMERIC_SYNTHETIC_ONLY`。

原 GPU tensor blob没有留在这份小档案中。独立报告复核源码、严格执行路径、receipt、hash及资源记录；没有声称重新计算了全部GPU输出。

## 局部 micro 怎样站住

[micro 源码](21-gdn-conv-source/micro/source/micro.py)沿用 V21 stationary block 时序：A走原 concat/product/sum/nn.silu/tail view，B走独立融合核；每次调用都新建计算并 `mx.eval(act,new_tail)`。八个不同的 synthetic raw/tail bank 固定轮换，权重只用同一小片 L0 conv。

每个 block 固定 prime128/repeat4096。重复数在计时前就设为 V21 的8倍，因为这里只测一个轻量卷积组；没有按结果调整。顺序是 `AA,BB,AB,BA,AB,BA,AB,BA,AA,BB`：6个均衡 AB/BA 对与4个同 variant 位置控制。两 block 相邻，中间没有 hash、JSON 写盘或资源探针；这些操作都放在完整 pair 外。八个 bank 在计时前后再次做 strict bytes 和 readonly，前后输出 signatures 也必须一致。

| 局部 host-wall 指标 | 本轮结果 |
|---|---:|
| A 中位 / call | 115.489939µs |
| B 中位 / call | 101.392166µs |
| `1-B/A` | 12.2069275% |
| B 获胜对数 | 6/6 |
| AA 的 slot2/slot1 | 1.032318、0.988284 |
| BB 的 slot2/slot1 | 1.020919、1.006247 |

门槛与 tight V21 一致：每 variant 两个 control 的中位在 `[.98,1.02]`，每个 ratio 在 `[.95,1.05]`；随后才看局部中位降幅至少5%、至少5/6胜。本轮全部通过，原状态为 `EXPLORATORY-PASS`。[独立 micro 复核](21-gdn-conv-source/micro/review/REVIEW.md)检查20个 block、bank/输出、pair位置、资源及退出记录；它也是 source＋receipt replay，不是再次抓取GPU blob。

本轮固定 AC Power，和历史 V21 battery run 分属不同电源域。micro 的23次资源样本均 thermal0、pressure1，Swapouts与swapMB没有增长；数值运行另有前后两点资源记录。两轮 owned child 均 wait/reap，另存[数值](21-gdn-conv-source/numeric/results/post-tool-external-absence.json)与[micro](21-gdn-conv-source/micro/results/post-tool-external-absence.json)的外部 PID absence 证据。采样点不代表连续 GPU 时间线。

## 下一步仍是完整资格

这12.21%只属于单组卷积的 Python构图/eval/sync host wall。它包含局部执行和提交成本，不能乘48层来宣称每token省多少，也不能换算为HTTP端到端收益。当前 `adopted=false`、`model_qualified=false`、`E2E_run=false`。

下一道门是从冻结25源基线构造候选，捕获真实 preconv 输入，并检查完整模型 logits/IDs、最终 conv/recurrent/KV cache、正常EOS、连续步和 abort/filter/reuse 生命周期。只有这些通过，才做未instrument的固定C1配对与整机收益门。内部 flags 未直接观察这一限制继续保留。

[小档案目录](21-gdn-conv-source/README.md)保存原型、PSO executed-v1 src/build、数值/micro源码、原 receipts、资源及独立报告；[manifest](21-gdn-conv-source/ARCHIVE-MANIFEST.json)记录原路径、字节数和SHA。源码中的历史绝对路径按原样存档，供审计；它不是任意checkout可直接运行的重定位包。本章没有改26个engine Python文件、旧证据或用户文件。
