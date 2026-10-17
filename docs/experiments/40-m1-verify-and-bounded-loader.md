# 40：先修正 verify 数值，再测吞吐—延迟

V117 已通过完整目标模型的本轮严格数值资格：普通逐 token A 与投机 B 各得到96个旧 cohort token ID，49次实际 verify 的提交缓存和最后已计算 logits 均严格一致。这是进入性能实验的先决约束，还不是优化收益。目标仍是以正确性为约束，改善 throughput–latency Pareto。

之前的 V114 在加载 DFlash 时触发系统 memory pressure2，由 owner终止并释放。它已经完成 A96，尚未执行 B verify，不能记为算法失败。源码把全部 BF16 原权重留在字典中，又对所有权重建立惰性量化图，直到最后统一求值，延长了源数组存活期。

V116 的最小修改只调整加载生命周期。默认仍关闭；开启后按原加载顺序取出完整 tensor，使用相同 affine4/group64量化参数，对三份输出一起求值，再释放这一份输入；norm和codebook原数组不改。没有切碎 tensor、改量化数学或替换 target参数。独立 draft-only实际对照中175个输出参数逐字节一致，原A175只读，350个A/B输出对象独立。

下面省略了原有名字映射与未量化分支，只展示量化对象的生命周期。

```python
# 旧：raw owners 与所有量化图一起存活到最终 eval
for name, value in weights.items():
    packed, scales, biases = mx.quantize(value, ...)
    out[name], out[base+".scales"], out[base+".biases"] = packed, scales, biases
model.load_weights(...)
mx.eval(model.parameters())

# 新：仅展示需要量化的分支，其他 norm/codebook 保留原对象
for name in tuple(weights):
    value = weights.pop(name)
    packed, scales, biases = mx.quantize(value, ...)
    mx.eval(packed, scales, biases)
    out[name], out[base+".scales"], out[base+".biases"] = packed, scales, biases
    del value
```

目标 verify 的另一处修正是保留逐行 M=1 的投影数学。普通 M=T batched QMM 与逐行 M=1 在旧同输入诊断中并非全部相同；V113 的 broadcast 形式对497个实际同输入投影都匹配原 M=1。V117 把该形式严格限制在实际 target verify 调用中，包括该调用的lm_head；普通prefill、drafter及其借用的head在作用域之外，finally恢复所有实例标记。W/scales只加singleton view，不拼接或重新绑定权重。

```python
# batch=T，但每个 batch 的矩阵行数 M=1
mx.quantized_matmul(
    x[:, None, :], W[None, ...], scales[None, ...], None,
    transpose=True, group_size=16, bits=4, mode="nvfp4",
).reshape(T, N)
```

局部同输入相等不等于整模型传播正确，因此 V117 保留独立普通 M1参考。A、B各自从同一prompt预填；A用自己的argmax推进，B只用自己的draft/verify结果。比较位置是实际计算并提交的 `start+committed_inputs`，不是最后发出但尚未计算的anchor。

| 实际资格 | 结果 |
|---|---:|
| A普通路径 / B投机路径旧cohort IDs | 各96，严格一致 |
| 实际事务 | 49（verify T1…5） |
| 独立A/B初始缓存 | 49×128 =6272组件严格一致 |
| 独立A/B提交缓存 | 49×128 =6272组件严格一致 |
| 提交位置最后logits | 49次严格一致 |
| literal同次调用prefix / history / input只读 | 全部通过 |
| 目标参数名/对象/原始字节 | 1349个保持只读 |
| 实际受限verify Linear调用 | 24,353 =49×497 |
| 预算1/2/3/5、reset、changed-prefill、结束清理 | 全部通过 |

Attention只比较真实已使用prefix，并分别要求两侧未来tail为零；GDN完整conv/S字节比较。原始每步producer owner与literal history仍保留，不重算较短前缀来充当reference。三个请求末尾canonical旧44缓存128组件也分别128/128 hash相等，这是额外诊断，不能扩写成旧132输出的完整资格。

加载阶段的MLX计数也支持生命周期修改有效：旧108在加载DFlash结束时记录的active peak为19,993,643,532B，当前117对应阶段为16,399,649,292B，相差3,593,994,240B，正好等于待量化BF16源数组总量。这是allocator口径的历史峰值对照，不是物理RSS下降或速度结果；本次完整资格后峰值为18,348,106,041B，包含后续literal history等活对象。

本次75个资源样本均AC、thermal0、pressure1，swap使用量和Swapouts无增长。owner39460/child39470退出0且已wait/reap，外部严格确认两进程不存在。原114压力STOP与旧数值失败保留，没有重新标为通过。

随后 V120 已尝试实际 C1 HTTP screen。两次都在 speculative 服务 ready 后触发 pressure2，尚未发出 speculative 请求，按资源门停止并回收。第一次仅 ordinary arm 完成3warm+3measure，完整请求延迟中位数1.250373s、总吞吐25.592356token/s；它不是配对结果，不能声称投机路径提升，也不与旧轮次混池。第二次未完成任何arm。

随后 V122 把 allocator cache限制为64MiB，仍在首个warm之前以pressure2停止，HTTP请求为0。启动记录只有一次target和一次draft加载：active/peak均16,399,485,440B、cache44,564,968B；没有重复模型加载证据。该新资源配置的失败和外部释放证明保留，不据此继续重复DFlash GPU重试或推断物理内存根因。

第二次终端打印 `all_owned_reaped=false`，原因是谓词比较了计划的两个模式与实际已启动/回收的一个server；actual server已wait/reap，PID和端口严格不存在，外部release也确认owner/server均释放。这里保留原始停止事实，不把计数谓词当成未释放或改写成性能通过。

当前仍需先恢复满足资源门的条件，再取得配对吞吐—完整请求延迟证据。历史M4七配置的联合前沿仍是oMLX C1（1259.77ms、25.40token/s）与native15 C4（3291.09ms、38.89token/s），见[历史M4 Pareto](../progress/2026-10-17-m4-pareto/m4-throughput-latency-pareto.png)。正确性是先决约束；本轮没有新的speculative性能点或Pareto胜出结论。

实际证据为 `fullmodel-v117broadcast-bounded-attempt1`；本页只从保存receipt提取短摘要，不是独立GPU复算。紧凑计数、资源口径与原证据SHA见[V117 实际摘要](40-source/v117-actual-summary.json)。

本章发布包只含V117原始receipt无损gzip、指定资格源码/源码审查、V120源码与白名单资源/释放摘要；不复制原始process argv或privatebulk。V120两次停止的紧凑证据见[summary](40-source/v120-attempt-summary.json)。
