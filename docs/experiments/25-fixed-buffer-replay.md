# 25：从每步构图，走到固定 GPU buffer 的原生重放

前面的 `mx.compile` 复用计算图，底层仍需要编码和提交 Metal 命令。本轮参考 Lithos 的固定权重、state 和执行序列，先实现一个更小的原型：捕获真实 kernel 与绑定，在自有 arena 中重放同一个 decode 步。对应源码是固定 commit 的 [Engine buffer 复用与 ICB 构造](https://github.com/lithos-ai/lithos-metal/blob/5485ad6bac0a5649d66a347f3e01a655ea29d937/monolith/runtime/engine.py#L58)，其中58–85行准备 buffer 和 dispatch，86行构造 ICB。

**本篇归档两项数值资格，不包含性能结果：**27B 固定 L20 重置后重放两次，logits 和128个缓存全部逐字节相同；普通 MLX 路径改用固定容量256后，连续32个 greedy token 和每步缓存也全部相同。这两项为原生连续生成准备固定形状；接续的 GPU 状态反馈实验单独记录。

## 这一轮替换哪一层

原路径每步调用 Python 模型，构建并求值新的算子图：

```python
logits = model(tokens, ctx)
mx.eval(logits, *new_cache_arrays)
```

实验捕获一次真实执行所用的 PSO、完整 buffer/bytes 绑定、offset、threadgroup memory、dispatch 尺寸和同步依赖。重放端用 C++ 绑定自有 arena 的对应位置，提交一个 command buffer，再等待完成。**这是原生重新编码的单 CB 重放，还没有 Metal ICB。**

下面是实际重放循环里的关键调用；`mapped` 为只读权重复用原 buffer，为其他数据选择自有 arena，`mappedOffset` 保留原 offset：

```objective-c
[e setComputePipelineState:cmd.p];
[e setBuffer:mapped(x.b) offset:mappedOffset(x.b,x.offset) atIndex:arg.first];
[e dispatchThreads:cmd.grid threadsPerThreadgroup:cmd.group];
// 完整循环还恢复 bytes、动态 threadgroup memory 和 barriers。
[e endEncoding];
[b commit];
[b waitUntilCompleted];
```

[完整原生实现](25-fixed-buffer-replay-source/prototypes/v33-native-replay-build4/replay.mm) 的 `v33_restore_replay` 还处理另一种 dispatch 类型；这几行不能单独构成完整捕获器。

```text
原来：Python 模型 → 算子图 → Metal 编码 → 多次提交
实验：已捕获的命令序列 → 固定 buffer 绑定 → 一次提交
```

这里的“固定”不能只固定指针名字。原 MTLBuffer 的身份、同一 buffer 内的不同 offset，以及输出只是 allocation 中间一个 view 的情况，都必须保留。卷积的 `xp[:, 1:]` 就是这样的真实 tail view；复制一份连续 tensor 会掩盖映射错误，因此资格脚本直接验证原地址。

## 先修实际遇到的同步行为

小探针的前三轮依次停在未支持的 fence、event signal 和空 command buffer 判断，均保存原 STOP。第4版才通过32 MiB权重零复制、内部输出 view、两次重置重放及原输入只读检查。

原生实现记录实际 fence/event 的 producer、CB、queue 和先后顺序。对于已证明的同 queue 依赖，在单 encoder 中改用 buffer barrier；不重新 signal 原来的单调 event 值。全新空 CB 也需要整个生命周期没有 encoder/event/未知操作的证据，不能把不认识的工作直接丢掉。

这些检查让本次捕获可重放，不代表通用 Metal 图捕获器已经完成。其他 queue、未知绑定或不支持的命令仍停止；本次只在测试子进程注入库。

## 27B 的实际结果

V33 native4/worker3 加载一次模型，先正常 warm 出32个固定 greedy IDs，再重新 prefill 并捕获 L20 的一步。捕获含2005次 dispatch、84个非空 CB，另有1个被证明为空的提交边界；两次原生重放各检查129个输出：logits、128个完整缓存。

两次所有输出均有限且字节相同；131个原 seed 输入，以及原捕获输出也保持只读。这仍是**同一输入重置后的单步检查**，不是连续生成32步，也没有作为 Pareto 点。

当前原型 arena 约3.751 GB，另有同等规模的 first-bind CPU 快照；约11.748 GB的大权重通过真实参数地址范围和只读反射复用。小权重暂被复制，下一版会同样依据实际参数与只读证据复用它们，减少额外占用。不能把这次正确性原型的内存和 reset 开销忽略后宣称加速。

## 为什么接下来可以只留一张固定形状的图

当前 KV 容量本来就是256。原路径把 attention 的 `max_len` 设为20到50；实验只把它固定为256，仍用每个请求的真实 `lens` 生成布尔掩码。固定版本的短序列 attention kernel 会跳过 mask 为假的位置计算。

V34 direct 实际检查了同一模型的两条独立状态链：A 使用原动态长度，B 固定256。31步中3999项 logits/缓存检查严格通过，32个 greedy IDs 相同，原输入保持只读。运行时保持48个 GDN gates 编译开启，其他实验功能关闭。两次整模测试均为 AC、thermal nominal、pressure1，进程已回收并经外部不存在检查。V33 的整机资源样本还记录了压缩和 pageouts 增加30；本轮没有性能结论。

原生连续反馈还要把新 state、GPU argmax token 和下一位置写回固定输入区域，并验证重置及换 prompt。这些属于接续实验；本篇不把它们与上面的两项资格合并。实际服务配对计时和同机官方 baseline 复测也另行记录。

[实际源码与范围说明](25-fixed-buffer-replay-source/README.md)、[文本收据索引](25-fixed-buffer-replay-source/evidence-index.json) 和 [原始证据压缩包](25-fixed-buffer-replay-source/evidence.tar.gz) 保留本轮通过与停止结果。主运行时尚未替换。
