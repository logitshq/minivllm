# D2：让下一步输入留在设备上

[六篇目录](README.md) · [上一节 D1](d1-command-reuse.md) · [下一节 D3](d3-service-ownership.md)

前置：[C3 状态与双缓冲](c3-state-buffers.md)、[D1 命令复用](d1-command-reuse.md)。命令能重放，还不等于能连续生成：若每步恢复同一份seed，模型只会反复计算同一个位置。真正的decode必须消费上一步的新token和新状态。

## 从重复seed到一次初始化

下面是流程伪代码，省略指针、资格判断与错误检查：

```text
之前：每步 restore(seed) → replay
之后：每请求 upload(seed) 一次
      每步 graph → feedback(new_state, argmax, pos+1, len+1)
      CPU仅观察本步prediction，下一步输入来自GPU
```

图输出logits、全部缓存及三个反馈标量。logits用于观察；新缓存、预测token、位置和长度成为下一步输入。保存的标准续写只能用于比较，不能作为下一步输入，否则测到的是答案回放。

## 省掉哪一段搬运

初版为安全先把输出复制到stage，再从stage写回输入。若实际注册的所有输入/输出范围在两套bank下均不交叉，可以改为直接反馈，见[真实实现](../../experiments/31-native-direct-profiling-source/prototypes/v57-native-direct-feedback-build1/replay.mm)：

```text
之前：compute → 完整输出stage快照 → stage写回输入
之后：compute → 输出直接写回输入
```

这省去中间复制，不减少投影或attention计算；旧stage的allocation也未因此释放。只看逻辑shape不足以证明安全，必须核对真实buffer、offset、范围及别名。compute/blit的fence仍保留，不能用“同一队列”代替依赖证明。

## 异步之后，什么还能读

后继可能覆盖共享输出。诊断模式给每张票保存完整非S快照，检查必须赶在下一次复用前；生产模式每槽只保留4字节prediction。生产路径只有队列完全交付、没有discard的请求末尾，才允许读完整终态。少复制状态，不等于每步都取得了完整状态证据。

后来的CAP64路径还要求在构造引擎、执行任何token之前统一设置模型和cache容量；只放宽位置guard不能改变已捕获图的长度。[适配差异](../../experiments/37-simd4-async-and-spec-handoff-source/prototypes/v102-native-cap64-direct-async-runner-build1/adapter.diff)保留了这一前置。

## 实际边界

连续反馈通过了原prompt、reset和changed prompt的整模对照；直接反馈又分别核对诊断与生产。生产资格检查请求末尾，不能冒充逐步全状态。旧单臂计时有的从seed上传后开始，有的包含上传，不能直接相减归因。逐encoder采样也会扰动执行，不能把诊断时间相加当生产份额。之后的HTTP轮次因资源守卫停止，未形成完整排名。

**理解题：** CPU每步读回4字节预测，为什么仍可称设备反馈？关键看下一步输入由谁写入，而不是完全禁止CPU观察。

详细实现与证据：[旧26章](../../experiments/26-gpu-feedback.md)、[旧31章](../../experiments/31-native-direct-profiling.md)、[旧37章](../../experiments/37-simd4-async-and-spec-handoff.md)。下一篇：[服务所有权](d3-service-ownership.md)。
