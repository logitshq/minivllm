# D3：算完了，不等于可以交付

[六篇目录](README.md) · [上一节 D2](d2-device-feedback.md) · [下一节 E1](e1-spec-transactions.md)

前置：[D2 设备反馈](d2-device-feedback.md)。独立脚本能连续生成，服务却还要排队、停止和复用资源。接入runner的关键是：**谁持有当前历史，谁有权交付结果，何时才能重用内存？**

## 先确定整条请求走哪条路

baseline每次`run`由普通模型更新缓存。新适配器仍让普通runner做prefill；首token产生后，合格的C1 greedy请求上传一次seed，再由原生路径持有实时状态。开关默认关闭，原`run/drop`保存作回退。已进入原生后不能突然回普通缓存：那份MLX缓存只停在seed位置。连续性破坏应报错，不能假装安全回退。

[初始适配源码](../../experiments/29-native-runner-http-source/prototypes/v41-native-runner-build1/native_runner.py)也要求普通`_ahead`为空，避免两条路径同时推进同一历史。

## 从同步等待到两张FIFO票

下面是[异步适配器](../../experiments/30-native-ahead-source/prototypes/v56-native-async-runner-build1/native_runner.py)的流程伪代码，省略原生ABI与断言：

```python
# 原来：submit(); wait(); read_prediction()
while len(pending) < 2 and submitted < budget:
    pending.append(submit_step())
    submitted += 1
wait_step(pending.popleft())
prediction = copy_current_prediction()  # 先消费，再允许refill
return prediction
```

两票让CPU提前提交，GPU仍按token依赖顺序计算，不能跨token并行猜算。读的是最早交付票的slot和状态bank，不是最近提交的bank；command buffer及捕获的参数owner必须活到最后使用结束。若prefill已交付1个token、输出上限32，原生预算只有31，不能为了补满队列再多算一步。

## 停止时先排空，再释放

EOS可能在后继已提交后才被CPU看见。必须等待旧工作结束，把未交付票标成discard，再清队列、清active并调用原`drop`；下个请求此后才能reset。已完成但丢弃的计算不算可见输出，也不计吞吐分子。

普通ahead同样要区分当前已提交缓存和未来候选；request、位置、采样或输入改变就不能复用。原生FIFO则不能靠普通ahead的旧holder完成中途迁移。

## 用一个早停例子理解证据

实际测试在第7个输出停止：prefill交付1个，原生交付6个，另有1张未来票完成但丢弃；随后fresh请求可重新开始。它证明交付、drain和复用，不证明早停六步都逐项核过全部缓存。完整生成和回退另有对应检查。早期HTTP对照因电源域变化停止，生命周期通过不能写成性能获胜。

**理解题：** 为什么不能先reset下一请求，再慢慢等待旧票？因为旧GPU工作仍可能写同一输入或状态区域，污染新请求。

详细代码、失败修正和证据仍在[旧29章](../../experiments/29-native-runner-http.md)与[旧30章](../../experiments/30-native-ahead.md)；[FIFO core](../../experiments/30-native-ahead-source/prototypes/v55-native-async-build1/pipeline_core.h)保留票、slot与bank规则。
