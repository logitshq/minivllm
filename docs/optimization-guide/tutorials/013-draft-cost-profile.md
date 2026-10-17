# E013：先测剩余draft成本

2026-10-17；运行前计划，基线`490bb25`。E012已移除每位置两次host同步；本轮先测剩余
上下文投影、词嵌入、5层backbone、最终norm、目标lm_head与selector，避免继续优化小头。

固定历史长度32/256/1024、block3/5/8，E012全程ON。原19-token前缀加固定文本补齐长度；
目标prefill只提供已存在历史的特征，保留最后4个历史token作为新上下文输入。
拆开的运算必须与实际`draft()`的hidden/logits/全部10个KV和IDs严格相等。
计时保留第一调用、两次预热、7次正反顺序组件轮询；另直接计时未装hook的原始完整draft。

同步组件时间不应相加成请求时间。`call_ms`包含Python调用与算子提交，selector/full里的
CPU取值也会等待GPU；`eval_wait_ms`不是纯GPU核时间。若lm_head超过完整draft的40%，
优先调查词表投影的矩阵路线；否则调查最大的组件。这里只定位成本，不凭profile宣称优化。

同时准备官方oMLX外部MTP对照：固定M4已定位的`b9b764dada894b8c416f89cbc8cf99bc7e20b831`，
target revision保持不变。下载不等于资格；须独立证明加载、attach、实际route和verify计数，
再做未插观察器的计时。M4的结果不替代M5资格。
