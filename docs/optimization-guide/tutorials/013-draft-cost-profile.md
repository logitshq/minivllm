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

## 尝试1：启动失败，未加载模型

脚本最初命名`profile.py`，遮蔽stdlib `profile`。transformers导入torch/cProfile时回到
本脚本，引发循环导入；没有形成plan或GPU计时。guard保存完整trace、资源和进程回收，
服务恢复成功。改名`costs.py`后先运行`--help`验证导入，再以新v2目录执行同一计划。

## 尝试2：完成成本分解

9组hidden/logits/10个KV共108个数组严格通过；第一调用及全部441个正式组件计时保留。
5层backbone在所有case都是最大组件；lm_head仅为完整draft的22.9%–28.9%，未到40%分界。
1024历史时，block3/5/8的backbone分别2.825/2.924/3.840ms，完整draft为
3.883/4.056/5.938ms。同步边界会改变重叠，不能把这些组件相加，也不能把比例当Amdahl精确占比。

[69文件归档](../../experiments/m5-ultra-256gb/E013-draft-profile-v2/)同时保留失败v1、成功v2、
源码、所有样本和资源。4个正式运行资源样本pressure1/thermal0，swap/Swapouts不变，
子进程已回收，外部服务同身份恢复。下一步拆两种backbone成本：E014有界KV避免长历史
反复复制；E016只测显式状态图复用上限。两项互相独立，主线程统一安排GPU测试。
