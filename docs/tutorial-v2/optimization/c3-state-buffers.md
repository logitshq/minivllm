# C3：固定地址之前，先分清状态的所有权

[六篇目录](README.md) · [上一节 C2](c2-weight-reuse.md) · [下一节 D1](d1-command-reuse.md)

## 问题：怎样让下一步留在 GPU？

先读[第六章 cache](../06-cache.md)、[第八章 batch](../08-scheduler.md)。`mx.compile` 复用计算图；本篇进一步固定真实 Metal buffer 绑定，并将命令编码留在 ICB。指针稳定只是起点，同一 allocation 的 offset、view、读写角色和同步依赖也必须正确。

## 从逐步构图，到固定输入输出

下面是教学伪代码，不是通用捕获 API：

```python
# baseline：每步构图，返回新的状态数组
logits, new_states = model(token, positions, lens, states)
eval(logits, new_states)

# 原生路径：每请求播种一次，其后由设备反馈
reset_and_upload_request_seed()
execute_icb(bank)       # 同一捕获图，S 绑定随奇偶步切换
feedback_states_token_position_and_lens()  # token 来自 GPU argmax
observe_prediction()   # 只读取本步4B token
```

原型先实现原生重编码，再建立两套真正的 Metal ICB；二者不是同一件事。权重只读、常量 bank、可变 arena 和 ICB 都需声明资源使用；fence/event 的先后关系也不能因为合成一次提交就删掉。

## 不变量：状态不原地覆盖，容量不等于长度

GDN 的 S 用两个完整 bank：本步读 A 写 B，下步读 B 写 A。仅当真实绑定证明该 allocation 独占、输入只读、输出只写且无其他别名时才允许轮换。结果必须读本步实际写入的 bank；新请求 reset 回起点。conv tail 的内部 view 也必须按原 buffer 加 offset 映射。

Attention 另有两个数：capacity 是物理槽位，lens 是有效历史。固定 capacity 不表示所有槽位都可见，仍须逐请求 mask。19-token prompt 生成32个输出只需31次 decode，原请求写到位置49；20-token changed prompt 写到50，容量64在该负载下足够。

缩小物理容量仍需新的资格。原测试对64内无效尾部查零，再在 CPU 补零到旧256形状核对完整 gold；这只是比较表示，没有偷偷让 GPU 仍分配256。更长请求不能借用这个容量结论。

## 记录能证明什么

固定单步两次重置通过，只证明同 seed 重放；连续生成还要检查 base、reset、changed prompt，并让 token 来自自己的 GPU argmax，不能喂保存的未来答案。S 轮换减少 copy，ICB明显减少记录中的主机编码区间，但当时 B-only 时间没有公平配对或 HTTP 胜出证明。

旧36还记录了另换 RMSNorm 舍入族：它属于数学 profile 改变，旧 ID 门曾失败，后来另立参考。不能把这个变化偷偷合并进“buffer 优化”并宣称旧基线仍通过。

**检查理解：** 物理空间还有64格，为什么长度20时不能读其余44格？为何看到输出地址相同，仍不足以证明下一步输入映射正确？

## 回到完整记录

[旧25](../../experiments/25-fixed-buffer-replay.md)、[旧27](../../experiments/27-native-icb-pingpong.md)、[旧36](../../experiments/36-cap64-and-rms-profile.md)保留同步失败、单步/连续资格、独立 buffer 克隆负结果及 profile 变化。[原生重放源码](../../experiments/25-fixed-buffer-replay-source/prototypes/v33-native-replay-build4/replay.mm)、[S轮换源码](../../experiments/27-native-icb-pingpong-source/prototypes/v36-native-pingpong-build1/replay.mm)、[ICB源码](../../experiments/27-native-icb-pingpong-source/prototypes/v38-native-icb-build1/replay.mm)、[CAP64来源与证据](../../experiments/36-cap64-and-rms-profile-source/README.md)可逐项检查；各原章的 source map/manifest 固定完整来源。
