# 26：让 GPU 把自己算出的 token 和 state 接回下一步

上一轮只证明同一个 seed 可以重放两次。这一轮固定 attention 容量256，让 GPU 把新 state、argmax token、下一位置及长度写回输入，连续生成。实际通过原 prompt、重置同 prompt、改变 prompt 三次请求；原生路径输出96个 token、93个 decode 步，12276项输出检查严格相同。首次速度探索约24.95 token/s，尚不足以采用，也没有新的 HTTP 或 Pareto 点。

## 从单步重置，换成连续反馈

原单步验证每次调用 `v33_restore_replay()`，把整个 arena 恢复成同一份旧输入。它不能用来生成后续 token。

新路径每个请求只初始化一次：

```python
native.call("v34_reset")
for original, current in zip(seed_holders, request_holders):
    native.call("v34_write_input", original.address, original.nbytes, current.address)

for step in range(31):
    native.call("v34_replay_step")
    # 只观察4字节 prediction；不把 CPU 算出的 token 喂回 GPU。
```

上面省略了实际脚本中的 ctypes 指针包装。完整实现见 [资格脚本](26-gpu-feedback-source/prototypes/v34-native-fullqual-build1/fullqual.py)。本轮仍由 C++ 每步重新编码 Metal 命令，还不是 ICB。

捕获图除了 logits 和128个缓存，再输出三个标量：

```python
prediction = mx.argmax(logits, axis=-1).astype(mx.int32)
next_positions = positions + 1
next_lens = lens + 1
```

因此输出共132个，输入共131个。logits 供观察；128个新缓存和三个标量通过131条 GPU feedback link 成为下一步输入。下一 token 来自本轮 GPU argmax，不来自保存的标准续写。

## 为什么先经过 staging

第一版先把全部132个输出拷贝到独立 staging buffer，再把131个对应范围拷回输入。这样即使某些逻辑范围重叠，也不会先写坏另一个尚未读取的输出。compute、两段 blit 之间使用自有 fence，并等待该步完成；不能只凭“编码顺序相同”忽略 untracked buffer 的同步。

这条保守路径每步搬运约342 MB数据，按每次 copy 的数据长度计。它便于建立正确性证据，后续要减少这部分开销。

所有实际参数 buffer 都经地址范围和只读反射验证后复用，约15.134 GB的 allocation 不再复制；arena/first-bind 快照各约366 MB，低于上一轮各3.751 GB。数字是实际 buffer allocation 的大小，不是 safetensors 文件大小。

## 三个请求检查了什么

- 原 prompt19个输入 token，生成32个输出；每步与原动态长度模型比较 logits、128个缓存及三个额外标量，共132项。
- 请求结束后重置，再运行相同 prompt；32个输出与原模型及固定续写相同。
- 在用户内容中插入一个已有合法 token，使 prompt 长度变为20；实际有129个 seed 输入发生变化，原生路径仍与该新 prompt 的逐步参考相同，生成32个输出。

三次共93步、12276项严格字节比较，原 seed、原捕获输出和各请求 seed 保持只读。实际计数确认每个请求只上传一次131个输入范围；decode 内没有 reset、恢复全 state 或 CPU token 上传。CPU 总共观察372字节 prediction。独立审计复核了源码、保存的比较记录及计数，没有独立运行 GPU 重算。

## 首次速度探索

去掉计时区间里的参考模型调用及全状态 CPU 比较后，在一个新进程中执行3次 warm、3次测量。每次重新 prefill、reset 和上传 seed，随后计时31步；区间后再核对全部132个最终输出的哈希。实际测量中位数：

| 边界 | 结果 |
| --- | ---: |
| 连续31步 decode 的平均每步时间 | 40.087 ms |
| 31 / decode窗口 | 24.946 token/s |
| 32 / prefill到完成窗口 | 22.534 token/s |

这是候选自身的探索，没有公平 A/B 或 HTTP 对照，不能与旧官方点拼成新曲线。本篇的数值通过也不代表服务生命周期已经完成。运行记录保留原资源检查、整机 paging 变化和进程释放证明；不把这次结果当作获胜。

[源码与范围说明](26-gpu-feedback-source/README.md)、[原始文本索引](26-gpu-feedback-source/evidence-index.json) 和 [证据压缩包](26-gpu-feedback-source/evidence.tar.gz) 保留本轮结果。接下来分别减少 state 搬运与每步命令编码，再重新检验。
