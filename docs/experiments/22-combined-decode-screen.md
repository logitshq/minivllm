# 22：融合和图复用，实际 HTTP 快了多少？

这一轮把两条从实际实现中读出的路线接到实验引擎：oMLX 的 GDN 卷积预处理融合边界，以及 Lithos 固定执行图所启发的 decode 图复用。组合版通过了数值和生命周期检查，C1 HTTP 三对测试也都变快；但中位交互速度只增加 **1.54%**、总吞吐只增加 **0.76%**，仍没有证明超过 oMLX/Lithos 的 Pareto 前沿。两项功能保留独立开关，默认关闭。

## 原来的调用换成了什么

GDN 原路径把旧 tail 和新的投影拼起来，做四项卷积，再做 SiLU：

```python
xp = mx.concatenate([st.conv.astype(qkv.dtype), qkv[:, None]], axis=1)
convolved = (xp * w.T[None]).sum(axis=1)
activated = nn.silu(convolved)
st.conv = xp[:, 1:]
```

实验路径把这部分换成一次自定义 Metal kernel：

```python
activated, new_tail = conv_silu_raw_tail(qkv, st.conv, w)
st.conv = new_tail
```

**tail 保存原始投影，不保存 SiLU 后的值。** 这一版保留原 BF16 乘法、归约和激活舍入规则。第21篇的 synthetic micro 并不等于真实模型加速；本轮才补齐真实模型与服务检查。

另一处替换在 `ModelRunner` 的普通 decode：

```python
# 原来：每步由 Python 重新构造模型计算图
logits = self.model(tokens, ctx)

# 实验：普通 B1 greedy decode 复用编译图，缓存张量是显式输入和输出
logits = self._normal_decode_forward(tokens, ctx, [r.sampling for r in decode])
```

helper 根据长度、容量和输入形状等信息缓存最多64张图。每次把所有 KV/GDN 状态传入，返回 logits 和新状态；不捕获某个请求的旧缓存。不同长度仍需首次编译，因此本轮三个相同长度 warmup 后，实测每进程31次 miss、62次 hit、31个 entry。**这是 MLX 计算图复用；还不是 Lithos 的 Metal ICB，也没有消除 Metal 编码和缓冲分配。**

完整27文件实验快照见 [源码](22-combined-decode-source/candidate/tutorial_code/miniengine/engine/model_runner.py) 和 [functional helper](22-combined-decode-source/candidate/tutorial_code/miniengine/kernels/functional_decode.py)。仓库主运行时代码未被这份实验快照替换。

## 先通过了哪些检查

- 真实模型32个 greedy token，逐步 logits、128份完整缓存和输入只读检查通过；组合图复用及不同 token 的同长度输入也通过。
- 引擎生命周期共185个生成 token、179步：开关复位、提前停止、重新使用引擎、并发请求过滤和新请求补入均通过。
- 第一次生命周期脚本错误地要求 B2 输出等于 B1 固定续写，整个尝试判 FAIL 并保留。旧引擎本身已有 B1/B2 舍入差异；修正后的脚本先运行同一并发场景的 OFF 对照，再严格检查 ON 与 OFF 的每个请求相同。通过不依赖放宽这一比较。

## 六个新进程的 HTTP 结果

A 是最终基线所用的25文件引擎，gates 编译开启；B 是本轮27文件组合，额外开启 conv 和 functional decode。顺序 AB、BA、AB，每进程3次 warmup 和3次正式请求；相同19个 prompt IDs、greedy、正常 EOS、32个实际输出 token、无 draft、prefix cache 关闭、allocator 上限2 GiB、Metal buffer 控制50/50/20/0。

X 沿用前次比较的客户端交互速度估计，Y 为实际输出 token 数除以 POST 到 SSE DONE 的窗口，包含排队、prefill 和 HTTP。两个轴都不是孤立 GPU kernel 吞吐。

| 配对 | 顺序 | X：B/A变化 | Y：B/A变化 |
| --- | --- | ---: | ---: |
| 1 | AB | +0.6656% | +0.4384% |
| 2 | BA | +1.6323% | +0.7563% |
| 3 | AB | +1.5445% | +1.0382% |
| 配对变化中位数 | | **+1.5445%** | **+0.7563%** |

36个请求共1152个输出 IDs 全部等于 raw IDs、文本重编码 IDs 和固定续写。独立 CPU 复算确认原计时边界、102份各进程原始资源样本和清理记录：AC、thermal nominal、pressure1，无 swapout 或 swap 使用量增长；外部检查确认8个进程不存在、6个端口拒绝连接。

这轮没有启动官方 oMLX/Lithos，不能把旧官方点与新引擎点混成一次获胜对比。相对旧 C1 差距，本轮改善仍然不足，因此未采用、未更新官方 Pareto 图；继续定位权重流和 GPU 执行耗时。

[独立复核](22-combined-decode-source/INDEPENDENT-REVIEW.md)、[完整文本证据索引](22-combined-decode-source/evidence-index.json) 和 [压缩证据](22-combined-decode-source/evidence.tar.gz) 保留原始 PASS、FAIL、源码、计时与资源记录。压缩包不含权重、原始张量文件或原生可执行文件；其中路径对应实际测试目录，复现需适配，不能将旧收据当作新测试。
