# 15：把 GDN 的小门计算一起编译

上一章把兼容的多个 prompt 一起 prefill。这一章回到单请求 decode：每生成一个 token，48 个 GDN 层都要计算两组很小的门。我们保持数学、量化投影和状态递推不变，只给这部分纯计算加一个显式开关。

**本章变化：`gdn.py` 一文件增加局部编译路径，默认关闭；本章 example 在引擎加载完成后启用。** 实测完整请求吞吐提高约0.94%；这次没有同时重跑 oMLX/Lithos，不能据此宣布超过它们。

## 1. 先找到时间花在哪里

第十四章的单请求仍慢于受测 oMLX 配置。先分开观察一次32-token请求的 API 调用，而不是先猜 scheduler 慢。

正常流程先跑六个未加观察器的请求，再在空闲时为一个请求加入时间戳。诊断请求实际包含1次 prefill、31次 decode；其中29步是已有 ahead 结果、并提交下一步的稳定阶段。

| 稳定阶段，29步 | 每步 wall 中位数 |
|---|---:|
| 构造 successor 模型调用 | 2.455ms |
| `mx.async_eval` 提交 successor | 33.645ms |
| 显式求值当前 token/cache | 0.028ms |
| token 的 `tolist()` | 0.0019ms |
| runner 外的 engine 工作 | 0.011ms |

`async` 不保证 host 调用立即返回。上表记录 CPU/API wall，可能包含构图、分派和等待，不能把33.645ms叫成某个 GPU kernel 的耗时。外层与内部计时也不能相加。

这些数据让我们继续检查计算图的组织；**它们没有证明 GDN 门就是主要瓶颈。** 本章把一个边界很小、可独立核对的候选做完，再用完整请求判断价值。同步、cache 求值与输出计时边界都保留。

## 2. 从 MLX-LM 学一个可独立核对的边界

实际安装的 oMLX `0.7.1.dev1` 环境中，`mlx_lm/models/gated_delta.py:18–20` 有这段局部编译代码：

```python
@partial(mx.compile, shapeless=True)
def compute_g(A_log, a, dt_bias):
    return mx.exp(-mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias))
```

这是 MLX-LM 的纯函数边界参考，不能写成实际 oMLX 服务每个 token 必经这里。实际 oMLX 的 Qwen3.5 服务走 `mlx_vlm/models/qwen3_5/language.py`，还能安装 `omlx/patches/qwen35_gdn_prework.py` 的更大融合；源码路由不等于每个历史测点都已证明命中。其FP32短卷积等算术会改变我们现有的BF16中间舍入，本章只取纯门边界，不移植整段融合。安装文件身份与位置保存在 compact 证据中。

注意接口差别：上游这里返回的是衰减系数 `decay`；我们的 `_gates` 返回的是 `g_log`，即衰减系数的对数。我们的 GDN Metal kernel 内部还会做一次 `exp(g_log)`。

```text
上游 compute_g  → decay
我们 _gates    → g_log → 原 Metal kernel 内 exp(g_log) → decay
```

因此我们借鉴“把小块纯计算一起编译”的边界，保留自己的表达式。直接把上游的返回值喂进我们的 kernel，会多取一次指数。

## 3. 之前：每层直接建立门的计算图

回顾第四章：`b` 决定新信息写入的强弱；`a` 与学到的 `A_log/dt_bias` 决定旧记忆怎样衰减。decode 时，`b/a` 各是 `[1,48]`，每个 value head 有一个门。

[gdn.py](../../tutorial_code/miniengine/model/gdn.py) 原有实现保留为参照：

```python
def _gates(self, b, a):
    beta = mx.sigmoid(b.astype(mx.float32))
    g = -mx.exp(self.A_log) * nn.softplus(a.astype(mx.float32) + self.dt_bias)
    return beta, g                                # ← 关键①：返回log-decay
```

原 decode 在量化投影后调用它：

```python
beta, g = self._gates(self.in_proj_b(x), self.in_proj_a(x))
o, S = gated_delta_step_metal(q, k, v, g, beta, st.S)
st.conv, st.S = xp[:, 1:], S                      # ← 原状态更新继续保留
```

48层、31次 decode 会重复建立这块小计算图。这里只改变这些门的计算组织；QMM 的形状、NVFP4 权重与 scales、卷积、GDN Metal 递推、norm 都沿用上一章。

## 4. 本章之后：四个输入显式传给纯函数

先从方法中抽出同一表达式。下面是本章新增的完整函数：

```python
def _gates_explicit(b, a, A_log, dt_bias):
    beta = mx.sigmoid(b.astype(mx.float32))
    g = -mx.exp(A_log) * nn.softplus(a.astype(mx.float32) + dt_bias)
    return beta, g
```

使用它时才创建 shape-specialized 的编译包装。下面省略资格判断，只展示包装和调用；完整 guard 见源码：

```python
fn = object.__getattribute__(self, "_compiled_decode_gates")
if fn is None:
    fn = mx.compile(_gates_explicit, shapeless=False)  # ← 关键②：只编译这块
    object.__setattr__(self, "_compiled_decode_gates", fn)
return fn(b, a, self.A_log, self.dt_bias)             # ← 关键③：权重也是输入
```

为什么不写一个偷偷引用 `self.A_log` 的闭包？因为之后替换权重时，编译函数不能继续使用旧值。四个 array 全部作为输入，真实测试还专门替换了两组小权重，核对第二次调用确实得到新结果。

开关与包装是私有 Python 元数据，不进入模型参数树。`set_gates_compile(False)` 会关闭路径并清空包装引用；构造函数默认就调用它。加载参数没有增加新的 checkpoint key。

新 decode 的替换点是：

```python
if _compile_gates and object.__getattribute__(self, "_gates_compile_enabled"):
    beta, g = self._decode_gates(self.in_proj_b(x), self.in_proj_a(x), x, st)
else:
    beta, g = self._gates(self.in_proj_b(x), self.in_proj_a(x))
# 后面的原Metal调用、conv/S更新、norm与out_proj不变。
```

资格仅限已验证的普通 B1 decode：BF16 `[1,5120]` 激活、48个门、FP32 gate 参数、NVFP4 group16/bits4 投影，以及 MLX0.32.3 的已识别旧架构后端。context 与 state 类型、record/capture/speculative/verify 标志也要通过检查。

B2/B4、prefill、记录/捕获/投机验证、直接调用默认 `_decode`、未知后端等情况继续调用原 `_gates`。启用开关并不等于绕过这些 guard。

## 5. 三层证据，分别回答三个问题

| 检查 | 实际范围 | 回答的问题 |
|---|---|---|
| 小门数学与 micro | 48层真实小权重 × 10组输入 × beta/g_log，共960项一致 | 局部表达式在这些输入上算得对、测得更快吗？ |
| 完整模型资格 | off → on → reset；每轮32个实际输出ID与所有层状态，共8256项数组比较一致 | 接入模型后，会不会改变logits、状态、参数树和完整续写？ |
| HTTP端到端 | 三对fresh process，AB/BA/AB；每进程3预热+3测量 | 用户拿到完整答案是否更快？ |

前两层先检查有限值，再比较 canonical little-endian FP32 bytes/hash；不是证明任意输入或所有原生存储布局都相同。全模型原始 tensor 没有保存，独立 CPU 审计复核记录、执行源码和断言，不能离线重新计算 GPU 数组。

完整模型正常 on 轮实际命中 `48×31=1488` 次 compiled gate，prefill 的48次仍走原式；off/reset 各1536次原式。1349个模型参数的 key/shape/dtype/对象身份保持一致。另有14个真实 fallback 案例通过；spec范围只到分派/捕获检查，未认证完整投机续写。

## 6. 局部快51%，为什么整体只快约1%

micro 的固定交错测试有10对，顺序各5次AB/BA，全部保留；6项稳定性检查通过。每轮计算48层门、构造相同的新输入并求值全部输出，两arm的 wall 中位数分别为0.855ms和0.417ms；10对相对减少量的中位数为51.30%。这是含 host 输入构造、Python graph/eval 的局部 wall，不是纯 GPU kernel 时间。

用一个**假设**理解 Amdahl：原请求100ms，其中门只占2ms。即使这2ms减半，也只是 `98 + 1 = 99ms`，整体约快1%。这个例子不代表我们测出了门占2%；不能从 micro 的51%反推出实测请求的组成。

下面才是完整请求实测：同一 M4 Max 48GiB、AC供电、同checkpoint、19-token prompt、greedy、正常EOS、实际输出32tokens、无draft与prefix复用。只切换48个GDN开关，其他源码与50ops/50MB/BFS20/FAST0设置相同。

| fresh pair | 执行顺序 | 吞吐y：off → on，tok/s/Mac | y变化 |
|---|---|---:|---:|
| 1 | off → on | 24.943893 → 25.167813 | +0.898% |
| 2 | on → off | 24.930806 → 25.241628 | +1.247% |
| 3 | off → on | 24.934536 → 25.141654 | +0.831% |

每个进程先取3个测量组的中位数，再对每个arm的3个进程中位数取中位数：y从24.934536到25.167813，约 **+0.936%**；请求E2E从1283.361ms到1271.465ms，约 **−0.927%**。TTFT从161.442ms到161.732ms，约 **+0.180%**；pair2/3分别变差0.192%/0.222%，没有首token改善。

y是实际输出tokens除以POST到收到`[DONE]`的时间，含排队、prefill、HTTP；客户端x是首个文本chunk后的交付速率代理，此次从27.633918到27.920443，约+1.037%，不把它叫GPU逐token速度。36条请求含预热共1152tokens，正式18条/576tokens，全部32个原始ID一致。

全部104份资源记录为实际AC100%、thermal0、pressure1。测量包络Swapin/Swapout/Pageout为0，但压缩/解压并非始终为0；进程完整加载到卸载也有Pageout。系统采样不能证明GPU独占或把波动归因于某应用。最初的startup harness错误产生0请求、0性能点，单独保留，未混入这三对。

三对都朝同一方向，改善幅度仍小，且只认证本机这个C1工作负载。本轮没有官方引擎同场对照，也没有C2/C4新结果；旧Pareto不因此替换数据。本章保留为显式opt-in步骤。

## 7. 运行与查看真实差异

[最小example](../../tutorial_code/chapter15_gdn_gates.py) 仍用原 `LLMEngine` 加载 checkpoint，并直接接收已有 token IDs。启用动作发生在引擎构造完成后，和实际E2E实验相同：

```python
engine = LLMEngine(model_dir, max_seqs=4, max_len=2048,
                   max_batched_tokens=512, dtype=mx.bfloat16)
gdns = [layer.linear_attn for layer in engine.model.layers if not layer.is_attn]
for gdn in gdns:
    gdn.set_gates_compile(enabled)                 # ← 关键④：只在example显式启用
```

在仓库根目录、已有MLX环境下分别运行；`MODEL`指同一27B NVFP4模型目录：

```bash
python -m tutorial_code.chapter15_gdn_gates --model-dir "$MODEL" --gates off
python -m tutorial_code.chapter15_gdn_gates --model-dir "$MODEL" --gates on
```

example打印输出IDs与实际创建的包装数，用于读代码和smoke；它不是上述HTTP配对测速器。分词器和OpenAI serving仍不属于本章实现范围。

正式example已分别在两个fresh进程运行off/on：两次32个实际IDs逐个相同，并与全模型资格记录相同；包装数分别为0/48，退出时引擎清空。10份实际资源采样均为AC100%、thermal0、pressure1。全程系统Pageouts增加200、压缩1830276、解压1575175，保留这些非零计数；smoke只验证入口，不是新的性能点。

本章的两个本地阅读点是：`tutorial-15-gates-optin`只保存受测的一文件；`tutorial-15-v1`保存讲解、example、索引与[compact证据](../../tutorial_code/examples/chapter15-gdn-gates.json)。数值资格、三对端到端结果及实际example smoke分别留有原始receipt和独立审查，不把它们合成同一种证据。

```bash
git diff tutorial-14-ac-evidence tutorial-15-gates-optin -- \
  tutorial_code/miniengine/model/gdn.py
git diff tutorial-15-gates-optin tutorial-15-v1 -- docs tutorial_code
```
