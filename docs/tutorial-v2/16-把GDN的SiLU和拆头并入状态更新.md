# 16：把 GDN 的 SiLU 和拆头并入状态更新

第十五章把两组小门一起编译。这一章继续普通 B1 decode，把**卷积之后、状态递推之前**的准备工作移进 Metal kernel：输入不再先生成独立的 SiLU 结果和重复后的 Q/K，而是直接交给状态更新核。

**本章是未采纳的独立实验，默认关闭，只保存在实验分支与标签，不合并当前 `tutorial/step-by-step` 核心。** 局部耗时降低约13.96%，完整模型数值资格通过；端到端吞吐仅+0.632%，未达预声明+1%门槛。局部改善不能写成整模型收益。

## 1. 之前 → 本章之后

```text
第15章：原卷积 → BF16 SiLU → split/reshape/repeat → FP32输入 → 原状态核
第16章：原卷积 → raw BF16 u ───────────────────────────→ 新状态核
                                                    内部SiLU/按头取数/原递推
两条路径：原来的门计算 → beta/g_log；原conv tail → 下一步conv状态
```

这里的 `u` 就是卷积结果 `[1,10240]`，不是投影前的输入。卷积、gate公式、tail、norm和out_proj继续使用原表达式；第十五章的编译开关也独立保留。

| 本次替换 | 本次保留 |
|---|---|
| 独立SiLU、拆头与Q/K重复的准备路径 | BF16卷积的乘法/求和及tail切片 |
| 用头号映射读取原始u | FP32状态递推表达式及归约顺序 |
| kernel内部暂存Q/K和一片V | beta/g_log公式、norm、量化投影 |

## 2. 先在第十五章调用点找到替换段

正式应用后的源码入口：[gdn.py](../../tutorial_code/miniengine/model/gdn.py)、[qwen3_5.py](../../tutorial_code/miniengine/model/qwen3_5.py)、[新kernel](../../tutorial_code/miniengine/kernels/gdn_postconv_metal.py)。源码身份与结果摘要见本章[compact证据](../../tutorial_code/examples/chapter16-gdn-fusion-results.json)。

以下节选来自冻结的 `model/gdn.py`：第十五章159–176行，本候选213–234行；省略函数头和不变的投影。这里展示真实替换段，不新增另一套实现。

**之前：** 卷积之后先准备好Q/K/V，再调用原核。

```python
w = self.conv1d_weight.astype(qkv.dtype)
xp = mx.concatenate([st.conv.astype(qkv.dtype), qkv[:, None]], axis=1)
qkv = (xp * w.T[None]).sum(axis=1)               # ← 关键①：原BF16卷积保留
q, k, v = self._split_heads(nn.silu(qkv), B)      # ← 本章替换这里的准备工作
if _compile_gates and object.__getattribute__(self, "_gates_compile_enabled"):
    beta, g = self._decode_gates(self.in_proj_b(x), self.in_proj_a(x), x, st)
else:
    beta, g = self._gates(self.in_proj_b(x), self.in_proj_a(x))
from ..kernels.gdn_metal import gated_delta_step_metal
o, S = gated_delta_step_metal(q, k, v, g, beta, st.S)
st.conv, st.S = xp[:, 1:], S                    # ← 原tail与状态整体替换
```

**之后：** `_decode` 通过初始资格检查才进入 `_decode_postconv`。它保留同一段卷积与上面的gate分支，把卷积后的 `qkv` 直接传给新核：

```python
if self._postconv_eligible(qkv, g, beta, x, st, _postconv_normal):
    from ..kernels.gdn_postconv_metal import gated_delta_step_postconv_metal
    o, S = gated_delta_step_postconv_metal(qkv, g, beta, st.S)  # ← 关键②：raw BF16 u
else:
    q, k, v = self._split_heads(nn.silu(qkv), B)
    from ..kernels.gdn_metal import gated_delta_step_metal
    o, S = gated_delta_step_metal(q, k, v, g, beta, st.S)
st.conv, st.S = xp[:, 1:], S                    # ← 与之前相同
```

后面的 `self.norm(o.astype(x.dtype), z)` 和 `self.out_proj(...)` 相同。旧 `_decode` 分支仍在，拒绝新路径时先回到旧计算；所有资格拒绝都发生在重新赋值 `st.conv/st.S` 之前。

## 3. SiLU 的公式相同，还要保留 BF16 舍入

数学上 `SiLU(u) = u × sigmoid(u)`，但有限精度的中间值也是接口。原路径先得到BF16 sigmoid，再做BF16乘法，最后才转FP32供递推使用。把它改成“FP32算完再转一次BF16”可能得到不同位。

新核复用冻结版本中的native sigmoid/multiply算子体，并显式保留两个BF16变量（`gdn_postconv_metal.py` 11–34行）：

```cpp
bfloat16_t sigmoid_value = V18NativeSigmoid()(u);
bfloat16_t activated = V18NativeMultiply()(u, sigmoid_value);
return {sigmoid_value, activated};               // ← 关键③：两个BF16边界
```

`V18NativeSigmoid` 使用 `metal::precise::exp(metal::abs(x))` 和按符号分支；编译选项为 `math_mode="safe"`。这不是任意换成一个近似sigmoid。读取 `activated` 后才 `float(...)`，然后沿用原状态核的FP32表达式。

独立检查枚举全部65,280个有限BF16输入，sigmoid和SiLU的原始BF16位均与原操作相同。它回答此算子在该冻结MLX/Metal环境的舍入问题；NaN/Inf不在此枚举内，也不因此认证其他后端。

## 4. 不复制 Q/K，按 value head 找原 head

本模型有16个Q/K head、48个V head，每头128维。旧 `_split_heads` 把每个Q/K head连续重复3次。新核对于value head `h`，直接从 `h // 3` 取Q/K。

| 原始u区间 | 元素地址（j为头内维度） |
|---|---|
| Q：0–2047 | `(h / 3) * 128 + j` |
| K：2048–4095 | `2048 + (h / 3) * 128 + j` |
| V：4096–10239 | `4096 + h * 128 + dv` |

Metal中的 `h / 3` 是无符号整数除法。每个threadgroup为 `32×4=128` threads，负责一个head的一片32列V。128个thread合作各读一个Q/K维度；第一组32个thread读取V片，写入threadgroup缓存后先做barrier，再执行原递推。

```cpp
uint j = sp * VALUE_TILE + dv_local;
q_storage[j] = float(v18_bf16_activation(u[(h / 3) * 128 + j]).activated);
k_storage[j] = float(v18_bf16_activation(u[2048 + (h / 3) * 128 + j]).activated);
```

以上是新核49–61行的一部分。它省去外部准备与分派，却增加重复激活：一层一个token从原来的10,240次，变成55,296次；逻辑threadgroup暂存也从512B增到1664B。计数为 `48 heads × 4 tiles × (128 Q + 128 K + 32 V) = 55,296`。不同head/tile会重新算相同Q/K，因此融合**可能更慢**，必须实际测量。

状态更新中的decay、delta、S_new和o继续使用原式及原归约顺序。这里没有把卷积或门公式再融合进来，也没有重排递推数学。

## 5. 独立开关与完整模型的通行条件

构造函数新增 `self.set_postconv_fusion(False)`。setter只接受bool，保存私有元数据，不增加checkpoint参数；它与 `set_gates_compile` 互不代替。

```python
for gdn in gdns:
    gdn.set_gates_compile(True)                  # ← 第15章的已接受路径
    gdn.set_postconv_fusion(enabled)             # ← 关键④：本章独立开关
```

`qwen3_5.py` 每次forward用实际context与 `capture` 参数确认普通decode，再把 `_postconv_normal` 经DecoderLayer传给GDN。证明只在本次调用传递，不存在模型上供下一次复用的临时“允许”标志。

GDN仍检查精确的context/state类型和字段、B1 BF16 `[1,5120]`、BF16卷积权重与tail、FP32 `[1,48,128,128]` 状态、48个FP32门及已识别后端。prefill、B2/B4、record/capture/speculative/verify、未知元数据和直接默认调用不会获得这条通行路径。

已完成18个真实fallback案例；完整spec续写、drafter inference和模型重新加载不在本次资格内。启用开关不代表这些场景获准融合。

## 6. 三层证据，不能互相替代

| 层次 | 真实结果 | 能回答什么 |
|---|---|---|
| saved15局部micro | A150.520060µs → B129.508057µs，耗时−13.959603% | 固定输入准备到消费o/S_new的host-wall |
| 完整模型正确性 | accepted15、candidate OFF/ON/reset OFF四臂严格一致 | 这次接入是否改变受测模型结果 |
| HTTP端到端 | 吞吐y +0.632032%，未达+1%门槛，**不采纳** | 完整请求改善是否足够采用 |

micro使用15banks（9真实+6合成）、10对（5AB/5BA），10/10方向改善；前后AA/BB controls和两arm drift均通过原定5%门槛。每次fresh函数调用并 `mx.eval(o,S_new)`，计时包含host提交与求值，不称为纯GPU kernel时间。

局部15banks的o/S_new共11,888,640个FP32元素，每次完整审计均按原始bytes严格相同，含±0；永久raw输出可由CPU核对。输入与计时资格各有冻结记录，不将局部13.96%外推到请求。

全模型四臂各19个prompt IDs→32个normal greedy IDs；每臂4128个canonical FP32签名严格相同。ON实际新核命中 `48×31=1488` 次；prefill和fallback不命中新核。1349个参数的身份及开关重置检查通过。

全模型永久保存128份logits raw；conv/S/有效KV永久只保留canonical FP32签名/hash。CPU可重读logits raw与签名记录，**不能重新读取不存在的state raw，也不能重新计算GPU状态**。

E2E预先规定：两arm都启用第十五章gates，只切换本章开关；三对fresh process按AB/BA/AB，每进程3预热+3测量。采用要求三对吞吐y都改善、聚合y至少+1%、交付速率代理x不退步，以及数值/源码/资源/协议检查全部通过。TTFT变差也必须照实保留。

本轮唯一正式E2E已完成，未重试或改门槛。实际结果如下；每arm先取3测量组中位数，再取3进程中位数，预热不计入：

| pair | 顺序 | 吞吐y off → on（tok/s/Mac） | y变化 | TTFT变化 |
|---|---|---:|---:|---:|
| 1 | AB | 25.227849 → 25.397032 | +0.670621% | +0.191479% |
| 2 | BA | 25.238984 → 25.446036 | +0.820366% | -0.263082% |
| 3 | AB | 25.237523 → 25.387278 | +0.593385% | +0.011925% |

聚合y为 25.237523 → 25.397032，**+0.632032% < +1%**；x交付速率代理+0.771936%。请求E2E为 1267.953292 → 1259.989750ms（-0.628063%）；TTFT为 161.783792 → 161.803084ms（+0.011925%，变差）。全部正负值保留。

36请求共1152个实际输出token，正式18请求/576token；每请求32个raw/visible/expected IDs严格一致。106份实际资源记录为AC、thermal0、pressure1，swap用量稳定；测量包络Pageout/Swapin/Swapout为0，但压缩0–12931、解压84–19458，不能把系统计数归因于模型。

**结论：`DO_NOT_ADOPT_KEEP_UNPROMOTED`。** 三对y都改善，但聚合未达预声明门槛；保留实验历史，不进入当前教程核心。原实际receipt、CPU重放与8个进程wait/reap及外部absence证明均保存，compact记录各证据SHA。新example GPU smoke仍未做。尚无同轮官方引擎挑战，不能宣布超过oMLX/Lithos。

## 7. 运行入口与逐步 commit 阅读点

[example](../../tutorial_code/chapter16_gdn_fusion.py)只在引擎构造后设置两个开关，用同一19-token prompt、greedy、max32、正常EOS、`cache_limit_gb=0`、无drafter。预期32个IDs来自已完成的全模型资格；新example目前只通过CPU语法检查，运行成功后才可另记smoke。

这套源码只在 `tutorial/16-gdn-postconv-experiment` 分支与实验标签可运行；当前 `tutorial/step-by-step` 不含它。在该实验worktree、已有MLX环境下分别开fresh进程运行：

```bash
MODEL=/Users/ryanli/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-nvfp4/snapshots/5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6
python -m tutorial_code.chapter16_gdn_fusion --model-dir "$MODEL" --fusion off
python -m tutorial_code.chapter16_gdn_fusion --model-dir "$MODEL" --fusion on
```

example核对并打印实际IDs，不是HTTP性能实验。正常EOS策略不代表第32个token就是EOS。配置和预期IDs见[JSON](../../tutorial_code/examples/chapter16-gdn-fusion.json)。checkpoint是 `mlx-community/Qwen3.8-27B-nvfp4`，精确snapshot为 `5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6`。

本章分两步保存；第一步已固定真实受测源码：commit `33152a7ff0b0c9313568e84564bc24b2dc3d1ecf`，tag `tutorial-16-experiment-code`。第二步解释与结果待独立审查后commit；它们属于独立实验历史，主教程分支继续停留在第十五章核心：

| 顺序 | 拟定tag | 只保存什么 |
|---|---|---|
| 1 | `tutorial-16-experiment-code` | 冻结source26中的gdn.py、qwen3_5.py与新kernel |
| 2 | `tutorial-16-experiment-v1` | 本章、example、JSON、README及后续实际证据状态 |

步骤1必须按 `source-manifest-v18-gdn-postconv-build2.json` 逐byte应用；其SHA256为 `198750c027274fb50eb364416db0acd9becbbd4d2dc1484b3991fe76a200ed58`。26个核心文件中仅3个路径变化，其余23个保持原byte；精确的3个相对路径为 `model/gdn.py`、`model/qwen3_5.py`、`kernels/gdn_postconv_metal.py`。旧tag保留，不把本实验合并到当前教程核心。

用以下命令逐步阅读；实验文档tag在独立审查后创建，源码tag先保存：

```bash
git diff tutorial-15-v1 tutorial-16-experiment-code -- tutorial_code/miniengine
git diff tutorial-16-experiment-code tutorial-16-experiment-v1 -- docs tutorial_code
```
