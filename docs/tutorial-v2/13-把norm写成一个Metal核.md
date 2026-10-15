# 13：把 norm 写成一个 Metal 核

第十二章先用 `mx.compile` 优化原来的 norm 表达式。这一章继续缩小执行边界：一个线程组读完整行，算均方、缩放、权重和可选门控，在一个 Metal 核中输出这一行。

本章基于 `tutorial-12-v1`，固定版本为 `tutorial-13-v1`。源码精确对应通过独立复核的 v4.2，仅增加 [norm_metal.py](../../tutorial_code/miniengine/kernels/norm_metal.py)，再修改 [layers.py](../../tutorial_code/miniengine/model/layers.py) 的 norm 调用；第十二章的流水线、VC32 与原 GDN 前处理保持原样。

## 1. 先看一个四元素例子

设一行输入为 \(x=[1,2,3,4]\)，权重全为1。RMSNorm 使用平方的平均值，不先减去均值：

\[
 m=\frac{1^2+2^2+3^2+4^2}{4}=7.5
\]

\[
 y_i=x_i\cdot\frac{1}{\sqrt{m+\varepsilon}}\cdot w_i
\]

先求整行的一个数 m，再把同一个缩放系数用到每列上。实际模型的 D 是128、256或5120等；平方、归约和最后的乘法都有有限精度舍入。

GatedRMSNorm 多一步：把 norm 输出乘以 `SiLU(gate)`。这一乘法的位置与数据类型也是原语义的一部分。

## 2. 上一个 tag 的表达式是什么

`git show tutorial-12-v1:tutorial_code/miniengine/model/layers.py` 中的普通 norm 是：

```python
@partial(mx.compile, shapeless=True)
def _rms_norm_explicit(x, w, eps):
    xf = x.astype(mx.float32)                         # ← 关键①：先用FP32
    y = xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return (y * w).astype(x.dtype)                    # ← 关键②：weight后才回BF16
```

门控版本是：

```python
xf = x.astype(mx.float32)
y = (xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + eps) * w).astype(x.dtype)
return y * nn.silu(gate.astype(mx.float32)).astype(y.dtype)
```

第十二章把整段交给编译器，能优化表达式以及可融合的逐元素运算，但 `mean` 的整行归约与后续逐元素计算仍有执行边界。一个Python函数不等于一次Metal发射；这里没有实际dispatch trace，因此不填猜测的kernel次数。

新实现直接把“读整行→归约→缩放→输出”组织进同一个自定义核。它减少中间结果的执行边界，是否改善完整请求由后面的真实服务检查回答。

## 3. 从 MLX 的实际归约源码学习线程所有权

本步的数值参照是固定版本 [MLX v0.32.3 的 row reduction](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/kernels/reduction/reduce_row.h)，结合 [reduce.cpp](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/reduce.cpp) 的线程组选择。每个线程先负责四个**连续**元素，然后跨过整个线程组已覆盖的一块，再读下一块：

```cpp
for (uint first = tid * 4; first < D; first += TG * 4) {
    for (uint j = 0; j < 4 && first + j < D; j++) {
        float xf = float(x[(size_t)row * D + first + j]);
        float sq = xf * xf;                           // ← 关键①：FP32平方
        part = part + sq;                             // ← 关键②：有序FP32累加
    }
}
float total = metal::simd_sum(part);
```

例如TG=32，一轮覆盖128个元素；线程0读0—3，线程1读4—7。更长的行再按128元素一块继续。它不是每个lane读 `lane, lane+32, ...` 的另一种加法排列。浮点加法的分组会影响舍入，所以布局要对应我们原来的归约路径。

实际选择是：D≤512使用32线程，D≤1024使用128线程；更长的行把 `ceil(D/4)` 向上取32的倍数，最多1024线程。每行一个线程组，rank3的前两维展平成行数。

一个SIMD组先合并32个线程的局部和。若线程组含多个SIMD组，再合并各组的和：

```cpp
if (lane == 0) sums[sg] = total;
threadgroup_barrier(mem_flags::mem_threadgroup);       // ← 关键③：各组和已写完
if (sg == 0) {
    float group_part = lane < nsg ? sums[lane] : 0.0f;
    float final_sum = metal::simd_sum(group_part);
    if (lane == 0) sums[0] = final_sum;
}
threadgroup_barrier(mem_flags::mem_threadgroup);       // ← 关键④：最终和可供所有线程读
```

这段只在 `nsg>1` 时执行；整个线程组进入同一个分支。尾部读写检查 `i<D`，没有线程在barrier之前提前退出。最终每个输出元素有一个writer。

源码对应的是MLX实际row路径；形状与数值测试另行确认它在本例里的结果。源码拓扑相似不能自动证明任意输入都逐位一致。

## 4. 一个核里仍保留原来的舍入边界

归约后，均方和缩放的关键代码是：

```cpp
float mean = total * (1.0f / float(D));
float inv = metal::precise::rsqrt(mean + eps[0]);
float normalized = xf * inv;
T weighted = T(normalized * wf);                      // ← 关键⑤：weight后回输入dtype
```

均方使用FP32的 `sum * (1/D)`，对应 [MLX mean 的实现](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/ops.cpp#L2350)。平方、累加、均方和epsilon加法继续是原来分开的FP32舍入步骤。kernel header使用 `#pragma clang fp contract(off)`，防止编译器把乘法和加法合成改变中间舍入的一条FMA。

对于zero-centered weight，先按原权重dtype计算 `1+weight`：

```cpp
if (ZERO_CENTERED) wf = float(W(1.0f + wf));            // ← 关键⑥：保留weight自身的舍入
```

T是输入dtype，本例BF16；W是权重dtype。不能因为后面使用FP32，就跳过BF16权重测试里这一次W转换。

门控版本的关键顺序是：

```cpp
T weighted = T(normalized * wf);
// value/sig均是FP32，sig使用MLX稳定sigmoid的precise-exp公式。
T activated = T(value * sig);                         // ← 关键⑦：FP32 SiLU后回BF16
weighted = T(float(weighted) * float(activated));      // ← 关键⑧：两个已舍入BF16值相乘
```

norm/weight后先得到BF16；gate用FP32计算SiLU后也先得到BF16；最后才做BF16输出乘法。把两条支路一直留在FP32，到最终输出才cast，会改变这一语义。

## 5. 为什么不直接替换成 fast RMSNorm

[MLX v0.32.3 fast RMS 的 Metal 代码](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/kernels/rms_norm.metal#L63) 在weight前就把normalized值转回T：

```cpp
w * static_cast<T>(x * inv)
```

当T是BF16时，路径是“FP32 norm→BF16→weight”；我们的旧表达式是“FP32 norm/weight→BF16”。二者在实数公式上相同，舍入边界不同，不能把它称作对旧实现的逐位替换。FP32 fast路径也还要检查归约拓扑与均方运算顺序。

这解释为什么要手写一个保留原语义的核，没有据此得出模型质量下降或上游kernel错误的结论。

## 6. 入口与回退也是实现的一部分

本章v4.2的普通norm通过layer wrapper选择Metal。关键调用如下，完整源码还检查weight/gate是支持的浮点dtype：

```python
if (x.dtype == mx.bfloat16 and x.ndim >= 2
        and self.weight.shape == (x.shape[-1],)
        and self.weight.dtype in (mx.bfloat16, mx.float16, mx.float32)):
    return _metal_norm(x, self.weight, self.eps, zero_centered=self.zero_centered)
# 不满足资格时，继续旧的weight处理与compiled helper。
```

gated norm同样调用 `_metal_norm(..., gate=gate)`，并额外要求 `gate.shape == x.shape` 及支持的gate dtype。weight必须有完整的一行D个元素，广播weight会沿用第十二章编译表达式。

rank1的归约可能走MLX all-axis路径，所以回退；广播gate或weight也回退，避免Metal的平坦地址读过分配范围。FP32等其他x dtype继续回退。原compiled helpers保留，没有另写一个替代数学定义。

这不是任意shape/dtype的通用认证：空维度、任意特殊值、其他设备与一般epsilon仍不在本次有限证据里。新核是内部接口，layer guards决定受支持的实际模型路径。

## 7. 验证按版本与范围分别读

以下GPU检查由唯一GPU负责人串行执行。教程整合者仅做CPU源码、报告、Markdown与Git核对。

| 证据 | 实际范围 |
|---|---|
| frozen v4：160个micro比较 | 四形状 `[1,5120]`、`[4,5120]`、`[1,24,256]`、`[1,48,128]`，普通/gated norm，有限随机输入 |
| frozen v4：33个extended比较 | 十一形状×普通/gated/zero-centered，包含D17/D257尾部、D1024/D2048与多行输入 |
| frozen v4：七个完整模型案例 | fixed33、normal EOS、mixed lifecycle、跨cache边界、晚到chunked prefill、可变采样、输入修改及runner复用 |
| v4.1：12个API补充比较 | rank1、广播gate、非连续输入、32/33/64行prefill；不是完整七案例重跑 |
| v4.2：22个API补充比较 | 保留原12个案例，新增weight广播、FP32/FP16回退、非浮点weight/gate；均通过，未重跑完整七案例 |

193个micro比较均记录bitexact=true、差异计数0。首份报告的“seed”是循环索引/缩放因子；脚本只在开头固定一次RNG seed，不把它误写成160次独立随机重播。

micro报告没有原始数组，因此CPU只能核报告标志与脚本的比较方法，不能从JSON重新生成数值等价证明。七个完整案例的输出字典与committed-cache记录已独立比较；fixed33的38次、chunked-late的42次完整logits记录相同。hash针对shape和FP32转换后的有限值，不证明任意dtype或storage布局。

最新guard只改变资格判断与源码出处说明，未改变受测模型输入所走的Metal正文。独立复核已确认实际模型路径仍选择同一核，且norm模块的可执行AST与frozen v4完全相同，因此本章复用frozen v4的193+7案例。v4.2没有重新完整运行七案例或native HTTP测速，不能把旧报告写成这次又重跑了一次。教学25个源码文件逐个SHA256匹配 `source-manifest-v4.2.json`。

## 8. 真实服务screen：有改善，胜负还没闭合

实际机器是M4 Max 40 GPU核、48GB、Apple9。目标checkpoint `mlx-community/Qwen3.8-27B-nvfp4@5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6`；group16/NVFP4-4bit、BF16激活、同19个prompt IDs、cap32、正常EOS、greedy、cache0、无drafter、并发1。每个版本三次完整预热、三次测量。

| 顺序screen的身份 | x客户端交付估计（tok/s） | y完整请求吞吐（tok/s） |
|---|---:|---:|
| 第12章三文件组合 | 26.526 | 23.996 |
| frozen v4单Metal norm | 27.468 | 24.830 |
| fresh官方oMLX balanced C1 | 28.717 | 24.637 |

相对第12章，这次screen的x约+3.55%、y约+3.48%。这是两个顺序测量点的差异，没有交替paired实验，不能当成稳定因果增益。193次正确性比较也不能把这个screen变成胜出证明。

x是首个非空文本块之后的客户端交付proxy，y是从POST到SSE DONE、包含队列/prefill/decode/传输的完整请求吞吐。自方六次原生检查均交付冻结的32个IDs，首块一个token。跨引擎baseline是同checkpoint、同输入、同实际32个生成token的dense工作量比较，不要求跨引擎greedy逐位相同。

oMLX报告没有actual GPU token-ID数组，仅提供可见文本的roundtrip tokenization；它的可见序列与自方不同。不能把这些IDs叫作已取得的GPU原始IDs，也不从差异推断质量。当前自方y略高，x仍落后，且尚非paired，不能宣布超过oMLX。

v4正式测量pressure1、swap2140.31MB稳定、Swapouts增量0；全局Pageouts增加12页，即0.1875MiB。这是系统全局计数，不把它归因于某个引擎。加载与预热的资源变化在源报告单列。

[完整证据](../../tutorial_code/examples/chapter13-metalnorm-screen.json)保存各报告原文与SHA256，明确测试版本和源码复用边界。失败的pressure guard请求没有测量点，不混入表格。

## 9. 阅读代码与运行

用相邻tag查看这一步：

```bash
git diff tutorial-12-v1 tutorial-13-v1 -- \
  tutorial_code/miniengine/model/layers.py \
  tutorial_code/miniengine/kernels/norm_metal.py \
  docs/tutorial-v2/13-把norm写成一个Metal核.md
```

完整生成入口沿用第十二章。下面是已有0.8B核心模型的运行例子，不是表中27B HTTP screen的复现：

```bash
cd /Users/ryanli/code/minivllm-tutorial
/Users/ryanli/code/minivllm/.venv/bin/python -m tutorial_code.benchmark_core \
  --model-dir /Users/ryanli/code/minivllm/models/Qwen3.5-0.8B-nvfp4 \
  --max-new-tokens 32 --repeat 3 --warmup-runs 3 \
  --out /private/tmp/chapter13-core.json
```

## 出处

本核独立手写，参考MLX **v0.32.3 / MIT** 的 `reduce_row.h`、`reduce.cpp`、`ops.cpp` 与 `unary_ops.h` 的归约及稳定sigmoid公式，源文件保留固定版本出处。oMLX与Lithos的源仓库是Apache-2.0；第十二章的执行流程/列所有权启发继续保留，本步没有引入它们的卷积融合或芯片配方。
