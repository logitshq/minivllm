# V19 Q/K 范数复用：代码讲解与静态边界

[实验阅读地图](README.md) · [第16章入口](../tutorial-v2/16-实验篇从哪里开始.md) · [本阶段结果](17-gdn-norm-reuse-status.md)

本讲解解释V19如何复用Q/K范数；正文的代码阅读没有执行GPU。**后续状态：build3完整模型检查已通过，正式E2E尝试因资源条件中止；V19仍未获得默认启用资格。** 旧暂停已由用户随后恢复测试撤销，详见阶段结果。早期两次token0前STOP只描述当时尝试，不再作为本页当前状态。

先抓住一个改动：多个value head和输出列会重复计算同一Q/K范数，候选先计算并保存，再供状态更新复用；各head自己的状态仍独立。源码固定于`tutorial-17-static-reading-v1`，以下链接指向同一Git版本，后续实验不会改写它。
调用链：[`GatedDeltaNet._decode:192`](https://github.com/logitshq/minivllm/blob/tutorial-17-static-reading-v1/tutorial_code/miniengine/model/gdn.py#L192) → [`_decode_norm_reuse:213`](https://github.com/logitshq/minivllm/blob/tutorial-17-static-reading-v1/tutorial_code/miniengine/model/gdn.py#L213) → [`gated_delta_step_norm_reuse_metal:196`](https://github.com/logitshq/minivllm/blob/tutorial-17-static-reading-v1/tutorial_code/miniengine/kernels/gdn_norm_reuse_metal.py#L196)，后者依次 `_prepare` 与 `_state`（201–202）。
入口还要求 normal=True、enabled=True、精确类型/shape/dtype/backend，见 [initial eligibility:181](https://github.com/logitshq/minivllm/blob/tutorial-17-static-reading-v1/tutorial_code/miniengine/model/gdn.py#L181) 与 [最终 eligibility:104](https://github.com/logitshq/minivllm/blob/tutorial-17-static-reading-v1/tutorial_code/miniengine/model/gdn.py#L104)；不满足时走旧实现（229–231）。

## 实际形状与复用方向

固定 B=1，Hk=16，Hv=48，Dk=Dv=128；卷积输出 u 为 BF16[1,10240]=Q2048+K2048+V6144，状态 S 为 FP32[1,48,128,128]。
[旧 `_split_heads:118`](https://github.com/logitshq/minivllm/blob/tutorial-17-static-reading-v1/tutorial_code/miniengine/model/gdn.py#L118) 把 Q/K 从 [1,16,128] 沿 head 轴 repeat 3 次成为 [1,48,128]，不是沿 token 轴复制。
新实现保存未归一化的 activated Q/K（转 FP32）各 [1,16,128]，V[1,48,128] 和 scales[1,16,2]；[state:93](https://github.com/logitshq/minivllm/blob/tutorial-17-static-reading-v1/tutorial_code/miniengine/kernels/gdn_norm_reuse_metal.py#L93) 用 h//3 回到共享 Q/K head。

## 旧实现（教学伪代码，11 行；对应模型 201–208、split 118–123、旧 kernel 18–50）

```text
u = causal_conv(...)
q, k, v = split_heads(nn.silu(u))  # Q/K 各 repeat 3 次
for h in 48 value_heads:
  for dv in 128 columns:
    for sp in 4 splits:
      qn, kn = 0.0f, 0.0f
      for d in range(128): qn += q[h,d]*q[h,d]; kn += k[h,d]*k[h,d]
      qs = rsqrt(qn + 1e-6f) * rsqrt(float(128))
      ks = rsqrt(kn + 1e-6f)
      kv_part = serial_sum_over_this_split(ks * k[h,d] * S[h,d,dv])
      update_state_and_output_with_same_split_reductions(...)
```
旧真实位置：[gdn_metal.py:33](https://github.com/logitshq/minivllm/blob/tutorial-17-static-reading-v1/tutorial_code/miniengine/kernels/gdn_metal.py#L33)，grid/threadgroup 在 [90](https://github.com/logitshq/minivllm/blob/tutorial-17-static-reading-v1/tutorial_code/miniengine/kernels/gdn_metal.py#L90)。

## 新实现（教学伪代码，11 行；对应 prework 51–77、state 93–118）

```text
for h in 16 QK_heads:  # prework；128 lanes 并行做 activation
  q_ready[h,:], k_ready[h,:] = typed_BF16_silu_then_FP32(u_Q[h,:], u_K[h,:])
  barrier()
  if lane == 0:
    qn, kn = 0.0f, 0.0f
    for d in range(128): qn += q_ready[h,d]*q_ready[h,d]; kn += k_ready[h,d]*k_ready[h,d]
    scales[h] = (rsqrt(qn + 1e-6f)*rsqrt(float(128)), rsqrt(kn + 1e-6f))
v_ready = typed_BF16_silu_then_FP32(u_V)  # prework 另外 48 groups
for h, dv, sp in unchanged_state_grid:
  q, k, (qs, ks) = q_ready[h//3], k_ready[h//3], scales[h//3]
  unchanged_state_and_output_math(q, k, v_ready, qs, ks, ...)
```
新真实位置：[prework:61](https://github.com/logitshq/minivllm/blob/tutorial-17-static-reading-v1/tutorial_code/miniengine/kernels/gdn_norm_reuse_metal.py#L61)、[串行范数:65](https://github.com/logitshq/minivllm/blob/tutorial-17-static-reading-v1/tutorial_code/miniengine/kernels/gdn_norm_reuse_metal.py#L65)、[state 读 scales:93](https://github.com/logitshq/minivllm/blob/tutorial-17-static-reading-v1/tutorial_code/miniengine/kernels/gdn_norm_reuse_metal.py#L93)。

## 它是哪一种 normalization

对激活后的向量 q、k，q_scale=rsqrt(Σq[d]²+1e-6)·rsqrt(Dk)，k_scale=rsqrt(Σk[d]²+1e-6)。这是带 epsilon 的 L2 normalization，Q 再乘 Dk^(-1/2)。没有先除 Dk 取均值，因此不是 RMSNorm；也不是 softmax 或 attention 权重计算。
模型输出后的 `self.norm` 是另一个 GatedRMSNorm（[定义:44](https://github.com/logitshq/minivllm/blob/tutorial-17-static-reading-v1/tutorial_code/miniengine/model/gdn.py#L44)、[调用:233](https://github.com/logitshq/minivllm/blob/tutorial-17-static-reading-v1/tutorial_code/miniengine/model/gdn.py#L233)），不要与本次 Q/K L2 混称。
二维教学例子：设 BF16 activation 后再转 FP32 的 q=(3,4)、k=(0,2)，串行累加分别 0→9→25、0→0→4；qs=rsqrt(25+1e-6)·rsqrt(2)，ks=rsqrt(4+1e-6)。同一 Q/K 的三个 value heads、各列/分段都能共享这两个标量；实际维度仍为 128。
该例保留 epsilon 在平方和之后、rsqrt 之前的位置；不把两个 rsqrt 合成一个，不把 scale 预乘到 ready 向量。

## 计数依据与不能混淆的基线

旧 kernel 每个 thread 都算完整 Q/K 平方和；逻辑 normsets=48 heads×128 columns×4 splits=24,576 对；新 prework 每个 Q/K head 的 lane0 算一对，共16对（单个标量范数是49,152→32）。
value tile=32，故旧有4个 tiles/head，但每 tile 又有32×4个线程；不能把旧 normsets 误算为48×4=192。以上是源码逻辑执行次数，不能当作硬件指令或耗时测量。
V19 activation 元素逻辑数=2×16×128+48×128=10,240（prework 61、62、76）。
55,296=48×4×(2×128+32)，已由[第16章固定版本的计数说明](https://github.com/logitshq/minivllm/blob/tutorial-16-experiment-v1/docs/tutorial-v2/16-把GDN的SiLU和拆头并入状态更新.md)证实属于 V18 中间 postconv 实现的静态对照；V18 未作为本次 stationary A。
实测 A=V17 decode isolated public 片段：nn.silu→split/repeat→原 gated_delta_step_metal；B=N2 public 两 kernel，norm-reuse 源码同 V19。A 不是 V18；A/B 的 SiLU 逻辑元素均10,240，相对 A 没有 SiLU 元素数减少。
计时范围为 fixed saved15 ready-u→consumed o/S_new local host-wall，包含准备、框架与消费开销；55,296 不能解释本次13.332419%收益，也不是硬件 profiler 实测。

## 等价意图、代价与证据界限

共享成立因 scale 仅依赖同一 Q/K 向量，不依赖 dv、sp 或共享它的 value head；state 各 head 自己的 S、g、beta、V 仍各自计算。
必须保留 [typed activation:10–31](https://github.com/logitshq/minivllm/blob/tutorial-17-static-reading-v1/tutorial_code/miniengine/kernels/gdn_norm_reuse_metal.py#L10)：`bfloat16_t sigmoid_value = V18NativeSigmoid()(u)`，再以两个 BF16 操作数执行 multiply，结果保存 BF16，之后转 FP32；不要替换为一次 FP32 sigmoid/乘法后统一 round。
必须保留 FP32 sum 的 d=0..127 串行顺序、epsilon、两个 rsqrt 的乘法顺序及 state 表达式结合顺序；ready 保存 raw activated FP32，不是提前乘好 scale 的向量。新编译选项为 math_mode=safe（8）。
GDN Metal 状态路径由一个 kernel 变为 prework+state 两个 dispatch；还输出10,272个FP32临时元素（41,088字节）。旧路径原有原生 activation/repeat/cast，因此不能只据此断言整个 forward 的净 dispatch 数或净峰值内存增加多少。
[状态文档](17-gdn-norm-reuse-status.md)保留局部13.33% host-wall降低、早期两次token0前STOP，以及后续build3完整模型PASS和E2E资源STOP；局部比例不能当作完整服务收益。
静态阅读只解释“迁移相同子表达式并复用”的意图。完整模型正确性以阶段结果中build3的有限测试为准，27B端到端提速仍未由这一实验取得正式结论。

```mermaid
flowchart LR
  U[卷积输出 BF16 u] --> P[prework: SiLU与16对Q/K scales]
  P --> S[state: 48 heads按h//3复用]
```
