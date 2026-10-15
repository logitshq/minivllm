# 14：把多个 prompt 一起 prefill

第八章已经能同时服务多个请求。但“同一轮接纳多个请求”和“用一次模型前向计算多个 prompt”是两件事。这一章补上后者：把兼容的新请求排成多行，共享一部分线性计算，每条请求继续拥有自己的历史和状态。

本章从 `tutorial-13-v1` 继续，只修改六个文件。模型路径先保存在 `tutorial-14-batched-ops`，runner 接入后保存为 `tutorial-14-batch-prefill`，讲解与证据保存在 `tutorial-14-v1`。三个版本都保留可运行的完整生成入口。

## 1. 先确认替换的是哪里

第十三章的 [model_runner.py](../../tutorial_code/miniengine/engine/model_runner.py) 已经允许一个 step 有多条 prefill。它仍逐请求调用模型，构建各自的计算图，再统一求值。下面省略投机分支、采样与 cache 收集，只保留这个顺序：

```python
for req, n in out.prefill:
    ids = mx.array(req.all_token_ids[start:start + n])
    logits = self.model(ids, ForwardContext("prefill", caches), last_only=True)
    # 为本请求采样，把结果加入 to_eval。
mx.eval(*to_eval, *self.batch.arrays())              # ← 关键①：最后合在这里求值
```

不是每条请求都有一个显式 `mx.eval`，也不能由 Python 循环推断 GPU 严格串行。问题是这些 prompt 尚未成为同一次模型调用中的多行输入。

本章之后，已被 scheduler 选中的兼容请求会走：

```python
ids = mx.array([r.prompt_token_ids for r in reqs])   # [B,T]
ctx = ForwardContext("prefill_batch", layer_caches,
                     prefill_shape=(len(reqs), n))
logits = self.model(ids, ctx, last_only=True)        # ← 关键②：一次调用，得到B行logits
```

continuous batching 决定每轮哪些请求进入、继续或退出；batch prefill 决定已选中的多条 prompt 怎样一起前向。我们继续使用原 scheduler，没有增加等人凑批的时间窗口。

此前真实 C4 诊断中，四条独立的19-token prefill 分两轮接纳，两轮含 prefill 的 step 共占680.6—706.1ms。每两请求建图约10ms，scheduler约0.003ms。它支持继续检查 prefill 计算；这些带观察器的 wall/eval 区间不是 GPU kernel 时间，也不是正式性能点。

## 2. 第一步：让模型明确知道 B 和 T

先看两条长度都是3的 prompt。B=2 是请求数，T=3 是每条请求的 token 数：

```text
请求A: a0 a1 a2 ─┐                  ┌─ Attention A ─ KV_A
请求B: b0 b1 b2 ─┴─ token行[B*T,H] ├─ Attention B ─ KV_B
                                   ├─ GDN A沿时间递推 ─ S_A / tail_A
                                   └─ GDN B沿时间递推 ─ S_B / tail_B
```

线性层分别处理每个 token 行，可以把前两维展平；Attention 和 GDN 则必须知道哪几个 token 属于同一条序列。把六个 token 当成一条长度6的 prompt，会让B读到A的历史。

本步在 [context.py](../../tutorial_code/miniengine/context.py) 加 `prefill_shape=(B,T)`，cache 排成 `caches[layer][row]`。原单序列 prefill 和 decode 的 context 接口仍保留。

[qwen3_5.py](../../tutorial_code/miniengine/model/qwen3_5.py) 的 grouped 分支先展平输入 IDs；各层中不涉及序列关系的计算使用 `[B*T,H]`：

```python
B, T = ctx.prefill_shape
x = self.embed_tokens(input_ids.reshape(-1))        # ← 关键①：每个token一行
# 逐层执行；Attention/GDN通过ctx恢复各自需要的布局。
```

上一章的 `last_only` 用 `x[-1:]` 取整张表最后一行。现在需要取每条请求的最后一个 token：

```python
# 之前：只有一条序列。
x = self.norm(x[-1:] if last_only else x)
# 之后：grouped且last_only时。
x = x.reshape(B, T, -1)[:, -1]                     # ← 关键②：保留B行
x = self.norm(x)
```

完整 logits 恢复成 `[B,T,vocab]`；last-only 输出 `[B,vocab]`。后者的输出投影使用每条请求原来的 M=1 调用，避免又在采样前改变矩阵乘法路径。

### Attention：批量维隔开请求，时间维只看自己

替换点是 [attention.py](../../tutorial_code/miniengine/model/attention.py) 的分派入口，新增 `_prefill_batch`。原 `_project`、`_prefill`、`_decode` 继续保留。

投影与 q/k norm 后，恢复批量、头、时间、维度四个轴。下面只展示布局与 cache 边界，省略投影和 gate：

```python
q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
k = k.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
v = v.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
q = apply_rope(q, 0, self.rotary_dim, self.theta)
k = apply_rope(k, 0, self.rotary_dim, self.theta)
for row, cache in enumerate(caches):
    cache.update(k[row:row + 1], v[row:row + 1])     # ← 关键③：每请求独立cache
o = mx.fast.scaled_dot_product_attention(
    q, k, v, scale=self.scale, mask="causal" if T > 1 else None)
```

causal mask 限制每个 row 内的时间关系；batch 轴不会变成可互相访问的 key。起始 RoPE 位置为0，因为入口只接收真正未开始的新请求。各自调用原 `cache.update`，继续保留256-token容量取整和各自的 active offset。

### GDN：可以合并独立的头，不能合并时间

替换点是 [gdn.py](../../tutorial_code/miniengine/model/gdn.py) 的 grouped 分支。卷积先按请求分别调用原 `causal_conv1d`，继续原 tap 顺序与 BF16 舍入。不能让B开头的卷积读到A结尾。

```python
for row in range(B):
    y, tail = causal_conv1d(raw[row], w)
    convolved.append(y)
    tails.append(tail)                              # ← 关键④：各自保留卷积tail
```

拆头后，原 chunk 算法的输入是 `[T,Hv,D]`。不同请求的不同头彼此独立，可以把 request 与 head 合在一起：

\[
[B,T,H_v,D] \longrightarrow [T,BH_v,D]
\]

```python
q = q.reshape(B, T, self.Hv, self.Dk).transpose(1, 0, 2, 3)
q = q.reshape(T, B * self.Hv, self.Dk)               # ← 关键⑤：T仍是原来的时间轴
# k/v/g/beta同样重排，再调用已有recurrent或chunk函数。
```

chunk 大小仍由每条序列的T决定。算完后把状态恢复成 `[B,Hv,Dk,Dv]`，逐请求写回 `S`、conv tail 和 offset=T；输出按相反顺序恢复 `[B*T,H]`。

本章没有修改 GDN Metal kernel，也没有接入另一候选的 decode 卷积融合或 prepared recurrence。第十三章的 decode 路径保留。

## 3. 数学相同，为什么第一次仍然没过数值检查

线性层在实数上逐行独立，但增大矩阵的行数M可能让库选择不同的计算方法。我们曾直接把 M=T 改成 M=B*T，真实27B的首个B2/T19案例出现 normalized RMS=0.011733，超过事先规定的0.005，候选因此停止，没有跑性能。

分解诊断把输入投影固定相同后，GDN head-fold 的输出/状态与 Attention RoPE/SDPA 在该案例都相同。差异先出现在部分 NVFP4 矩阵乘法。

固定 [MLX v0.32.3 的量化 dispatch](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/quantized.cpp#L1081) 会根据 `ceil(M/32)` 等量选择 split-K：把K方向拆成几段，各段的部分和先存入输入 dtype，再做最终归约。本例输入是BF16。M19变M38时，GDN z 的K分段从2变1，而 MLP gate 仍为1。部分和在哪里舍入改变了，即使实数公式一样，输出也可能不同。

不能把这解释成 GDN 串了别人的状态，也不能为了合批放宽误差门槛。修正放在 [layers.py](../../tutorial_code/miniengine/model/layers.py) 的 `prefill_linear`：只在保守 guard 证明两种形状都走固定 splitK1 路径时共享矩阵乘法，其余投影逐请求调用。

```python
# 所有guard满足时才共享；完整资格判断见源码。
return op(mx.contiguous(x))
# 不满足时维持每条请求原来的M=T。
return mx.concatenate([op(row) for row in x.reshape(B, T, -1)], axis=0)
```

guard 固定 MLX0.32.3、当前GPU、已识别的 architecture generation<17、BF16、NVFP4 group16/bits4、权重/scale dtype及矩阵维度。T≥14、K>4096与对齐检查使它保守地避开小矩阵 qmv 路径；原M和合并M的32行/列tile乘积都要大于256，使 splitK 都为1。

架构限制对应 [MLX 的 NAX availability](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/device.cpp#L887)，小矩阵边界对应 [qmv selector](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/quantized.cpp#L82)。这些是固定实现的资格条件，不是对未来MLX版本或其他GPU的保证。未知版本/架构直接回退，不按prompt内容或某个请求ID开特例。

MLP 新增 `_prefill_batch`，分别用这个 helper 处理 gate/up/down；原 `MLP.__call__` 没有变成另一套 compiled MLP。真实直接激活记录中，每次有193个投影走共享路径，敏感投影仍逐row计算。

到这里保存 `tutorial-14-batched-ops`。模型已具备完整新路径，runner 仍走原单请求 prefill，因此中间版本继续可运行。

## 4. 第二步：只把兼容请求交给新路径

现在修改 runner，保存为 `tutorial-14-batch-prefill`。它先按完整prompt长度分组；一组至少两条请求，并且每条都必须是 fresh：

- `num_computed_tokens == 0`，本轮块长等于整个prompt长度。
- 真正的 cache offset=0，且没有已有KV、conv state、GDN state或record。
- 没有 drafter/spec；后端、NVFP4配置和BF16激活符合入口资格。

一组里的 prompt 可以完全不同。B1、异长、partial chunk、continuation、已有状态、record或spec仍走原路径。capture 不进入此分组入口；显式调用 grouped+capture 会被模型拒绝。

逐请求的 cache 先按层转置成 `layer_caches`，然后一次前向、一次批量采样，再把行对应回请求：

```python
caches = [self.staging.setdefault(r.req_id, make_cache(self.model)) for r in reqs]
layer_caches = [list(layer) for layer in zip(*caches)]
ctx = ForwardContext("prefill_batch", layer_caches, prefill_shape=(len(reqs), n))
logits = self.model(ids, ctx, last_only=True)
prefill_batch_toks = sample(logits, [r.sampling for r in reqs])
for row, r in enumerate(reqs):
    prefill_toks.append((r, prefill_batch_toks[row:row + 1]))
    grouped.add(r.req_id)                           # ← 关键⑥：旧循环跳过这些请求
```

这里特意使用 `prefill_batch_toks`，不覆盖同一步 decode 的 `toks`。随后按 `out.prefill` 恢复交付顺序；采样结果仍由原来的统一 `mx.eval` 求值，再按原请求接口读回。

当新请求进入 decode，已有 staging→DecodeBatch 的搬运、rebuild、drop 与 ahead 失效规则继续使用。优化必须连同这些状态边界一起验证，不能只看第一个token。

## 5. 最终代码实际验证到了哪里

最终25个教学源码文件逐SHA256匹配冻结的 accepted13+纯batch-prefill 候选，manifest SHA256为 `2a990456522fdd00d0f331011964be4e9b755e47b2b6eeccd293b1a8c435baad`。六文件之外的19个文件及全部kernel与第十三章相同。

GPU负责人实际使用同一27B checkpoint `mlx-community/Qwen3.8-27B-nvfp4@5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6`、NVFP4 group16/bits4、BF16，比较本章路径与 accepted13 独立prefill。最终17个案例通过，共88,268条有限值 canonical FP32 byte-SHA比较记录一致。

这里先检查shape/dtype与有限值，再把当前比较数组转换成FP32计算byte hash。它是这些数值与shape的有限验证，不是所有原生dtype存储布局的证明。原始tensor未保存，独立CPU审计复核的是源码身份、断言方法和记录的hash/count，没有离线重算全部tensor。

| 实际案例 | 覆盖内容 |
|---|---|
| direct B2/B4 T19 | 所有位置logits、最后argmax、各层active KV/conv/S、offset及独立cache对象 |
| permutation/neighbor | 排列[2,0,3,1]，只改row1，其他row的logits/cache与完整后续IDs保持独立 |
| greedy B1/B2/B4 T19 | 正常EOS设置下每请求完整32个实际IDs；包括排列和改变邻row后的完整续写 |
| B2 T129 | 一次完整grouped prefill跨过GDN内部64-token块，再做31次decode；不是scheduler分块接纳 |
| mixed与drop | decode1+fresh3、decode2+fresh2；cap[1,3,5,32]结束、压紧、ahead与最终清空 |
| fallback | 异长、budget16下partial/continuation、recorded cache；spec仅核对captured-prefill选回旧路径 |

17例包含一个续写独立性的汇总案例和一个spec选择案例，它们各新增0个tensor比较；不把17例说成17次完整生成。没有测 drafter算术，未认证任意输入长度/dtype/backend或inactive KV padding/capacity。

数值核对期间采样为电池28%→23%、thermal0；其后另有AC24%充电/thermal0记录，不能把数值运行改称AC测速。pressure采样均为1、swap2190.50MB稳定、Swapouts增量0，但全程系统Swapins312、Pageouts631；不是零paging或某引擎独占资源的证明。

本章整合只做CPU语法、AST边界与25文件SHA核对，复用同一受测源码的实际报告，没有重复运行GPU。前一个ops tag没有单独重跑这17例。

## 6. prefill阶段变快，完整请求仍要另测

此前同一loaded model的V5→build3交错诊断，各3次预热、3次测量，得到下面的materialized prefill时间。它的parent含另一套decode候选，不是本章accepted13组合的正式HTTP对比：

| B | 原独立prefill中位数 | grouped中位数 | 时间变化 |
|---|---:|---:|---:|
| 1 | 175.791ms | 176.085ms | +0.17%，近持平 |
| 2 | 350.676ms | 330.035ms | -5.89% |
| 4 | 745.692ms | 602.204ms | -19.24% |

这说明该记录中的prefill阶段有收益。完整core请求的B4波动明显，不能用prefill的19.24%宣布端到端稳定提速，也不能据此宣布超过oMLX。正式AC、fresh-process、交替配对的C4 HTTP记录正在单独取得；本版证据标为pending，不把旧电池或诊断点移过去。

[compact证据](../../tutorial_code/examples/chapter14-batch-prefill.json)保存25源码hash、原始报告/脚本SHA、17例计数与scope、资源摘要和prefill三对原始时间。完整raw留在原实验目录，不把88k比较记录全部复制进教程。

## 7. 用三个版本阅读真实改动

```bash
# 模型/context/helper具备完整grouped路径；runner仍使用旧入口。
git diff tutorial-13-v1 tutorial-14-batched-ops -- tutorial_code/miniengine
# runner启用资格分组、映射与fallback。
git diff tutorial-14-batched-ops tutorial-14-batch-prefill -- \
  tutorial_code/miniengine/engine/model_runner.py
# 文档和compact evidence；核心源码与上一个tag相同。
git diff tutorial-14-batch-prefill tutorial-14-v1 -- docs tutorial_code/examples
```

完整生成仍使用第十一章的 `tutorial_code.benchmark_core` 入口；GPU命令需在具备相应模型与MLX环境时运行。固定tag用于查看阶段代码，最终证据绑定本章完整六文件组合，不能让中间版本继承不存在的测速。

本章的计算组织由现有Attention/GDN与cache接口推导，量化guard参考固定MLX源码。第十二章已有oMLX流水线和Lithos列所有权出处继续保留；没有把它们的其他实现或硬件配方作为本章新优化来源。
