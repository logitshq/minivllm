# 06：只计算新 token——KV Cache 与 GDN 状态

第五章已经用自己的模型生成，但每轮都从头计算全部前缀。本章保存中间状态，让后面的轮次只输入最新 token。

| 之前 | 本章之后 |
|---|---|
| 输入长度逐轮为 `19,20,21,...` | 首轮输入 19，后续输入 `1,1,1,...` |
| 每层重算历史 | Attention 保存 K/V；GDN 保存矩阵状态与卷积尾巴 |
| 仅整段 prompt | 同一条 prompt 也可以分块输入 |

代码版本：`tutorial-06`；上一个版本是 `tutorial-05`。主要变化在 [cache.py](../../tutorial_code/miniengine/cache.py)、[context.py](../../tutorial_code/miniengine/context.py)、两个 mixer 和 [generate_mlx.py](../../tutorial_code/generate_mlx.py)。参考原 [minivllm cache](../../minivllm/cache.py)。本章只处理单请求，多请求在第八章。

## 1. 为什么过去的结果可以复用

因果模型里，权重、位置和已计算前缀不变时，追加新 token 在数学上不会改变旧位置的 hidden states。旧 K/V、已经更新好的 GDN 状态都仍然有效。换一条 prompt 或改动旧 token 时，不能直接沿用这份 cache；本章为新请求重新调用 `make_cache(model)`。

```text
无 cache： [A B C] → D；[A B C D] → E；[A B C D E] → F
有 cache： [A B C] → D；      [D] → E；        [E] → F
            存状态             接续状态             接续状态
```

首轮把 prompt 填进状态，称为 prefill。后面逐 token 接着计算，称为 decode。它们共享模型权重与计算含义，区别在新输入长度和初始状态。

注意：生成了 D，不代表状态已经包含 D。首轮状态包含 `A B C`，其末尾 logits 预测 D；下一轮才把 D 输入模型，状态增长到 `A B C D`。

## 2. Attention cache：保存 K/V，位置接着往后

在 HF 的 [Attention.forward](../../ref/modeling_qwen3_5.py) 中，RoPE 之后原本有：

```python
if past_key_values is not None:
    key_states, value_states = past_key_values.update(
        key_states, value_states, self.layer_idx,
    )                                           # ← 关键① 追加，返回全部历史 K/V
```

我们自己的版本，省去形状转换：

```python
offset = cache.offset if cache is not None else 0
q = apply_rope(q, offset, self.rotary_dim, self.theta)  # ← 关键② 位置不能每轮归零
k = apply_rope(k, offset, self.rotary_dim, self.theta)
if cache is not None:
    k, v = cache.update(k, v)                    # ← 关键③ 新 K/V 写入，读有效历史
```

逻辑上的形状变化：

```text
prompt 三个 token 后： K/V [1, Hkv, 3, D]，offset=3
再输入一个 token：    新 K/V [1, Hkv, 1, D]
attention 实际读取：  K/V [1, Hkv, 4, D]，offset=4
```

缓存底层按一段容量增长，避免每个 token 都重新扩容；`offset` 才是有效历史长度，容量中多出来的位置不能参与 attention。

分块 prefill 还有一个细节。历史已有 3 个 token，本次输入 2 个 token，mask 是 **2 行对 5 列**：

```text
             旧0 旧1 旧2 新3 新4
新 query 3    ✓   ✓   ✓   ✓   ×
新 query 4    ✓   ✓   ✓   ✓   ✓
```

因果三角形要按右侧对齐。若写成从左上角开始的普通 2×5 三角形，新 query 就读不到正确历史。MLX 算子处理这个位置关系。

## 3. GDN cache：保存两种不同的状态

GDN 不存全体 K/V，保存的是：

| 状态 | 形状 | 表达什么 |
|---|---|---|
| `S` | `[Hv,Dk,Dv]`，fp32 | 前面所有 token 累积后的记忆 |
| `conv_state` | `[K−1,C]` | 最近的卷积前 QKV 投影 |
| `offset` | 整数 | 已经计算的 token 数 |

HF 的 GDN 也分别读 conv state 与 recurrent state。我们的单 token 接续主线是：

```python
conv_state = cache.conv_state if cache.offset > 0 else None
qkv, new_conv_state = causal_conv1d(raw_qkv, weight, conv_state)
S0 = cache.S if cache.offset > 0 else None
output, S = gated_delta_rule_recurrent(q, k, v, g, beta, S0)  # ← 关键④ 从旧 S 继续
cache.conv_state = new_conv_state
cache.S = S
cache.offset += T                               # ← 关键⑤ 各层接续相同新位置
```

完整实现对多 token 使用纯 MLX 的 chunk 计算，单 token 使用第四章的逐步递推。chunk 在实数数学上等价，但不保证浮点结果逐字节相同；第九章再展开它的数学和 Metal 三角求解优化。两种路径都接受同一个初始 S，返回末尾 S。

忘记 conv state 会让每轮卷积都误以为序列刚开始；忘记 S 会丢掉长期记忆。仅保存其中一个，通常仍能生成文字，但不再是同一个模型的输出。

## 4. 把 cache 接进完整模型

模型只需把一个 context 逐层传下去。以下是生成循环的关键节选：

```python
caches = make_cache(model)
ctx = ForwardContext("prefill", caches)
current = prompt_ids
for step in range(max_new_tokens):
    logits = model(mx.array(current), ctx, last_only=True)
    next_id = int(mx.argmax(logits[-1]).item())
    output_ids.append(next_id)
    current = [next_id]                          # ← 关键⑥ 下轮只喂最新 token
```

本章 `prefill` 指单序列的可接续前向路径，`T=1` 也可走它。第八章的 `decode` 模式才表示“一行对应一个请求”的批量路径，名字相似但形状含义不同。

```diff
- logits = model(mx.array(ids), last_only=True)
- ids.append(next_id)             # 下轮继续传全部 ids
+ ctx = ForwardContext("prefill", make_cache(model))
+ logits = model(mx.array(current), ctx, last_only=True)
+ current = [next_id]              # 全部历史已经在 ctx 中
```

## 5. 怎样确认 cache 真的正确

不能只看文字像不像。检查同一份权重、同一串实际 token IDs，分别用三种方式算：

1. 整个前缀无 cache 一次前向。
2. prompt prefill，再逐 token 接着算。
3. prompt 按 7 或 13 个 token 分块，带 cache 接着算。

对照各个位置的 logits、argmax 和每层 offset。分块跨过卷积窗口、Attention mask 以及 GDN 初始状态，能发现只测单 token 看不到的错误。

```bash
python -m tutorial_code.check_cache
python -m tutorial_code.generate_mlx \
  --trace-json tutorial_code/examples/chapter06-trace.json
git diff tutorial-05 tutorial-06 -- tutorial_code/miniengine tutorial_code/generate_mlx.py
```

结果见 [chapter06-checks.json](../../tutorial_code/examples/chapter06-checks.json)；逐轮输入长度见 [chapter06-trace.json](../../tutorial_code/examples/chapter06-trace.json)。检查包括固定 12 个接续输入的 logits 和两种 prompt chunk size。浮点运算组织变化会带来小误差，因此同时比较数值与 argmax。

现在完整生成已接入单请求 cache。本章没有请求调度、批量状态或投机回滚；下一章先让加载器认识优化引擎常用的量化 checkpoint。

[下一章：07 量化](07-quantization.md)

## 本机验证记录

完整结果见 [chapter06-checks.json](../../tutorial_code/examples/chapter06-checks.json)。这是本章代码实际运行的结果：

```json
{
  "chapter": "06",
  "dtype": "float32",
  "prompt_tokens": 19,
  "decode_tokens_checked": 12,
  "cached_max_abs_error": 4.1961669921875e-05,
  "chunk_sizes_max_abs_error": {
    "7": 7.486343383789062e-05,
    "13": 5.8650970458984375e-05
  },
  "integration": "generation uses single-request cache; no scheduler/batch yet"
}
```
