# 07：让权重更小——接入量化 checkpoint

前面用原始权重的 fp32 版本核对数学。比较 oMLX、运行更大模型时，需要读懂 MLX 转换过的量化权重。

| 之前 | 本章之后 |
|---|---|
| Linear 直接读取浮点矩阵 | Linear 也能读取 packed 权重与 scale |
| 只支持原始 checkpoint | 支持本地 MLX 格式的量化 checkpoint |
| 与 HF fp32 参考核对 | 量化结果与**同一个量化 checkpoint**的参考核对 |

代码版本：`tutorial-07-v2`（原 `tutorial-07` 保留）；上一个版本是 `tutorial-06`。主要变化在 [layers.py](../../tutorial_code/miniengine/model/layers.py) 与 [loader.py](../../tutorial_code/miniengine/model/loader.py)，参考原 [minivllm 层](../../minivllm/model/layers.py) 和 [权重加载](../../minivllm/model/loader.py)。

## 1. 量化在省什么

一个 fp32 权重用 32 bit。若主要矩阵的一个权重只用 4 bit，可以把 8 个值打包进一个 32-bit 存储单元。

```text
浮点矩阵：  w0       w1       w2       ...   每个值单独占 32 bit
4-bit 打包： [q0 q1 q2 q3 q4 q5 q6 q7]         合起来占 32 bit
附加信息：  每组的 scale，某些格式还有偏移
```

这减少权重存储与读取流量，但还需要 scale 等附加数据，也有保持浮点的参数。不能用“参数量×0.5 字节”当作整个运行进程的准确内存。

一个只说明思想的玩具量化：若 `w≈s×q`，`s=0.1`，则 `[0.2,0.5,0.9]` 存为 `[2,5,9]`。还原后仍是这三个值。`0.53` 只能近似为 `0.5`，精度损失由此出现。

这个例子采用简化的整数表达。**NVFP4 的具体表示不等于上面的整数公式**，不能把 affine 的 q、scale、bias 规则套到 NVFP4。我们把 checkpoint 中的 `mode/bits/group_size` 原样交给支持该格式的 MLX 算子。

## 2. 原本的 Linear 如何改变

第二章以及原始 HF 的 Linear，本质是：

```python
y = x @ weight.T                                # weight 是浮点 [out,in]
```

我们新增另一条分支：

```python
if self.q is None:
    return x @ self.weight.T                    # ← 关键① 保留原始浮点路径
return mx.quantized_matmul(
    x, self.weight, self.scales,
    getattr(self, "biases", None), transpose=True,
    group_size=self.q["group_size"],
    bits=self.q["bits"], mode=self.q["mode"],     # ← 关键② 匹配 checkpoint 表示
)
```

不先把整个大矩阵还原成 fp32，再做普通 matmul；那会丢掉量化存储的内存优势。矩阵乘本身仍由 MLX GPU 算子执行，教学代码负责参数与格式的正确组织。

形状示例：`in=128,out=64,bits=4,group_size=16`。

```text
浮点 weight：     [64,128]
packed weight：   [64,16] uint32       # 128×4/32 = 16
scales：          [64,8]              # 每 16 个输入维度一组
```

scales 的存储类型也取决于格式；不要假定它永远是 fp32。

## 3. Embedding 与输出头也要认识量化

读输入 Embedding 时，只需要本次 token 对应的行。以下是省略格式参数的阅读节选：

```python
selected_weight = self.weight[ids]
selected_scales = self.scales[ids]
x = mx.dequantize(selected_weight, selected_scales, ..., mode=self.q["mode"])
                                                   # ← 关键③ 只还原用到的行
```

若 `tie_word_embeddings=True`，同一份 Embedding 权重还承担输出头。输出头需要对整个词表算分数，应使用量化矩阵乘；不能只还原输入 token 那几行。

```text
输入：少量 IDs → 取这些行 → dequantize
输出：hidden → quantized_matmul 全词表 → logits
```

同一个权重对象承担两个方向，正是 `Embedding.__call__` 与 `Embedding.as_linear` 的区别。

## 4. 加载器的一个隐蔽细节：Norm 的 +1

原 HF 的零中心 RMSNorm 计算：

$$y=\operatorname{normalize}(x)\,(1+w_{HF}).$$

MLX 转换文件里的这些 norm 权重已经包含 `+1`，即 `w_MLX=1+w_HF`。教学类仍沿用 HF 的 `1+w` 公式，所以 loader 要减回去：

```python
if converted_mlx and is_zero_centered_norm(name):
    value = value.astype(mx.float32) - 1.0       # ← 关键④ 还原存储约定
```

小例子：HF 保存 `w=0.2`，实际倍率是 1.2；转换文件保存 1.2。如果不减 1 就加载进我们的类，倍率变成 2.2，模型会算错。

上面是条件的简写，源码依据文件名前缀识别转换格式，并只对明确列出的 norm 做变换。GDN 输出的普通 gated norm 没有这个 `+1`，不能一律减 1。名字映射、卷积布局与 strict 加载沿用第五章。

## 5. 本章 diff 与运行入口

```diff
  class Linear:
-     return x @ self.weight.T
+     if self.q is None:
+         return x @ self.weight.T
+     return mx.quantized_matmul(x, packed_weight, scales, ..., mode=mode)

  # loader
+ if converted_mlx and zero_centered_norm:
+     weight = weight - 1.0
```

```bash
python -m tutorial_code.check_quantized
python -m tutorial_code.generate_mlx \
  --model-dir /Users/ryanli/code/minivllm/models/Qwen3.5-0.8B-nvfp4 \
  --dtype bfloat16 --max-new-tokens 8
git diff tutorial-06 tutorial-07-v2 -- tutorial_code/miniengine/model
```

`check_quantized` 默认使用本地 0.8B NVFP4 checkpoint，结果见 [chapter07-checks.json](../../tutorial_code/examples/chapter07-checks.json)。参考是 [同 checkpoint 的 MLX-LM 记录](../../tests/data/mlxlm_0.8b_nvfp4.npz)，不是原 HF fp32 的输出。

检查同时记录 logits 误差、argmax 一致率与 greedy 共同前缀长度。不同 backend 的浮点计算顺序也可能影响接近并列的最大分数，因此不能由前 8 token 相同推出以后所有 token 都相同。

加载器也可用于符合该配置与权重格式的大模型。0.8B 验证不等于已经验证任意 27B checkpoint；是否实际运行以及相应模型路径，要看后续单独保存的运行报告。

现在完整生成可以使用量化模型。下一步让多个请求共用一次模型执行，开始写引擎的调度部分。

[下一章：08 调度与批量](08-scheduler.md)

## 本机验证记录

完整结果见 [chapter07-checks.json](../../tutorial_code/examples/chapter07-checks.json)。这是本章代码实际运行的结果：

```json
{
  "chapter": "07",
  "baseline": "mlx-lm same nvfp4 checkpoint reference",
  "quantization": {
    "group_size": 16,
    "bits": 4,
    "mode": "nvfp4"
  },
  "max_abs_error": 1.04833984375,
  "prompt_argmax_agreement": 1.0,
  "greedy_prefix_match": 11,
  "greedy_tokens_checked": 12,
  "generated_ids": [
    79852,
    18887,
    220,
    95761,
    21966,
    220,
    95789,
    98618,
    99580,
    97792,
    97989,
    96172
  ],
  "note": "quantized outputs are not required to equal raw HF fp32"
}
```

## 修复：量化权重和激活 dtype 是两件事

在 `tutorial-07`，`load_model(dtype=mx.float32)` 转换了浮点参数，却没有转换量化 Embedding 解出来的向量。`mx.dequantize` 在这个 checkpoint 上默认产生 bfloat16，因此激活和 logits 仍是 bfloat16。第八章做批量验证时发现了这个接口问题。

`tutorial-07-v2` 明确保留 packed 权重与 scale 的整数格式，同时让 Embedding 的输出服从所请求的浮点 dtype：

```diff
+ model.embed_tokens._output_dtype = dtype
- return mx.dequantize(packed_rows, scales, ...)
+ return mx.dequantize(packed_rows, scales, ...).astype(self._output_dtype)
```

上面是阅读节选，完整参数在源码中。float32 和 bfloat16 两种设置都已检查 Embedding/logits 的真实 dtype；packed 权重仍为 uint32，NVFP4 scale 仍为 uint8。结果见 [dtype-checks](../../tutorial_code/examples/chapter07-dtype-checks.json)。默认 bfloat16 下的量化参考结果仍是前 11/12 token 匹配，不把这一修复解释成跨实现完全一致。

```bash
git diff tutorial-07 tutorial-07-v2 -- tutorial_code/miniengine/model/layers.py tutorial_code/miniengine/model/loader.py
python -m tutorial_code.check_quantized --dtype bfloat16
```
