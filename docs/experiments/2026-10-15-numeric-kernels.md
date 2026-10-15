# 数值内核实验档案（2026-10-15）

此档案从 accepted15 `f976d6a` 分叉；主教程仍采用第15步，没有新整体胜出结论。
本分支仅存纯函数与证据，未接入实际引擎正常路径，也没有默认启用入口。
完整路径、SHA256、参考实现与控制原值见 [结果元数据](2026-10-15-numeric-results.json)。

## 残差加法与 faithful norm

目标是合并一次算子边界；以下是拟议调用边界，未替换引擎代码：

```python
# before：独立 BF16 加法，再沿用 oldnorm 的约定
r = x + delta
n = oldnorm(r, w, eps)
# after：实验纯函数，同时返回两个输出
r, n = residual_norm(x, delta, w, eps)
```

[纯函数](../../tutorial_code/miniengine/kernels/residual_norm_metal.py) 按已验证原文件逐字节复制。
其原始文件头保留准备阶段文字；当前数值结论以本档案与 sealed proof 为准。
范围为 BF16 `[1,5120]` 两输入、BF16/FP32 `[5120]` 权重、三个已测 eps；MLX 0.32.3。
实际 330/330 例的残差与归一化输出逐 uint16 位相等，含 NaN/Inf payload/sign，输入只读 PASS。
独立 CPU replay 已核验原始字节；不代表全模型、物理非别名或速度资格化。

## 满前缀 SDPA 去掉全真 bool mask

仅已证明的普通 B1、满有效前缀 `lens == L`：D256、GQA6、L19..50，BF16。
Q=`[1,24,1,256]`，K/V=`[1,4,L,256]`，固定 scale=0.0625。
以下是合成实验的 before→after 示意；实际引擎仍保留原 mask 路径：

```python
# before：本范围内 mask 全真
mask = mx.arange(L)[None, :] < lens[:, None]
y = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.0625, mask=mask)
# after：仅上述已验证条件的实验调用
y = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.0625, mask=None)
```

实际 352 banks / 704 calls 逐位相等，含全部 64 个非有限输出 bank，输入只读 PASS。
该结论不覆盖 padding、非满前缀、其他形状/stride/runtime；未接引擎、未全模型资格化。

## 三次无效控制各自保留

这些是局部 host-wall 同臂控制，单位 µs/call；不能跨算法计算收益，也不是 GPU time。

| 独立尝试 | 控制原值 before→after | 变化 | 判定 |
|---|---|---:|---|
| residual build1，warm512/repeat2048 | AA 108.684610840→114.285197266 | +5.153062961% | INVALID |
| residual build2，唯一 long-window，warm2048/repeat8192 | AA 109.155283569→115.388178467 | +5.710117453% | INVALID |
| SDPA build1，warm2048/repeat8192 | AA 120.176681519→125.598007202；BB 100.803359985→114.937235596 | +4.511129460%；+14.021234622% | AA PASS；BB INVALID |

三次均为零 AB/BA candidate pairs；后置控制与漂移未完成，无速度结论；不 pool、不挑赢家、不再重试。
各次 before/finally 数值与只读审计 PASS，numeric-after 未到达；不可包装成完整性能 PASS。
残差 build2 driver=-15、cleanup=null，明确 NOT_ATTESTED；strict process absence 已证明 GPU 释放。
SDPA driver cleanup=true，owner52756/driver52767 strict absence 已证明 GPU 释放。
完整 27B model-ready 仍因 pressure2 HOLD；Chrome 先前取消退出，不擅自关闭应用或改变资源状态。
