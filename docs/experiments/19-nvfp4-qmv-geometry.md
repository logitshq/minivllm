# NVFP4 QMV 几何实验：R4控制通过，R2未过收益门

2026-10-16，M4 Max 40 GPU / 48GB、MLX0.32.3。独立低层实验V22；`STOP-MICRO-GATE`，未采用，无fullmodel/E2E或AC前沿资格。

## 唯一改动与原路径

仅层0 `mlp.gate_proj`：BF16 x[1,5120]、U32 weight[17408,640]、U8 scales[17408,320]，payload=50,135,040B。原调用：

```python
y = mx.quantized_matmul(x, weight, scales, transpose=True,
                       group_size=16, bits=4, mode="nvfp4")
```

先手写wrapper调用官方fast QMV，R=4作控制；全部控制输入通过后，才把每SIMD输出行数改为2。两次均保留2个SIMD/group：8行/group→4行/group，2176→4352组。

```python
# before: R = 4（数值控制）；after: R = 2（唯一候选）
y = kernel(inputs=[weight, scales, x], template=[("T", mx.bfloat16), ("R", R)],
           grid=(64, 17408 // (2*R), 1), threadgroup=(64, 1, 1),
           output_shapes=[(1, 17408)], output_dtypes=[mx.bfloat16])[0]
```

Metal body调用 `fp_qmv_fast_impl<T,16,4,false,R>`；K/N为constant全局，tile.x固定0，sg=local/32、lane=local%32。保留官方fp4/fp8 decode和fast_impl原算术；未设置新math policy。

每行K5120含320个16值量化组。每lane取16连续值，32 lanes每轮覆盖512列，10轮完成一行；R不改变这些K工作，也不改变bits=4或group_size=16。完整头文件含无关GEMM依赖，原型把所需定义逐段原样展开，附MIT全文和SHA。

[MLX0.32.3 dispatch](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/quantized.cpp#L435-L503)只对M5 Max大N自动选R2；这提供假设，不能推断M4收益。[源码映射](19-nvfp4-qmv/source-map.json)及原研究保留MLX/Lithos出处，未搬用Lithos不同pack/算术。

## 数值、启动与计时边界

R4对原QMM：8个synthetic加1个历史真实L0 MLP输入，共9项严格BF16 byte通过；R2分别对QMM/R4再做18项，共27项通过。包含finite/shape/dtype检查；任一失败即停止且不计时。

真实x来自旧V17/gates ON首次normal decode，不是本轮capture；计时只轮换8个synthetic。27项是有限样本，不等于完整模型、所有输入或cache/lifecycle资格。

首次调用host wall另列（含编译/准备/执行，不是纯编译耗时）：原QMM 5.483916ms，R4 122.476541ms，R2 44.466667ms。全部在正式block外；没有用它们推算摊销收益。

A始终是原QMM，B是R2；相同prime128/repeat512，每call eval全部输出，两块完成后才hash。固定前AA/BB→3AB+3BA→后AA/BB，170s owned-child上限。

| 配对 | 顺序 | A μs/call | B μs/call | 降低 |
|---|---|---:|---:|---:|
| 2 | AB | 241.925619141 | 241.118246094 | +0.333727800% |
| 3 | BA | 240.009765625 | 241.182046875 | -0.488430647% |
| 4 | AB | 221.749755859 | 220.114419922 | +0.737469104% |
| 5 | BA | 221.978027344 | 219.326824219 | +1.194353854% |
| 6 | AB | 222.172769531 | 218.162433594 | +1.805052863% |
| 7 | BA | 241.406087891 | 241.085205078 | +0.132922419% |

六对中位A=231.091267578μs、B=230.599812500μs，降低0.212667092%，5/6胜。未达跑前固定的至少5%且5/6胜门，保留负对，结论STOP；不能称作稳定kernel提速。

AA slot2/slot1=0.999156080/1.013800621；BB=1.006487403/1.004035520。两类各两个控制全部保留；中位[0.98,1.02]、每对[0.95,1.05]门通过。

这是同步完整调用host wall，不是GPU kernel timestamp或token速度。24份资源快照电量均26%、Battery、thermal0、pressure1；SO/SI/PO/Compressions增量0、Decompressions+5，不与AC或正式服务前沿混算。

owner31180的child31181 wait/reap exit2；supervisor与root外部strict ps均rc1且stdout/stderr字面空。原始收据在压缩证据内，原临时目录未回改。

## 归档与停止条件

[证据目录](19-nvfp4-qmv/README.md)保存原result、完整10对、资源/启动/输入/byte摘要、退出收据、原型参考包、MIT与source map。独立结果review复算原统计与SHA并确认STOP；它未重建未保存输出。原型依赖本机路径、checkpoint、旧输入和资源工具，需适配，不是自包含发行包。

本轮不重跑、不同R sweep、不接入默认；26个引擎源码和V19 default OFF不变。[教学版](../tutorial-v2/19-把量化矩阵乘拆开看.md)解释逐层资格的作用。
