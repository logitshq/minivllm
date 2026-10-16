# 输出行拼接评估：两条局部路线均未采用

2026-10-16，M4 Max 48GB / MLX 0.32.3。当前教学引擎的 26 个源码文件逐字不变，V19 default OFF；本章只归档独立低层实验。

## 候选改动到底是什么

GDN 四个 Linear 读取同一个 `x`，原顺序为 qkv/z/b/a；MLP 的 gate/up 也读取同一个 `x`：

```python
qkv, z, b, a = (op(x) for op in gdn_ops)  # 原四次 QMM
# 或：gate, up = gate_proj(x), up_proj(x)   # 原两次 QMM
```

实验的 generic helper 只沿输出行连接原 NVFP4 存储；以下是省略 guard 和事务回滚的教学节选，未接入当前引擎：

```python
weight = mx.concatenate([op.weight for op in ops], axis=0)
scales = mx.concatenate([op.scales for op in ops], axis=0)
mx.eval(weight, scales)                   # ← 先完成一次加载期复制
# 原 Linear 随后重绑为 slab 的连续行 slices，供独立 fallback 使用。
y = mx.quantized_matmul(x, weight, scales, None, transpose=True,
                       group_size=16, bits=4, mode="nvfp4")
boundaries = [10240, 16384, 16432]         # MLP 是 [17408]
outputs = tuple(mx.split(y, boundaries, axis=-1))
```

K=5120，weight 为 U32[N,640]，scales 为 U8[N,320]。GDN 的 N 为 10240/6144/48/48，合计16480；MLP 为17408/17408，合计34816。不解量化、不重新量化、不补行、不转换 scales。

存储字节保持只说明权重表达未变。N 改变可能改变 QMV/QMM 的 tile、split-K、累加次序和 BF16 舍入，所以仍须逐输出比较 finite、shape、dtype 和原始 BF16 bytes。字节相同仍不说明更快：读权重总量没有减少，host 构图、分派、split 与硬件状态都计入真实调用成本。

## 结果及不可越过的边界

下表 A 为原独立 Linear，B 为拼接后一次 QMM；时间是一次调用覆盖所选3层所有输出的 host wall 中位数，不是单个 kernel 或 token 延迟。

| 实验 / 电源域 | A μs | B μs | 降低 | B 胜对数 | 结论 |
|---|---:|---:|---:|---:|---|
| 旧 GDN AC attempt2 | 893.9583125 | 697.8281250 | +21.9395% | 3/5 | 保持旧 STOP |
| v20 GDN Battery | 466.4866484 | 465.7998047 | +0.147238% | 4/6 | STOP |
| v21 MLP Battery | 761.2115898 | 762.1476240 | −0.122966% | 3/6 | STOP |

旧 AC 的 AB 为3/3胜、BA为0/2胜；5/5均是第二个 block 更快。旧计时结构为 resource probe→首 block→CPU hashes→次 block，位置准备不对称；物理原因仍未知，不能拿21.9%当作收益，也不回改旧 raw 或旧 STOP。

v20 两条路径各 prime16/repeat64，固定6个 AB/BA 与前后各 AA/BB；hash 放到整对计时之后。120个 byte记录通过，输入仅8个 distinct synthetic（zero/cancellation/random amplitudes），没有新真实 norm 输入或完整模型调用。

v20 AA slot2/slot1 为0.988740/0.964418；BB为0.923016/0.881353，中位分别0.976579/0.902185。它们通过当时预声明的宽门（中位[0.90,1.10]，各对[0.80,1.20]），仍显示位置偏差；5% median、至少5/6胜的性能门未过。

v21 是独立新方向，跑前固定 prime128/repeat512，控制门收紧为中位[0.98,1.02]、各对[0.95,1.05]。AA为1.003758/1.000395；BB为0.995425/0.994880，均通过；性能门仍未过。没有用新门重判 v20 或重跑 GDN。

v21 的66个 byte记录通过：8个 synthetic 加每层1个历史真实 MLP输入，再核验B2/B4重绑 fallback。真实输入来自旧V17/gates ON首次 normal decode，三个 BF16[1,5120] 文件各10240B，hash/来源已核对；计时仍只用 synthetic，不冒充新完整请求。目标权重只读取0/32/63层 gate/up 的所需tensor，payload=300,810,240B；未加载27B整模型。

两次新实验各27份资源快照均 Battery、thermal=0、pressure=1、Swapouts增量0。v20电量32→32%，v21为30→29%；电源域不与旧 AC 混算。owned child 29080/29827各 wait/reap exit2，控制器与root外部 strict ps均 rc1且stdout/stderr字面空；exit2表示正常完成性能 STOP。

## 归档与当前决定

- [证据目录](18-projection-packing/README.md) 保存两个原 result、全量配对/控制/资源等小型压缩证据、CPU检查、独立 replay 收据和原SHA清单。
- v20/v21 独立 replay 均确认 `STOP-MICRO-GATE`；仅重算已有统计与核对元数据/hash，不声称重建未保留的输出张量。
- [教学讲解](../tutorial-v2/18-先确认优化真的更快.md) 解释为何通过正确性后仍应停止。
- 两方向均不再重跑、不采用；无新 fullmodel、cache/lifecycle 或 E2E 资格，没有相对 oMLX/Lithos 的新胜出结论。
- selected3 的 active-memory下降是局部allocator观察，不等于物理buffer共享、全部48/64组常驻内存或模型收益证明。
