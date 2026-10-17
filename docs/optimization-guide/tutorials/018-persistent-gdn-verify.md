# E018：在短验证块内保留 GDN 状态

2026-10-17，M5 Ultra 256 GiB。阶段：CPU 准备，未 GPU 资格；独立 micro 模块，默认核心不使用。

## 假设与范围

原生 `gdn_metal.gated_delta_step_metal` 每 token 读旧 S、写新 S。若已知整个验证块的 q/k/v/g/beta，在一个 threadgroup 中串行处理 T=3/5/8，可以将每线程负责的 32 个 FP32 状态元素保留在寄存器，并省掉相邻 token 之间的重复状态读取与 kernel dispatch。为支持严格的前缀 commit，仍写出每个位置的完整 S；不以压缩、重建或改变递推公式换速度。

27B 的每层状态为 48×128×128 FP32，即 3 MiB。48 个 GDN 层保留 T 个前缀约需 T×144 MiB；T=8 约 1.125 GiB/请求。256 GiB 内存允许先探索这种存储换计算方案。该内存预算只是结构推算，不是已测整模驻留量。

本实验只比较“原始同一 helper 串行 T 次”与“一个 kernel 内串行 T 次”。它**不证明**当前 chunk-WY verifier 与普通 decode 等价，也不修复 M4 正在诊断的 QMM 几何或整个投机事务问题。没有接入 `SpecRunner`、model 或 cache。

## 保留的算术

候选限定 B=1、H=48、Dk=Dv=128。保留原 kernel 的 SPLIT=4、VALUE_TILE=32、q/k norm 循环、分段 kv 累加、四段固定顺序归约，以及逐元素 `s = exp(g) * old_s + ks * k * delta` 和输出归约。每步 s 仍为 FP32，并写入 literal history。

新增的唯一同步边界是相邻 token 之间的 threadgroup barrier：否则 sp>0 的线程可能在上一 token 的 sp0 读完 red 之前覆盖共享归约内存。最后一 token 不添加这个额外 barrier。代码表达式相同不自动证明 compiler 的寄存器分配、FMA 或循环处理仍一致，必须做逐字节设备资格。

## 冻结的第一轮实验

`bench/m5_ultra_256gb/gdn_verify/screen.py` 从实际 target 第 0 个 GDN 层取得 q/k/v/g/beta。先用固定 19-token prompt 建立真实初始 cache，再给首层输入归档 continuation 的 8 个 token；因为首层没有上游 decoder 层，其输入不需要重复执行全模型。此步骤不声称是 fresh 整模 greedy rollout。输入以保留 dtype 的 safetensors 冻结，记录 source/模型 SHA 后卸载目标权重，再运行 micro；结束时重验 source、workload 和模型文件 SHA。

12 个资格点为 T∈{1,3,5,8} × initial S∈{真实、全零、独立随机}。所有位置输出、所有位置 FP32 S、每个 keep=0..T 从同一初始 S 独立重跑的前缀，必须逐字节一致；initial S 必须保持不变。保留所有失败，不使用 allclose。任何资格点失败就不计时。

通过后只对真实初始 S 的四个 T 计时：固定 2 GiB allocator pool，记录首个资格后调用，2 warm/arm，3 ABBA blocks。reference 保留原 helper 输出的数组列表，不额外 stack 复制历史来人为放大对照成本。全部原始计时入盘。

报告分别记录输入准备阶段的峰值、释放 target 后 active memory，以及重置峰值计数后的 micro 峰值，避免将加载权重占用误写成 kernel 驻留需求。资格阶段的 history/临时状态引用在计时前释放并清理 allocator cache，另记计时前 active memory。CPU 检查只覆盖固定 shape/dtype 的拒绝路径、launch 参数和 `--help` 无 GPU import；它们不证明 Metal 编译或数值正确。

进入下一轮的门：T=3/5/8 加速几何平均至少 1.25×，T=1 至少 0.97×，任何长度不得低于 0.97×。这只是单进程 micro 初筛；通过后仍需 fresh process 重复、真实多层输入与整模事务资格。

## 论文阅读与不能套用的部分

[Persistent-State Dataflow Accelerator v1](https://arxiv.org/html/2603.05931v1)，首次 2026-03-06，读 §III–IV。论文用 FPGA BRAM 让 GDN 状态跨 token 保留，并讨论状态读写与算术占比。这里只借用状态驻留的思路；M5 threadgroup/register 生命周期、容量和带宽都不同，未复现其加速，也不采用其改写的输出公式。

[Evaluating Losslessness Under Finite Precision v2](https://arxiv.org/html/2609.15504v2)，v2 2026-09-27，读 §1、§3.5、§4.2。作者在 Orthrus/Qwen3-1.7B 上观察到低精度下的轨迹差异，说明算法证明与实际 token 一致性应分别验证；它未定位本机误差来源，不能拿来放宽本实验逐字节门。

已有 TreeWY 阅读表明其三角重建改变计算表达，不能作为本实验 exact 性质的依据。
