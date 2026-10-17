# E018：在短验证块内保留 GDN 状态

2026-10-17，M5 Ultra 256 GiB。阶段：单进程 micro 初筛通过，三个 fresh process 的整体重复性门未通过；默认核心不使用。

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

## v1 实测与重复性预注册

初筛12 cases、165个输出/状态/前缀比较全部exact。T1/3/5/8分别为0.982453/1.199745/
1.219095/1.578766×，T3/5/8几何平均1.321746×；原始48正式、16warm、8首调用全部保留。
目标卸载后active 8 bytes，计时前3,443,720 bytes，micro峰值93,329,576 bytes；
输入准备峰值15,492,008,675 bytes。子进程exit0/reap，服务恢复S。没有整模性能结论。

预注册固定再跑**三个**独立进程，使用相同screen/candidate、同输入生成过程与2 GiB池、
原warm/ABBA顺序及同样的门。初筛单独呈现，不与复测池化；每个复测都必须独立通过，
否则不宣称重复性通过，不补跑挑选。原始资格和所有计时不省略。此轮只检验单层重复性，
多层、非连续输入、整模状态事务尚待下一项独立资格。

## 论文阅读与不能套用的部分

[Persistent-State Dataflow Accelerator v1](https://arxiv.org/html/2603.05931v1)，首次 2026-03-06，读 §III–IV。论文用 FPGA BRAM 让 GDN 状态跨 token 保留，并讨论状态读写与算术占比。这里只借用状态驻留的思路；M5 threadgroup/register 生命周期、容量和带宽都不同，未复现其加速，也不采用其改写的输出公式。

[Evaluating Losslessness Under Finite Precision v2](https://arxiv.org/html/2609.15504v2)，v2 2026-09-27，读 §1、§3.5、§4.2。作者在 Orthrus/Qwen3-1.7B 上观察到低精度下的轨迹差异，说明算法证明与实际 token 一致性应分别验证；它未定位本机误差来源，不能拿来放宽本实验逐字节门。

已有 TreeWY 阅读表明其三角重建改变计算表达，不能作为本实验 exact 性质的依据。


## 三 fresh process 结果与独立审计

[完整初筛和复测归档](../../experiments/m5-ultra-256gb/E018-persistent-gdn-verify-v1/README.md)
已通过独立 CPU 审计；按原预注册，**整体重复性未通过，不进入生产采用**。
三个复测各自的完整结果如下，初筛保持单独呈现，未池化或补跑挑选。

| 复测 | T1 | T3 | T5 | T8 | GM T3/5/8 | 单进程门 |
|---|---:|---:|---:|---:|---:|---|
| 1 | 1.023888× | 1.220317× | 1.461622× | 1.530151× | 1.397486× | PASS |
| 2 | 0.939084× | 1.270615× | 1.413632× | 1.478124× | 1.384694× | REJECT |
| 3 | 1.066419× | 1.256273× | 1.483812× | 1.504110× | 1.410092× | PASS |

第 2 次复测的 T1 为 0.939084×，低于 0.97× 门，即使该进程 T3/5/8 GM 超过 1.25×，
仍必须拒绝整体重复性。T≥3 的三个复测点全部有收益，是后续独立假设的线索，不替代本次失败门。

四进程共 48 cases、660 个输出/FP32 history/keep-prefix 比较全部 exact；独立审计逐项核对
shape、dtype、reference/candidate SHA、keep 与 history 对应关系、各 T 的共同前缀，
并直接解析四份 safetensors 验证 BF16/FP32 类型、offset、完整输入 SHA 与跨进程 tensor bytes。
三个不同 initial-state 的 hash 也不同。CPU 审计没有重新执行设备数学。

所有 192 formal、64 warm、32 first-post-qualification 样本全部保留，ABBA 顺序和 median/speedup
重新计算。被测 34 份源、workload、模型 metadata、两轮 lease/jobs 和所有资源/owner 记录入档。
模型大权重由每个 screen 前后校验；独立审计没有重复读取大 payload。188 个 archive members
的大小与 SHA 全部通过。四个子进程全部 exit 0、wait/reap 与 strict absence 通过，两次服务恢复 T→S。

下一步若仅研究 T≥3，必须另立新的实验范围与资格，不能追改本实验的 T1 门。
多层、非连续输入、整模 cache/rollback 事务以及 HTTP 性能仍未获此实验的证明。
