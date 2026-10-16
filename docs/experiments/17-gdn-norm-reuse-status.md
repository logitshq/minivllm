# V19：GDN norm reuse 独立实验状态

本分支保存实验源码和已核验的证据；V19 default OFF、未采用，教程章节尚未完成。
本机 build3 已独立核验 FULLMODEL_QUAL_PASS；E2E actual=0、待测，未获得默认启用或性能晋级资格，未证明模型收益或相对官方引擎胜出。
完整模型资格测试的 accepted15 基线 `f976d6a547a9dc1946b52b1de6db376d799b34d7`；实验源码提交 `b158c0fbf95b84fb3567e0039b0ab1ffb1a46bb6`。
本次更新状态、结果 JSON 和 256GB Mac 交接包；26 个实验源码逐字不变，不合入教程主分支。

## 已完成的局部核验

- 独立 CPU source review：SOURCE_CPU_PASS；default OFF、严格布尔参数、normal eligibility 和 fallback 已核验。
- 纯 kernel 15 banks（9 real + 6 synthetic）严格数值 PASS；前、后、释放前数值及只读审计 PASS。
- saved15 局部 host-wall micro：A=V17 decode isolated public 片段（nn.silu→split/repeat→原 gated_delta_step_metal），148.959340 μs；B=N2 public 两 kernel（norm-reuse 源码同 V19），129.099457 μs；降低 13.332419%，10/10 pairs 胜；A 不是 V18。
- 上述 PASS 和 13.33% 只属于局部算子，不是完整模型收益或 E2E 吞吐结论。

## 两次旧完整模型资格测试分别为 STOP（历史保留）

- build1（旧 AC Power）：accepted load 完成并同步后，model-ready guard pressure=2，token 0 前 STOP；candidate 未启动。
- build2（新 Battery Power，Chrome 正常退出后）：accepted load 1 次并完成同步，model-ready guard pressure=2、thermal=0，token 0 前 STOP。
- build2 accepted normal forward=0、candidate 未启动/forward=0、comparison rounds=0；未开始数值或状态比较。
- 这是资源守卫 STOP，不是 numeric FAIL，也不是 fullmodel PASS；两次电源域证据分开，不合并。
- build2 独立原始证据审计 AUDIT PASS / QUAL STOP；原 owner/driver/worker STOP 与 `GPU_released=false` 保留。
- build2 owner 58471、driver 58484、accepted 58498 均 wait/reap，exit=1；实际工具返回后外部 strict ps 均 rc=1、stdout/stderr 字面空。
- 后续 GPU_RELEASED 由进程退出和三 PID strict absent 证明；不宣称进程内卸载成功。
- accepted finally 的 pressure warning cleanup 保留，driver/owner cleanup errors 为空；首个 STOP 与 cleanup 分开。
- 两次旧 STOP 的原始数字和 proof SHA 永久保留；本次文档归档不授予新的资源 GO。

## build3 完整模型严格资格 PASS

- 用户已明确恢复本机测试；build3 在实际 AC Power 域完成 accepted/OFF/ON/reset-OFF 四轮，每轮 32 raw IDs 严格一致。
- 128 份保留的 FP32 logits 逐字比较通过；16,384 state signatures、完整 cache/observation signatures 和 18 个独立 fallback 严格通过。
- ON 六条生产路径每路实际 48 layers × 31 = 1,488 次调用；prework/state 各创建一个真实 kernel；OFF/reset-OFF 走 legacy。
- 每 worker 1,349 个参数 fingerprint 在所需边界稳定；跨 worker 比较 key/shape/dtype 等稳定字段。object_id 仅在各 worker 内部稳定；receipt 没有参数 raw-byte hash，不能声称全权重逐字节比较通过。
- 22 份原始资源快照均 AC、pressure=1、thermal=0、swap=7571.62MB，Swapouts delta=0；Pageouts 63979→64543（+564），原 guard 未要求零 Pageouts。
- owner/driver/worker 8222/8236/8249/8407 在实际工具返回后严格 PID absent；driver 和两个 worker wait/reap exit=0。原 owner/driver GPU_released=false 保留，外部 GPU_RELEASED 是单独释放边界。
- 独立结果审查 FULLMODEL_QUAL_PASS；[审查文本](../handoffs/256gb-mac/FULLQUAL-BUILD3-REVIEW.md) SHA256 `dfc5c77df0f28c45d178de45c64b744235538440a6f2cf6eb2fe52c0c41533cb`；[审查收据](../handoffs/256gb-mac/FULLQUAL-BUILD3-REVIEW-RECEIPT.json) SHA256 `af7b23397e16d615d19f8de4704d5c57d038797940d37bc3597c3056d6b78177`。
- 这只证明本次有限 fullmodel 正确性；E2E actual=0、待测，整体 win=0，default OFF、不采用、不合并 main。

## Chrome 与 E2E 的边界

- 旧暂停已由用户随后明确恢复测试撤销；当前完整模型资格为 build3 PASS，后续实际运行仍须读取最新运行状态并遵守单 owner/资源 guard。
- Chrome 已成功正常退出一次；观察到系统物理 free 增加、压缩器物理占用下降，不能全部归因于 Chrome。
- RSS 降幅不等于回收物理 RAM；系统资源改善不保证 27B load 或稳态安全，不能由资源改善单独推断资格；最新资格由 build3 独立结果审查给出。
- E2E build2：CPU 27 checks、1421 frozen pins；仅 CPU_BIND_READY_QUAL_PENDING，尚未独立审查、实际 E2E=0，不可采用。
- build2 结果审计只核验过 E2E gate 拒绝实际 fullqual STOP，不构成 E2E harness 的独立审查或执行。
- 原 E2E 准备、load-only diagnostic 和原局部算子结果均不能替代完整模型资格 PASS。

## 精简证据索引

- [代码讲解与静态边界](17-gdn-norm-reuse-walkthrough.md) 说明源码、数学、shape 和成本，仅解释源码；完整模型资格见上方 build3。
- [结果 JSON](17-gdn-norm-reuse-result.json) 保存必要 metadata、原始 proof 路径及 SHA256，不复制大 raw。
- Source manifest SHA256 `d86be282646c1fc9d26155ad54014cf461f608c8222e32cd66d11088e180a32e`（26 sources）。
- 原局部算子 receipt SHA256 `999ced1e1d1c8facb1db1c61f97b062b66afabff8a1f774605360fb354cbdfd4`。
- build1 STOP receipt SHA256 `59dac27678206422a00e6ac8553e08b3479ce64b70fd584f36223175f917b070`。
- build2 STOP receipt SHA256 `970d402bbbee0fabde5bc3b9f8b28effc2a6e56505bb53c171d80f72851dbafc`。
- build2 release SHA256 `818e8ce9c0b4d146db6f93a699de2021137c5ad48e8e80b79d8eae46b804bfe8`。

- [256GB Mac 交接包](../handoffs/256gb-mac/README.md)：约 4MB 源码/历史参考压缩包；目标机 harness 迁移与重新资格测试仍待完成。
