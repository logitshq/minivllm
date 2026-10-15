# V19：GDN norm reuse 独立实验状态

本分支只保存已核验的实验源码和现有证据；V19 默认关闭（default OFF），尚未采用。
它不是已完成的教程章节，也不赋予完整模型、E2E、默认启用或性能晋级资格。
基线：教程 accepted15 `f976d6a547a9dc1946b52b1de6db376d799b34d7`。
源码仅改 `model/gdn.py`、`model/qwen3_5.py`，新增 `kernels/gdn_norm_reuse_metal.py`；其余 23 个源码逐字不变。

## 已完成的局部核验
- 独立 CPU source review：SOURCE_CPU_PASS；default OFF、严格布尔参数、normal eligibility 和原 fallback 已核验。
- 纯算子 15 banks（9 real + 6 synthetic）严格数值 PASS；stationary 前、后、释放前数值审计及只读检查 PASS。
- 固定 saved15 的局部 host-wall micro：A 148.959340 μs、B 129.099457 μs，降低 13.332419%；10/10 pairs 胜。
- 以上只覆盖局部算子，不是模型加速或端到端吞吐结论。

## 完整模型与 E2E 尚未通过
- 完整模型资格测试唯一已执行的尝试：accepted15 在 load 后、model-ready 同步时 pressure=2，资源守卫 STOP。
- 停止点在 token 0 之前，0 normal forwards/observations；candidate 尚未启动，未发生候选数值比较。
- 这不是数值 FAIL，也不是完整模型 PASS；不重试、不改写该次 STOP。
- V19 E2E 仅有 CPU harness 准备和审查，实际 E2E 未执行；后续 load-only diagnostic 不能替代资格测试。
- 未证明完整模型收益或相对官方引擎的胜出（no official win）；保持独立实验、不采用，不合入教程主分支。

## 证据（本机原始工件）
- Source manifest：`/private/tmp/minivllm-engine-optimization-20261014/v19-norm-reuse-engine-build1/source-manifest.json`
  SHA256 `d86be282646c1fc9d26155ad54014cf461f608c8222e32cd66d11088e180a32e`（26 sources）。
- Source review：`/private/tmp/minivllm-engine-optimization-20261014/v19-independent-review-build1/SOURCE-REVIEW.md`
  SHA256 `251198e2c0d39c8e025c670492cf9cf2c3e8198c9c95b9356dcb5716b150f6d9`。
- Operator/micro receipt：`/private/tmp/minivllm-engine-optimization-20261014/norm-reuse-stationary-micro-build1/norm-reuse-stationary-final-receipt.json`
  SHA256 `999ced1e1d1c8facb1db1c61f97b062b66afabff8a1f774605360fb354cbdfd4`。
- Fullmodel STOP receipt：`/private/tmp/minivllm-engine-optimization-20261014/v19-fullmodel-qual-build1/external-owner-attempt1/fullmodel-final-receipt.json`
  SHA256 `59dac27678206422a00e6ac8553e08b3479ce64b70fd584f36223175f917b070`。
- Fullmodel freeze：`/private/tmp/minivllm-engine-optimization-20261014/v19-fullmodel-qual-build1/frozen.json`
  SHA256 `50dcc23ac7aa6a7058754082a79e9b001eda82d9f2dd0d9d4e555283c9e31bb1`。
