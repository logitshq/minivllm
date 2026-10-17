# V112 保存结果独立审查

结论：`PASS_INDEPENDENT_SAVED_EVIDENCE_AUDIT_V112`。本审查仅用 Python 标准库读取冻结源码、保存 JSON 和 raw；未导入 MLX/NumPy、加载模型、执行 GPU 或修改原证据。

- actual receipt SHA256：`09c7f55cfbd46a803b4bd014eb24f694573f62de48e2a3fdee3568ec1910992a`，原状态为 `DIAGNOSTIC_COMPLETED_V112_SAME_INPUT_BATCH_VS_M1_NO_ADOPTION`。76 个 source pins 全匹配，保留全部 61 个 V108 pins；实际导入路径为同一候选，before 23 / final 26。
- 978 个捕获算子独立复算展开为 1138 项比较：497 Linear = 146 exact + 351 different，差异元素合计 1281；其余 641 项全部 exact。161 RMSNorm、48 GatedRMSNorm、192 gates、64 SiLU、64 sigmoid、48 softplus、32 RoPE、32 SDPA 的分组与保存 summary 完全相符。
- 首差为捕获完成顺序 order 9：`layers.0.linear_attn.out_proj`，5 个 BF16 元素不同，maxabs `4.76837158203125e-7`。这不是 GPU 调度顺序。首个坐标 `[0,709]`，bits 12974 vs 13008；数值由 BF16 bits 独立解码匹配。
- 首层 QKV 的实际输入 `[5,5120]`、51200 B raw 已完整重读、SHA/有限值校验；SHA `f4c0c896eab78d21f4747a97e018905b86064f1f0229afc61819a8c15321e08d` 与 row0 norm 输出和 QKV input signature 一致。raw 保留原处，本目录只封存身份。
- “相同输入”指每个替代 M1 调用切分该算子原 batch 输入、使用原权重；上游其他算子继续消费原 batch 输出。`op_audit.py:36–48,55–69,74–81,84–104,107–119` 均返回原 `y`。`diagnose.py:125–127` 完成原 posterior/features/literal histories 后，才在 146 行构建替代图；154–155 行断言原输入/输出 bytes 未变。没有把替代结果传播成新全模型轨迹。
- prompt19 重新 prefill 的 live anchor 为 79852；仅用 actual108 保存的真实 draft proposals `[6297,220,95761,95779]`，复现 posterior `[18887,220,98435,100606,98277]`。anchor/posterior 保存值只作断言。128 个初始缓存 signatures 精确来自 actual108 第一个事务的 history[0]；keep0 后全部128及 offsets 恢复的断言完成（`diagnose.py:104–116,158–167`）。
- 1349 个参数的 before signatures/object identities 已保存，最终 identity/bytes readonly 断言完成；after signatures 未单独保存，审查不声称重新读取已释放设备参数。1138 对输出 raw 未保存，能重算全部分组/首差 metadata 与 hashes 的一致性，不能离线重算缺失 tensor 的逐元素差异或全体 finite 检验。
- 11 份 owner resource raw + 1 份外部 raw 全校验：均 AC、thermal0、pressure1；owner swap 4470.88 MB 恒定、Swapouts17460761 恒定；release swap4462.88 MB，Swapouts不变。raw matcher 没识别该命令，9 份 during derived inventory 按 owner 源码20–24行追加显式 owned PID20383，原 raw 未改。
- owner20374 / child20383，rc0、wait/reap、保存的双 PID strict absence 与 `PASS_EXTERNAL_RELEASE` 匹配；any-power release 实际 AC。这里只核保存释放证据，未执行新进程/资源/lease 探测；数值 any-power 策略不构成性能资格。

未观察实际 PSO、dispatch 或 reduction 实现；未修复 full96 ID、未改变 V108 FAIL、未提供性能或 adoption 结论。最多说明原 batch 轨迹上 Linear 存在局部 batch/M1 形状敏感性。

复验（只读本地原证据，标准库，无 GPU）：

```sh
python3 -B /private/tmp/minivllm-engine-optimization-20261014/v112-independent-results-review-build1/review.py
```

`raw-source-seals.json` 记录 103 个原始源/证据身份；`receipt.json` 保存独立复算结果；`manifest.json` 封存本审查文件。原 tensor/native/MLX 二进制仅验证身份，不复制 payload。
