V97 独立保存结果审计：PASS；没有发现阻止另立新 profile 做 B-only 探索测时的证据问题。CPU-only，未加载 MLX/NumPy/native library，未运行 GPU、资源探针或 Git，未改任何冻结文件。

实际 receipt SHA `6c9621109e8f23e20b88244324b7b104e7672ca67a89342af52c4ebfe0b65947`；状态仍是 `DIAGNOSTIC_COMPLETED_NEW_REFERENCE_STRICT_NO_ADOPTION`，不是旧 profile PASS。

- 当前 142 个 source pins 全部重算一致，17 个实际导入源码路径均在 pins 中；worker、V89 native library/source 身份与 owner receipt 一致。全部五个 GDN 源文件等于 V44，GatedRMSNorm 类 AST 也等于 V44。
- 新 A 是独立 checkpoint raw BF16 gamma + nn.RMSNorm 的普通 MLX 路径；新 B 是独立所有者的 cached BF16 gamma + mx.fast.rms_norm，经实际 native ICB 执行。161 对 gamma 原始字节一致、地址不重叠；322 个 gamma 有保存的 C memcmp 只读证明，1349 个模型参数身份保持，161 个 B gamma 的实际 native full-range readonly binding 可核。
- 三次独立 A/B prefill：393 个 seed 原始字节检查、3 个末尾 logits 检查、3 个首 token 检查均严格一致，A/B seed 分别保留，B 只从自己的 prefill 播种。
- base/reset/changed 各 31 步，共 93×132=12,276 个有限输出；逐记录重算新 A/B 严格一致和 old38/44 共识，旧 hash 为 241 相同、12,035 不同。93 步均记录零 input upload、4B 预测只读回；每请求仅 seed131 一次，reset 次数 1/2/3。
- 实体 KV 全为 CAP64；32 个 KV 输出/步共 2,976 条 tail-zero/canonical 记录。256 只用于 CPU 原始 BF16 每头补零。独立提取 canonical 函数验证保留负零原始位且拒绝非零物理尾部；未向 GPU 提供 CAP256 state。
- 实际 ICB 为两 bank、每 bank 1,609 dispatch，累计 93 次执行；48 个 GDN S ping-pong、83 条 staged link、无 CPU 每步全 state restore。native 顶层 `scope` 字符串仍是继承的 teacherforced 描述，本审计依据 `feedback.icb` 实际证据，不将旧字符串作性能资格。
- warm/base/reset 新 raw32 IDs 相同，并与保存的 oMLX 可见文本 tokenizer roundtrip32 相同；它不是内部 raw generation ID 记录。base/reset 与旧 raw32 各 19 个位置不同，首次 index4；changed 与旧 native raw32 相同。所有新 A/B 的 96 IDs 互等且无 EOS。
- owner1104/child1119：rc0、waited/reaped、严格 absence 和 external release PASS。22 外层 +34 子进程 +1 外部快照：pressure1、AC100%、thermal nominal、swap5006.88MB 与 Swapouts17460761 恒定。已核 afterload rawSHA；记录只能证明采样和已知进程范围，不证明未识别 GUI/GPU 活动不存在。

结论：可以为“vendor plainnorm leaf profile”另立 B-only 探索测时，绑定这份新 A/B 参考、CAP64、同一候选/库/参数和明确的新 protocol/pins，并在计时运行继续检查该 profile 的输出/ID 与生命周期。它支持的是此 161 plain RMSNorm leaf rounding profile 在三个提示词、32 tokens 范围内的执行保真；不把不同 profile 的绝对耗时标成保旧数学的加速或 Pareto 胜利。新的实现/范围需另行资格，当前没有性能结果。

V93 原始 FAIL receipt SHA `0cb2058cb050b0c82f14ff67e2ee636ea17843fa814b6f136f2d84e00ddcc575` 已复核并保留；V97 不重判它，也不主张整个 oMLX 数学相同、旧 profile 同值、模型质量等价、服务适用或采用。实际张量未全量归档，本次独立复核重算保存的哈希/计数并审查执行源码，不能重新执行历史 C memcmp。

详见 verification.json、refsSHA.json 和可复跑的 stdlib verify.py。
