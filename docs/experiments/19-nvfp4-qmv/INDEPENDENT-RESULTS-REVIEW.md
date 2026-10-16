# Independent V22 results replay — PASS audit; STOP-MICRO-GATE retained
2026-10-16；仅CPU stdlib source/hash/记录/统计/原resource复算，无新GPU、MLX、tests、模型、探针或process命令；raw与driver未改。
Driver SHA256: dd071185991c93b41a9c01997912f4baae1bb20c7e9a4a72d6c3ffa8d5d5c279；protocol/source-manifest SHA与运行identity一致。
冻结7文件及5 upstream SHA复验；官方helpers/fast_impl按1-based片段逐字与SHA再匹配；runtime math_policy_override=false。
仅L0 gate K5120/N17408、U32 weight+U8 scale共50135040B；无full model/full shard loading或adoption。
R4/R2实际first call完成；27数值记录顺序严格为R4-QMM全部9项→R2-QMM/R2-R4各9项。
覆盖8 synthetic+1历史L0 real；全部[1,17408] BF16/34816B，finite/exact真、差字节0、左右hash相同且每输入三路hash一致。
历史input文件/固定旧receipt SHA复验，非本轮capture；原GPU输出tensor未保存，CPU核的是记录与hash，不冒称重跑GPU计算。
qualification=STRICT-R4-R2-BYTES-PASS/checks27/timing_allowed true；冻结source保证资格失败即停、不进入timing。
10对schedule、20block prime128/repeat512、1组/1输出全部吻合；duration=end-start、ns/call、时序及pair两侧output hash逐项复算。
A保持原mx.quantized_matmul，B为R2；9-input资格pass仅支持本target/样本，不能推广全部shape/math policy。
控制A ratios=.9991560802/1.0138006211，中位1.0064783506；B=1.0064874030/1.0040355202，中位1.0052614616。
tight control各[.95,1.05]及中位[.98,1.02]均通过，不把控制门解释为稳定提速证据。
A原QMM中位231091.267578125ns，B R2中位230599.8125ns；复算下降0.21266709178392462%，5/6胜。
5/6仅满足wins门；下降不足预声明5%，整体仍STOP-MICRO-GATE，与result.json完全吻合，禁止单凭5wins改PASS。
24个样本Battery26%/thermal0 nominal/pressure1；known matcher0的scope不排除未知或GUI GPU任务及采样间变化。
原vm_stat文字逐样本重解析一致：SO/SI/PO/Compressions增量0，仅Decompressions+5；swap原文字4719.56MB恒定。
MLX peak50571536B只属该小target过程；不据此推完整模型、全层或物理allocation共享/释放资格。
child31181/ppid31180贯穿资源样本；cleanup记录exit2及finally退出，controller与root-posttool均ps rc1、stdout/stderr literal empty。
adopted/full_model_qualified/E2E_run/AC_frontier_entry均false；无fullmodel/E2E结果或模型级收益。
14个原始文件SHA+bytes seals、27项覆盖、完整pair/控制/资源/退出复算见INDEPENDENT-RESULTS-RECEIPT.json；封存前所有raw seals二次核验未变。
