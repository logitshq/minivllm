# Independent V23 numeric runner CPU review — PASS for bounded synthetic qualification
2026-10-16；总10CPU工具轮（含前置path分析）；无MLX/GPU/model/Git/engine改动；未执行GPU argv。
Runner SHA256: efa9c2c58987f14606310852d555094947b77b2058245e03a6e4bbc3f20a6c5f
CPU checker SHA256: 73da9eb908bcf6f8fe9a094d9e0da8c5d04f64f5c084565eae74b5be3b85df14
Source-pins SHA256: b0d867a66c626c10b51e27fcb7084034670e12f00d317a594955352413a95f2f
完整runner/banks/reference/probes/README及最终修段已读；check_cpu与--help均rc0，末轮SHA复验未变。
独立AST核对reference：accepted15 w/xp表达式仅替换绑定名；显式product内联回sum后AST相同，nn.silu及xp[:,1:]不变。
单一81,920B BF16 conv weight range读取、reshape(C,-1)同loader；不mx.load完整shard/构造27B模型。saved g/beta/S shape/dtype/file SHA独立复核。
6个确定synthetic preconv case+8step独立tail/state链；真实weight和保存g/beta/S不等于真实preconv激活，receipt明确该scope。
显式sum-y4先绑定root独立PSO M1024/width32；无按output挑树或自动branch scan。
layout_probe在candidate前读取原product GPU descriptor，ensure_row_contiguous=False；shape/strides不符立即停。
row/column contiguous明确标为从stride推导，internal_flags_observed=false/pending保持；不能把它改称native flags或实际reduce dispatch已观测。
实际mlx/mlx-metal0.32.3版本在GPU import前核；device_name/architecture必须精确匹配PSO证据，两项原review建议均已修。
candidate是归档prototype，conv_probe仅将act store替为conv store并重命名tail；CPUchecker逐文本验证，shared 662B HEADER/math_options不改。
equal先shape/dtype再strict finite raw bytes，记录shape/dtype/bytes/SHA；conv/act/tail/probe-tail先过，才原递推o/S_new。
每case全部raw/tails/w/g/beta/states输入前后SHA相同；链步分别推进reference/candidate状态与tail，未把同一结果冒充两臂。
无容差、argmax替代或micro计时；首次异常FAIL_STOP保存receipt，完整logits/IDs/cache/lifecycle始终未资格。
saved state并非新real preconv capture；默认无real_bank，PASS只能是PASS_CONV_NUMERIC_SYNTHETIC_ONLY。
runner自身不带owned child/resource supervisor；root外层numeric_owner.py承担fresh guards/180s timeout/owned wait-reap/strictabsence，不能归功于runner。
小array memory/cache limits是MLX分配限制，不是process/compiler RSS硬上限；无全模型资源资格。
本PASS只是精确SHA下的CPU审计，不是GPU GO或numerical PASS；后续实际结果仍需独立复算，不能继承V18资格。
完整本地file SHA与state provenance/CPU命令记录见INDEPENDENT-CPU-RECEIPT.json；prototype、raw、26-source未改。
