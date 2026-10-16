# Independent V23 numeric results replay — PASS, synthetic scope only
2026-10-16；CPU stdlib读取source/receipt/hash/原resource；无MLX/GPU、新tests、模型、probe或process命令；原raw未改。
实际runner SHA efa9c2c58987f14606310852d555094947b77b2058245e03a6e4bbc3f20a6c5f，owner/child记录与当前file一致；依赖pins及全部runtime_sources再核一致。
6个预声明synthetic case+8步synthetic chain，共14组、84项strict checks、46,678,016B，全部记录strict_equal=true。
conv/act BF16[1,10240]，tail/probe-tail BF16[1,3,10240]，o FP32[1,48,128]，S_new FP32[1,48,128,128]；shape/dtype/bytes逐项核对。
每组8类输入before/after SHA相同且input_readonly=true；chain后一步oldtail/S输入SHA链接前一步tail/S_new输出SHA，8步完整。
全部product GPU descriptor实测shape[1,4,10240]/strides[40960,10240,1]；probe ensure_row_contiguous=False。
内部flags依旧未observed；row/column flags是实际stride推导值，receipt保持pending，未冒称native flags或实际reduce dispatch观测。
仅Y4，独立PSO M1024/width32证据先于数值选择；未扫树或按结果选择Y。
真实checkpoint仅L0 conv BF16[10240,4,1]81,920B定range读取；保存g/beta/S仍只是复用state，非真实preconv激活。
actual mlx/mlx-metal0.32.3、M4 Max/applegpu_g16s匹配PSO证据；V18 HEADER不变，本结果仅证明这些V23样本。
原GPU输出tensor未留档；CPU复核的是严格执行路径、记录及hash一致性，不描述为再次计算全部GPU tensor。
root wrapper审读：direct owned Popen、180s wait/超时terminate-kill/wait-reap，before/after资源guard；未构造模型或服务。
注意wrapper热态guard允许0或1；此次两端actual均0/nominal。资源只有before/after2点，不声称全程连续监测。
原vm_stat文字独立重解析，SO/SI/PO/Compressions/Decompressions五counter增量全0；swapMB不增长，AC/pressure1/known matcher0。
owner39075/child39086，child return0/waited_reaped=true及strict ps rc1/literal empty在owner raw中可核。
root另报告两PID外部exact absence均rc1/empty；该独立工具输出未作为raw文件提供，此处不冒称文件复核或自行重poll。
PASS_CONV_NUMERIC_SYNTHETIC_ONLY与结果完全一致；real_preconv_cases=[]、engine_enabled=false、logits_cache_lifecycle_qualified=false。
尚无真实raw/tail/w激活、fullmodel/logits/IDs/cache/lifecycle或性能资格；不可凭synthetic pass声称补齐C1缺口。
完整raw SHA seals/source对齐/逐项统计/VM及owner范围见INDEPENDENT-RESULTS-RECEIPT.json；保留本scope，继续CPU准备候选。
