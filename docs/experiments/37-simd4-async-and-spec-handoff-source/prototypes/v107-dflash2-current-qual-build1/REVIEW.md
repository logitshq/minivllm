V107 HOLD：现有接口不能给出要求的“同一次 verify 轨迹的 accepted-prefix 中间cache”严格字节见证，尚未 numerical-ready。

candidate44 的 SpecRunner 调用一次多token prefill，返回 produced/new_feats；GDN chunk 仅返回(o,last_S)，S每chunk结束才更新。block5的verify选chunk8且只有一个chunk，根本没有逐token中间S对象。tape仅存raw/q/k/v/g/beta；partial rollback另用prefix调用默认chunk64，这不是同一次verify轨迹。小shape freshprefill或重跑tape不能伪称它的严格bitgold。

后续可研究保存原chunk geometry8而非默认64，但这只能减少已知不一致因素，尚不能证明同轨迹中间state。需要先定义/暴露producer的authoritative prefix checkpoints，再做逐字节rollback比较。Attention rollback只改offset，未来列仍留在物理buffer中；精确cache比较的有效前缀和物理tail政策也必须明记，不能默默归一化后称旧tensorgold一致。

已核old38/44/89三请求prompt+各32IDs完全一致，共96个ID硬门已预声明；没有在107执行任何数值gate。目标仍同27B，external DFlash2用106所核snapshot，raw文件3,848,817,896B；未来1349 target parameter身份/readonly、reset/changed、earlybudget/partialaccept都必须真实观察，额外draft内存/量化峰值完整记录。gold只作assert，绝不作为draft/B未来token输入；旧16/128记录不替代本次PASS。

qualify.py 默认/--inert只做stdlib source identity并返回HOLD。--execute-gpu在任何MLX/native/model加载前明确拒绝(rc3)。本轮因105性能窗口暂停，入口已做AST编译检查但未运行；不补GPU实现、不改任何冻结源、shader/native/perf。未来比较仍须允许oMLX的强spec配置，不能把无draft balanced.03当最好。
