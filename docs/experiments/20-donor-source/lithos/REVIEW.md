# Lithos 实际执行路径复读（CPU/source only，2026-10-16）
L=/private/tmp/minivllm-remaining-20261014/lithos-source，固定 HEAD 5485ad6bac0a5649d66a347f3e01a655ea29d937；A=/private/tmp/minivllm-engine-optimization-20261014/candidate-v17-gdn-gates-build1-root/tutorial_code/miniengine。
任务上下文：M4 Max40/Apple9/48GiB，MLX0.32.3，Lithos accelerator=on、无draft、C1、steps_per_cb=8/in_flight=3；这些参数来自上游实验记录，本文没有运行或重证。
结论：尚无“已证实际启用且有增益”的新 epilogue 候选；确认一项尚未落地的架构机制：固定缓冲、预编码 Metal ICB、GPU StepState 推进多token。近端可手写项是保持本引擎数学的 conv+SiLU+tail 单核；不重试已否决的投影拼接或参数扫描。

## serving → Metal 的明确调用链
- L/monolith/generate.py:632–645：普通 streaming 获取 T1 decoder；每轮 dec.run(min(8,need), steps_per_cb, in_flight)，返回后回调输出；:650–655 非streaming可一次run剩余全部token。
- L/monolith/runtime/engine.py:58–86：按角色分配/共享稳定缓冲（params独立），编译pipeline，绑定所有buffer/offset/grid，构造一次Icb；:91–96只声明实际绑定资源，weights/params标read-only。
- 同文件:31–43 pipeline缓存键含device、source、macros、language、fast_math、function；缓存复用不保留模型buffer。:98–107 run默认reencode=False，走native runner。
- L/runtime/src/metal_core.mm:224–251：ICB预写pipeline、buffer、threadgroup及barrier；ICB禁止setBytes，参数必须在buffer；offset必须≤32bit。没有Apple10分支。
- 同文件:327–375：一个serial encoder中n次execute同一ICB；每CB只commit一次；pending到上限等最老CB，退出前全部wait。不是常驻GPU内核，也不是一个CB跨step任意并发。
- 静态重要推论：当前streaming每次max_steps≤8且steps_per_cb=8，通常每次run只有一个CB；退出必drain，因此in_flight=3不代表此路径同时排24步。非streaming长run才可能填满3个CB。
- 同文件:269–299：队列保留资源引用；默认用全局residency set按buffer寿命登记，关闭环境开关才每CB useResource；:22–27释放时注销。复用要求旧GPU工作完成前不能释放/重绑或由CPU改写。
- L/monolith/compiler/emit.py:1440–1459追加advance并调用barrier pass；L/monolith/compiler/barriers.py:38–74按底层buffer读写冲突分组，第一条总有barrier，:3–17明确别名按整个base buffer判定。
- L/kernels/common/argmax.metal:77–98在设备写token；advance.metal:15–36将token写序号ring、pending_tokens，更新position/step，检查EOS/stop_at/context/ring overflow。
- L/runtime/src/metal_core.mm:310–324 CPU不信任并发可见的head，按每slot的64bit(sequence,token)读ring后发布tail；:335–344完成CB后检查error/done。照抄批提交却丢掉这些协议会出生命周期/取消错误。

## M4 条件及 epilogue 证据边界
- L/monolith/backends/metal/m4_max/backend.py:1–6继承普通MetalBackend；config-40c.json:5–22是Apple9/native fallback、validation=unmeasured、accelerator默认off、无融合recipe。显式on不把设备变成Apple10，也不补充测量证据。
- L/monolith/backends/metal/m5_max_40c/scheduling.py:12–18的GDN mega fusion要求apple10、40核、静态T8、非spec、commute_norm、accelerator=on及recipe；不适用本M4/T1。
- L/monolith/compiler/emit.py:350–362输入norm是否融入GEMV取决于tuner choices全部fuse_norm；否则缓存一份norm_scratch，siblings共享一次norm_apply。不能由accelerator=on推断已融合。
- 同文件:363–411输出stat_value/STAT_OUT把平方和partial写入producer epilogue，并统一每个variant的partial数量；kernel元数据记录fused_norm，需实际emitted program/recipe才能确认此benchmarkengagement，现证据为UNKNOWN。
- L/kernels/common/gemv_nvfp4_rows.metal:100–157确有gate/up同threadgroup、barrier、silu_mul和rounded输出平方和epilogue；:18–19要求gate/up伙伴完整留在同组；不是把两次PythonLinear简单合成一次即可。
- L/kernels/common/activation_math.metal:18–30默认SILU_ROUND=0；即使启用也用float gate/(1+exp(-gate))，不等同MLX BF16稳定sigmoid的typed exp/abs中间舍入。不能标为无损移植。

## 与手搓引擎的差异及最小落地点
- A/engine/model_runner.py:78–98已做device-token一拍ahead；但:90–94仍每token重新构造ctx/model/sample，:88–89用新array句柄保护cache。它降低host等待，未在本实现暴露固定MetalBuffer或可重放ICB。
- A/model/layers.py:10–21只编译norm子图；A/model/gdn.py:90–94只编译gate子图；现有mx.compile不能据此宣称完成Lithos式ICB。完整迁移需native Metal层、固定arena/双槽state/KV指针、设备采样/advance/ring、读写依赖、完成栅栏、取消/abort/admission/eviction/reuse处理，属于运行时工程，非一个装饰器。
- 不应先跨token原地复用现有array/缓存：in-flight旧图/已承诺state与未来state必须保持隔离；变batch、采样参数、输入token或positions后旧ICB/step资格必须失效。pipeline缓存与稳定buffer生命周期是两个问题。
- 近端最小变更：只替换A/model/gdn.py:164–167的xp concat→BF16乘积→四tap顺序sum→BF16 SiLU，新增一个手写_decode_conv_silu(raw,old_tail,w)->(act,new_tail) kernel；:174用new_tail提交；_split_heads与recurrence/norm/projection保持原样。
- 首版仅已知B1/T1、conv4、BF16、形状/stride受控；每channel一个owner；old_tail只读，new_tail复制原始投影[t-2,t-1,t]，不复制激活值；不能让Hv sibling覆写仍在读的共享q/k窗口。
- 数值成本：每tap BF16 product及每次有序BF16 add都保留，SiLU复用MLX0.32.3同typed表达式（包括BF16 exp返回/稳定sigmoid边界）；禁用跨这些边界的FMA/reassociation。FP32累加只在末尾cast不合格。
- 生命周期/存储成本：C=10240时B1新tail=3*C*2=61440B、act=C*2=20480B；输出仍需独立分配并随state提交；可移除4*C*2=81920B的显式xp临时量，但实际分配/流量须测，不承诺收益。
- 验证仍由GPU owner执行：逐中间位/尾状态、chained steps、强制logits/token、cache hash及服务取消/缩batch/reuse；本文未运行MLX/GPU/模型/测试，未改engine/Git。当前没有可报告的新性能数字。
