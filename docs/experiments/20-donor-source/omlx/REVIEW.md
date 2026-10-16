# 实际 oMLX 源码复核（CPU-only，2026-10-16）
结论：优先核对已证的**MLX allocator缓存上限差异**；真conv4＋SiLU＋tail＋gate单dispatch仅列为次级数值资格候选，均无新收益结论。
本次只读实际代码、已有日志与 stdlib 写本报告；无 MLX import、GPU、模型、测试、engine/git 改动。

1. 身份：O 为实际安装 0.7.1.dev1 site-packages（路径与逐文件 SHA 见 source-map.json）；未把当前 upstream main 当历史跑分代码。
2. 调用链：O/omlx/scheduler.py:14075–14086 → O/mlx_lm/generate.py:1872–1884 → GenerationBatch.next:1439–1449 → _step:1355–1414。
3. 普通 self.model 在 :1367；:1405 async next tokens/logprobs；:1410 eval current、:1411 tolist。因此 one-ahead 本身已在双方实现，不能再提为新优化。
4. Qwen3.5命中条件：O/omlx/patches/qwen35_gdn_prework.py:1864–1897，exact class、B1/T1、GPU FP16/BF16、mask=None、普通无spec/lengths/padding ArraysCache、conv4、Dk=Dv128。
5. 实际路线 :2053–2099：分开投影→gdn_prework_fused→gated_delta_update→norm→out_proj→两个state一起commit；不是 standalone fused_greedy_decode。
6. 不只“eligible”：既有 freshscreens-ac-frontier-attempt1/omlx-balanced-c1/server.log:58 记 VLMBatchedEngine，:60 记 “Qwen B1/T1 fused GDN prework engaged”。此日志不能证明 B2/B4。
7. 真conv primary：该patch :108–120 在读取old tail与raw projection时直接计算conv4＋SiLU；:177–194同dispatch输出原始tail；:769–803为80个32线程head组（Hk16/Hv48）。
8. 该Qwen3.5 fused prework不计算gates：输入仅qkv/tail/conv_w/q_scale/k_scale（:757–765）；gates经 :2063、:2079–2088进入另一个函数。不能把Qwen4大融合说成实测路线。
9. Qwen4 :243–324示范在v-head lane0计算gate并输出tail，但它是另一路线，gate BF16语义也与我们FP32语义不同；只作结构参考，未证明本checkpointengaged。

10. Own实际accepted A/model/gdn.py:162–174仍为 raw qkv→concat旧tail→BF16逐tap乘＋sum→SiLU→repeat；:169编译gates为独立调用；A/kernels/gdn_metal.py:17–60做FP32 q/k归一化、exp(g)和递推。
11. v18的差别明确：V/model/gdn.py:219–227仍完整保留concat/乘/sum，仅把卷积**之后**的SiLU/映射融合进递推；因此v18<1%并未测本候选的真conv消除。
12. 最小手写形态：保留4个投影与原递推不变；一个Metal prep读取raw qkv、不可变旧tail、conv_w、a/b、FP32 A_log/dt_bias，输出q/k/v、new_tail、FP32 beta/g。
13. 用80个32线程组；每lane拥有连续4channel。q/k owner将activated值写入3个原有重复head槽，v owner写自身；只在v-head lane0算gate；tail每channel唯一owner。这样不改递推寻址/归一化/规约/exp(g)。
14. 这是从实际Qwen3.5真conv机制与既有own gates合成的候选，**不是直接移植oMLX数值实现**；compact q/k可另议，不与首版混在一起。
15. 首版仅exact B1 BF16、Hk16/Hv48/D128/C10240/K4、普通ForwardContext＋BatchState(conv,S)、nvfp4现有shape与无record/capture/spec/verify；任何不符退回原路线。
16. 本checkpoint形状静态适配（A/model/gdn.py:71–85与V:181–190）；新候选尚不存在，因而**未证明own engaged**；不得把旧gate计数当新prep命中。
17. M4可手写：仅标量/SIMD与独立output，无MPP/apple10要求；是否更快未知。80小组写重复q/k、gate稀疏lane及额外tail写回也可能抵消节省。
18. 数值硬门：先核原product的actual layout/flags与kernel M/Y，再按C1-GDN-CONV-REDUCTION-CONTRACT.md预声明分支保留BF16 product/reduction；M≥128/Y4才可钉死tap0→1→2→3。禁止oMLX FP32累加替代；保留typed BF16 SiLU与既有FP32 gates边界。
19. 最低资格成本：CPU先冻结单独prep源码/输入输出契约；GPU owner一次loaded parent先确认原layout/flags及actual M/Y，再抓raw qkv/old_tail/w/a/b与参考intermediates，覆盖零/非零tail、head边界与合成抵消；不按结果挑规约树。
20. 先逐byte比较conv、activation、q/k/v(含repeat)、beta/g、raw tail；再接原递推检查o/S及连续steps。若有一处不等，STOP；不要先做整机配对。
21. 通过后才做原19→32 IDs/logits、最终cache、正常EOS及ahead-abort/filter/reuse资格，再一次未instrument固定C1配对；<既定收益门仍STOP。

22. 其余方向未立新项：installed omlx所有.py未搜到max_ops_per_buffer/max_mb_per_buffer显式控制；未发现本decode路径的whole-model mx.compile，不能凭接口存在承诺Graphcompile收益。
23. 专用worker stream确实存在：O/omlx/engine_core.py:348–364建单线程executor＋new_thread_local_stream；scheduler.py:3420–3429传给BatchGenerator，generate.py:1879进入stream；own shim:38已有单worker线程。
24. 专用stream注释明确服务多model并发；固定C1单model无实际加速证据，故不列为本轮候选；也不改commit/eval节奏或HTTP行为。
25. 本轮推荐级别：**一项数值资格候选，零新增实测收益结论**。如只接受已有bytePASS与收益PASS的项，本轮没有新可合入优化。

26. allocator实证：m4-final-native15-official-attempt2/client.py:52启动child.py；child.py:35未传cache_limit_gb，故A/engine/llm_engine.py:15–18的默认2实际调用set_cache_limit(2<<30)。本attempt不能套用旧core_http的0参数。
27. oMLX client.py:24–35→official_entry.py:13–14→actual CLI；cli.py:452–463在total_mem>0时设allocator limit=total_mem，与prefix-cache开关独立（server.py:9215同）。本attempt已证策略为native2GiB vs oMLX total_mem，不是双方allocator0。
28. allocator-source-observations.json保留launch证据；对该attempt output全部resource/measure/metadata JSON键与字符串搜索MXcache/mlx-cache/cache_memory未命中，故池内实际字节/复用率未知，不是0。下一诊断先记录limit/active/cache，再提单变量候选；无速度因果结论。
