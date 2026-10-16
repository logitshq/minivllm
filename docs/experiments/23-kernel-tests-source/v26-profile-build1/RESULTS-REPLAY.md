# V26 build1 raw results replay
Runner author performed independent raw arithmetic replay; root independently executed and reviewed.
PASS diagnostic capture only: one model load,3×32 exact warm requests,4 measured B1 direct forwards.
336/336 command buffers completed/status4;7896 recorded dispatches (1974/step), including497 NVFP4 QMV/step.
Actual GPU envelope sum140.569378ms; union140.569375ms; mean35.1423445ms/step.
Mixed groups140.329669ms (99.8295%); no per-kernel GPU attribution is justified.
Largest4 buffers each contain lm_head NVFP4 QMV +Add+norm,1.408250–1.437375ms; component time cannot be split.
Host graph API wall4.366–5.500ms/step; eval+wait36.175–36.428ms/step; cProfile/encode instrumentation perturbs all values.
Dispatch-family counts are observed/shape-derived, not GPU-time shares: {"GDN_quantized_projection": 576, "GDN_recurrence_or_conv": 192, "MLP_quantized_projection": 768, "attention": 192, "attention_quantized_projection": 192, "elementwise_reduction_or_misc": 5524, "logits_quantized_projection": 4, "quantized_unattributed": 448}
8 internal resource samples AC/thermal0/pressure1, Swapouts and swapMB delta0.
Owner49706/child49707 reaped; external literal rc1/empty for both; tool exit0.
Native raw SHA 3dbcfdb60ce309496d4ba9110f0c8d2c924718f0e646f25ebf810fb9416eaee8
Receipt SHA a880d7daf229398d1756981fc2763b17866f8fbff58cfbe8e2ab0acf5aeec47e
CPU summary replay exactly matches saved summary.json. No production/Pareto/adoption claim.
