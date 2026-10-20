# Latest research reading — 2026-10-20

Read primary sources before applying their conclusions. These notes define hypotheses; no new external result was reproduced on our hardware.

[Adaptive DHiSparse, arXiv2610.13708v1, submitted October9](https://arxiv.org/html/2610.13708v1), architecture and discussion: protect required KV/tail/staging storage, lend only recoverable optional history, and reuse staging after its readers finish. Deduplicating retrieval across a verification block preserves each query's ordered selection and causal boundary. The authors keep proposal/acceptance rules unchanged. Their own discussion reports that hierarchy overhead loses when resident histories fit, and that speculation can lose aggregate throughput when its memory reduces active slots. Their evaluation is one GPU platform and two sparse-attention models; online block selection is future work.

Our inference: carry over allocation lifetimes and accepted-prefix invariants, not the paper's throughput numbers. Shared index positions do not imply shared payload bytes. On unified-memory Apple hardware, first establish a real copy/bandwidth bottleneck before adding a host/device cache hierarchy. A bounded draft window is applicable only if it preserves that drafter's actual visibility contract.

[MLX issue4198](https://github.com/ml-explore/mlx/issues/4198), August12 report, closed August16: the reporter measured an M<=32 split-K routing discontinuity on M5 Max for affine4-bit shapes. A maintainer closed it because the team prioritized actual inference cases; closure is not evidence of a merged fix. Our inference: inspect the installed NVFP4 dispatch and actual model shapes before importing a threshold or expecting those gains. An observed kernel effect is not automatically a full-model optimization.

Next concrete gate: measure draft+verify+commit/rejection cost divided by useful accepted output. Choose width/config by actual workload and latency objective; disabling speculation must remain a valid result. Do not multiply E054/E057 speed ratios or use metadata byte savings as measured DRAM traffic.
