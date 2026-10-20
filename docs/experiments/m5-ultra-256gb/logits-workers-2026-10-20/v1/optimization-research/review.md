**Review — minivllm M5 Ultra 256GiB, Qwen3.8-27B NVFP4, GLM-5.3-Flash — 2026-10-20**

Scope read-only: `docs/optimization-guide/tutorials/036-concurrency-specific-configurations.md:1`, `bench/reviews/2026-10-19-adaptive-verification-and-upstream-qmv.md:1`, `bench/m5_ultra_256gb/rowwise_mlp_shared/candidate.py:1`, `bench/m5_ultra_256gb/nvfp4_multirow_shared/candidate.py:1`, `bench/m5_ultra_256gb/rowwise_native_verify/candidate.py:1`.

**Current measured state (no new Pareto point):**
- E045 v2 `036:33-55` — 5 processes each C, no config meets 3% gate; best C1 MB200 +1.50% throughput/1.48% p50 speed vs dual-50 baselines; drift 0.04-0.41%. C8 disqualified: token IDs differ on 4/8 fixtures at baseline MB50 already `036:40-45` (D04/D06/D07/D08 first diff 59/22/187/209), so timing not attributable to `MLX_MAX_MB_PER_BUFFER`. `MLX_MAX_OPS_PER_BUFFER=50`, `BFS_MAX_WIDTH=20`, `METAL_FAST_SYNCH=0` fixed; MB is commit threshold not DRAM. Raw SSE rechecked 1030 steps/8192 IDs `036:65`.
- E054 C1 verifier cost `reviews:44-48`, `036:60`: audited medians at used=1023, k=T: native AR(T) 85.47ms (T4)/168.49ms (T8) vs verifier+commit 59.69/108.67ms; paired savings 25.72/59.66ms is optimistic budget assuming all T accepted, excluding draft/accept/scheduler/HTTP and block-latency. Not engine speedup.
- Upstream: `reviews:10-17` MLX PR3764 merged 2026-06-26 `qmv_wide` already reuses dequantized weight across rows for NVFP4; `fp_quantized.h:570,651` SHA `359a1ea...` — generic reuse not novel, grouping changes rounding. MLM PR1922 `reviews:19-22` open, affine 4-bit only, +22.8% at B8 reported on short workload with 2.3e-3 error — not transferable to NVFP4 or our bit contract.

**Correctness gates (must not relax):**
- Exact token IDs per fixture across all warm/probe `036:18-20`; wire count = committed tokens in POST→DONE window, EOS not counted `036:24`; same entry/workload vs oMLX/Lithos; original-byte qualification before timing `036:49-51`, `reviews:15`. C1/C8 may diverge config `036:60` but each needs its own qualified table.

**Ranking from first principles (simple, large impact):**

**1. C1 speculative verification with small shared K5120 MLP/head projector — highest TCO ceiling.** MLP gate/up =44.5% of packed projection bytes, K5120 coverage 64.9% `036:60-62`; candidate `rowwise_mlp_shared/candidate.py:70-110` shares across T=2/4/8 while preserving per-row mixer/postnorm/SiLU/down/residual.
- Benefit when: `E[accepted] * c_AR - (c_verifier(T)+c_draft+c_sched_overhead) >0` and p50 block latency `c_verifier(T)` acceptable. Break-even: `E[acc] > (c_verif+c_draft)/c_AR`. E054 budget 25.7ms (T4) bounds draft+overhead; need measured acceptance (EVICT product-of-probs `reviews:26-30`, DSpark budget `reviews:32-35` — not yet measured for Qwen/GLM).
- Switch: `verify(...,enabled=False)` default OFF `candidate.py:37`; `build(enabled=False)` returns None `nvfp4_multirow_shared/candidate.py:60`. Separate from C8 GDN/graph.
- Reject if: E056 shadow canary layers 0/32/63 gate/up vs native fails `036:62-64`; E057 full logits/selected-state/continuation bit-mismatch; or cost table shows `c_verif` ≥ `T*c_AR` after graph padding jumps `reviews:34`.
- Architectural vs tweak: new arithmetic path + scheduling; not MB tweak (which gave <1.5% and failed C8).

**2. C8 native batched decode throughput — batched GDN + attention shape stability.** C8 is 98.8% B8 steps `036:6`; C1 GDN prework only hits B1. E055 `036:66-68` proposes batched GDN flag with frozen OFF observer, native HTTP/count, 6×8warm+64formal ON/OFF. No device result yet.
- Benefit when: `c_B8(with GDN+graph) < c_B8(baseline)` per decode step, without extra memory traffic. Weight reuse already in upstream qmv_wide; gain is from avoiding per-row dequant repeat + attention cache layout.
- Switch: `batchedGDN flag` OFF default, independent of C1 `036:64-68`.
- Reject if: token inequality persists at MB50 baseline (current 4/8 fixtures) — indicates scheduler/prefill-split/graph bug not MB; fix diagnosis `036:46` (eager vs graph same schedule) before timing. Also reject if throughput/p50 <3% vs dual baselines with tight drift `036:18`.
- Architectural: changes decode critical path for saturated batch; launch tweaks insufficient.

**3. Memory-traffic / copy cost trimming (row-contiguity, shared tiles) — enabler for 1+2.** `nvfp4_multirow_shared/candidate.py:20-50` uses 5760B shared tile (1280 U32 +640 U8), one barrier, `ensure_row_contiguous=True` cost included in timing `candidate.py:60-65`; K=5120, N%4.
- Benefit when: `T * weight_bytes + activation copies > shared_load + barrier + contiguous staging`. For NVFP4 gate/up (5120→17408), naive T× loads dominate; shared reduces weight traffic ~1/T but adds staging. Break-even: `saved_DRAM(T-1)/T * bw > staging+sync`.
- Switch: `enabled=False` no kernel compiled; C1/C8 may retain direct path.
- Reject if: `supported()` false, `header` SHA mismatch, or measured `c_verif` not improved vs baseline verifier `reviews:14` on actual M5 Ultra `applegpu_g17d` `candidate.py:70`.

**One next experiment (falsifiable, no new bench claimed):** Run E056 canary on device (T4/T8 layers 0/32/63) → gate on exact `rowwise_mlp_shared` numerical PASS, then E057 cost table at used1023; compare to E054 before any draft/acceptance work. Separately, fix C8: log scheduler outputs at MB50 deterministic schedule and compare eager vs graph token streams — timing invalid until bit equality restored. No final fastest claim; all speedups hypothetical until same-workload oMLX/Lithos engine measurement.
