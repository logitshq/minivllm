# Real-input GDN qualification and fresh Apple references

Read2026-10-17 while the observer-OFF oMLX benchmark owns the GPU. These are source/research notes; no candidate device experiment or runtime upgrade occurred.

## October16 implementation report: synthetic parity can miss real failure

The author's [Qwen3.5 NKI kernel card](https://huggingface.co/kernels/jburtoft/qwen35-gdn-deltanet-neuron-kernels) reports a v15 prefill correction on2026-10-16: previous chunked cores passed random tests but failed some real checkpoint layers, including nonfinite outputs. The new gate exercises real model inputs. Its September updates also remove projection/norm launches and distinguish prefill from decode regressions. This is an author implementation report on Trainium, not a peer-reviewed M5 speed result. Its approximate numerical criteria and hardware-specific reductions do not establish our original-dtype bit equality.

Local consequence: keep the existing full-vocabulary/state/ahead checks and real serving histories. Do not reuse short-route qualification for a newly enabled long-route fusion. CPU mocks check wiring and failure handling; they do not qualify Metal arithmetic.

## October9 paper: Minions addresses machine availability

[Minions, arXiv2610.12970v1](https://arxiv.org/html/2610.12970v1), submitted October9, distributes independent tasks across idle Macs using atomic ownership and recovery. Its reported benefit concerns availability and task queues; it does not present a single-M5-Ultra Qwen D512 HTTP frontier. The applicable operational lesson is explicit experiment ownership. Keep only one GPU experiment on this M5; parallel agents can inspect sources. No distributed server/database is added to our small runtime.

## October15 Apple measurement: relevant hardware, different protocol

Apple's [local/distributed MLX report](https://machinelearning.apple.com/research/mlx-distributed-inference) measures Qwen3.8-27B in BF16 on512GB Mac Studios with8,192/65,536-token contexts and128-token generation using mlx_lm.benchmark. It also describes experimental MLX_METAL_FAST_SYNCH for CPU/GPU coordination in distributed inference. This differs from our256GiB NVFP4 D512 full-HTTP measurement, so no published Apple point is placed on our Pareto axes and no numeric win is inferred. Fast synchronization remains a separately qualified future hypothesis; frozen runtime/environment remain unchanged.

## Concrete local candidate: unbind existing GDN fusion from short dispatch

Source inspection of the installed oMLX Qwen3.5 B1/T1 branch confirms it uses one prework kernel for conv/SiLU/split/qk normalization/new-conv, with original projections and recurrence. Our existing native fused helper is only reached through a short dispatch tied to projection packing. The prepared candidate reuses native arithmetic with the original five Linear calls, keeping attention bounds and packing unchanged. It installs per instance, defaultsOFF, and falls back outside exact shape/dtype/backend requirements. Correction after inspecting the actual MLX0.32.3 headers: default CompileOptions already use Safe, as does the fused helper. This is not an established math-mode difference. Fusion boundaries and lowering still require exact long-state qualification.

A second isolated hypothesis computes common q/k scalar normalization once perSIMD instead of32 repetitions, keeping serial accumulation order. Compiler hoisting may already remove the work; neither candidate has measured benefit. Prepare source/mocks now, execute only after the current server cycle closes and original service recovery is audited.

## Earlier papers that sharpen the experiment choices

[MSInfer's GDN report, July 18 2026](https://arxiv.org/html/2607.16831v1) separates small decode launch/shape choices from chunked prefill reformulation. Its own local comparison reversed under a stronger official baseline. The public implementations it discusses suggest one-warp, small value tiles and direct bindings as candidates; those timings concern B200 and cannot predict M5 performance. Its state/output algebraic rewrite changes floating-point operation order, so it is not adopted under our exact-state criterion. Local action: qualify unbound prework and graph dispatch first; if they leave a measured GDN bottleneck, test smaller independent value tiles without changing recurrence reductions.

[Memory-Bound but Not Bandwidth-Limited, May 28 2026](https://arxiv.org/html/2605.30571v1) studies batch-one NVIDIA decode and reports that graph replay helps the faster-memory GPU more. Quantization benefits also depend on the actual kernel path. It supports testing dispatch overhead as a hypothesis rather than declaring every decode memory-bandwidth saturated. This is older supporting evidence, not a newly published Apple result. Locally we need separate observer-OFF full HTTP timing and narrowly scoped diagnostic timing; neither a kernel speedup nor a lower logical weight size establishes a serving gain.

The working decision order is therefore: remove avoidable dispatch/prework from the actual long route; measure the full request again; only then spend effort on shape-specific state traffic or more concurrency. Keep the same weights and arithmetic contract while comparing scheduling/runtime changes. Model-quality changes need a separate evaluation and never enter these exactness-qualified experiments by implication.


## ANE is a separate experiment, not free GPU capacity

Read 2026-10-17: [MacPaw Neural Engine study](https://research.macpaw.com/projects/neural-engine), indexed as October 10. Its LFM2.5-2.6B / DSpark work uses M2/M3/M4 Pro and Core AI, not our M5 Ultra NVFP4 target. It reports 98.4% teacher-forced top-1 agreement against BF16; this is not our exact-state contract. Its useful design observation is that extra tokens per invocation amortize ANE weight reads, while attention can move to the GPU at longer contexts.

Inference for this project: evaluate heterogeneous execution only as a separate backend with its own numerical contract and transfer accounting. Do not call ANE utilization an existing Pareto gain. Our immediate path remains fewer launches and state movements on the already qualified GPU route. This source does not establish an exact NVFP4 ANE implementation or a deployable Qwen3.8/GLM5.3 path.
