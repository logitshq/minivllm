# E025 same-split NAX v2 preregistration

Status before execution: **DEVICE UNRUN**, default OFF. This bounded operator
screen uses the unchanged E025 candidate and seed, with complete raw comparison
pairs and a separately registered resource/lifecycle wrapper. No model loads.

Three T19 shapes: GDN z (K5120,N6144,split2), MLP down
(K17408,N5120,split2), output (K6144,N5120,split3). Keep full original row
strides, partition boundaries and BF16 partial stores. Native QMM is the first
oracle: reconstructed SIMD reduction must equal it before NAX is considered.
No tolerance, alternative oracle or changed quantization is admitted.

Each shape runs five normal checks and fallback M1/11/12/32/33/128, giving
33 checks and, only on PASS, 68 self resource phases and six kernel receipts.
Every check saves both original-dtype tensors, including passing prefixes and
fallbacks. First failure stops; a compile failure is not numerical PASS. Inputs
are saved before/after and X must be finite. The original candidate and screen
are unchanged; v2 only adds evidence and resource controls.

Root reran 10 screen CPU tests and the real CPU-only owner preparation with
all source/runtime pins. Eight corrupted-plan challenges were rejected. These
checks do not exercise Metal. Independent source review found no blocking
contract mismatch; actual compilation and numerical behavior remain unknown.

Registered limits: child120s, owner work450s plus cleanup30s; MLX observed
active+cache and peak≤1GiB, cache0, physical readings≤2GiB, ≥50GiB real free,
no Swapouts growth, swap-use growth≤16MiB, AC power and original thermal rules.
Stable existing pressure2 is admitted only for this small numerical diagnostic;
full-model/performance pressure1 is unchanged. These are observed bounds, not
OS hard caps. Sole GPU ownership and the existing authorized SIGSTOP/SIGCONT
lease surround the direct child; the original service must be restored.

Full plan, jobs and CPU receipt are adjacent. Root must commit these exact
sources before executing jobs. No elapsed diagnostic time is a performance
measurement; no result from this screen alone enters the throughput-versus-
complete-request-latency Pareto plot. Whole-model and HTTP qualification remain
separate requirements even if all local comparisons pass.
