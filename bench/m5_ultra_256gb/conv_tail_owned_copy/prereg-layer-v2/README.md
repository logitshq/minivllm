# E026 layer v2: separate OS footprint observations

Frozen before this new attempt. Plan SHA `313384a618fe1ddf4724ff2fafb8c5e07fbd6f923bb04fe5f5654fd4caf2eef2`; 608 execution pins, root 32 CPU tests and independent auditor 47 corruption-rejection checks PASS. This registration has no device result yet.

[V1 is permanently archived](../../../../docs/experiments/m5-ultra-256gb/E026-conv-tail-layer-v1/README.md). Its first probe returned unequal top-level and auxiliary current bytes, both below budget, and the old equality assumption stopped before the first case. V2 validates all three raw physical readings independently against the same 2 GiB limit; only the actual auxiliary historical peak is required to be monotonic. Derived maxima do not claim to be historical counters. Cancellation is distinguished from SIGALRM.

All model computation, weights, inputs, switch cases, limits, deadlines and exact-bit acceptance remain as originally specified. The [full v2 hypothesis](../layer-preregistration.md) is pinned. The real plan/jobs live in `.artifacts/m5-ultra-256gb/conv-tail-layer-v2-inputs/`, with new owner and service-lease directories. The existing `conv-tail-layer-v1-weights` bundle is reused unchanged. No original attempt is overwritten or relaxed.

Run only through the existing suspended-service lease using the exact `jobs.json` command; audit after all children exit and the same service PID is restored. This is a numerical/state/backing diagnostic, not a throughput/latency experiment or full-model correctness qualification.
