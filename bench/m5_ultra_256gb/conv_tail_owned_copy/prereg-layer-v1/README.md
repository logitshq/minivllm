# E026 layer v1: frozen before execution

This registration extends E026 from a pure copy to the original real-weight layer-0 GDN computation. It contains no device result. The objective remains correctness-gated whole-machine throughput versus complete-request latency; no single-layer diagnostic duration may enter the Pareto chart.

The plan SHA is `f01fa0ac7ce85975a92ef07188417dde096c8f4b37abcea9467408f516e3ce08`. It pins 608 source/runtime/bundle files. Root reran 27 CPU tests and the independent auditor's 31 corruption-rejection checks; independent source/schema review found no blocking issue. The CPU-only bundle proof checks all 14 tensor slices against the original shard; the preparation receipt additionally records the full 5.34 GB shard SHA. Runtime source copies are made by the owner before spawning any child.

Three fresh processes run OFF, ON-lazy and ON-materialize, with T=1,19,935,978 and fresh/continuing synthetic finite state. Each ON must equal OFF exactly before the next arm starts. See [the fixed hypothesis, limits and stop rule](../layer-preregistration.md). The original service is temporarily paused under the existing SIGSTOP/SIGCONT authorization, then restored. No cold restart is authorized by this registration.

The real local plan and jobs are under `.artifacts/m5-ultra-256gb/conv-tail-layer-v1-inputs/`; this directory preserves exact copies. Output must be new: `conv-tail-layer-v1-owner` and `conv-tail-layer-v1-service-lease`. Run via the existing `bench/m5_ultra_256gb/suspended_service.py --pid 74384 --jobs ABS_JOBS --out ABS_LEASE`. Audit afterwards with `audit_layer.py --owner ABS_OWNER --lease ABS_LEASE --plan-sha256 SHA --out NEW_JSON`.

No run is complete until every direct child is reaped, the service is restored with its original identity, and the independent raw-data audit passes. A failed or incomplete attempt is preserved and cannot be reclassified by changing this plan.
