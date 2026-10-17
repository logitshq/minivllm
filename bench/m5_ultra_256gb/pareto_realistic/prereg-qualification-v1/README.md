# E023 numerical qualification preregistration v1

Fixed Qwen3.8-27B NVFP4 revision and eight frozen D fixtures. Two independent engine instances at a time compare eager reference with the existing native compiled/fused/gates candidate. All full-vocabulary logits, persistent state and emitted IDs must match bit-for-bit for the same discrete schedule. Normal EOS remains enabled; output caps are not forced output lengths.

Run D-C1 (8 requests), D-C8 (16 requests), then 13 short lifecycle requests; at most 12,411 raw IDs per arm. The child has an 1800-second total deadline, including weight checks and loading; the owner has 1830 seconds plus bounded cleanup. A resource failure or first mismatch rejects the attempt and preserves evidence. The separately authorized service lease restores PID 74384 on exit and has a durable watchdog.

This is numerical qualification only. Instrumented durations are never throughput or HTTP latency results. No new Pareto point is admitted from this run. Existing 256-token decode-graph guards remain unchanged. E024's long decode graph is a separate future qualification.

Plan, model manifest, frozen core manifest, source/runtime hashes and exact job are saved here before execution. CPU owner tests cover timeout, resource failure, child failure, termination fallback, reap and first-error preservation. GPU correctness is pending.
