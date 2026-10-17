# E026: copy a small state, release a large prefill buffer

2026-10-17. This default-off candidate has now passed its isolated device bit/backing diagnostic. A scoped integration is CPU-reviewed; real-model numerical qualification and performance measurement remain pending.

The recurrent state we need after a prefill is small, but a small view is not necessarily a small allocation. The frozen GDN implementation stores `xp[T:]`: three BF16 rows of width 10240, or 60 KiB per layer. MLX 0.32.3 slicing shares the parent buffer. With a roughly 1000-token prompt, that handle can retain about 20 MiB per layer. Across 48 GDN layers, this makes ownership worth investigating before changing the math.

The isolated candidate is one operation: `mx.contiguous(tail.view(mx.uint16)).view(mx.bfloat16)`. In this pinned MLX version, contiguous copies even an already row-contiguous view when the backing exceeds logical bytes by more than 16 KiB. The integer word view avoids a floating-point conversion. The helper defaults to OFF, checks its exact domain, and does not patch production code. A separate materialize switch makes an immediate `mx.eval` explicit; a lazy copy cannot be assumed to free its parent immediately.

The first question is measurable without model weights: do the original bits survive and does the retained output own a small buffer after old aliases are dropped? The public `get_array_buffer_size` API counts unique full allocator buffers. Use it alongside active/cache/peak memory; summing shapes or counting Python objects cannot establish storage ownership. Hold parent/oracle aliases only until comparison completes, then remove them before measuring release.

This source path could account for about 3.46 GiB of excess reachable backing across E023's two arms at step515, but that is not an observed footprint reduction or the established cause of the resource abort. Other aliases, workspace and allocator lifetime still matter. Complete engine correctness and a separate uninstrumented throughput-versus-complete-request-latency experiment remain necessary after a device diagnostic passes.

Root reran seven CPU checks for OFF identity, uint16 transfer, separate materialization, fallback guards, source pins and forbidden imports. The static contract passed. These checks do not establish GPU equivalence or speed.

[Candidate and source reasoning](../../../bench/m5_ultra_256gb/conv_tail_owned_copy/README.md) · [Cache capacity analysis](../../../bench/reviews/e023-cache-memory-budget.md)


## Device result and what it proves

After preregistration `e03ff2d`, three fresh processes completed 48 cases and 528 recorded phases. Exact original uint16 words survived both lazy and immediate copies, including the full payload domain; all finite fixtures stayed finite. For T978, the original tail retains 20,103,168 allocator bytes, while either ON form retains 65,536 bytes after old references are dropped. Every case returns active arrays to zero on final release. This is an actual ownership result, not a projection from shape arithmetic.

Full active-memory deltas include other temporary buffers: the finite-validation branch releases 98,440 bytes even in OFF. The retained-buffer API and alias union, rather than that aggregate delta, establish the optimized storage. This is why a memory graph needs explicit roots and phases.

The observed pressure-2 system state stayed unchanged under this separately preregistered no-model diagnostic; free physical pages stayed above 69.53 GiB and no new swapping appeared. Model and performance gates still require pressure1. All experiment processes were reaped and the original service was resumed, not restarted.

[Full evidence and memory figure](../../experiments/m5-ultra-256gb/E026-conv-tail-tiny-v1/README.md). No new point is added to the throughput/latency Pareto chart until complete model correctness and an independent HTTP performance run pass.

## Integrate only the ownership change

The diagnostic hook keeps the original prefill code object and convolution output.
It replaces only the returned tail through a private function-global mapping,
limited to selected ordinary eager prefill instances. Both switches remain OFF
by default, and temporary bindings are restored in `finally`. Root
reran 12 CPU checks and independent review found no blocking issue, including
MLX Module parameter-dictionary behavior. This establishes routing and cleanup,
not real-device model equality. The wrapper is diagnostic instrumentation, not
a performance candidate to benchmark with frame inspection enabled.

The next finite test can read only the first real GDN layer
([feasibility and known byte counts](../../../bench/reviews/e026-single-layer-feasibility.md)).
Its approximately 62 MiB weight payload avoids loading an entire 5 GiB shard.
A new resource contract and real-weight numerical checks are still required;
it does not replace the original 37-request full-model qualification.

For a later production-placement candidate, the pinned ordinary runner already
has one shared evaluation at `model_runner.py:158`. Its inputs include decode
batch arrays but omit staging prefill tails. Appending only eligible copied
staging tails to that existing evaluation could avoid 48 separate per-layer
synchronization points. This is a source-based hypothesis, not implemented or
measured here; copy-only, immediate-per-layer, and shared-boundary evaluation
are three distinct interventions and require separate qualification.

## Freeze a real-weight test before measuring performance

The [layer-v1 registration](../../../bench/m5_ultra_256gb/conv_tail_owned_copy/prereg-layer-v1/README.md) now pins the real 14-tensor bundle, original GDN computation, three switch configurations, eight input/state cases per configuration, and resource/deadline rules. Root passed 27 CPU tests; an independent auditor passed 31 negative checks. Full original output, convolution output and both states must compare exactly. This remains a numerical/state diagnostic, with no HTTP timing or new Pareto point.

The first real-weight attempt stopped before a case: an overly strict equality check on two OS footprint readouts rejected a 64 KiB discrepancy, although each was below 2 GiB. [Its failure evidence is preserved](../../experiments/m5-ultra-256gb/E026-conv-tail-layer-v1/README.md). Bound each raw observation independently; do not increase the budget to fix a parser assumption.
