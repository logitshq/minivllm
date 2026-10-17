# E026: copy a small state, release a large prefill buffer

2026-10-17. This is a default-off candidate with CPU/source checks only. It has not yet passed device numerical or allocation checks, been integrated into the model, or produced a performance point.

The recurrent state we need after a prefill is small, but a small view is not necessarily a small allocation. The frozen GDN implementation stores `xp[T:]`: three BF16 rows of width 10240, or 60 KiB per layer. MLX 0.32.3 slicing shares the parent buffer. With a roughly 1000-token prompt, that handle can retain about 20 MiB per layer. Across 48 GDN layers, this makes ownership worth investigating before changing the math.

The isolated candidate is one operation: `mx.contiguous(tail.view(mx.uint16)).view(mx.bfloat16)`. In this pinned MLX version, contiguous copies even an already row-contiguous view when the backing exceeds logical bytes by more than 16 KiB. The integer word view avoids a floating-point conversion. The helper defaults to OFF, checks its exact domain, and does not patch production code. A separate materialize switch makes an immediate `mx.eval` explicit; a lazy copy cannot be assumed to free its parent immediately.

The first question is measurable without model weights: do the original bits survive and does the retained output own a small buffer after old aliases are dropped? The public `get_array_buffer_size` API counts unique full allocator buffers. Use it alongside active/cache/peak memory; summing shapes or counting Python objects cannot establish storage ownership. Hold parent/oracle aliases only until comparison completes, then remove them before measuring release.

This source path could account for about 3.46 GiB of excess reachable backing across E023's two arms at step515, but that is not an observed footprint reduction or the established cause of the resource abort. Other aliases, workspace and allocator lifetime still matter. Complete engine correctness and a separate uninstrumented throughput-versus-complete-request-latency experiment remain necessary after a device diagnostic passes.

Root reran seven CPU checks for OFF identity, uint16 transfer, separate materialization, fallback guards, source pins and forbidden imports. The static contract passed. These checks do not establish GPU equivalence or speed.

[Candidate and source reasoning](../../../bench/m5_ultra_256gb/conv_tail_owned_copy/README.md) · [Cache capacity analysis](../../../bench/reviews/e023-cache-memory-budget.md)
