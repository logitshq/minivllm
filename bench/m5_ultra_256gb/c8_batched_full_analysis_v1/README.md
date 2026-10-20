# E051 complete recorded-history audit

The actual offline audit passed on its first attempt: exit0, 6.395 seconds.
It independently rejoined the original E047 raw SSE/trace and streamed all
553,394,208 ledger bytes (1,034 lines: 1,030 steps and four pair-lifecycle events)
through the frozen full scheduler/array checker. No prefix iterator or shortened
finish is used. Both complete route streams reached EOF and matched their hashes.

- All 16 requests drained, with all 8,192 raw and committed output IDs matching
  the recording and both replay arms; zero terminal EOS in this captured workload.
- 1,324,017 bit-equality receipts and 2,648,034 finite receipts: 2,063 complete
  vocabulary logits, 1,318,912 state arrays, and 3,042 ahead-token arrays.
- Decode B0: two steps; B2/B4/B6: four each; B8: 1,016 steps.
- 1,014 consumed-ahead events, six partial retirements and six batch refills.
- 1,028 graph route calls: bound1023 ×90, bound1536 ×938, no route fallback;
  actual route domains include B2, B4, B6 and B8.
- All 48 distinct GDN hooks installed before tracing and fully restored.
  Aggregate Python candidate/fallback branches are 384/0. These include tracing
  and do not provide a per-B layer breakdown or GPU execution count.

This qualifies the actual batched GDN+graph candidate against eager **only for
the complete registered E047 history**. E047 captured the previous graph-only
HTTP engine: neither its timings nor this numerical result qualify new candidate
HTTP performance. A new candidate capture requires its own same-schedule replay.

Successful raw full tensors were not retained. The audit checks complete,
source-bound device comparison receipts, not offline tensor-byte or arithmetic
recomputation. Resource/restoration review is separate. The 73,514,815,896-byte
MLX peak is an allocator receipt, not measured physical footprint.

Evidence under `.artifacts/m5-ultra-256gb/`:

- `c8-batched-full-v1-independent-audit-attempt1`: original command, source
  snapshot, stdout/stderr, exit status and complete audit JSON; actual PASS.
- `c8-batched-full-audit-cpu-v1`: six CPU tests PASS, including full synthetic
  16-request drain and rejection of missing state, nonfinite/unequal arrays,
  missing release, false completion counts and unmatched route tail.

The audit and CPU-test child blocked MLX imports, process launches and signals.
No device rerun or frozen-source edit occurred.
