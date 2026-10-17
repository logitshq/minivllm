# Candidate: skip identity gathers during refill

2026-10-17. **Design only, default OFF. No implementation, GPU run, measured speedup, or adoption qualification.** This is the single next candidate from this review. It preserves the current arithmetic and only removes a redundant state-selection expression.

## Exact route and scope

`tutorial_code/miniengine/engine/model_runner.py:170–182` calls `DecodeBatch.rebuild(keep, new)` when decode membership changes. `tutorial_code/miniengine/cache.py:122,128,140–141` creates an index array and gathers old K/V, conv and S even when `keep == list(range(old_batch_size))`.

The proposed bench-only switch applies **only** when `new` is nonempty, every existing row is kept in the same order, and the K/V capacity L is unchanged. Pass the original arrays into the existing padding/concatenate expressions in that branch. Preserve output row order, padding, allocation capacity, all model operations, and the necessary combined output. All other transitions call the original implementation. Keep retirement (`model_runner.py:190–204`), lookahead invalidation, scheduling and EOS handling unchanged; do not defer retirement or combine two scheduler steps.

Reviewed current sources, also matching the frozen E020/E023 core:

| File | SHA-256 |
|---|---|
| `tutorial_code/miniengine/cache.py` | `056a3bf3bc42163e93f55eb914fa35d5c081f803da9b5faa740588a67057a791` |
| `tutorial_code/miniengine/engine/model_runner.py` | `14fd964c57814846887585a9cbbdf1e11d2500ab80c23b10c945d2aa9316b7b1` |

## Existing evidence and its limits

This review read only the final 2 MiB of `.artifacts/m5-ultra-256gb/realistic-qualification-v1-owner/qualification/steps.jsonl`. In D-C8 step 514, the previous committed rows are `c8-4..7`; decode adds `c8-8, c8-9`. Step 515 similarly keeps `c8-6..9` and adds `c8-10, c8-11`. Both transitions therefore have `keep=[0,1,2,3]`, B4→B6 and L1536; both subsequently retire two rows to B4. This establishes actual eligibility in the recorded workload. The prior independent audit binds that ledger to SHA `5c58f5ad169b5d1b79f192d5d46a2f1eb525545cb2e0e06242d589b7643beb8f`; the ledger remains a resource-aborted, incomplete qualification, not a successful full workload.

The old B4 payload selected by those gathers is:

```
4 * (48 * 48 * 128 * 128 * 4       # GDN S, FP32
   + 48 * 3 * 10240 * 2           # conv, BF16
   + 16 * 2 * 4 * 1536 * 256 * 2) # attention K+V, BF16
= 1,018,429,440 bytes = 971.25 MiB
```

There are 128 gather expressions across 64 layers. This is logical payload and Python expression count, **not measured Metal traffic, launch count or saved memory**. MLX may already eliminate or fuse some work. The required B6 concatenate remains.

Existing timing supports investigating state movement, but does not measure this candidate: recomputing the recorded `37 delivered / 11 remaining` transition from all three C64 processes in `pareto-batch-frontier-v1` and `retire-batch-frontier-v1` gives medians **3831.228 ms → 151.104 ms**, 14 samples per implementation including warmup. These are unpaired whole `Engine.step` timings for the already adopted E005 retirement fix. They cannot be attributed to identity-gather removal. E023's step records have no elapsed-time field. **Identity-refill cost and speed benefit are currently unknown.**

## Smallest useful future validation

1. Use a default-OFF, per-engine batch adapter, with an explicit ON branch witness. No global monkeypatch, weight repacking, new kernel or new audit framework. First run a bounded device test of one actual-shape GDN state and one attention state, then the complete rebuild output: B2→4, B4→6 and B6→8 at L1536; controls for reorder, row removal, no new rows and L growth must use the original path.
2. Compare all original-dtype output bytes, IDs/lens and input bytes, including unused K/V padding. Include offset/noncontiguous inputs and independent subsequent writes/evaluation: the new output must not rewrite retained old states or staging states. Use the same evaluation boundaries in OFF/ON; do not force extra materialization on the reference. Record first call, complete build+eval time and active/peak/backing measurements separately. Stop if no measurable redundant work remains.
3. Only after that gate passes and full-model resource admission is available, reuse the existing same-schedule qualification to cover actual refill, retirement, lookahead and all logits/cache/token checks. Then run fresh paired HTTP OFF/ON with the unchanged D fixtures, C1 control and C8 refill workload. The result belongs on the throughput–latency plot only after correctness passes: actual committed tokens divided by the common request window versus complete POST→DONE latency, with p95 and TTFT retained. No micro result substitutes for that point.

## Alias and lifecycle risk

Removing a gather preserves the old array's layout and backing reference until concatenate materializes; the old gathered intermediate had different ownership. Existing views, observer snapshots and lazy graphs can therefore change liveness even with identical values. Keep the original concatenate and lookahead invalidation, verify delayed evaluation and old-state immutability, and measure unique backing rather than summing aliases. Do not claim a 971.25 MiB physical reduction or a leak fix from the payload calculation. This proposal is separate from E026's prefill-tail ownership change and from the rejected E025/E027 arithmetic replacements.
