# Ordered state-log CPU reference — default OFF

This is a falsifiable, stdlib-only reference experiment, not an engine adapter.
It does not import MLX, load a model, install hooks, access a GPU or benchmark
latency. No production optimization is enabled.

## Paper versus hypothesis

[DeltaLog, arXiv:2608.15533v1, 16 August 2026](https://arxiv.org/html/2608.15533v1)
§3 describes a dense base plus bounded update log with periodic merging, using
cumulative decay and low-rank updates. §5.5 explicitly reports **tolerance
equivalence**, not bit equality: accumulation order and mixed precision can
change results. Its GDN maximum reported output relative error is 0.00303.
The evaluated platforms are NVIDIA RTX 4090/H200, with serving on H200; these
results do not qualify an Apple Metal implementation. §6 warns that replay
and bookkeeping can offset saved state writes at small batch/state sizes.

Our separate hypothesis: deferring *storage* can preserve a specified sequence
of rounded scalar operations if each state element replays each original update
in order. It must not fold decay factors, sum deltas, distribute a dot product
over the log, or otherwise reassociate operations. This variant is not a
reimplementation or validation of the paper's kernel.

## Exact toy contract

`R` rounds a scalar to binary32 using Python `struct`, retaining subnormals and
rejecting nonfinite results. Each multiply/add/subtract is separately rounded;
there is no FMA. `ex`, `ks`, `qs`, key/query/value and beta are supplied FP32
inputs. The toy computes no exponential or normalization.

For each value column, with ascending key-index reduction:

```
kv = fold_R_add(R(R(ks * key[i]) * old_state[i,j]))
delta[j] = R(R(value[j] - R(ex * kv)) * beta)
new_state[i,j] = R(R(ex * old_state[i,j]) + R(R(ks * key[i]) * delta[j]))
output[j] = fold_R_add(R(R(qs * query[i]) * new_state[i,j]))
```

The independent `Dense.step` constructs the full updated matrix. `Deferred.step`
computes its own delta from its own logical old state; no oracle delta/state is
fed into it. Its immutable entries preserve `(ex, ks, original_key, delta)`.
`cell` replays those entries in order. Every Mth update commits the replayed base;
other updates append factors. At most M−1 entries persist. Immutable log tuples
may be shared by clones; reset/append/retire replace each handle's own objects.

The local native source has the related `old S → kv → delta → new S → output`
structure at `tutorial_code/miniengine/kernels/gdn_norm_reuse_metal.py:98–120`.
However, native uses Metal `exp`, normalization, split reductions, compiler
math/FMA behavior and model-specific input conversion. This toy establishes
**none** of those contracts and cannot grant Metal, model, HTTP or performance
qualification.

## Checks and limits

The explicit run covers M=1/2/4/8 × 64/128 updates × three input banks (24 cases),
8×6 state, full original-FP32 state and output comparison after **every** update.
Banks exercise random inputs, signed cancellation, one-ULP factors, subnormal
values, and zero/unit decay. Each case resets after a pending log, clones a
live state, advances the clone independently, verifies original state unchanged,
retires the clone, and rejects reads/steps/reset/clone on the retired handle.
M=1 necessarily has no pending log. Merge boundaries and partial tails are kept.

The two-step scalar counterexample starts at 2²⁴, key=0.5, ex=ks=beta=1,
values `[8388610, -25165824]`. The original delta formula yields
`[2, -33554432]`. Ordered updates give FP32 **0**; combining the two deltas first
gives FP32 **1**. It disproves a general bit-equality claim for that algebraic
rewrite under this toy contract, not for every input or every possible kernel.

The write counts are an analytical model: dense writes N·Dk·Dv words; a merge
writes Dk·Dv, an append writes 2+Dk+Dv factors. Reset segments determine merge
counts. We report any pending tail and cost of an optional final flush.
Initialization/reset/clone, output, Python temporaries, verification snapshots,
log reads and replay computation are excluded. The Python implementation
reconstructs state frequently and may be much slower; **no measured speed or
physical memory-traffic saving is claimed**. A real implementation must first
preserve its native arithmetic contract and then measure total traffic, register
pressure, replay cost and serving latency independently.

## Reproduce

No arguments only describes the experiment. A run requires a new output file:

```sh
python3 -I -B bench/m5_ultra_256gb/state_log_exact_reference/toy.py
python3 -I -B bench/m5_ultra_256gb/state_log_exact_reference/toy.py --run --out /tmp/state-log-new.json
python3 -B -m unittest bench.m5_ultra_256gb.state_log_exact_reference.test_toy_cpu -v
```

The JSON records each completed case, trajectory hashes, exact-comparison totals,
lifecycle checks, theoretical writes, the counterexample, and first failure if
any. The explicit output uses exclusive creation. The first actual execution
and test stdout/stderr/source copies are retained separately; failed attempts
are not overwritten. A PASS applies only to these enumerated toy trajectories.
