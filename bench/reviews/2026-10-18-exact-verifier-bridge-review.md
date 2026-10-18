# Exact verifier bridge: source review and smallest next diagnostic

Source-only review, 2026-10-18. No backend import, test, device measurement,
new qualification or source change to a candidate/engine. The direction in
[the preceding note](2026-10-18-exact-verifier-direction.md) is useful: amortize
target weight work over committed tokens. The first deliverable should be a
falsifiable numerical bridge, not a drafter or another lifecycle framework.

## Actual boundary and likely first differences

There is no separate `SpecRunner.verify` method. Its target transaction is
[`SpecRunner.step:51–59`](../../tutorial_code/miniengine/engine/spec_runner.py#L51):
`snapshot()` each staging cache, `model(block, ForwardContext('prefill', caches),
capture=...)`, evaluate posterior/features, then `rollback(n+1)`. `T` below is
**total block width including the anchor**; T2/4/8 offers 1/3/7 draft tokens.

Ordinary C1 instead calls `runner.decode_model` with one token, one explicit
position, lens=position+1 and max_len=position+1
([model_runner.py:58–80](../../tutorial_code/miniengine/engine/model_runner.py#L58)).
Its optional ahead call at lines82–102 consumes the device argmax on a separate
state. A teacher-forced experiment must not consume that ahead state or let
candidate logits determine the next input.

| Stage | Native C1 decode versus the current verifier | Consequence |
|---|---|---|
| First layer input norm | Both BF16 paths use `norm_metal`; threadgroup size depends on feature width, not row count ([layers:31](../../tutorial_code/miniengine/model/layers.py#L31), [norm_metal:60](../../tutorial_code/miniengine/kernels/norm_metal.py#L60)). | A useful first checkpoint, not a new device proof. |
| Quantized projections | `Linear.__call__` passes all rows to QMM. T>1 can select a different reduction than M1 ([layers:79](../../tutorial_code/miniengine/model/layers.py#L79)). | Earliest plausible divergence is first-layer QKV/z/b/a, before any recurrent state update. E027 already observed native T2 vs independent M1 inequality on synthetic N10240 input; real activations still need this bridge. |
| Conv and gates | Prefill conv sums four separate products with Python `sum`; decode uses `(xp*w).sum(axis=1)`. Prefill calls `_gates`; eligible C1 decode uses compiled `_decode_gates` ([gdn:26](../../tutorial_code/miniengine/model/gdn.py#L26), [gdn:142](../../tutorial_code/miniengine/model/gdn.py#L142), [gdn:208](../../tutorial_code/miniengine/model/gdn.py#L208)). | Projection equality alone cannot qualify the path. Preserve gates ON / norm-reuse OFF of the target serving configuration; do not silently substitute an OFF baseline. |
| GDN state/output | T2/4/8 prefill uses chunk8: cumulative log-decay, triangular solve and matrix products. Decode uses native `gdn_metal`, SPLIT4, sequential normalization/dot/reduction and `delta=(v-ex*kv)*beta` ([gdn kernels:43](../../tutorial_code/miniengine/kernels/gdn.py#L43), [Metal:33–58](../../tutorial_code/miniengine/kernels/gdn_metal.py#L33)). | Algebraically equivalent forms have different rounding, exp/rsqrt and reduction routes. Even **T1 prefill is not the C1 oracle**: it uses pure-MLX recurrence after a different conv expression. |
| Attention | Prefill uses scalar-offset RoPE and T-query causal SDPA; decode uses tensor positions, KV scatter, one query and an explicit boolean mask over `max_len` ([attention:39](../../tutorial_code/miniengine/model/attention.py#L39), [attention:79](../../tutorial_code/miniengine/model/attention.py#L79)). | Same valid context does not prove the same SDPA route or bytes. Do not batch queries in a purported exact verifier without a new gate. |
| Later MLP/head | Gate/up/down and vocabulary projections also change M from1 toT; residual/activation shapes differ. | Testing only the GDN input projection cannot establish full-vocabulary equality. |

These are source-derived risks, not a diagnosis of an unrun full-model failure.
E027's failed case stopped before shared-W was invoked. Its FAIL remains FAIL;
passing smaller shapes do not qualify a replacement. See the
[original result](../../docs/experiments/m5-ultra-256gb/E027-native-bridge-v1/README.md).

## Small entry, complete gate

Proposed new file `bench/m5_ultra_256gb/exact_verifier_bridge/qualify.py`, default
no action; explicit original `--plan --plan-sha256 --out` for an already owned
child. Use the existing cold observed owner and original E023 source/model/D8
bindings. No HTTP server, new scheduler, downloaded drafter or fake SpecRunner
is necessary. The plan fixes T=[2,4,8], two original D fixtures (shortest/longest),
early/after128-committed-decode anchors, normal EOS, source/settings and cap.
That is at most12 bridge cases, stopping at the first original-bit difference.
Early EOS means missing coverage, not synthetic continuation or ignore-EOS.

1. Obtain each canonical initial state from a real target-only C1 continuation.
   Retain all128 arrays, valid length, allocated KV capacity and input token.
   Use `runner.decode_model` with the **actual registered native flags** as the
   reference; it retains short compiled/fused behavior where eligible. No new
   E024/GDN/E028 installation is part of this first native-verifier bridge.
   Check independent loaded parameter values and canonical state before arms.
2. Build the teacher-forced block from a separate reference-only continuation.
   Preserve its explicit token IDs and positions. Both arms consume those same
   fixed IDs; no candidate argmax is fed back. Each case starts from independently
   reloaded/materialized copies of the same canonical state, with enough of the
   original padded KV capacity to avoid accidental capacity-route differences.
3. Oracle: T ordinary single-token decode-context calls, collecting the entire
   vocabulary logit vector after each call. Candidate: the exact existing
   prefill transaction expression from `SpecRunner.step`, with `last_only=False`,
   and its normal snapshot/record semantics. No drafter cost or acceptance is
   involved yet. If capture is requested, pin actual capture-layer IDs and also
   preserve its returned features; do not invent a trained drafter identity.
4. Canonicalize **representation only**: BatchKV K/V versus AttnCache keys/values,
   and BatchState conv/S versus GDNCache conv_state/S with the singleton batch
   dimension restored. Compare T×full-vocab original-dtype logits and all128
   final arrays, including the full allocated KV backing; require finite and
   unsigned bit equality. Check cache offsets and unchanged seed arrays.
   Different cache Python classes/record bookkeeping must be recorded separately,
   not confused with a numerical tensor difference or silently discarded.
5. Reuse E023 `snapshot`/`Comparator` unchanged, including saved first-failure
   safetensors. Reuse isolation `dump` and `parameter_values` for original bytes
   and provenance. Keep source/model pre/post gates, failure prefix and cleanup.
   A full bridge PASS covers only these fixed teacher-forced blocks.

The initial bridge does not need per-layer observations. On a failure, one new
bounded diagnostic replays the **same failed block** and checkpoints input norm,
first GDN projections, conv, q/k/v+gates, recurrent output/state, residual/MLP,
then the first attention layer and final head as needed. Stop localizing once
both sides have equal inputs but distinct outputs. Such observations are never
performance samples.

## Rollback is a separate, concrete blocker to check

`GDNCache.rollback(n<T)` recomputes with `gated_delta_rule_chunk`'s default64,
whereas short verify used chunk8
([cache:56–75](../../tutorial_code/miniengine/cache.py#L56)). Even a forward PASS
would not establish accepted-prefix state equality. `AttnCache.rollback` only
changes offset; rejected suffix KV bytes remain. Thus a strict **entire backing**
comparison against serial prefix state can fail even when valid-history bytes
match. Do not quietly mask those bytes or reinterpret the gate.

After forward qualification, compare each keep=0..T against an independently
executed native prefix and then the same next token. Record both valid-history
and full-backing results; under the proposed full-state-exact contract either
failure rejects that implementation. A future verifier may choose exact saved
per-prefix states (including backing restoration) rather than chunk replay,
but that is a new implementation and its copy/release cost must be measured.
Recurrent S alone is144MiB per boundary; T+1 boundaries are432/720/1296MiB at
T2/4/8, excluding conv/KV/features/temporaries. These are dimension-derived sizes,
not physical peaks or an admission decision. Normal EOS, rejection/cap, actual
SpecRunner accept/rollback, and scheduler/ahead transitions remain later gates.

## Reuse, then a measured go/no-go

- `pareto_realistic/qualify_native.py:71,127`: metadata-preserving snapshots,
  original dtype/finite/bit comparator and first failing tensors.
- `long_decode_graph/qualify.py:234`: real B1 decode contexts, all64layers/128
  arrays, independent loading and direct-call gates. Do not copy its hardcoded
  gates-OFF diagnostic setting over the current serving plan.
- `long_decode_isolation/qualify.py:112,124,159`: raw dump, value-based parameter
  receipts and independent state reconstruction; reuse helpers, not its matrix.
- `gdn_prework_unbound/qualify.py`: actual early/late D-fixture input acquisition
  and normal-EOS coverage handling. Its layer-local PASS is not this bridge.
- E018's `gdn_verify/verify_history` carries the literal native recurrence and
  all prefixes in one kernel, but accepts only T1/3/5/8. Its repeated performance
  gate was REJECT. It neither covers T2/4 nor full-model verification; do not
  silently extend its enum or use its micro speed as end-to-end evidence.

If the native prefill bridge fails, a defensible larger direction is a new
layerwise verifier: expose all T already-known activations to **M1-preserving**
projection rows, then execute original per-token conv/GDN/attention in order;
proceed to the next layer only after all current-layer outputs exist. Projection
weight reuse remains the hypothesis. E027 does not yet qualify all actual
projection geometries, real activations or T8. Full-vocabulary head is included.
This may save weight passes but retain recurrent arithmetic and explicit prefix
state traffic; no gain follows merely from fewer Python calls.

For a block, let c be actual committed tokens (accepted drafts plus bonus,
subject to EOS/cap), and cost=t_draft+t_verify+t_state/rollback/release. A necessary
local condition is cost < E[c]·t_AR. T2/4/8 therefore have perfect-acceptance cost
ceilings2/4/8 ordinary target steps; at half that committed-token yield the
ceilings are1/2/4 steps. For1.25× local speed, the allowable cost is0.8·E[c]·t_AR.
Measure the actual acceptance distribution and all three costs separately, then
measure the total without double-counting overlapped work. Keep cold compile and
warm steady costs separate. Current HTTP first-output/post-first means are not
GPU decode times and cannot supply t_AR. Only a subsequent observer-OFF full
POST→DONE comparison can establish improvement over the measured oMLX C1 point.
