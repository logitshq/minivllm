# C1 full-verifier cost screen: proposed, not executed

E042 qualified fixed forward prefixes, not performance. Its 58.171206-second
producer duration includes parameter checks and roughly23 GiB of snapshots; it is
not a verifier latency. Reuse the qualified rowwise candidate and shared head
without editing their sources. No GPU work, model hashing or timing was done for
this design.

## Smallest useful domain and reuse

Start with the original shortest D fixture at **used1023, T4/T8**, covering the
1023/1024 boundary without the separate short-prompt diagnostic. Keep the exact
E042 model, MLX0.32.3, gates ON/norm reuse OFF,1536 backing and four MLX environment
variables. One model; both arms use its same original parameters. No compilation
or parameter repacking. Used1024 is a follow-up confirmation if the screen wins.

- `rowwise_native_verify/qualify.py:76` validates the existing child plan/source
  closure; `seed_cases`, `seed_admission`, `prefix_ownership` and `counted_head`
  provide the existing shape/provenance checks.
- `rowwise_native_verify/candidate.py:21` is the complete layer-major verifier;
  its pinned identity is
  `4ce1b1720dc256b01d30a0f9b19b109bde9949050e98b50a31d308f9e71c8f15`.
  Pass the already-qualified `nvfp4_multirow_shared/candidate.py` projector
  (`d1826b3c4d07d44adbae071ef2da43197348c09b2d8be1035eb1a0c2788d132e`).
- `long_decode_isolation/qualify.py:159` restores fresh cache descriptors from
  evaluated canonical arrays. `exact_verifier_bridge/qualify.py` provides the
  original canonical-state naming and teacher-only token generation.
- Actual E042 evidence is
  `docs/experiments/m5-ultra-256gb/E042-shared-head-prefix-v1/README.md` and
  `.artifacts/m5-ultra-256gb/rowwise-head-shared-v1-child`. Read only the selected
  seed/prefix files when executing; no need to copy/requalify all96 files.

A proposed new default-OFF `rowwise_native_verify_cost_v1/screen.py` can inject
these existing functions; no new lifecycle framework is needed. Its future CLI
would require `--run --plan FILE --plan-sha256 SHA --out NEW`. This module and
plan do **not** exist yet. Reuse an existing pause/resource owner after separate
source/CPU review; do not run it during another device experiment.

## Timing boundary and fair baselines

Load/verify model, establish the canonical seed, create independent arm storage,
and warm each route outside timing. Each sample constructs fresh contexts and
operations; never reevaluate an already-evaluated output. Include fresh descriptor
setup inside both arms. Do not clear the allocator cache asymmetrically or include
file I/O, parameter hashing, comparator or full snapshot serialization.

The candidate measured interval is: start after an idle synchronize; construct
all64 layers and shared head; evaluate **all T full-vocabulary rows and all T
complete prefix states**, as E042 did; compute/return greedy IDs; select a
registered prefix; publish it into an isolated `DecodeBatch` holder; drop
unselected prefixes/temporaries; final synchronize; stop. Record total wall ns.
Keep that total as primary: internal host build/submit timings are overlapping
observations and must not be added as GPU-exclusive time.

The frozen native `model_runner.py:84–102` already keeps the next greedy token on
device and submits ahead. Therefore two baselines have different meanings:

1. Original E042 sequential teacher: T fresh native `decode_model` calls, each
   logits+state evaluation and host argmax feeding the next call. This is the
   numerical oracle and a secondary synchronous-reference cost, not the strongest
   existing C1 engine.
2. Token-major native **device-dependent AR(T)**: same native B1 calls, next token
   from that call's own device argmax, per-step `async_eval(token,state)` as the
   frozen native route does, independent KV wrappers, all full logits retained,
   final logits/state materialization and IDs returned once. It cannot use the
   known teacher IDs to skip causal dependence. This new submission order must
   first pass byte comparison against the original oracle; it is not assumed
   qualified by E042. Compare candidate against the faster valid baseline.

The second baseline is a forward-cost screen, not the complete production ahead
scheduler. A win still needs comparison to the current qualified GDN+graph C1
engine under its exact contract, then drafter/acceptance/HTTP measurements.

Use3 untimed warmups and5 groups of10 paired samples, alternating AB/BA. Retain
all individual elapsed ns and order, report group medians and paired ratios, plus
p50/p90 totals. Resource probes are shared external interference: keep their
boundaries in the result; never discard a slow valid sample post hoc.

## Prefix commit is new work and must be verified

For prefix k, transfer its64 cache wrappers to an isolated one-request
`DecodeBatch`, preserve req_ids/backing, and set lens to `[seed.position+k]`.
The next pending token is greedy row `k-1`; its next position is
`seed.position+k`, not the previous final row's bound. Release all rejected-prefix
references before final synchronization. No GDN replay, casting or KV truncation.
This is a local holder commit, not request/scheduler publication.

First validate k=1,T/2,T outside timing: exact selected full logits and128 state
arrays, untouched canonical input, then one native continuation from the committed
holder versus independent sequential original AR(k+1). Retain both original-byte
outputs and fail before timing on any mismatch; T8/k8 continuation is a new ninth
step, explicitly outside E042's old eight-prefix coverage. It must not silently
reuse the old qualification flag.

Main ratio uses **k=T only**, matching T cache advances/output rows to AR(T).
For k=1/T/2, report commit mechanics and wasted verifier cost versus AR(k), never
call AR(T)/candidate a realized speedup. These teacher blocks contain correct
original tokens; choosing a shorter prefix is a commit diagnostic, not evidence
of real draft rejection handling or an acceptance rate.

Required receipts: exact source/model/runtime/seed identities; full numerical
comparison status; chosen k and committed lens/next ID; raw timing order/durations;
all failures and memory observations; source-bound head call count per block;
producer+reap+restore closure. Keep `http_qualified`, `performance_qualified` and
`engine_pareto_point` false. Only if the complete k=T cost beats the strongest valid
AR(T) comparator is there any measured budget left for drafting and acceptance;
otherwise prioritize a larger kernel/algorithm change rather than the shared
head's isolated gain.
