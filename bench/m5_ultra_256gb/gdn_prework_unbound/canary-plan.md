# Real activation canary, prepared but not executed

`qualify.py` and its 12 fake CPU test methods are source-only preparation. No
tests, MLX imports, model loading or GPU execution were performed while the
oMLX formal run was active. Candidate and all earlier frozen sources are intact.

The five-field child plan is:

```json
{
  "schema": "GDN-prework-real-layer-plan-v1",
  "qualifier_sha256": "<SHA of qualify.py after review>",
  "candidate": {
    "path": "/Users/ryanli/code/minivllm/bench/m5_ultra_256gb/gdn_prework_unbound/candidate.py",
    "sha256": "1bf5a837c6515ca505bc5d1beaa1718b88e1ccd52609f6814bb995afbe6c1260"
  },
  "e024_plan": {
    "path": "<actual original E024 plan path from inherited owner plan>",
    "sha256": "5025d37768a5fa74d6e3e256a96ff8b8e4e92b31837183d5ffe031d9473e7123"
  },
  "deadline_s": 1800
}
```

The device command, only under root's sole GPU owner, is:

```text
<original pinned Python> -I -B <qualify.py> --plan <child-plan.json> --plan-sha256 <SHA> --out <NEW qualification directory>
```

The existing `cold_cycle_v2/owner.py:prepare` has the needed child-launch API:
it runs the real original E024 environment/source/model preflight, saves its
inherited launch, then executes `experiment.qualifier` with `experiment.plan`
as a single direct child. **The current lifecycle cannot directly follow the
ongoing oMLX formal-v2 cycle.** `cold_cycle_v2/controller.py:209–231` only accepts
the fixed v1/Lithos formal predecessor protocols or the same v2 lifecycle. It
rejects `E023-omlx-D512-formal-cold-journal-v1`, including the retry controller
`8023d0b114c02e7a25cf8bfc305ef0986db3d7086f84cefd6f9c50436a5ded30`
and restoration auditor
`03476fd4f63087209e242ca4e8b072296c1983bca281bf3671e9e5134520c15e`.

Root therefore needs a reviewed, separately versioned predecessor-admission
adapter/lifecycle manifest, with actual new entry/source identities and the
future archived journal/audit pair. Reuse the old single-child prepare,
resource/reap/recovery algorithms; do not modify frozen v2, spoof its `HERE`,
or synthesize a prior recovery. Only that narrow migration is missing, not
another experimental server or generic owner framework.

Use phase `gdn-prework-real-layer`, unchanged 1800-second child,
1830-second work and 15/15-second TERM/KILL budgets. Pin the six child support
sources emitted by `qualify.py`, the actual candidate's seven core dependencies
via the inherited full source manifest, and the future independent auditor as
outer evidence. The lifecycle manifest and *actual archived* predecessor
RESTORED journal/audit must be supplied by root; this document creates neither.
The existing owner/resource/restoration gates remain separate from canary PASS.

## Fixed experiment

Use the shortest and longest of the eight frozen D512 prompts (ties ordered by
fixture ID), without retokenization, repetition, truncation or altered prompts.
The original eager model has gates ON, norm reuse OFF, no graph/packed decode
and no drafter. The first fixture records decode steps 1–3; the second records
129–131 after actual ordinary greedy generation. Normal EOS is respected;
insufficient coverage is INCOMPLETE, never a replacement token or selected
different fixture. The late chain has actual used lengths over 1024.

Capture the first, middle and last of 48 GDN layers (ordinals 0/24/47, actual
indices recorded and loaded structure checked). Only the selected instances'
`_decode` methods are observed; every original return value is used unchanged
by the full model. Save full original BF16 x, conv and FP32 S at each selected
node, its full original output/state, and the six actual full-vocabulary logits
that produced the next reference IDs. The unselected rollout steps retain
their causal token trace, not tensor snapshots.

After reference capture, install the default-OFF candidate explicitly on those
three idle instances. Each fixture/layer chain loads its own initial state
once. Subsequent candidate calls receive reference activations but the previous
candidate state; no later reference state is injected. Separate evaluated
storage and original-input immutability are checked. Every output `[1,5120]`,
conv `[1,3,10240]` and S `[1,48,128,128]` must have exact dtype, shape, finite
values and bit equality under the frozen Comparator. All 18 candidate output
triples are saved before the bit gate, retaining the first failure. The real
adapter's per-call counter must increase exactly once with no fallback; full
evaluation follows the call. These are evaluated Python dispatch observations,
not profiler evidence about individual compiled Metal kernel invocations.

The three selected layers' entire original parameter bytes are hashed before
capture, after reference capture and after candidate replay. Full model shard,
fixture and source pins are checked before and after. Instance hooks restore
on errors; a foreign wrapper is rejected instead of overwritten. Output includes
`events.jsonl`, `captures.json`, original-dtype safetensors, parameter receipts
and a terminal `complete.json`. First-failure and partial artifacts survive.

## Limits and next gate

PASS means only these 18 local calls / six three-step state chains. The actual
candidate never generated the subsequent model activations, and this is not
a candidate whole-model rollout. B>1 fallback, special/strided inputs, changed
parameters, scheduler/ahead/retirement, HTTP, and combinations with E024/E028
remain unqualified. No forward or wall duration is a throughput/latency result.

If any bit differs, stop and retain the raw evidence. Do not reuse short packed
fusion qualification to waive a difference. If these calls pass, the next
necessary gate is the existing complete 37-request same-schedule full-logit,
128-cache/ahead comparison with per-instance install/restore, then real HTTP
capture/replay and observer-OFF formal measurement. An independent raw audit
must verify the saved three-array pairs and causal chain before that promotion.

`audit.py` and 13 small stdlib raw-fixture tests are now prepared, **not run**.
The auditor imports neither producer nor MLX/numpy. Its explicit invocation is
`audit.py --child <closed qualification directory> --plan-sha256 <SHA> --out
<new audit JSON>`. It checks all 60 saved tensor files, independently parses
BF16/FP32 payload bits and finiteness, compares all 54 full output/conv/S pairs,
checks 24 state-continuation pairs and recomputes six complete-vocabulary argmax
IDs. It also validates the exact event chronology, 18 evaluated dispatch deltas
with fallback 0, all 162 device comparison receipts, fixed contract/configuration,
source/plan identity and selected-layer parameter before/after hashes.

The candidate input-before/input-unchanged checks are device receipts: those
historical candidate input payloads were not saved separately. For later steps,
the preceding candidate output state *is* stored and compared offline with the
next original reference input state. Uncaptured reference steps' token IDs are
causal producer receipts, not independently recomputed model inference. The
three layers' parameter tensors are not archived; their complete value hashes
are checked for consistency without claiming offline weight reconstruction.
Missing terminal/raw evidence is INCOMPLETE, and a failed producer or differing
byte remains unqualified. Resource, lifecycle and restoration are separate.
