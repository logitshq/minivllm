# C1 full-verifier cost screen — prepared, not run

This default-inert probe measures complete T4/T8 forward cost at used1023 with the
frozen E042 rowwise verifier and shared head. It loads the original E042 canonical
seed and model; no model source is modified, no engine integration is
added. A bounded external owner is now prepared; no GPU result is claimed here.

New qualification is required before timing: an independent original native host
AR(9) generates causal teacher inputs, full logits and state. For each T, the
candidate materializes all T full-vocabulary logits and only selected prefix
k=1,T/2,T, publishes its 64 wrappers into an isolated DecodeBatch holder, and runs
one native continuation. Original-byte comparisons include all selected 128 state
arrays and full KV padding. T8/k8's ninth continuation is new coverage. Both the
host-sequential and device-dependent native AR(T) routes must match that oracle.
Canonical buffers, IDs, lens and model parameter values are checked. Both sides of
qualification comparisons are saved before comparison; failures stop timing and
remain in the fresh output directory.

The primary native baseline keeps causal token dependence on device and performs
per-step async_eval(token,state), then evaluates all full logits/state and returns
IDs. The host-synchronous oracle is also timed as a secondary comparator; removing
its host synchronizations is not called an optimization. Use the faster valid
baseline when judging any future measured budget for drafting.

Ownership-set checks and counted-head audit wrappers run only in the untimed
qualification gate. Timed calls pass the actual projector directly.

Each timed sample creates fresh contexts and operations and includes full logits,
selected state, host-returned IDs, local holder commit, rejected-state reference
release and final synchronization. It does not repeatedly evaluate a completed
lazy array. Three untimed warmups precede 5 groups × 10 alternating AB/BA pairs for
each T/baseline, retaining all 400 individual timings and order. Snapshot IO,
parameter hashes and comparisons are outside timing. Only k=T is timed; shorter
prefixes are commit diagnostics without a speed claim. The probe does not force
all rejected states solely for audit. This policy itself passes the new selected
state and continuation checks.

Future owned CLI (requires a separately sealed plan and review):

```text
.venv/bin/python -I -B bench/m5_ultra_256gb/rowwise_native_verify_cost_v1/screen.py --run --plan ABS_PLAN --plan-sha256 SHA --out ABS_NEW_OUT
```

The exact plan keys are schema `C1-verifier-cost-plan-v1`, `screen_sha256`, pinned
absolute `{path,sha256}` refs `e042_plan` and `e042_result`, and a positive finite
`deadline_s` ≤1800. The E042 result must bind that original plan and pass its
forward-prefix qualification. The source validator reuses its frozen dependency,
model, fixture, environment and runtime pins. Both live arms share original model
parameters; each arm separately loads evaluated canonical seed storage. A future
external owner must enforce resource observations, direct child ownership,
absolute deadline, reap and restoration. This file performs no service action.

Scope remains local forward/holder commit cost: no compiled qualification, actual
draft rejection/acceptance rate, scheduler publication, HTTP qualification,
performance qualification or Pareto point. Timed sample outputs are checked for
IDs/lens but their full tensors are not saved each iteration; saved full-byte
qualification covers the admitted deterministic routes and initial states.

The owner has19/19 guarded CPU checks (627aad), source snapshots preserved in
`.artifacts/m5-ultra-256gb/c1-verifier-cost-owner-cpu-v1`. It binds the restored
E052v2 service and uses child1500s/owner1650s/outer1800s deadlines. ACK arrival
rechecks parent identity and deadline before exit, then the actual owner reaps.
Metadata prepare1abb6c produced parentf613b5df1494e9064ba84e2dd025b437177891b00bf5d98bcf18d03e5eb960ce,
child3d10d60dc72eff7e2c70444239c61be09c865b6e15109c7f43b8cdf8a6a66c2c.
