# Prepared combination gate

`qualify.py` is the next device **producer**, not an independent auditor or
service controller. It imports only stdlib until explicit complete execution
arguments. Empty arguments describe scope. Old candidate, integration, harness
and lifecycle sources remain unchanged.

The child plan has exactly five fields:

```
schema: GDN-graph-combined-serving-plan-v1
qualifier_sha256: SHA of qualify.py
qualification: {path: original E023 plan, sha256: 21ba52e4…547fb}
sources: exact helper/source_pins() map
deadline_s: 1800
```

Generate it with a new, non-existing destination:

```sh
.venv/bin/python -I -B bench/m5_ultra_256gb/gdn_graph_combined/qualify.py \
  --prepare-plan /absolute/new/combined-plan.json \
  --qualification-plan /absolute/existing/E023-plan.json
```

The external owner launches the child with the established three arguments:

```sh
<pinned-device-python> -I -B bench/m5_ultra_256gb/gdn_graph_combined/qualify.py \
  --plan /absolute/new/combined-plan.json --plan-sha256 <printed-SHA> \
  --out /absolute/new/qualification
```

Use the existing single-child cold owner with experiment phase
`gdn-graph-combined-serving`, qualifier/ref above and `child_seconds=1800`.
Keep its original inherited4dff CPU preparation, work1830, resource gates,
unknown-launch/reap and finally recovery. Do not invoke the device command
outside that admission. The latest actual predecessor must be accepted by the
chosen lifecycle version; neither old GDN nor E024 numerical qualification is
authority to roll back service history. This module implements no lifecycle.

The frozen E024 `adapt`, `parameter_checks` and `require_complete` execute
unchanged against a new installation bridge. The base E023 work remains 8 C1,
16 C8 and 13 lifecycle requests, with full logits, all 128 cache arrays, the
original schedule, sampling, own generated histories, ahead and retirement
checks. The same 42 idle direct calls remain, at used1023/1024 and B1/8/1.
Their exact changed/restored parameters are `norm.weight`,
`layers.0.linear_attn.in_proj_a.weight` and `.scales`. This does **not** add a
conv-parameter replacement test or assert all-parameter/hot-reload coverage.

`route-events.jsonl` retains the original route-call and parameter receipts and
adds `combined-graph-call`. Each combined event records installation, phase,
step/direct label, B/used, all48 before/after counters and actual compiled
callable invocation delta. A trace must choose the same original GDN branch
in all48 layers: candidate for B1, fallback for B>1. Several Python passes during
one compiler invocation are recorded without assuming one trace. A zero-delta
compiled call is allowed only after that installation has actually observed a
trace for the same bound/B. This labels Python tracing versus callable reuse;
it is **not** a kernel execution/completion count or compiled graph inspection.
Failed submissions retain their before/after counts and original failure.

No extra candidate forward, evaluation or token substitution is introduced by
the witness. The original numerical comparator evaluates the same output/state
trees. Constructor failures unwind the original observer before removing graph
and GDN scopes. Normal teardown removes the numerical observer, graph witness,
compiled caches and GDN hooks in that order.

Complete producer PASS requires the original42/37 gates and two complete
48-layer installation/restore lifetimes, B1 traces in both graph bounds,
multirow fallback tracing, and compiled reuse. `complete.json` records
`same_schedule_checks_passed`; it leaves `independent_audit_required=true`,
`http_qualified=false`, and `performance_qualified=false`.

Before treating device PASS as qualification, bind a new independent audit to
this plan/source and reuse the original ledger arithmetic/FCFS/state checks,
E024 parameter/route gates, and the new trace rows. Original frozen auditors
hard-bind their old schema/source and cannot be presented unchanged as auditors
of this new combination. Actual HTTP capture/replay and observer-OFF timing are
later gates. No device qualification has been run for this new entry.
