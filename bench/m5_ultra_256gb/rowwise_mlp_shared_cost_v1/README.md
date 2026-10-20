# E057: shared MLP + shared head verifier qualification and local cost

**Prepared source only; no E057 plan, owner preparation or GPU run.** Default OFF.
Actual run refuses missing/failed E056 canary or independent raw/provenance audit.
The fixed E056 auditor source is sealed; its future actual plan/result/audit refs
remain unavailable until the canary completes. No placeholder result is accepted.

`screen.py` is a 171-line explicit adapter around the frozen E054 `device` code
object. A private `FunctionType` namespace replaces only `candidate_run`, `write`
and the helper object used for parameter receipts; frozen globals/code are not
changed. The original 400 timing calls, native device primary/host secondary
baselines, raw tensor comparisons, canonical checks and cleanup remain in that
same device loop. The new top result does not pretend to be an original E054 run:
it records the actual candidate/source pins, old plan ref, exact inherited
contract, and explicit adaptation description under its own schema.

Both shared paths are required: 128 actual gate/up calls per forward plus one
shared head call. MLP SiLU comes from the source-pinned original model class's
`nn.silu`; runtime enrichment is a fresh namespace. Original M1 norms, mixers,
down projections and residual operations remain in the candidate. No fallback
or shadow-only replacement is used. Each of six untimed qualification calls
checks actual ordered live parameter handles and records all128 MLP callbacks;
original counted-head receipts are retained. Timing forwards use the projector
directly and do not add that observer. Python call counts are not GPU counters.

Before any timing, all six k=1,T/2,T cases at T4/T8 must pass fullT logits,
selected128 states, committed lens/pending and native continuation checks.
Together with the two native baseline checks per width, this retains the exact
16 output/state pairs, 32 raw files and2064 tensor pairs. Canonical storage checks
remain unchanged. An extra `parameters-before-timing.json` rechecks every original
parameter value after numerical comparisons; only a weak model reference is
retained for this gate, preserving cleanup. Original before/after parameter and
post-timing canonical checks remain. Failure prevents timing or successful status.

## API for the existing external owner

```python
prepare(e054_plan, e056_plan, e056_result, e056_audit, deadline_s=900) -> plan
validate(plan) -> (base_module, candidate_module, old_e054_plan, q, deps, seed)
```

Every reference is exactly `{"path": ABSOLUTE_PATH, "sha256": ACTUAL_SHA}`.
The child plan schema is `C1-MLP-shared-cost-plan-v1`, with exactly:
`schema, adapter_sha256, e054_plan, e056_plan, e056_result, e056_audit, deadline_s`.
The deadline must be finite, positive and at most1800seconds. The owner separately
binds E056 resource/restoration closure, process identity, runtime, source/model
and physical limits; this adapter creates no new lifecycle or resource policy.

```sh
.venv/bin/python -B bench/m5_ultra_256gb/rowwise_mlp_shared_cost_v1/screen.py
# Only under root's later bounded owner, after E056 closure:
.venv/bin/python -B bench/m5_ultra_256gb/rowwise_mlp_shared_cost_v1/screen.py \
  --run --plan ABS_CHILD_PLAN --plan-sha256 ACTUAL_SHA --out FRESH_NUMERICAL_DIR
```

Result schema: `C1-MLP-shared-cost-result-v1`; success: `PASS_LOCAL_COST_ONLY`.
Required flags: `numerical_screen_qualified`, `pre_timing_parameters_unchanged`,
`device_cleanup_completed`, `source_model_pins_rechecked`; all must be true.
`pid, ppid, plan_sha256` bind the child. `qualification` has6 original head
receipts; `raw_comparisons` has16; `timings` has4 cases/400 samples.
`mlp_qualifications` has6 `{T,k,status,call_count:128,scope,calls}` receipts, with
ordered `{layer,projection,T,K:5120,N:17408,weight_shape,scales_shape,input_dtype,
output_dtype}`. Root-level original file names are retained; there is no `base/`.

The independent E057 auditor can reuse frozen E054v2 raw layout/pair/statistical
functions unchanged, adapting actual owner/source/plan/result bindings and
checking the6 new128-call receipts plus pre-timing parameter receipt. Original
canonical receipt names keep `root/root/`, as in E054v2. All model/HTTP/Pareto
qualification flags remain false. No drafting, acceptance or rollback timing is
measured. E054 head-only timings are historical context, not a paired third arm.

## Actual CPU evidence

- v1 (`151b0d`,0.0454s): pre-test guard initialization failed because replacing
  `socket.socket` before importing unittest.mock broke stdlib SSL class creation.
  No tests or forbidden operations ran; exact attempt/source/output is retained.
- v2 (`6f1ead`,0.0678s): pre-imported unittest.mock before enabling the same guards;
  **8/8 PASS**, no failures/errors/skips and forbidden=[]; production/test sources
  unchanged. Guards forbid MLX imports, process creation, signals and sockets.

Both attempts are in `.artifacts/m5-ultra-256gb/rowwise-mlp-shared-cost-cpu-v{1,2}`.
Tests cover source loading, default/rejection, ordered128 calls, original SiLU
injection, untimed-only observation, pre-timing mutation rejection and weak-ref
cleanup. They do not establish device numerical equivalence or speed.
