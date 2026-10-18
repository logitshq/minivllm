# Small next execution boundary — prepared, not run

2026-10-18. `owner.py` and ten fake CPU test methods are source-only and have not
been executed or AST/hash checked. No service, model or performance result is
created. Existing active formal sources and plans are untouched.

The existing multi-server owner is the right operational model: its Python
process directly owns each model-server Popen. Wrapping `capture.run` in a child
of a generic single-child owner would make the model a grandchild and probe the
wrong PID. Existing controllers also hard-code their two/four/twelve phases;
they cannot truthfully accept three diagnostic modes by borrowing a formal name.

This module therefore stops at a **single-phase glue**, not another controller
or a second stop/recover implementation. `main()` describes only; there is no
launch CLI and no executable registration yet.

## Direct outer-owner call

`prepare_modes(formal_spec_ref, output, nonces, capture_source=...,
server_sha256=..., order=('off','runner','full'))` prepares three specs from the
same pinned, validated C1 combined-ON formal spec. Exactly one occurrence of
each mode and distinct drain nonces are required. Original jobs, configuration,
deadlines, tokenizer and count qualification stay identical. All modes run
eight warm plus eight diagnostic requests; nothing becomes a Pareto point.

For each phase, the actual registered outer process calls:

```python
result = await profile_owner.run_mode(
    spec, phase_dir / 'trial', capture_source=registered_capture_ref,
    registered_spawn=spawn, resource_probe=probe, strict_reap=release,
)
```

The callback shapes directly match the existing nested functions in
[`route_http_formal/owner.py`](../route_http_formal/owner.py):

- `spawn(mode, popen_thunk)` uses the existing STOPPED lock, service absence,
  durable global/phase intent and returned-Popen tracker. The glue retains the
  **same actual server object** before outer identity or receipt work can fail.
- `probe(label, pid, cleanup=False)` keeps original service-absence,
  pressure1/AC/thermal/swap/process checks; the glue prefixes labels with mode.
  It runs off the HTTP event loop and settles before propagating cancellation.
- `release(mode)` retains the existing TERM/KILL/wait and exact PID/port queries,
  writing `phase_dir/release.json`. The glue checks this against the actual
  inner release; a clean capture requires natural exit0 in both records.

This is API compatibility with those functions, **not** a claim that the old
frozen twelve-phase run body accepts these new phases or results unchanged.
The glue does not patch imported globals, alter `HERE`/`__file__`, fake formal
point fields, or make any callback return stand in for real release files.

The callback subprocess/read/wait bounds and outer hard owner deadline remain
mandatory. Python threads cannot be safely killed: a cancelled resource/reap
callback is settled before control returns. If one fails or never settles,
there is no successful phase/recovery proof. Unknown Popen preserves the inner
spawn intent and `unknown_launch=true`; this module never emits NO_CHILD_STARTED.
The outer owner's known-PID finally cleanup and unknown-launch refusal remain
authoritative. Repeated cancellation remains FAIL even after successful reap.

## Smallest missing outer adaptation

1. After current timing closure, run the capture15 and glue10 fake tests under
   the guarded CPU runner; freeze the actual capture/server/glue dependency set.
   Then exercise the recipe on one real immutable C1ON formal spec.
2. Register three fresh phases in one chosen order with the existing bounded
   outer-owner algorithms. Keep original157 runtime/source preflight, full
   model hashes/pre-post, ancestor service checks and actual server resource
   attribution. Insert `run_mode` in its per-phase execution point; accept
   `C1-pipeline-owned-phase-v1`, not a 64-request formal point.
3. Add only that explicit three-phase plan/journal acceptance to a new cold
   adapter. Reuse underlying stop/recover/phase-reap algorithms, including
   first/late no-Popen, known-null identity and unknown-launch refusal. Bind the
   real latest independently RESTORED predecessor after it exists.
4. The outer driver must preflight the actual owner CLI before stopping the
   service, retain its invocation uncertainty, and call recovery in `finally`.
   Recovery remains conditional on strict owner/child termination evidence.
   A new diagnostic offline binder must inspect raw counts/closure/profile,
   resources and restoration; neither `glue-result.json` nor callback success
   alone grants those independent qualifications.

The recipe and glue are usable building blocks, but steps2–4 are deliberately
**not implemented or claimed ready** here. No fake controller/auditor file or
future predecessor is supplied to make registration look complete. The result
keeps `full_runtime_model_verified=false`, `restoration_verified=false`, and
numerical/performance flags false. Shared control overhead and overlapping wall
spans still require a real profile OFF/runner/full overhead comparison.
