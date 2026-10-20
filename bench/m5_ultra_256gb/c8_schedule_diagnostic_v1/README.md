# E047: observe the C8 baseline before further optimization

Fixed MB50 / ops50 / BFS20 / sync0, current combined loader with GDN OFF and graph ON. The unchanged frozen worker receives eight warm requests and, after their client cleanup completes, the same eight bodies in a screen wave. Both client phases are explicitly unqualified `qualification` captures. Their labels identify waves, not performance samples.

The new server adapter changes only argument/identity/observer glue: combined `main` and installation are reused verbatim, and the existing route QualificationObserver receives the actual `graph_handle`. The original combined qualification entry required both candidates ON; that frozen entry is untouched. The observer may change arrival order and scheduling, so even a reproduced difference is diagnostic evidence only.

`trial.prepare(base_spec_ref)`, `validate(spec)` and `async run(spec,out,checkpoint=...)` expose a single labelled result compatible with the repaired E045 owner. `runner.py` and `execute.py` rebind private globals to reuse its resource checks, cancellation settlement, exact child identity, bounded cleanup and pause/restoration. The inherited outer envelopes retain their E045 schema/scope text; this E047 plan and per-trial result explicitly carry the diagnostic scope. No new numerical or timing qualification is inherited.

Preparation, default-inert description, and device execution are separate:

```sh
.venv/bin/python -I -B bench/m5_ultra_256gb/c8_schedule_diagnostic_v1/execute.py
.venv/bin/python -I -B bench/m5_ultra_256gb/c8_schedule_diagnostic_v1/execute.py --prepare
```

Preparation checks closed E046 and writes `prereg-v1/{plan.json,commands.json,specs/c8-mb50-two-waves.json}`. Root reviews/tests/commits and executes the exact `commands.json` run invocation. Preparation does no model import, GPU work, pause, or model-weight hashing. Actual execution retains unchanged full runtime/model verification before pausing the original service. Existing 64 GiB child / 240 GiB system / no swap-growth policy is unchanged.

After capture and restoration close, `commands.json` also gives the CPU-only raw join. It rechecks each raw wave, constructs an explicit derived 16-request view, and invokes the frozen `http_join.join` to check every received SSE byte, full prompt/output IDs and actual scheduler event sequence. The original wave groups retain their individual barriers, timestamps and cleanup receipts. The derived view invents no common barrier or throughput. `joined-input/recording.json` has the existing `http_replay` schema and retains both schedules for the next same-schedule eager/graph comparison, particularly D06. It is replay input, not a numerical replay result.

Next device diagnosis should compare eager A/A and graph/ahead A/B on the same observed history, preserving exact prefill chunks and batch memberships. Cross-schedule variation remains distinct from graph errors. This capture cannot establish a cause for E045 by itself.
