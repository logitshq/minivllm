# v12 display policy, fixed before the E024 ON results

These are prepared source files, not a generated result. Neither script has been run. v11 remains current.

- Keep all 14 v11 process observations and their metric values unchanged.
- Original point dictionaries remain intact; their old nondominance annotations retain the v11 fourteen-point scope and are not used by the v12 renderer.
- Add all 12 E024 OFF/ON observations only after the complete independent formal audit, its integrated raw resource checks, independent restoration, and archive verification pass. No successful prefix is promoted. On FAIL or INCOMPLETE, do not generate a v12 speed update.
- Show 26 actual observations in separate p50 and p95 POST-to-DONE panels. Historical native points use gray hollow markers and remain a separate n=1/C configuration. They are never pooled into paired native OFF n=3/C.
- Show process-metric medians and observed min–max only within the same configuration and concurrency. These are not pooled request percentiles or confidence intervals.
- Show the six adjacent ON/OFF ratios from the frozen independent auditor. Do not infer pairing with historical measurements, identical realized arrival schedules, graph-hit counts, a global fastest engine, or an unmeasured frontier.
- Keep GDN canary/37-request qualification separate. It adds no speed point and does not qualify a combined E024+GDN configuration.

After the root confirms the actual archive is complete, the proposed commands are:

```sh
/Users/ryanli/code/minivllm/.venv/bin/python -I -B /Users/ryanli/code/minivllm/docs/optimization-guide/progress/2026-10-17-v12/prepare.py --run
/Users/ryanli/code/minivllm/.venv/bin/python -I -B /Users/ryanli/code/minivllm/docs/optimization-guide/progress/2026-10-17-v12/render.py --run
```

Preparation copies only closed small receipts and source files, binding them to the verified archive manifest and the fixed plan/auditor identities. It does not reread request ledgers, model/runtime payloads or the tar stream. Rendering revalidates copied receipts and recreates the display data before importing Matplotlib. Both commands refuse overwrites and are inert without `--run`.

Save actual commands/stdout/stderr separately when executing. Inspect the PNG/SVG and record visual QA before changing any root or guide “current” link. Neither script changes such links; a render receipt explicitly leaves visual review incomplete.
