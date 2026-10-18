# v13: same display policy, corrected offline tokenizer-path audit

Prepared source only. Neither v13 script has been run. v11 remains current; v12 source and its actual `NO_NEW_SPEED_CHART` result are preserved unchanged.

The first E024 formal auditor rejected the registered Hugging Face `tokenizer.json` symlink before opening HTTP raw data. A separate v2 auditor permits this one exact registered snapshot reference after checking its model-manifest SHA, original pre/post target inode/size/mtime and actual regular target payload hash. The generic receipt path gate, raw HTTP metrics, resource thresholds and ownership checks are unchanged. The original FAIL and archive remain retained. This is a post-run parser correction; no timed experiment was rerun.

The v13 builder binds the corrected audit source/schema, the actual eight regression tests, and the complete verified `E024-route-http-formal-v2-reparse-v1` archive. It refuses to produce output unless every formal/resource/restoration/retention gate passes. An incomplete or failed reparse leaves v11 current.

The display policy is exactly the one fixed in v12 before the ON results:

- Preserve all 14 v11 process observations unchanged; append all 12 E024 OFF/ON observations only together.
- Show 26 actual observations, 10 configuration/concurrency summaries and six audited adjacent pairs. Historical native n=1/C stays gray/hollow and separate from paired native OFF n=3/C.
- Keep p50 and p95 POST-to-DONE panels separate. Compute process-metric medians and observed min–max only within the same configuration/C; no pooled percentiles or confidence intervals.
- Display frozen auditor ON/OFF ratios. Do not claim identical realized arrivals, formal graph-hit observation, a global fastest engine, a fitted frontier or proof of every future schedule.
- Keep GDN canary/37-request qualification separate, with no GDN speed point or E024+GDN composition claim.

After the root confirms the corrected archive and reviews these sources:

```sh
/Users/ryanli/code/minivllm/.venv/bin/python -I -B /Users/ryanli/code/minivllm/docs/optimization-guide/progress/2026-10-17-v13/prepare.py --run
/Users/ryanli/code/minivllm/.venv/bin/python -I -B /Users/ryanli/code/minivllm/docs/optimization-guide/progress/2026-10-17-v13/render.py --run
```

Both commands are inert without `--run` and refuse overwrites. Preparation reads closed small receipts and source files, not request ledgers, model/runtime payloads or tar contents. Rendering rederives display data from copied receipts before importing Matplotlib. Record actual commands/stdout/stderr, inspect the PNG/SVG and record visual QA before any current-link update. Neither script changes those links or marks visual QA complete.
