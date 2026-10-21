**Done**

- Verified `PASS_FEATURES_LOCAL_COST` and `PASS_CAPTURE_AND_RESTORE`; reducer on 60 rows matches `timing_summary` via JSON-normalized keys; independent sorted-central-two recompute matches.
- `device-timing-analysis.py` (stdlib only, no GPU/network) writes `device-timing-analysis.json` with 3-arm median/min/max ms, 5 group ratios, median ratio and capture_with/without ratio; `device-timing-analysis.md` concise summary; `device-timing-meta.json` holds command/exit and SHA hashes.
- Scope disclaimer included: 10 samples/arm/width, no CI, T=width not concurrency, native = ORIGINAL MODEL CAPTURE on teacher tokens, not draft/acceptance/throughput/Pareto; not end-to-end speedup.
