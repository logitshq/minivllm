# E054 local cost figure

Rendered from the closed **PASS** independent v2 audit, then visually inspected.
The original v1 auditor failure remains in the experiment evidence. The plot
script was adapted to v2 before its first render; the never-rendered v1-schema
draft is retained as `plot_cost_prepared_v1.py`.

- Actual render: `render-execution-v1/`, exit 0, 7.931 s (tool `aa7b90`).
- Exact input copies: `inputs/audit-v2.json`, `inputs/timings.jsonl`.
- Figure: `render-v1/local-verifier-cost.png` and `.svg`.
- Recomputed plot data: `render-v1/plot-data.json`.
- Visual review, values and file hashes: `visual-review.json`.

| T | Native median of group medians | Verifier median of group medians | Audited paired native/verifier ratio |
|---|---:|---:|---:|
| 4 | 85.494 ms | 59.700 ms | 1.43045× |
| 8 | 168.343 ms | 108.591 ms | 1.54952× |

These ms summaries are medians of five group medians; the audited ratio is the
median of five group median paired ratios. Neither is a ratio of pooled p50s.
Each arm/group has ten calls. All 400 original timing rows are rechecked; only
the 200 device-baseline calls are plotted. Host-sequential timing is excluded.

The chart measures local fixed-prefix verifier plus selected-state commit cost,
assuming all T proposed tokens are useful. Drafting, acceptance/rejection,
scheduler and HTTP overhead are absent. This is not an engine speedup or Pareto
point. Group spread is not a confidence interval.

The exact reproducible argv is in `render-execution-v1/argv.json`; use a fresh
output directory when repeating. Rendering uses the existing
`.artifacts/m5-ultra-256gb/progress-render-venv/bin/python` Matplotlib environment.
