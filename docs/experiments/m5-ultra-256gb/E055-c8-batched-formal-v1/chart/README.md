# E055 C8 paired performance figure — source ready

E055 is RUNNING. This directory contains source only: no current E055 result was
read, no image was rendered, and no result is forecast. `plot_pairs.py` is inert
without `--render`; only syntax was checked (tool `49d158`, rc 0). No model,
Matplotlib import, CPU test batch or service operation ran for this preparation.

## Render after the actual closed audit

```sh
.artifacts/m5-ultra-256gb/progress-render-venv/bin/python \
  docs/experiments/m5-ultra-256gb/E055-c8-batched-formal-v1/chart/plot_pairs.py \
  --render --audit ACTUAL_CLOSED_AUDIT_JSON --audit-sha256 ACTUAL_AUDIT_SHA256 \
  --out-dir docs/experiments/m5-ultra-256gb/E055-c8-batched-formal-v1/chart/render-v1
```

Root must confirm actual owner/resource/restoration closure first. Preserve
source, exact argv, stdout/stderr and exit receipt for every attempt; visually
inspect the resulting PNG before freezing. No generated image exists yet.

The figure uses the exact six raw-process `points` from the independently closed
`E055-six-process-metrics-ids-v1` audit. It rechecks all three pair ratios and
per-arm process medians/min/max; it does not rerun the raw HTTP audit. The script
pins auditor source SHA, explicit audit SHA and exact phase order. The left panel
shows aggregate committed output tokens/s against **full HTTP POST→DONE p50
seconds**, with one OFF→ON arrow for each of three pairs. Diamonds are per-arm
metric medians and whiskers are observed process min–max, not confidence bounds.
Both axes are visibly described as a C8 detail view with zoomed limits.

The right panel shows throughput gain `100*(ON/OFF-1)` and E2E p50 reduction
`100*(1-ON/OFF)`, with three pair values and their medians. These are performance
changes, not per-token latency. Six processes all have C8; the figure is not a
six-concurrency sweep. Each process has 8 warm +64 formal requests; only the 384
formal requests enter the displayed metrics. HTTP C is not a GPU batch claim.

`timing_valid`, E053 same-schedule numerical prerequisite, cross-schedule ID
comparison, speed gate and qualification scope stay distinct. Valid measured
points are retained when greedy trajectories differ. The plot retains
`performance_qualified=false`, `formal_full_state_replayed=false` and
`frontier_point=false`; it cannot independently promote a configuration.

## Smallest later v16 frontier update

Source inspection only: the frozen renderer/data are under
`docs/optimization-guide/progress/2026-10-19-v16/`. Do not overwrite that snapshot.
Once root records an explicit source-bound qualification decision plus actual
closed raw-metric and resource/restoration audits, prepare a new v17 extension:

1. Copy v16 `chartdata.json` verbatim as the historical input, with its SHA.
   Preserve all 50 original point objects and all 18 old summaries unchanged.
   Append all six E055 points (including OFF controls), with two separate C8-only
   configuration labels, e.g. `native-batched-off` / `native-batched-on`. This
   yields 56 observations and 20 configuration/concurrency groups, not a new
   concurrency grid. Do not pool fresh OFF processes into earlier graph groups.
2. Use the audited E055 `comparison.process_summary` for the two new rows, then
   reuse v16 `prepare.frontier(rows, x)` for each engine and joint p50/p95 sets.
   **Do not extend `summaries()` unchanged:** it requires every label to have
   both C1 and C8. Only replace the selected native C8 endpoint if it is actually
   non-dominated; preserve every old observation even when newly dominated.
3. Use a small fresh renderer or adapt the frozen renderer in the new directory.
   Its `main()` currently reopens E033-only inputs, checks literal 50/18 counts,
   and its concise plot hardcodes y≤200 and endpoint label positions. Replace
   these with the new input binding, dynamic counts/limits and actual selected
   labels; changing only a C8 coordinate would leave stale assertions/captions.
4. Preserve historical C1, oMLX and Lithos values and source dates. Label them
   explicitly “historical, frozen v16 snapshot 2026-10-19”; use original receipt
   timestamps for actual measurement dates when available, never relabel the
   snapshot date as a new comparator measurement. E055 is a new minivllm C8
   cycle, not a freshly interleaved three-engine benchmark.
5. Prefer the actual latency plot (p50/p95 seconds versus tokens/s). The old
   optional `512/p50` view may retain its explicit full-request-speed label; it
   must never become P90 interactivity, TPOT/ITL or tokens/dollar. C2–7 remain
   unmeasured. Frontier lines only connect measured non-dominated endpoints.

This update needs a small data append plus a renderer adjustment, not another
experiment or audit framework. No v17 source or data has been written here.

Actual update 2026-10-20: v2 source renders the closed independent audit; render-v2 PNG/SVG, plot-data and visual-review receipts are saved. Original v1 source remains as the earlier draft. All figures retain diagnostic/local-cost scope.
