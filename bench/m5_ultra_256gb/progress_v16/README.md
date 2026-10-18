# v16 source preparation — NOT RUN

No new performance data or figure is present. These scripts await the actual
closure of [E033](../../../docs/experiments/m5-ultra-256gb/E033-gdn-graph-combined-formal-v2).
During active formal timing, do not run the tests, preparation, rendering, or
archive work. Existing [v15](../../../docs/optimization-guide/progress/2026-10-17-v15/README.md)
and current figure links remain unchanged.

`prepare.py` consumes v15's fixed `chartdata.json` SHA
`7f44aa9223732127745684b5dc74646a7cb6e65720ef5615d78e4ac5264e61c7`:
all 38 observations and all 14 group summaries must remain exactly equal.
Only the full, independently audited E033 twelve-process result can add points.
The prospective result is **50 process observations and 18 configuration/C
groups**, not a current result.

The new cycle uses one entry with graph **ON** in both arms and GDN **OFF/ON**.
It measures the incremental GDN change on graph. Its graph-only baseline remains
separate from earlier E024 ON and both earlier OFF cycles. Combined ON is a
separately executed configuration; percentage gains are never added or multiplied.
E028 identity refill remains OFF. Historical native has N=1; each other group
has three fresh processes. Pairing applies only within a registered cycle, not
between engines or historical cycles, and does not imply identical arrival schedules.

The closed gate is bound to actual registered plan
`f4e19bd0a842b48ea26fb2d27f870fdf62409897c7f862067c7800de5949b6f1`,
independent auditor `2391e9d4…20b34`, and restoration auditor `7ab477f0…ec57d`.
It checks the actual E033 packer's flat manifest and schema-less verification
receipt, both successful audit executions, matching archive members and raw
inventory, exact owner terminal, RESTORED journal, and zero driver return codes.
Missing, partial, failed, or merely restored evidence is rejected **before creating
an output directory**. No partial prefix is drawn. A retained failed experiment
is valid archival evidence but is not a formal performance result.

This visualization does not rerun the independent HTTP audit or decompress/hash
the archive. It checks small closed evidence plus the archive's retained size,
preserving the successful member-verification receipt and exact source bindings.
The current local closed owner/driver small JSON originals are also required.
If the packer contract changes, review this source before use; do not weaken its
gates to make an unexpected receipt pass.

`render.py` retains v15's metric/plot structure and requires a fresh directory.
Before importing matplotlib it rechecks closed evidence and recomputes every
observation, process median/range, and frontier. The two PNG/SVG figures show:

- Aggregate committed throughput against `512 / median_process(p50 POST→DONE)`.
  This is normalized **full-request speed**, not ITL, TPOT, or post-first-token speed.
- Aggregate throughput against p50 and p95 full POST→DONE latency.

Frontier lines are computed separately within engines and jointly across all
measured configurations. No winner is hardcoded. An engine's frontier may select
different executed configurations at C1 and C8; a line segment is neither an
executable switch combination nor evidence about C2–7. Ranges are observed process
min–max, not confidence intervals or pooled request percentiles. Formal observers
are OFF, so formal graph/GDN call counts and unseen schedules are not qualified.

Eight focused fake-data tests are **source-prepared, NOT RUN**. They cover unchanged
history, separate cycles, dominance/reciprocal rules, failure/prefix/arm rejection,
scope/count/type gates, and inert/no-output behavior without closed evidence.
Actual schema compatibility and image legibility still require closed-data execution
and visual inspection. No progress result directory is created by source preparation.

After root confirms actual closure and authorizes CPU work, use exclusive output
paths (examples; not executed):

```sh
.venv/bin/python -I -B -m unittest discover \
  -s bench/m5_ultra_256gb/progress_v16 -p test_cpu.py -v
.venv/bin/python -I -B bench/m5_ultra_256gb/progress_v16/prepare.py \
  --run --repo "$PWD" --out .artifacts/m5-ultra-256gb/progress-v16-prepared-v1
# Select an existing Python with matplotlib first; do not install during timing.
MATPLOTLIB_PYTHON -I -B bench/m5_ultra_256gb/progress_v16/render.py \
  --run --repo "$PWD" --data .artifacts/m5-ultra-256gb/progress-v16-prepared-v1 \
  --out .artifacts/m5-ultra-256gb/progress-v16-render-v1
```

Capture exact commands, stdout/stderr, source hashes before/after, and the first
attempt. Preserve failed/earlier renders. Inspect both actual PNGs with `view_image`
and parse/check SVG labels before publishing; the renderer leaves visual review
PENDING. Only after that should a new dated progress directory and current links
be considered. No source preparation claims visual QA has occurred.

Visualization workflow:
`/Users/ryanli/.codex/plugins/cache/openai-curated-remote/data-analytics/1.0.11/skills/visualize-data/SKILL.md`.
