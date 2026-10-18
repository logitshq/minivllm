# v15 source preparation — no new result yet

Status: **PREPARED_NOT_RUN**. This directory contains unexecuted preparation/rendering source for the registered GDN-only twelve-process formal cycle. No new point, figure, winner, or measured gain is available here. [v14](../2026-10-17-v14/README.md) and the current figure link remain unchanged.

`prepare.py` requires the actual registered plan `02337f02…af52f5`, independent formal auditor `70705b8a…bf976` PASS, all twelve closed process trials, 864 checked requests (96 warm + 768 formal), independent restoration PASS, driver recovery exit zero, and the verified **E029-gdn-http-formal-v1** archive with matching small sidecars and raw-file manifest entries. FAIL, INCOMPLETE, missing, or mismatched evidence stops preparation. It does not rerun the HTTP audit or hash/decompress the archive stream; it records that scope. Missing evidence produces no synthetic points.

The exact 26-point v14 source remains an unchanged prefix. Twelve actual GDN-cycle points would add four independent summaries: GDN OFF/ON × C1/C8. The total would be 38 process observations and 14 summary groups. The older E024 OFF observations and the new GDN OFF observations are **not pooled**. Each cycle keeps its own three OFF/ON process pairs; older Native N=1 observations remain historical, not paired repeats.

Every configuration median and observed min–max is recomputed from its own process observations. Each engine's frontier may select among its separately measured configurations; the joint frontier is then computed from all summary points, with no predetermined winner. **E024 ON and GDN ON are distinct; no E024+GDN/E028 combination is implied.** Lines are visual guides between measured endpoints, not interpolated performance: C2–7 remain unknown, and HTTP concurrency is not GPU batch size. Min–max ranges are not confidence intervals or statistical significance claims.

The two views retain full POST→DONE p50/p95 and aggregate committed-token throughput. The third coordinate is `512 / median_process(p50_POST-to-DONE)`, which preserves point-estimate dominance but is not an ITL/TPOT measure or a pooled request-speed quantile. Prior actual-history numerical qualification does not prove unseen schedules, general quality, or GDN dispatch counts during observer-OFF formal timing.

The renderer adapts [v14/render.py](../2026-10-17-v14/render.py), removes its fixed winner assertions and fixed axis limits, retains the palette and process scatter/ranges, and uses different markers for E024/GDN/OFF/history. Both CLIs are inert without `--run`. Future execution, **only after closure and parent authorization**, uses fresh output directories:

```sh
.venv/bin/python -I -B docs/optimization-guide/progress/2026-10-17-v15/prepare.py \
  --run --repo "$PWD" --out .artifacts/m5-ultra-256gb/progress-v15-prepared-v1
MPLCONFIGDIR=/tmp/minivllm-plot-cache .artifacts/m5-ultra-256gb/progress-render-venv/bin/python -I -B \
  docs/optimization-guide/progress/2026-10-17-v15/render.py --run --repo "$PWD" \
  --data .artifacts/m5-ultra-256gb/progress-v15-prepared-v1 \
  --out .artifacts/m5-ultra-256gb/progress-v15-render-v1
```

`test_cpu.py` contains **8** small fabricated arithmetic/binding cases, not benchmark observations. It has **not been run** during formal timing. After closure, the whole module can be run with `.venv/bin/python -I -B docs/optimization-guide/progress/2026-10-17-v15/test_cpu.py`; there is no filtering. These fake checks do not exercise actual archive bindings or matplotlib. Actual preparation, rendering, full 26-point prefix verification, image inspection, and a dated result receipt remain pending. Rendering records `visual_review=PENDING`; a successful save is not a visual QA claim.
