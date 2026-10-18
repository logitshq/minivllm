# Partial C1 HTTP timestamp diagnostic

These are **unaudited partial results of the active native formal cycle**, not
new qualified performance points. Historical oMLX r1-C1 was independently
accepted by a later versioned audit supplement; its original failed audit is
retained. That earlier measurement is not a simultaneous paired control.

All three groups contain 64 requests to the same Qwen3.8-27B-nvfp4 snapshot and
the identical eight D512 fixtures repeated eight times. Each observed request
returned 512 committed output tokens. Native graph is ON in both current arms;
GDN prework is OFF versus ON. Normal EOS was enabled, but these requests all
ended at the length cap.

| Group | TTFT mean / median (s) | DONE−first mean / median (s) | E2E mean / median (s) |
|---|---:|---:|---:|
| Historical oMLX r1 C1 | 0.564235076 / 0.563858375 | 9.517902517 / 9.512168521 | 10.082137592 / 10.079908603 |
| Current native r1 C1 OFF | 0.636941341 / 0.635908167 | 10.553463620 / 10.555703562 | 11.190404961 / 11.189323688 |
| Current native r1 C1 ON | 0.638341262 / 0.637216333 | 9.957507003 / 9.956251667 | 10.595848264 / 10.591330875 |

ON minus historical oMLX has an additive mean E2E difference of 0.513710672 s:
0.074106186 s before first received output and 0.439604486 s afterwards. The
latter is 85.5743% of the difference. Dividing it by 511 yields 0.860283 ms as a
per-later-token diagnostic amortization target, **not actual ITL, TPOT or GPU
time**. Medians are calculated from each duration distribution; median
subtraction is not used as an exact decomposition.

The client's start clock precedes request construction/send; first output is the
first nonempty answer **or reasoning** text received. DONE−first includes model
generation, scheduling, host/transport and terminal overhead. These observations
do not identify a kernel or GIL bottleneck, and no backend/profile was run.

## Retained evidence and first failure

- `review-combined-v2-r1-c1-timestamps.py` is the original extractor. It read each
  original raw group once, verified request/count/timestamp/cleanup fields and
  retained all 192 request timestamps in `combined-v2-r1-c1-timestamps.json`.
  Its final assertion incorrectly required the two numerical-plan **paths** to
  match. This first calculation exited 1 (tool chunk `855a80`); the FAIL result
  and exact script remain unchanged.
- `finish-combined-v2-r1-c1-timestamps.py` reads that retained extraction and small
  specs/plans only. It confirms the two actual plan files are byte-identical at
  SHA `21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb`,
  verifies the complete fixtures/model/arm settings, and recomputes the means
  and medians. It exited 0 (tool chunk `d4fb89`). Its retained result has status
  `COMPUTED_TIMESTAMP_DIAGNOSTIC_ONLY`, never qualification PASS.
- `combined-v2-r1-c1-timestamps-v2.json` contains the full 192 timestamp records,
  original input paths/digests and computation checks. The adjacent Markdown
  table and `combined-v2-r1-c1-followup.md` describe the scope and falsifiable
  candidate observations.
- `numeric-plan-registered.json` and `numeric-plan-retained-copy.json` preserve
  the two original small plan files. `historical-omlx-summary.json` preserves
  the old audit-supplement summary. `spec-selected-fields.json` is explicitly a
  selected-field JSON extraction of original fixtures/model/configuration;
  it is not a byte copy of the full specs or a new runtime qualification.
- `copies.json` records exact source paths, sizes and byte-copy checks for nine
  original small files. Full raw groups remain at their recorded original
  paths; none were copied, rescanned or compressed here.

Both scripts retain their original fixed `/tmp` output names and refuse to
overwrite an existing output. They are preserved evidence of the calculation,
not an automatic live-cycle polling or rerun tool. To review without touching
the original groups, inspect the v2 JSON's per-request start/first/DONE values
and formulas; all statistics can be recomputed directly from those values.

The source review at
`bench/reviews/2026-10-18-c1-ahead-submit-and-delivery.md` independently checks
the measured oMLX wheel's burst boundaries. Native already has a dedicated
continuous worker, so its different scheduling boundary prevents importing a
burst speedup claim. No inference or serving code changed. No commit was made
as part of this diagnostic.
