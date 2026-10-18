# C1 HTTP timestamp decomposition

Current native r1 OFF/ON are **unaudited partial results**, each one completed process of the active formal cycle. Historical oMLX r1 matches the requested50.781567507558 tok/s point and its three-process throughput median; it is not a pooled three-process timestamp distribution.

| Process (n=64 each) | TTFT mean / median s | DONE−first mean / median s | E2E mean / median s | common-window tok/s |
|---|---:|---:|---:|---:|
| omlx_r1_c1 | 0.564235076 / 0.563858375 | 9.517902517 / 9.512168521 | 10.082137592 / 10.079908603 | 50.781567508 |
| native_r1_c1_off | 0.636941341 / 0.635908167 | 10.553463620 / 10.555703562 | 11.190404961 / 11.189323688 | 45.751905441 |
| native_r1_c1_on | 0.638341262 / 0.637216333 | 9.957507003 / 9.956251667 | 10.595848264 / 10.591330875 | 48.319060901 |

Means add within floating-point rounding. Each component median is computed directly from64 request durations; no subtraction of medians is used as an exact decomposition.

Native ON minus OFF mean deltas: TTFT 0.001399920 s; DONE−first -0.595956617 s; E2E -0.594556696 s.
Native ON minus historical oMLX: TTFT 0.074106186 s; DONE−first 0.439604486 s; E2E 0.513710672 s.

Same Qwen3.8-27B-nvfp4 snapshot and identical original eight fixtures, in identical order repeated eight times, normal EOS/cap512 and temperature0. All192 observed requests returned512 committed tokens. The two numeric-plan paths are copies with identical actual SHA21ba52e4…547fb. Native graph is ON in both arms; only GDN prework toggles.

First output means first nonempty answer or reasoning text received. The start timer precedes request construction/send. DONE−first includes model generation, scheduling, server/client transport and terminal overhead; it is not exclusive GPU time or pure TPOT. Historical oMLX was measured earlier, not interleaved with these native processes.

oMLX qualification lineage: original formal/resource audit FAIL retained, later versioned parser supplement independently PASS. The current native cycle has no completed independent final resource/restoration/performance audit yet.

## Exact input groups

- omlx_r1_c1: `/Users/ryanli/code/minivllm/.artifacts/m5-ultra-256gb/omlx-d512-formal-v2-owner/phases/r1-c1/trial/formal-group.json`
  SHA256 `cad300cfadff27af3a36faaf4300eabe96c917667b0a8fec22553471d3c8bea2`; n=64.

- native_r1_c1_off: `/Users/ryanli/code/minivllm/.artifacts/m5-ultra-256gb/gdn-graph-combined-formal-v2-owner/phases/r1-c1-off/trial/formal-group.json`
  SHA256 `11b076ece475d9f078a9edf32384fde75ccc1a3bec68dea11efd1d26d4a4af54`; n=64.

- native_r1_c1_on: `/Users/ryanli/code/minivllm/.artifacts/m5-ultra-256gb/gdn-graph-combined-formal-v2-owner/phases/r1-c1-on/trial/formal-group.json`
  SHA256 `d98df86f7878b970ccce42272cb9f384ef0532b04d8815501cda0c402fc98127`; n=64.

## Calculation and retained attempt

- First raw extractor: `/tmp/review-combined-v2-r1-c1-timestamps.py`.
- Its retained first result `/tmp/combined-v2-r1-c1-timestamps.json` contains192 validated timestamp rows; final path-equality assertion failed on legitimate same-SHA plan copies.
- Finish script: `/tmp/finish-combined-v2-r1-c1-timestamps.py`; reads only the retained extract and small specs/plans, never the raw groups again.
- Recomputed result and per-request rows: `/tmp/combined-v2-r1-c1-timestamps-v2.json`.
- Historical qualifying receipt: `/Users/ryanli/code/minivllm/docs/experiments/m5-ultra-256gb/E023-omlx-d512-formal-v2-audit-supplement/independent-formal.json`.
