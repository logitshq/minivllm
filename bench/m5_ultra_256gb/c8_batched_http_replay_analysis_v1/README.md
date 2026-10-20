# Independent audit of the candidate HTTP replay (E053)

Status: source and synthetic CPU preparation only. No actual E053 result has been audited. E052 v1 stopped at readiness before HTTP requests because its expected identity retained an obsolete `diagnostic_only` field. Its failed evidence remains intact; it supplies no history to this audit. The auditor now pins the corrected E052 v2 trial and E053 producer; actual offline execution waits for the replay to close and service restoration to be confirmed.

`audit.py` independently rejoins the actual two groups of eight requests from their original request bodies, raw SSE bytes, tokenizer and scheduler trace. Counts come from that new recording, including any terminal EOS. Neither 1,030 steps nor 1,028 route calls from E047/E051 are assumed.

The frozen full-history checker validates every retained device bit/finite receipt for logits, recurrent/KV state and ahead outputs, reconstructs scheduling, and requires all 16 requests to drain with their recorded IDs. Successful tensor payloads were not saved: this is an independent audit of source-bound device comparison records, not an offline recomputation or reread of all tensor bytes.

`routes.py` retains the original graph-capacity and semantic-call checks. Its only branch-policy change is batched GDN candidate support for B=1..8 with zero native fallback. It independently checks both captured and replayed streams, accepts zero Python counter delta only after tracing the same `(installation, bound, B)`, and consumes both streams to EOF. Python trace counts are not GPU kernel counts.

No model, service or MLX import is needed for this audit. CPU tests install an audit hook rejecting MLX imports, subprocess launches and signal operations. The separate resource/restoration audit remains required. A numerical PASS qualifies only this exact candidate HTTP history and does not grant a performance or Pareto point.

Preparation receipts retained:

- `.artifacts/m5-ultra-256gb/c8-batched-http-replay-audit-routes-cpu-v1`: 7 route-policy tests passed.
- `.artifacts/m5-ultra-256gb/c8-batched-http-replay-audit-cpu-v1`: 4 dynamic-history/full-checker/source-pin tests passed against the original E053 draft. These snapshots remain evidence of that source version.
- `.artifacts/m5-ultra-256gb/c8-batched-http-replay-audit-cpu-v2`: all 11 tests passed after binding the final v2 trial and revised producer. Actual execution returned 0 in 0.377 seconds; the guarded child performed no model import, subprocess launch or signal operation.

After the actual E053 child is closed and root authorizes the offline audit, use a fresh output directory and the real child plan SHA:

```sh
.venv/bin/python -B bench/m5_ultra_256gb/c8_batched_http_replay_analysis_v1/audit.py   --child ACTUAL_CLOSED_E053_NUMERICAL_DIRECTORY   --plan-sha256 ACTUAL_CHILD_PLAN_SHA256   --out FRESH_EXISTING_DIRECTORY/audit.json
```

Successful output schema is `E053-independent-http-replay-audit-v1`: `status=PASS`, `recorded_history_verified=true`, `http_history_qualified=true`, and `candidate_http_capture_qualified=true`. `performance_qualified`, `owner_resources_restore_qualified`, `offline_tensor_bytes_rechecked` and `weight_shards_rehashed` remain false.
