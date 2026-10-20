# Full recorded C8 history with batched GDN + graph

Source prepared. The guarded CPU contracts passed **8/8** in `.artifacts/m5-ultra-256gb/c8-batched-full-cpu-v1`, with no backend/process/socket attempts and unchanged source bytes. This first run had no failures. No plan has been prepared and no device execution has occurred. This new namespace preserves the frozen E049/E050 prefix evidence. Preparation requires the actual E050 `DIAGNOSTIC_PREFIX_COMPLETE` result, its unchanged plan, inherited numerical result, composition/cleanup receipts, and source closure.

The experiment uses the complete original E047 recording: **1030 global steps, 16 requests, original cap512, 8192 output IDs**. The first eight-request wave drains at steps512–515; the second arrives at516/517 and drains at1027–1030. “512 output tokens” is not a 512-global-step boundary. No job, cap, arrival, EOS rule, comparison, retirement, or recorded token is rewritten.

The frozen E050 `Bridge` installs the actual 48-layer batched GDN hooks before graph creation and before the original observers. The original `http_qualification_v2.adapted` keeps its full `run_case`, which calls the unchanged `http_replay.run_recorded_case`. Unlike the prefix experiment, no `PrefixSteps` iterator or boundary exception is installed. The original end checks require both engines empty, all jobs admitted/completed, no active requests, and the complete recorded batch histogram. Full logits, original-dtype finite/bit-equality receipts, all cache states, actual ahead reuse, retirement, and release remain in `base/steps.jsonl`.

The original `CapturedRoutes.finish` must consume the entire raw stream and verify its SHA. The expected1028 actual graph calls cover B2/4/6/8 at both bounds1023/1536, with physical KV1536 and max observed used1489. Route counters prove actual compiled-call dispatch, not completed arithmetic by themselves. The frozen bridge requires each of48layers to have positive aggregate Python candidate counts and zero fallback. It does **not** record a per-layer/per-B trace matrix. Neither counter is represented as GPU kernel executions.

## Memory evidence and admission

The older independently audited full E024 C8 replay reported **73,609,187,604 B =68.553898 GiB MLX peak**, in `docs/experiments/m5-ultra-256gb/E024-route-http-v1/c8-independent-numerical.json`, field `peak_bytes_diagnostic`. Its full schedule had1028steps and differs from this E0471030-step schedule; it is a warning about observed full-replay memory, not a prediction for the new candidate. E049 warm49 reported49.510929GiB MLX peak and51.012732GiB physical peak; extrapolating that to a guaranteed50–60GiB full-run budget is unjustified.

Retain the original **2 GiB free allocator cache** configuration. `LLMEngine.__init__` already calls `mx.set_cache_limit(2 << 30)`. Installed MLX API documentation describes this as unused buffer retention; reducing it cannot cap live arrays. `mx.set_memory_limit` is documented as a guideline, not a hard allocation cap. No cache0 mode or altered memory API is introduced here.

The separate owner must preregister physical sampling thresholds before execution. The proposed full-run policy is80GiB child /240GiB system /zero swap growth, with the existing bounded service pause and restore. These are **sampled stop thresholds, not hard allocation caps**. The child does not claim that it will fit. If observed thresholds fail, preserve the failure and cleanup; a separately registered cold-service attempt requires its own lifecycle evidence. This script performs no service operations.

## Prepared interface

After E050 independent audit and source freeze, metadata preparation is:

```text
.venv/bin/python -I -B <absolute qualify.py> --prepare --prefix-plan <E050 child-plan.json> --predecessor <E050 numerical/complete.json> --out <fresh registration folder>
```

Actual child command, owned by the external lifecycle:

```text
.venv/bin/python -I -B <absolute qualify.py> --run --plan <absolute plan.json> --plan-sha256 <sha256> --out <fresh numerical output>
```

No explicit mode only describes the experiment. The complete schema is `c8-batched-gdn-graph-full-result-v1`; success is `PASS`, with actual pid/ppid, new plan SHA, base result, complete case, steps1030, total_requests16, composition, full route comparison, and checked source/input pins. `full_history_qualified` means only this registered recording; `http_qualified` and `performance_qualified` remain false. The parent separately binds independent E050 audit and current service identity/resources. A future candidate HTTP capture, its full numerical replay, and observer-OFF formal timings remain separate work.

## Actual complete history result

E051 ran on device and passed:1030 steps,16 completed requests,8192 IDs,1028 compiled route invocations and zero fallback. Independent audit6c969e/session92141→2fa572 checked1,324,017 full-array bit-equality receipts and2,648,034 finite receipts, plus complete drain/ahead/retirement. Successful full tensor payloads were not retained; the independent check audits source-bound producer receipts rather than repeating GPU arithmetic. Lifecycleedb8f7/session89344→afb109 returned0 and restored same oMLX PID52818. Full archived evidence is [E051](../../../docs/experiments/m5-ultra-256gb/E051-c8-batched-gdn-graph-full-v1/README.md). Qualification applies to this exact captured history; no throughput result is granted.
