# E024 serving integration: one instance entry point, separate switches

2026-10-17. Read-only source review during the separate Lithos formal run. This note adds no implementation, device test, model qualification or HTTP speed claim. The existing forward-only measurements do not measure request latency or a Pareto point. It supplements [the serving plan](route-graph-serving-plan.md).

## Smallest benchmark adapter

Add a new bench-local `route_decode_graph/integration.py`, default `enabled=False`. OFF returns without inspecting the runner, importing the candidate or replacing any object. ON installs once, immediately after a real target-only `ModelRunner` is constructed and its gates/norm flags are configured, before requests or an `Observer` exist. Require empty batch/staging/feats, `_ahead is None`, `spec is None`, the original model/source/backend identity, and an unwrapped original decoder. Emptiness cannot prove an instance was never used; construction-time installation is a caller contract, as in E028. No live toggle or state migration.

Save the existing `runner.decode_model` as the short decoder and construct `RoutePreservingDecode(runner.model, enabled=True)` against the actual model. Assign **only** `runner.decode_model` to a small dispatcher:

```text
integer ctx.max_len in [257, 1536] -> frozen RoutePreservingDecode
everything else                 -> saved original decode_model
```

Do not replace `runner.model`, `run`, `_sync_batch`, sampling, scheduler outputs, or classes/globals. Do not call the candidate's `_eligible` twice, materialize state, pad caches, or synchronize. Inside the long domain the candidate performs its existing guard and falls back to the original eager model; the old short `CompiledDecode` also falls back to eager there. Outside it, retaining the saved decoder preserves the adopted <=256 compiled/packed route and ordinary eager fallback. The disabled adapter must preserve decoder object identity, not merely produce equal output.

Concrete source locations:

- `tutorial_code/miniengine/engine/model_runner.py:19–24`: original decoder construction; ordinary decode calls it at `:79`, actual ahead construction at `:97`. One assignment covers both. A consumed ahead at `:66–68` correctly performs **no new decoder call**.
- The same file's grouped prefill `:134`, solo prefill `:147–150`, and speculative capture `:74` call `model` directly and remain untouched. Ahead submission/evaluation and separate cache wrappers at `:82–102` also remain original.
- `engine/compiled_decode.py:24–40` limits the existing graph to used<=256; `:56–85` handles parameter capture and state return for that separate short route.
- `bench/m5_ultra_256gb/route_decode_graph/candidate.py:63–98` owns long eligibility, `:104–127` owns bound selection, captured model inputs and all state assignment. Do not change this frozen candidate to install it.
- For a new server version, insert installation after `pareto_realistic/native_server.py:373–374`, before startup identity construction and `observe_runner` at `:393`. Record enabled flag plus adapter/candidate hashes in identity. Old `http_qualification.expected_identity` (`:103–124`) must receive an explicit new binding; do not label new code as the old server.

The candidate verifies the whole frozen core source tree. Instance installation leaves that tree intact; editing core and reusing its old manifest would fail or misstate provenance. Initial D512 inputs use physical KV capacity1536 through `cache.py:120–121`. The candidate guard permits >=1536, but the registered initial serving workload should require observed capacity1536 in qualification. Any larger actual capacity needs its own exact state evidence; do not silently broaden a serving claim or force artificial capacity to obtain graph hits.

## Essential missing cases before a formal ON comparison

| Case | Exact gate and why it is necessary |
| --- | --- |
| Changed-value capture | Reuse `long_decode_graph/qualify.py:139–172`: independently replace `norm.weight`, layer0 `in_proj_a.weight` and `.scales` with genuinely changed valid values. Require changed eager logits, candidate full-vocabulary logits plus all128 cache arrays exactly equal eager, and original outputs after restoration. Reuse the **same** route object across both bound families and B1→B8→B1. Use serving gates ON/norm OFF. Equal-value replacement is insufficient. |
| Short/long transitions | Through the installed dispatcher, check used256→257, 1023→1024→1025 and 1536→1537 with valid backing for the final fallback. Check mixed-length B8 rows and batch-size reuse. Preserve true context metadata and verify short decoder/fallback identity; candidate-family construction alone is not proof of a completed graph call. |
| Ahead consumed | Produce a real future across the1023/1024 boundary, then consume it using the original request/position/sampling/token key. Pair its retained full logits/state with the reference's independent recomputation. Do not require a redundant candidate invocation on the consuming step. |
| Ahead discarded / retirement | Add arrivals, a mixed decode+prefill step, partial retirement with surviving rows, complete retirement and final empty state. Require invalidated futures cannot be consumed, committed arrays retain their values, and original-handle separation holds. `ModelRunner._sync_batch:174` and `drop_many:195` invalidate ahead; preserve both. Check cap termination; report actual normal-EOS coverage separately if no request emits EOS. Do not suppress EOS or inject output IDs to manufacture coverage. |
| Real HTTP schedule replay | Capture new ON C1/C8 eight/sixteen-request D512 traces, then replay recorded arrivals through independently executing reference/candidate engines. Compare actual schedule, every original-dtype full logit/state, delivery/raw/committed IDs, retirement and final drain. Require actual long graph calls in both families; the prior OFF capture is not an ON qualification. |

Parameter replacement is an **idle forward graph** contract here, not live serving hot reload. The actual ahead key (`model_runner.py:60–65`) has no parameter version. A weight change with a valid pending future can bypass the decoder and use stale output even if graph capture itself is correct. Freeze weights/configuration during requests; if hot reload is ever required, separately define explicit ahead invalidation or runner recreation. Do not imply the adapter solves it.

## Reuse qualification machinery; avoid another lifecycle implementation

Privately load the frozen `pareto_realistic/qualify_native.py`; a small `Observer` subclass installs only on the candidate before `super().__init__`. The original Observer wraps both callables (`:199–209`), checks committed/future separation (`:224–252`), and `pair_calls` (`:301–314`) matches consumed ahead by semantic key, not call number. Reuse its Comparator, `run_case`, original37-request workload, pair construction and release. A qualification-only route witness may record actual callable invocations and physical shapes; exclude it from formal timing. Restore observer first, then the installed decoder, and clear adapter/model links when releasing the pair. Exceptions must take the same cleanup path.

For real traces reuse `http_qualification.run_adapter` (`:147–164`) plus `http_replay.run_recorded_case`; only the private Observer installation and explicit new server/adapter identity need adaptation. Recorded tokens remain comparison data, never engine inputs. Preserve the existing float0.0 sampling contract and corresponding versioned audit; do not normalize ledger metadata. An ON trace qualifies that recorded scheduling history, not every later formal arrival schedule. Once these gates pass, use the existing observer-OFF formal transport and process owner instead of copying them.

## E028 is independent in code; combined correctness is not automatic

`identity_refill/integration.install` replaces only a fresh `runner.batch` (`:75–86`); E024 replaces only `runner.decode_model`. E028 creates the original `BatchKV`/`BatchState` types at `candidate.py:93,99`, so the long graph's exact cache-class checks are structurally compatible. Neither changes scheduler policy. Install E028 first, E024 second, then attach the Observer; both gates can be independently default OFF without modifying frozen class methods or migrating active state.

The first HTTP comparison should be E024 ON/E028 OFF against both OFF. Test the four switch combinations on CPU for installation/routing isolation; real combined ON requires full state comparison at identity refill immediately after sync and before decode, plus consumed/discarded ahead checks. E028's previous37-request PASS does not establish combined HTTP performance. The existing E028 `model_qualification.py:108–158` shows the precise128-array predecode refill snapshot hook and can be reused only for that later combined qualification.

No tests, runtime hashing, model imports, GPU calls or service operations were performed for this note; source paths and line numbers above refer to the read working-tree files. No frozen source was edited.
