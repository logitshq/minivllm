# E024: shortest default-OFF serving integration

Read-only design, 2026-10-17. No implementation, GPU, service operation or new qualification. Root reports gates-ON forward-only gains of roughly 6.7–7% at B1 and 5.7–5.9% at B8; these are not HTTP gains.

## 1. One existing interception point; preserve the current short route

`ModelRunner.__init__` selects `decode_model` (`engine/model_runner.py:19–24`). Both ordinary decode (`:79`) and ahead (`:97`) call it. Prefill and spec/capture use `model` directly. Therefore install one per-instance callable before any requests/Observer, leaving `run`, `_sync_batch`, `drop_many`, sampling and scheduling unchanged.

Proposed production API: `LLMEngine(..., route_decode_graph=False)` → the same strict-bool keyword on `ModelRunner`. Reject an enabled flag with a drafter before loading. No live toggle. With OFF, leave the original `decode_model` object untouched. With ON, retain the existing short decoder as fallback:

- `257 <= ctx.max_len <= 1536`: invoke the long route candidate, whose own guard chooses graph or original eager model.
- Everything else: invoke the original decoder. This preserves the existing <=256 compiled/packed path when `compile_decode/fuse_decode` are enabled.
- Do not call `_eligible` twice, copy/pad request caches, or add eval/synchronize. The frozen candidate already performs its 64-layer guard inside the measured forward. Long-domain fallback to eager matches today's `CompiledDecode` behavior because its guard rejects lengths >256.

The arithmetic remains `candidate.py:100–127`: explicit positions/lens/all128 states, new KV view wrappers, captured array parameters via `mx.compile(..., inputs=model)`, complete returned state assignment. Bound1023 below used1024 and bound1536 above it preserve the tested attention dispatch region; `model/attention.py:87–90` slices to that bound and masks by true per-row lens. Do not collapse back to bound1536 everywhere.

For the first actual HTTP experiment, a small instance adapter on the existing frozen core is cheaper than production edits. The candidate's `_load_runtime` verifies the **whole original core source tree**. Editing `ModelRunner`/`LLMEngine` then importing this bench candidate will fail that check. After HTTP acceptance, put the small route class into the core, use direct core imports and freeze a new source manifest; do not make production import a bench path or pretend old core pins still apply.

## 2. Keep the initial supported domain honest

The matrix qualified B1..8, used257..1536 with physical KV length1536, gates/norm OFF; the forward experiment adds gates ON at B1/B8 × used1023/1024. It did not qualify parameter replacement, ahead, retirement, full D512 or arbitrary backing lengths.

The frozen guard permits backing **>=1536**, but matrix evidence used exactly1536. D inputs886–978 + cap512 allocate1536 through `DecodeBatch.rebuild` (`cache.py:120–125`), so the first D512 serving experiment needs no capacity expansion. For a new production module, either initially guard physical K/V length==1536 or add explicit full-state checks for1792/2048 before claiming the broader existing guard. Never force larger backing just to record graph hits. Preserve eager fallback for underallocation, unsupported backend/quantization, B>8, capture/spec and out-of-domain context.

## 3. Small missing correctness gates; reuse existing functions

1. **Parameter capture:** reuse `long_decode_graph/qualify.py:139–173` (`changed_parameter`, `parameter_replacement`) and the direct full-state comparison body at `:236–253`. Change actual values of `norm.weight`, `layers.0.linear_attn.in_proj_a.weight`, and `.scales` on independent equal models; require eager logits change, candidate full logits/all128 states equal eager, and restored output equal original. Run B1→B8→B1 through the *same* route object and both bound families, with gates ON/norm OFF for the serving setting. Count actual compiled calls only in qualification. No new loader/validator framework.
2. **Ahead and retirement:** reuse `qualify_native.Observer`, `pair_calls`, and `run_case`. They already check semantic call keys, full logits/states, committed state not rewritten by ahead, different array handles, consumed/carried/discarded future state, final drained queues and retirement. Install the candidate before constructing the Observer, restore afterwards. Require genuine long-route hits at both sides of1023→1024, an actual consumed future, a discarded future on arrival/retirement, and partial retirement with surviving rows. Include normal-EOS/cap endings and a 1536→out-of-domain routing check. If normal EOS prevents required coverage, report insufficient coverage; do not change EOS semantics.
3. **Live weight changes are a separate contract:** `model_runner.py:64–68` can consume `_ahead` using a key containing request IDs, positions, sampling and tokens—no parameter version. A changed weight can therefore bypass the decoder entirely and deliver an old future. The minimal feature keeps weights and enablement fixed while requests/ahead exist. Direct idle graph replacement tests above do not establish a hot-reload API. Any future hot reload must explicitly invalidate ahead or recreate the runner; do not add a general versioning system for this experiment.

CPU tests need only adapter OFF identity, short/out-of-domain fallback, unsupported-layout fallback, setup validation before load, no spec/prefill interception, and scoped install/restore on exceptions. Actual device tests establish capture and numerical behavior.

## 4. Actual C1/C8: retain the existing join/replay, add only installation and identity

`http_qualification.run_adapter` and `http_replay.run_recorded_case` already replay real recorded arrivals through independently executing engines; recorded schedules/IDs are comparisons, never injected outputs. Reuse them plus the existing 37-request control workload. A private Observer subclass can install the new per-instance decoder on the candidate before `super().__init__`, as E028 already demonstrates. A small qualification-only callable witness records eligible/fallback counts and current/future call shapes; it must not require every short lifecycle call to compile.

Capture new ON C1/C8 traces using the same 8/16 D512 qualification requests, normal EOS, transport and allocator settings. Use a new narrow server adapter/version: instantiate at `native_server.py:365–374`, install before `observe_runner` at `:395`, and record the actual flag/source hash in startup identity. Existing `http_qualification.expected_identity` pins the old server/configuration, so its binding must be explicitly adapted; do not loosen `http_join` or relabel baseline receipts. This path is request-ID agnostic—no E028 UUID-prefix restriction.

Require the original full-vocabulary and state comparisons, per-request raw/committed IDs, admission/refill/retirement, and actual graph coverage. Then run existing formal N64 C1/C8 with observers OFF and the same fixed setup. Preserve OFF baseline points and report both axes: committed throughput and POST→DONE p50. A trace proves its recorded schedule, not every future formal arrival schedule; preserve that documented scope.

## 5. Likely limit on benefit

The forward timer already includes the candidate guard and explicit-state handling, but excludes prefill, input preparation, scheduler/batch rebuild, sampling, Python delivery, SSE and teardown. Existing ahead also overlaps host work with device work. If eligible forward accounts for fraction f of latency and its measured speedup is s, the ideal end-to-end bound is `1 / ((1-f) + f/s)`; do not project the full 6–7% onto HTTP.

Watch three concrete costs in the first actual replay/formal run: graph misses at batch/physical-shape changes (two callable families are not two machine graphs), surviving-row gathers in `DecodeBatch.rebuild`, and state traffic. The fixed BF16/F32 state is about254.6MB per row at physical1536 (~2.04GB for B8); changing graph submission does not remove that state work. Reuse E028's independently qualified identity-refill work later, as a separate switch, rather than combining two new interventions before this HTTP comparison.

## Source identities read

- `route_decode_graph/candidate.py`: `6531be25acb53713d6ddcd930d2c778f6640a3b1f53304fd5375d20b044525c3`.
- Current `engine/model_runner.py`: `14fd964c57814846887585a9cbbdf1e11d2500ab80c23b10c945d2aa9316b7b1`.
- Current `engine/compiled_decode.py`: `22111101109b9d9c69e8fd114aa114533fe708e462303d5e72bedd6d217da1bf`.
- Current `engine/llm_engine.py`: `f43ef9cdcd5c1c4bcc61ef767830dbbb65d00ade80bba1d33e5811349659cece`.
- Current `cache.py`: `056a3bf3bc42163e93f55eb914fa35d5c081f803da9b5faa740588a67057a791`.

Those four current core files equal the frozen E024 source manifest. No files in the repository were changed for this note.
