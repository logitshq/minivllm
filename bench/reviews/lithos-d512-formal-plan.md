# Lithos D512: minimal observer-OFF formal path

CPU/source design, 2026-10-17. The corrected retry qualification is executing separately; this
note does not claim it passed. Original v1 failed its full-prompt gate because
already-rendered prompt_text was wrapped as chat content a second time. No implementation, model/service operation,
benchmark or runtime/payload hash was performed for this design.

## Smallest implementation

Add a `lithos_d512/formal.py` with the existing `prepare` / `validate` /
`async run(spec,out,checkpoint,backend=None)` shape. Reuse the frozen Lithos
trial's direct-server/readiness/cleanup flow; add native formal's two finite
groups and a small raw-NDJSON/count checker. Do not edit or parameter-spoof the
frozen qualification trial. No new HTTP scheduler or tokenizer is needed.

| Existing file / function | Exact reuse / narrow change |
| --- | --- |
| `lithos_d512_retry/serve_probe.py:specification(...,observed=False)` | Already returns original `python -B -m monolith.serve`, real Lithos cwd, same target/pack/2048 capacity/128-exact/greedy/no-draft flags and environment. No probe process, monkeypatch or observer output in formal mode. |
| `lithos_d512_retry/trial.py:System` / `ready` / `run` | Keep authenticated `/v1/models`, exact model/capacity, lsof listener, direct Popen with cwd, absolute timeout, settled checkpoints, cancellation and TERM/KILL/wait/PID+port absence. Replace only qualification group with warm/formal groups. |
| `pareto_realistic/native_formal.py:prepare,count_contract,run` | Follow 8 warm + 64 formal construction, fixed fixture cycles, unique request IDs, phase transitions and retained points. Its native raw-ID checker and native qualification binder are **not** compatible with Lithos. |
| `pareto_realistic/lithos_client.py:run_group` | Use unchanged async closed-loop workers and NDJSON parser. Headers include `Authorization: Bearer KEY`. Existing qualifier map supports formal counts; no changes to `client.py`. |
| `lithos_d512_retry/join.py:wire,generation_log_window` | Reparse raw NDJSON for every warm/formal request; reuse pure log parser with `expected_logger='__main__'`. Do not call observer `join()` for formal runs: it intentionally requires 8/16 qualification requests and actual taps. |
| `lithos_d512_retry/owner.py:full_prepare,verification_contract,trial_one` | Keep the five actual Lithos/outer/runtime/source/model/pack groups, full pre/post verification, direct same-process await and tracked Popen. Native prepare alone does not cover Lithos. A separately versioned formal owner changes only trial spec/result/phase names and gates. |
| `pareto_realistic/native_formal_owner.py` + Lithos controller | Existing formal prefix/order and two/six-child receipt protocol is reusable. Frozen Lithos controller accepts only qualification owner/schema and two literal phases, so a narrow formal controller/reaper binding is still required; do not pass a new plan through the old frozen validator or wrap this runner as a GPU grandchild. |
| `lithos_d512_retry/audit_qualification.py`, `audit_resources.py`, `audit_restore.py` | Bind closed qualification audit and evidence; adapt raw resource label order to warm/formal and retain independent restoration. Qualification audit's `qualified` flag alone does not cover runtime/resources/restoration. |

## Fixed workload and process matrix

Eight original D fixtures (886–978 prompt tokens), **body.messages copied from
fixture.messages**, normal EOS, cap512, same target NVFP4 snapshot and existing
none pack. Never use rendered fixture.prompt_text as chat content. Official
CPU tokenizer must reproduce all frozen prompt_ids from each message body;
this was independently checked for all 24 corrected retry jobs before retry. Within each
group repeat fixture order 0..7, preserving the complete canonical bodies. Warm
has eight jobs; formal has 64. Each fresh server processes only its own warm/formal
pair at its declared HTTP C1 or C8. C8 is the official FIFO GPU route, not an
eight-row GPU batch. Warm group must fully DONE + drain + close before formal.

Six fresh servers in preregistered order `r1-c1,r1-c8,r2-c8,r2-c1,r3-c1,r3-c8`:
three independent process points per C, 48 warm + 384 formal requests total.
Never replace failed samples or pool all requests into one artificial process.
Show all three points and their per-C median. Existing native n=1 points are
historical comparisons, not a new paired three-repeat experiment. If engines
are to be called paired/contemporaneous, preregister and run fresh native trials
in an interleaved order too; this note does not silently add those runs.

Suggested explicit upper bounds, not claimed durations: readiness180s,
request600s, warm720s, formal2400s, close10s, TERM15s, KILL15s, overall3600s,
resource interval5s. These preserve native formal request/formal/cleanup limits
and the existing Lithos readiness bound. Register the total owner bound from
the actual six phases, preparation and cleanup before execution.

## What exactly is timed and counted

- Per request: existing client `start_perf_counter_s` is immediately before
  request construction/POST; DONE is receipt of bytes completing the unique
  newline-terminated `done:true` record. Thus includes client construction,
  transport, queueing, prefill, decode and terminal delivery. EOF/drain/client
  close is excluded from latency but must succeed to retain the point.
- Primary x = median of the 64 complete-request latencies; p95 = nearest rank.
  Primary y = sum of committed tokens / (last formal DONE − first formal POST).
  Not average request tok/s, not server `eval_duration`, and not all-run wall
  time. Warm/model loading/readiness/hash/log validation/reap are excluded.
- Recompute each raw terminal model/shape/finish/count/durations, prompt count,
  exact original request bytes/body SHA, response text concatenation, timestamps,
  positive POST→DONE intervals and complete cleanup. Check actual outstanding
  interval peak exactly C and all workers drained. Preserve all raw chunks and
  unsuccessful rows; no retry or shorter replacement queue.
- `length`: eval_count must be512; `stop`: 1..512, committed=eval_count−1 exactly
  once under the separately qualified normal-EOS contract. Allow variable
  committed counts, including zero on immediate EOS. No retokenization, no
  synthetic raw IDs, no previous qualification's tokens as a future oracle.
  Wire cache count stays unknown; this API has no request IDs or raw token IDs.
- A strict qualification binder must check the actual closed audit SHA/source,
  owner-plan/spec/rawjoin/source/runtime/model/pack identity and the matching C.
  Build `status='qualified'`, actual receipt SHA, original fixture body/path SHA,
  new complete request→fixture map, `raw_ids_count_rule='none'`, actual prompt
  lengths, protocol/cache/usage rule and normal-EOS fields required by frozen
  `lithos_client._qualification_errors`. It must not manufacture an audit PASS.

## Log windows are the one missing measurement primitive

Record ready→warm-start, warm-close/end, formal-start (before creating its task),
and formal-close/end byte offsets and `fstat` device/inode on the already-open
server stderr file. Warm and formal windows must be whole lines. Preserve
original full log plus each exact window's bytes/SHA/start/end/stat receipts;
after reap verify immutable prefixes and same inode, no rotation/truncation.
Warm window: exactly8 Generation lines, cached_prompt_tokens=0; setup may be
nonzero. Formal window: exactly64 lines, cache=0 and setup_ms=0 on every line,
verify_tokens=1. Reject missing/extra/malformed lines, unrelated traffic or
unresolved group cleanup. Do not guess the last64 lines of an unbounded log.

This boundary is supported by pinned official `monolith/serve.py`: Generation
logging at401 occurs before `Backend.complete` returns at409; worker `execute`
and result enqueue at536–541 precede `wire.finish` at601–602. Python logging's
stream handler flushes each emitted record. Waiting for all DONE plus response
drains therefore closes the request window before the next phase. Still prove
offsets from the actual file; source ordering alone is not a log receipt.
The count/window claim is **all entries in this isolated window**, not a
per-wire-ID log join. Repeated fixtures can affect cache behavior; enforce the
actual zero-hit logs without resetting cache or changing the official recipe.

## Offline gates and remaining dependencies

Add a small formal audit that independently reparses original NDJSON, recomputes
counts/latencies/overlap and both log windows, binds source/qualification/launch,
and consumes the actual raw resource + direct-reap + restoration evidence.
Preserve the pressure1/AC/thermal/swap/foreign-process policy unchanged. No
formal point is published before all these gates pass. Full-state equality
across frameworks and every future schedule remains unqualified; the honest
contract is observed Lithos route/count/text/cache qualification plus unchanged
source/config and independently checked formal transport/count/log evidence.

Missing before implementation/run: actual corrected retry qualification + resources +
restoration PASS artifacts and their immutable bindings; new formal runner and
strict count binder; bounded log-offset capture; minimal formal owner/controller
schema adapter; formal offline checker and focused cancellation/raw-count/log-
rotation/extra-request tests. Runtime, pack, jobs, NDJSON client, exact cold stop/
restore algorithms and raw log parser already exist. No new pack, model,
observer, HTTP scheduler or broad framework is needed.

## Freeze before the first formal POST

Use one immutable plan with these concrete values, not mutable defaults:

1. `schema`, exact six `trials` order, `repeat`/C on every spec, unique NEW output
   root, total owner deadline and every per-trial budget listed above. No "up to
   three" repetitions and no result-dependent new repetitions.
2. `qualification={owner_directory,owner_plan:{path,sha256},audit:{path,sha256},
   resource_audit:{path,sha256},restoration_audit:{path,sha256}}`. Bind matching
   actual C1/C8 specs, both rawjoin hashes and every original body/full prompt
   fixture. The qualifier/auditor source versions are explicit. Qualification
   scope is route/count/text/cache; do not label it native full-state replay.
3. Original target revision/path, existing pack manifest+weights, all five
   verification groups (including distinct server and outer Python versions),
   exact source pins, resolved executable, cwd, original CLI, no-draft route,
   observer_enabled=false, environment (four MLX variables plus existing flags),
   port/API-key policy and no unregistered traffic. `serve_probe.py` must not be
   in executed argv and no observer JSONL is expected/generated.
4. Immutable D8 fixtures SHA, each original `messages` body/path SHA, prompt IDs
   and lengths, greedy options, EOS IDs, cap512, 8/64 jobs and complete
   request→fixture maps. Exactly one use of each fixture during warm and eight
   during formal. No request stop/minimum-token/ignore-EOS overrides.
5. Exact client protocol/count rule/cache contract; all raw NDJSON retained;
   normal-EOS normalization once; timing origin/terminal rule, p50/p95 formula,
   process aggregation and max-outstanding C requirement. No old raw output
   tokens become a per-request expectation. No tokenizer-based token counting.
6. `log_windows`: whole-line offsets + dev/inode + bytes/SHA; expected8 warm,
   64 formal, cache0 both, setup0 formal, expected_logger='__main__', and no
   per-wire-ID claim. Boundary receipts precede the first POST and follow
   complete group DONE/drain/close. Final log recheck must precede eligibility.
7. Exact unchanged pressure1/AC/thermal/paging/inventory guards, resource
   cadence, service's immutable restored predecessor, direct owner/server
   PPID+identity, durable global and per-phase intents, original no-Popen and
   unresolved-spawn rules, TERM/KILL/reap/PID+port absence, restoration contract.
   A failed point stays retained and stops the remaining registered cycle.
8. Consumer output schemas: trial `count_eligible` remains separate from final
   `performance_qualified`; point retains C/repeat/64 request latencies, raw and
   committed counts, common window, p50/p95, EOS/cap counts. Closed offline
   audit must combine transport/count/log/runtime/resource/reap/recovery gates.

The frozen `owner.run` has qualification-specific result assertions and the
native formal owner has native-specific schema/cwd/qualification bindings.
Neither can be passed a different trial and called compatible unchanged. The
smallest honest code work is one Lithos formal runner/count checker plus one
narrow lifecycle schema adapter, importing the already-reviewed hashing,
resources, PID/reap and raw log functions; no wholesale duplicate tokenizer,
HTTP client, qualifier, pack preparation, or generic execution framework.
Only tests for new deltas are needed: 8→64 phase counts, original messages,
EOS-at-cap/short-EOS normalization, incomplete NDJSON/cleanup, empty/extra log
windows, inode rotation/truncation, and partial phase recovery. Keep numerical
qualification and this observer-OFF formal run as separate artifacts.
