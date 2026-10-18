# C1 repeated submission and delivery work: source-only candidates

No device profile or performance result exists for these candidates. The current
formal sources and qualification evidence remain unchanged. The diagnostic HTTP
entry has the retained 15-test result plus one later focused interleaving test;
that is not a rerun of the final whole 16-test module or device qualification.

## Actual pipeline and the smallest change

`native_server.BatchWorker.tick` calls `engine.step`, puts each actual result onto
its request queue, and immediately starts the next tick. The HTTP handler consumes
that queue, decodes text, writes SSE and flushes on its own thread. It does not
release the next engine step only after the client flushes. See
`pareto_realistic/native_server.py:133–158,278–307` and
`tutorial_code/miniengine/engine/llm_engine.py:55–76`.

In `model_runner.py:66–68`, an exact ahead key consumes the already submitted
token and state roots. There is no current model forward on that branch; the
forward at line 97 constructs the successor. The same current roots are submitted
again at line 88 despite having been submitted at line 101 of the prior run.
The final line-158 `mx.eval` and host token transfer remain necessary boundaries.
This is source evidence of a repeated API submission, **not evidence of repeated
GPU model computation** or measured time wasted.

New `ahead_submit_once/candidate.py` keeps the original run code object and uses a
private `mx` proxy in that one runner. Default OFF returns without file/runtime
access. ON may omit only the first async call in a run, when all of these hold:

- One decode row, no prefill or finished-ID transition, target-only greedy route,
  original batch order, and the exact original ahead key.
- The original cap/max-length lookahead conditions still hold. Known terminal
  input IDs and finished requests remain ordinary.
- The token and **every state root are the exact same objects** as a prior async
  call which returned normally through this scope. Weakrefs provide this proof
  without retaining retired model tensors; unsupported weakrefs mean fallback.
- The actual first call arguments match that certificate and the consumed ahead
  roots. All future async submissions, eval calls, arithmetic and final host
  transfers are forwarded. Exceptions propagate without retrying advanced state.

Official **matching-tag source** supports the weakref choice: MLX v0.32.3
[`python/src/array.cpp:286–292`](https://github.com/ml-explore/mlx/blob/v0.32.3/python/src/array.cpp#L286-L292)
registers `mx::array` with `nb::is_weak_referenceable()` at line 291. This rules
out missing weakref support as a source-level reason for unconditional fallback
on that version. It is not a test of the installed binary, a runtime eligibility
result, or numerical/performance qualification. No MLX import or weakref runtime
test was performed; the conservative unsupported-type fallback remains.

A newly installed scope cannot skip an ahead that it did not observe being
submitted. New requests, key mismatches, mixed prefill, larger batches, known
terminal inputs and cap/length boundaries all retain the original backend calls.
A device-resident current token can itself be EOS. Detecting that beforehand
would add synchronization, so this adapter deliberately does not inspect it:
its redundant current submission may be omitted, while the original EOS sampling,
update, retirement and speculative-successor behavior remain unchanged.

Install on a fresh runner. Optional composition is **frozen profile Session first,
then this candidate, then qualification Observer**; restore in reverse. The
candidate replaces only the profile's private `cloned` method in this explicit
mode. Its clone retains the profile's sample and mx proxies: actual calls still
produce profile events, omitted calls instead increment a scalar omission
counter. Unknown existing wrappers refuse. There is no global MLX patch.

The concrete benefit hypothesis is less CPU root traversal/API submission. It
might be too small or already cheap in MLX, and the identity checks have their
own cost. No throughput prediction follows from the source.

## Measured oMLX's executor burst is a different boundary

Read-only source check of the measured wheel at
`.artifacts/omlx-baseline/.venv/lib/python3.13/site-packages/omlx/` confirms:

- `settings.py:157–179` maps balanced mode to a 64-step cap and a 0.1-second
  single-request budget. `engine_core.py:276–297` supplies the separate
  0.03-second concurrent budget.
- `engine_core.py:438–494` runs several `scheduler.step()` calls in one
  `_step_burst` on the MLX executor. It stops when the latest output is a
  request's first generated chunk, work runs out, prefill eviction needs an
  async callback, the budget check expires or the step cap is reached. These
  budgets are checked between steps; they are not a hard bound on one step.
- `engine_core.py:516–561` awaits one executor call, then distributes the returned
  outputs in order to collectors on the event-loop thread. This amortizes that
  executor/event-loop handoff; later token delivery may arrive in bursts.

The actual historical r1-C1 spec records balanced mode and has
`mtp_enabled`, `dflash_enabled` and `vlm_mtp_enabled` all false. The retained
diagnostic's `spec-selected-fields.json` copies those original JSON fields and
identifies the complete spec by its existing digest. No installed backend was
imported or requalified for this source check.

Native `BatchWorker.run/tick` already runs consecutive engine steps on one
dedicated thread and puts results directly on request queues
(`pareto_realistic/native_server.py:133–158`). It does not pay the same
per-token `run_in_executor` round trip. Copying the oMLX burst loop therefore does
not establish a comparable saving. The wheel's explanatory comment contains its
own timing anecdotes; those are not measurements of our current gap. Handler
tokenizer/delivery work competing for the GIL remains a hypothesis for the
existing profile spans, not an established causal attribution or permission to
alter current serving cadence.

The lightweight comparison is retained at
`docs/experiments/m5-ultra-256gb/combined-v2-r1-c1-diagnostic-v1/`.
Current native r1 OFF/ON are completed individual groups in an unfinished,
unaudited formal cycle; historical oMLX is not a simultaneous paired control.
The ON-minus-oMLX mean E2E gap is 0.513710672 s, including 0.439604486 s after
first received output (85.5743%). Dividing the latter by 511 gives a diagnostic
amortization target of 0.860283 ms per later token, not measured ITL or GPU time.
Only means are additively decomposed; each median is computed independently.

## Larger delivery hypothesis, not implemented

`native_server.py:292–300` decodes the entire visible token prefix on each output,
trims trailing U+FFFD before terminal output, verifies monotonicity, then emits
only the new suffix. A 512-token non-EOS result passes 1+…+512 = 131,328 token IDs
to cumulative decoding (a source-derived count, not a timing measurement).

The installed tokenizers 0.23.3 API in
`.venv/lib/python3.13/site-packages/tokenizers/decoders/__init__.pyi:111–171`
provides a per-request `DecodeStream(skip_special_tokens=False)`. A future
handler-local adapter could decode only the newly arrived ID and retain one
original full decode at terminal for residual UTF-8/text validation. It must
first reproduce **each original emitted piece**, including split UTF-8, byte
fallback, literal U+FFFD, special/control tokens, empty pieces, EOS-at-cap and
immediate EOS. A final equality check cannot undo wrong bytes already sent.
Do not blindly concatenate single-token decode results or change usage/EOS rules.

## First measurement and correctness gate

Use the existing diagnostic entry with identical configured graph/GDN flags,
normal EOS, eight explicit warm requests followed by eight diagnostic requests,
and the normal DONE/explicit drain protocol. Compare OFF, runner and full profile
modes to report observer overhead separately. No profile result becomes a formal
Pareto point. Existing events suffice for the first decision:

1. `runner_step`: actual used length, ahead presence/match/consumption and current
   versus future call counts; separate first-call compilation and the 1024 route
   boundary rather than attributing all steps to one graph.
2. `existing_async_eval`, `decode_graph_submission`, `compiled_callable`,
   `sample_graph`, and `existing_eval`: locate the two submission sites in their
   fixed source order. Candidate counters count omitted API calls, not kernels.
3. `engine_step`, request/ordinal queue put/get, `text_decode`, write/flush and
   normal-DONE closure: locate delayed delivery and overlap with subsequent
   engine work. Queue put/get spans only bound the handoff instant.

Do not sum nested/overlapping spans into exclusive GPU time. Python remainder
also includes host token transfer; these events do not separate that transfer,
GIL contention, GPU execution or driver scheduling. Drained proves worker and
handler closure, not completion of all discarded ahead GPU work.

Before device timing: CPU contract tests must use the actual frozen run body,
normal-return submission certificates, mismatches/retirement, exception cleanup
and profile composition. Then real qualification must compare full logits,
raw IDs, all 128 state arrays and ahead/committed-state pairing under the same
scheduler, including actual EOS/cap, refill/retirement and sampling-key changes.
The module and tests here are source-prepared only; no tests were run while the
current formal cycle was active.
