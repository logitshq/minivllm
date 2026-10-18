# C1 timestamp gap and next falsifiable observations

Lightweight read-only diagnostic while the current formal cycle is active.
Calculation: `/tmp/combined-v2-r1-c1-timestamps-v2.json`; readable table:
`/tmp/combined-v2-r1-c1-timestamps-v2.md`. Both preserve the initial path-only
assertion failure and use its already extracted 192 timestamps. Each original
group was read once. The two small numerical plans were independently checked
as identical bytes with SHA 21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb.

Native current r1 ON minus historical oMLX r1 has mean E2E difference
0.513710672014 s: TTFT difference 0.074106186188 s and post-first difference
0.439604485826 s. The latter is 85.574333915% of the additive mean difference.
Dividing that post-first difference by 511 gives 0.860282751 ms per subsequent
token as a diagnostic amortization target only. It is not measured ITL, TPOT or
GPU time: first output is first nonempty received answer/reasoning text, and the
interval includes scheduling, transport, host work and terminal completion.
Current native r1 OFF/ON remain unaudited partial results; historical oMLX is an
earlier independently accepted point, not a contemporaneous interleaved control.

## Minimal observations for the two unqualified candidates

`ahead_submit_once/candidate.py` omits one API submission only after an exact
ahead-key and token/all-state-root object certificate from a normally returned
prior async submission. It does not remove a model call or prove redundant GPU
work. Existing scalar counters are sufficient to establish whether it operates:
record runs, exact_hit_eligible, omitted_current_submits, forwarded_async_submits,
fallback_runs and errors at the drained boundary. Require a positive omission
count before claiming this path was exercised; unsupported weakrefs or changed
identities must remain ordinary fallback. With the explicit frozen-profile-first
installation order, forwarded_async_submits must match retained actual
existing_async_eval events. Omissions must not exceed eligible hits. For matched
request histories, independently compare actual current/future decode-call,
sample and eval counts and the same complete numerical/state gate. A reduced
async API count with no net runner/HTTP improvement falsifies its performance
benefit under that measured workload; it cannot imply GPU kernels were removed.

`long_decode_onepass/candidate.py` combines the dynamic layer identity/cache
checks, flags and state collection; it still calls mx.compile with inputs=model.
It does not remove captured model flattening or change state writeback. Use the
same existing `decode_graph_submission` and nested `compiled_callable` spans in
both arms, through the explicitly verified new-class profile adapter already
described in the source review. Record actual callable invocation/key/used-length
and B, and separate cold first invocation, fallback, the 1024 route boundary and
steady eligible calls. The per-call outer interval minus its nested callable
interval measures surrounding Python work plus instrumentation; it is not a
pure guard timer. If that remainder is immaterial, onepass cannot explain the
remaining HTTP gap; if the candidate does not reduce it beyond observer noise,
its proposed host-work saving is falsified. A large callable interval instead
does not implicate onepass, since captured model traversal remains unchanged.

Keep OFF/runner/full observation-overhead controls, identical decoder/GDN flags,
normal EOS and the exact same request history. Do not sum overlapping spans into
exclusive GPU time or add synchronization to create a cleaner-looking clock.
Neither candidate has actual device qualification from this diagnostic; no
candidate execution, tests, backend imports, source changes or extra framework
were performed here.

## Additional source boundary supplied by the root review

The measured oMLX wheel's `engine_core.py:438–494` `_step_burst` retains multiple
`scheduler.step` calls on the same MLX executor (balanced: 0.1 s for one request,
0.03 s when concurrent, at most 64; a first chunk ends the burst immediately).
That reduces repeated async event-loop/executor and GIL transitions. Our native
`BatchWorker` already runs consecutive steps on a dedicated thread, so copying
that burst mechanism does not establish the same removable cost. The historical
formal oMLX settings have MTP, DFlash and VLM-MTP disabled; the comparison is not
explained by enabling those draft mechanisms. Delivery/tokenizer work competing
for the GIL is a profile hypothesis, not an attribution established by these
HTTP timestamps. Do not alter current serving cadence on this evidence alone.
