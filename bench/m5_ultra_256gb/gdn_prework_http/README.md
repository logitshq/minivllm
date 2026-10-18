# GDN-only recorded HTTP qualification — source preparation

These adapters are prepared source, **not an executed HTTP qualification**.
The separate GDN 18-call canary and original 37-request same-schedule numerical
qualification passed; the new HTTP history must pass its own capture, replay,
and independent audit. No performance point or E024/E028 composition is granted.

## Entries and required order

1. `http_capture.prepare(base_plan_path, base_plan_sha, C, *, python,
   environment, transport, deadlines)` rebuilds an immutable specification for
   C1 × 8 or C8 × 16 original D512 requests. `validate(spec)` reconstructs it.
   The real entry is `http_server.py`, with explicit `--gdn-prework on`,
   `--phase qualification`, `--qualification-trace`, and `--dispatch-events`.
   The server's switch defaults OFF. Original native worker, handler, request
   bodies, prompt IDs, normal-EOS rules and client are retained.
2. A future direct-child resource owner calls
   `await http_capture.capture(spec, NEW_DIR, checkpoint=..., backend=...)`.
   The default backend owns the actual server Popen. Capture must finish and
   strictly reap the server before replay can start; startup identity and a
   retained sidecar alone do not qualify arithmetic or process cleanup.
3. `http_qualification.py --prepare-only --plan PLAN --plan-sha256 SHA --out NEW`
   joins the retained raw SSE and four-event schedule trace using the pinned CPU
   tokenizer. `--run` instead runs the original two-model full-logit/state/ahead
   comparison against that history. With neither explicit mode it only describes.
   The exact eight plan fields are `schema`, `qualifier_sha256`, `qualification`,
   `serving`, `concurrency`, `capture`, `inputs`, and `sources`.
   `schema` is `GDN-recorded-http-plan-v1`; `serving` contains pinned `audit`,
   `plan`, and `complete` refs to the actual GDN37 prerequisite. The original
   numerical plan, 1800-second child deadline, flags and fixtures remain fixed.
4. After terminal replay, `audit.py --child CLOSED --plan-sha256 SHA --out NEW_JSON`
   independently rejoins the wire/trace, checks original full64-layer device
   comparison receipts, and derives both dispatch streams from the semantic call
   order. It also binds exact PID/port probe commands and closed capture bytes.
   Its `GDN-recorded-http-independent-v1` PASS covers only this captured history;
   the future outer audit must separately qualify resources and restoration.

The replay installs HTTP workload selection **before** frozen `serving.adapt`,
then directly calls original `qualify_native.main`. Calling the old
`run_adapter()` inside that scope would overwrite the case-label wrapper and is
intentionally avoided. Recorded tokens/schedules are comparison targets only:
both real engines generate their own greedy tokens.

## Routing and evidence limits

Only the 48 original GDN instances are installed. E024 route graph and E028
identity refill remain OFF; original short compilation/fusion/gates stay ON,
norm reuse stays OFF. The server observer reads Python counters and context
metadata only, with no token `tolist`, tensor eval, or full-state snapshots.
Actual current and future calls produce rows; consuming an existing ahead does
not fabricate a new invocation. SUBMITTED means a Python submission, not a Metal
kernel completion. Replay retains the original evaluated bit/finite gates.

The candidate supports ordinary B1 decode. B>1 uses the original fallback;
short packed calls and ordinary prefill are untouched. C1 must show real B1
candidate work. C8 must show actual multirow fallback and actual HTTP peak8;
its tail may legitimately shrink to B1 and hit the candidate. HTTP concurrency
is not GPU batch size. `prefill_batch` is explicitly rejected by the frozen
observer contract rather than silently treated as ordinary prefill.

Capture results use `GDN-prework-http-capture-result-v1`; replay results use
`GDN-recorded-http-result-v1`. Producer PASS sets
`replay_numerical_checks_passed`, while independent HTTP/resource/performance
flags remain false. Successful tensor payloads are not retained by the inherited
Comparator: offline audit validates source-pinned device bit/finite receipts,
not a fresh arithmetic recomputation. First failures remain failures with the
original failure evidence. Unobserved EOS, cancellation, sampling-key invalidation,
parameter replacement, future arrivals and combined candidates gain no coverage.

## Validation status

`test_http_cpu.py` contains **16 source-prepared CPU test methods; none have run**.
Only AST parsing and source review were performed while the unrelated formal
GPU benchmark was active. Fakes cover no-array-read dispatch, partial install
and observer rollback, both-scope cleanup, adapter ordering, C8 fallback,
closed-prerequisite binding, actual capture/reap receipt shape, rehashed wrong
PID rejection, consumed ahead, unchanged full-state audit gates, and incomplete
terminal refusal. They cannot prove numerical GPU behavior.

There is **no new outer lifecycle owner, controller, execution registration,
resource audit, or restoration predecessor in this directory**. Those remain
required before any real run. Existing frozen native, route graph and GDN
candidate/serving sources were not changed. `source-manifest.json` records this
source-only snapshot; it is not a test or execution receipt.
