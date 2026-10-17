# Native HTTP cleanup cancellation repair

The separately versioned `native_cleanup.py` privately loads the unchanged frozen SSE client and replaces only its cleanup function. Root reviewed the implementation and independently ran six CPU tests in the repository venv; all passed. The negative cases first reproduce the old false acceptance, then require rejection after either response close or client close raises CancelledError.

The corrected order is workers, response closers, then client close under one cleanup deadline. Valid DONE still immediately refills the request slot. Unresolved or cancelled closers prevent measurement eligibility. The caller must catch this adapter’s private GroupCancelled class, preserve partial results and reap its server. The original request/parser/statistics functions and source remain unchanged.

Tests also check bounded cancellation resistance, external cancellation, no public module mutation and source corruption rejection. This does not test real sockets or qualify a model, HTTP schedule, future arrival pattern or performance point. The receive-tail scope remains unchanged: it cannot attest to wire bytes never received after DONE.

The new capture phase derives fixed requests, CLI and startup identity from the registered numerical plan. Root independently passed its seven CPU lifecycle tests. A repeated-cancellation review issue was fixed before use: separate bounded cleanup still attempts TERM/KILL/wait and strict PID/port absence. Missing release uses a cleanup-incomplete callback with the owned PID; the outer owner must reconcile it before replay or service recovery. Successful capture is not numerical or performance qualification.

No actual HTTP capture was run. The next outer owner must bind the restored service identity, verify runtime/weight pins, run capture and fully reap it before creating the independent numerical replay model pair.

Independent draft_graph source review found no blocker in the final capture lifecycle. It specifically checked that private GroupCancelled partial results are saved before rethrow; repeated caller cancellation does not cancel the separate cleanup task; receipt/PID/port failures remain FAIL with the exact unresolved PID. This review did not rerun the CPU tests or operate any service.
