# E050 independent prefix audit

The actual offline audit passed (attempt2, exit0, 0.665s), reusing the frozen E049
full 49-step checker and independently binding the new candidate composition.
It verified 380 prefix IDs, 103 full-vocabulary logit pairs, 64,768 state-array
pairs and 134 ahead-token pairs: 65,005 bit-equality and 130,010 finite receipts.
Both eager and the new combined candidate match the captured warm49 IDs; their
full states/logits match according to the complete producer comparison receipts.

`inherited/plan.json` is deliberately the unchanged E049 replay contract. It does
not mean E050 executed the old graph-only candidate: the outer plan pins the new
`prefix.py` private FunctionType globals, engine-aware `bridge.py`,
`gdn_batched_graph/integration.py`, and batched GDN dependency. The audit checks
those source hashes, same-PID outer/inherited bindings, complete original ledger,
and the actual two-event `composition.jsonl` lifecycle without importing producers.

All 48 distinct GDN layers were installed before tracing and restored. Each layer
recorded five Python candidate branches and zero fallback branches: 240/0 total.
These counts include graph tracing; they are neither GPU execution counts nor a
per-B breakdown. The independently checked route sidecar contains 49 graph calls
(45 bound1023, four bound1536); its next unconsumed captured event is step50.

The result remains `INCOMPLETE_BY_DESIGN`: zero complete requests, eight active,
eight unadmitted. It is not a performance or full-history qualification.
Successful raw full tensors were not saved, so this is complete source-bound
per-array receipt verification, not offline tensor-byte or arithmetic recomputation.
The 56,315,007,048-byte MLX peak is an allocator receipt, not physical footprint.

Evidence in `.artifacts/m5-ultra-256gb/`:

- `c8-batched-prefix-v1-independent-audit-attempt1`: retained audit FAIL. The new
  auditor incorrectly expected a nonempty graph cache at `Facade.restore`.
  Frozen `RouteObserver.restore` first restores its original empty dictionary;
  traced functions live separately in the route witness. No experiment changed.
- `c8-batched-prefix-v1-independent-audit-attempt2`: PASS, with source snapshot,
  exact command, stdout/stderr, return code and complete nested audit result.
- `c8-batched-prefix-audit-cpu-v1`: seven mutation tests PASS, covering wrong
  identities, missing layers/events, fallback/unreached candidate, installation
  after tracing, incomplete restoration and false GPU-count scope.

Both executions blocked MLX imports, subprocess launches and process signals
inside the child. No GPU retry, service operation or frozen-source edit occurred.
