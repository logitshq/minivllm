# Incremental GDN on graph: prepared timing factory

This factory uses the same combined HTTP server for both arms: route graph ON,
GDN prework OFF versus ON. E028 stays OFF; context remains 2048, compile/fuse/gates
ON and norm OFF. Both observers are OFF. No performance result exists here.

`formal.prepare(base_plan, concurrency, repeat, *, arm, baseline_qualification,
route_qualification, combined_qualification, environment, transport, deadlines)`
returns a pinned specification. `validate(spec)` reconstructs it. The asynchronous
`run(spec, out, *, checkpoint, backend=None)` owns one fresh server; an external
resource/service owner must provide checkpoints and enforce restoration.
The command-line entry only describes the proposal.

The original native qualification supplies fixed D8 normal-EOS/cap512 request
bodies and count semantics. OFF additionally binds the actual E024 HTTP history;
ON binds the actual combined history. Both arms require the combined C1/C8 cycle
to have independent numerical, resource and restoration PASS receipts. Unsealed
cycle sources or missing actual receipts reject preparation. No future receipt
is synthesized, and installation identity does not establish formal graph hits.

The twelve fresh processes retain the existing alternating order:

```
r1-c1-off r1-c1-on r1-c8-on r1-c8-off
r2-c8-off r2-c8-on r2-c1-on r2-c1-off
r3-c1-off r3-c1-on r3-c8-on r3-c8-off
```

Each process sends eight warm and 64 formal requests. The original native run
code, SSE/count checker, POST-to-DONE timing, concurrent queue and bounded cleanup
are reused with a private validator dependency; imported module globals are not
changed. A point remains `performance_qualified=false` until an independent
complete-cycle audit. Report per-process metrics, three-process medians and six
actual pair ratios. Do not pool historical native trials or multiply earlier
GDN and graph speedups. These C1/C8 histories do not qualify larger concurrency
or every future arrival schedule.
