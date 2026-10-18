# B1–8 GDN prework: CPU contract results

2026-10-18 UTC. The first complete guarded CPU run passed **12/12**, with zero
failures, errors, skips, expected failures, unexpected successes, denied
operations or loaded backend modules. No source changes or retries were needed.
The 14 source snapshots and closed prerequisite receipts matched before and
after execution. These fake tests establish Python routing/ownership contracts
only; **no numerical, model, HTTP, combined-candidate or performance
qualification is granted**.

The original source-prepared README and candidate remain unchanged. This result
is later evidence, separate from their original freeze.

```sh
/Users/ryanli/code/minivllm/.artifacts/omlx-baseline/.venv/bin/python -I -B \
  /tmp/run-gdn-batched-prepared-cpu-v1.py --run
```

The runner blocked all real socket creation, subprocesses, signal delivery and
timers, device/backend imports, and model payload opens. It ran the whole
`test_candidate_cpu` module without filtering. The suite reports 0.006 s;
the guarded procedure records 0.01524 s. Neither duration measures device
performance.

Full [result](../../../.artifacts/m5-ultra-256gb/gdn-batched-prepared-cpu-v1/result.json),
[stdout](../../../.artifacts/m5-ultra-256gb/gdn-batched-prepared-cpu-v1/stdout.txt),
[stderr](../../../.artifacts/m5-ultra-256gb/gdn-batched-prepared-cpu-v1/stderr.txt),
[runner](../../../.artifacts/m5-ultra-256gb/gdn-batched-prepared-cpu-v1/runner.py)
and [artifact manifest](../../../.artifacts/m5-ultra-256gb/gdn-batched-prepared-cpu-v1/artifact-manifest.json)
are retained with original source copies and per-test/subtest outcomes.
The complete [portable CPU archive](../../../docs/experiments/m5-ultra-256gb/gdn-batched-cpu-v1/README.md)
also retains those bytes under `docs`, with an independent archive verifier,
so another checkout does not need the local `.artifacts` tree.

| Evidence | SHA256 |
| --- | --- |
| Candidate | `acd2a23802296b9e420108203cc215c3170cd1eccd9a859e1a787432f98a5353` |
| Test source | `85f160b405a801229fa96c4e1d03dbe4fc7c344eadb8da5ccbdbea2908f5f6b3` |
| Guard runner | `b5c7600f515182c07edfcf967c3233aa4aec7d35783f8eb1b020506463904bf3` |
| Actual result | `3d4451c6ab7030e9996edb1a8741eb645b45354a61d6ff83c1858af487ed2aa2` |
| Artifact manifest | `4f5617bde19803a9aab6cb620a0443dc305d7862621642eced413c3dd829d5e4` |

The separate [GDN HTTP16 CPU archive](../../../docs/experiments/m5-ultra-256gb/gdn-http-cpu-v1/README.md)
preserves its complete test and queue evidence. Both CPU procedures were gated
after the prior route formal cycle's independently verified restoration and
archive. Its historical formal-audit failure remains retained; this result
does not reinterpret that failure.

Next device evidence must compare original native batched arithmetic against
the candidate on independent continuing long-history states, first B2/B8 and
then the complete admitted domain. Full-model and actual HTTP qualification
remain separate later gates. No such run was performed by this CPU procedure.
