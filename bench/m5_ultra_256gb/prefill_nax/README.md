# E010: preserve per-sequence NAX reduction

The prototype broadcasts packed weights to `[B,N,Kpacked]`, keeping the input
matrix dimension `M=T`. It only batches projections whose original single-row
invocation has split-K=1. Other projections remain separate. This is not weight
dequantization, arithmetic approximation, or a custom universal compiler.

`screen.py` checks each layer/cache then screens whole prefill. `boundaries.py`
adds tile/chunk boundary lengths. `lifecycle.py` checks actual scheduling,
partial prefill, delayed admission, stops and retirement with graph/fuse decode.
All scripts write their plan and source hashes before GPU work.

The first HTTP run under `../prefill_http/` is **contaminated**, not accepted:
a pre-existing process titled `omlx-server` escaped the old command-line matcher.
No samples are deleted. `../resources_v2.py` covers native executable names;
`../prefill_http_v2/` tests the final integrated core with the updated probe.

`../prefill_integration/qualify.py` reuses the preserved numerical cases against
the actual core. It does not install `candidate.py`. The control only disables
runner grouping; individual prefill arithmetic stays unchanged.

CPU checks:

```sh
.venv/bin/python -B bench/m5_ultra_256gb/prefill_nax/test_integration.py
```

See [E010 ledger](../../../docs/optimization-guide/experiments.md) for exact
scope, provisional timing, interference, pending HTTP gates and raw evidence.
The service suspension helper requires explicit human authorization; it is not
an automatic prerequisite for running a benchmark on an otherwise idle machine.
