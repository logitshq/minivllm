# E022 v2: portable TCP_NODELAY boolean readback

Status: CPU implementation and local socket semantics checks only. No new execution plan has been prepared or frozen; no GPU, model, or HTTP server was started. E022 v1 remains a failed experiment with its original sources, plan and raw evidence unchanged. The owner reported that its ON arm was rejected because the actual accepted socket returned integer `4` rather than `1`.

## Scope and provenance

The seven Python files listed in `copied-from-v1.json` were copied from `../native_transport/`; that file records each original path and SHA-256. `common.py` and `test_lifecycle_cpu.py` remain byte-identical. `v1-to-v2.patch` records the complete changes to copied files. The additional probe and source evidence are CPU diagnostics, not part of the timed HTTP path.

The necessary correction is to validate `bool(raw) == requested` for the integer returned by `getsockopt`. The accepted-socket handler and the independent result validator both apply this rule. Receipts retain `tcp_nodelay_actual` as the unmodified raw integer and add `tcp_nodelay_actual_enabled`. Plan and socket receipt schemas use v2 so the old integer-equality contract cannot silently accept new evidence.

An actual opposite boolean state, a non-integer readback, a receipt error, or a later accepted-connection failure still rejects the whole process. The existing permanent failure receipt and whole-process exit code 70 remain. There is no retry exemption for an earlier successful connection. The standard `disable_nagle_algorithm` flag, SSE bytes, buffering, model settings, inherited E020 lifecycle, request timers, and launch order are unchanged.

## Independent Apple source check

Apple's published XNU code returns the `TF_NODELAY` bitmask for this getter; its definition is hexadecimal `0x00004`. This explains why an enabled option can read back as integer `4`. [Apple getter, pinned source line 2830](https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/bsd/netinet/tcp_usrreq.c#L2830), [Apple flag definition, line 433](https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/bsd/netinet/tcp_var.h#L433).

The Apple TCP manual describes this as a boolean option, and the setter treats a nonzero value as enabled. Normalizing the readback for comparison therefore preserves the option's intended semantics. [Apple manual, line 133](https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/bsd/man/man4/tcp.4#L133), [Apple setter, line 2347](https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/bsd/netinet/tcp_usrreq.c#L2347).

`apple-source-evidence.json` retains the pinned commit, URLs, file hashes and original line numbers. This published commit is source evidence for the semantics; it is not identified as the exact binary build running locally. Neither source inspection nor the probe establishes a latency improvement.

## CPU evidence

`cpu-checks.json` records **18 passing tests**, AST parsing of every Python file, source hashes, and successful `serve.py`, `run.py`, and `prepare.py` help exits. Only `--help` was invoked for preparation. The tests include raw `0`, `1`, and `4`, both opposite-state failures, inconsistent normalized receipts, invalid raw types, unchanged wire bytes, permanent fatal propagation through the real stdlib `ThreadingMixIn` exception handler, all existing gate failures, and the actual frozen E020 lifecycle under mocked IO.

`cpu-localhost-probe.json` records a separate bounded, pure CPU TCP probe on two ephemeral loopback listeners. OFF read back **0** and ON read back **4**. Direct setter inputs `0`, `1`, and `4` read back `0`, `4`, and `4` in both connections. The test bytes arrived unchanged. All accepted/client/listener descriptors closed (`fileno == -1`), and both retired listener ports returned `ECONNREFUSED`. No threads, subprocesses, HTTP services, model imports, or GPU operations were used by this probe.

To repeat CPU checks:

```sh
.venv/bin/python -B -m unittest discover -s bench/m5_ultra_256gb/native_transport_v2 -p 'test_*.py' -v
.venv/bin/python -B bench/m5_ultra_256gb/native_transport_v2/probe_socket_cpu.py --out /tmp/e022-v2-new-probe.json
```

The probe refuses an existing output file; retain every attempt. A repeated probe is diagnostic evidence and must not replace the saved result.

## Unchanged performance contract

Any future execution must first receive a separately reviewed, committed v2 plan. The inherited E022 schedule and thresholds are unchanged: three fresh process pairs **OFF→ON, ON→OFF, OFF→ON**, three warm plus three formal requests per process (**36 total; 18 formal**), C1, the frozen 19-token prompt and 32-token cap, normal EOS, target-only, no prefix cache, two-second cooldown, and the same frozen E020 model/client/core and resource/lease/reap contract.

Correctness and lifecycle remain hard gates. All three paired complete-request latency speedups must exceed 1, their geometric mean must be at least **1.05**, and every paired TTFT speedup must be at least **0.97**. Retain every process and request, including negative points. No pairing order or threshold was changed after v1 failed. A future v2 result must remain distinct from v1 and from E020; it cannot be pooled with or substituted for those measurements. No throughput-versus-complete-latency Pareto conclusion follows from these CPU checks.
