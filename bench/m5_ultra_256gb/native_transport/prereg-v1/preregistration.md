# E022 preregistration: native accepted-socket TCP_NODELAY

Status: CPU implementation; no listener, model, or GPU experiment has run. E020 remains frozen and unchanged. This is a new experiment, not a correction or replacement of its measurements.

## Question and local source evidence

Does enabling the standard library's existing `disable_nagle_algorithm` setting reduce native C1 HTTP delivery latency while preserving the same engine calls and exact SSE bytes?

The E020 first native process was reported by the owner as approximately 0.716 s complete-request latency and 0.131 s TTFT, versus archived E010 configuration medians 0.622 s and 0.082 s. These are different experiments, workloads of observations, and instrumentation, so the subtraction is not a causal measurement or a preregistered benefit. E020 runs and negative points must all remain.

Local CPython 3.13.16 `lib/python3.13/socketserver.py:788–845` establishes:

- `StreamRequestHandler.wbufsize = 0`; `_SocketWriter.write` calls `sendall` directly. Its `flush()` does not disable TCP packet coalescing.
- `disable_nagle_algorithm = False` by default. Setting it to `True` calls accepted-socket `setsockopt(IPPROTO_TCP, TCP_NODELAY, True)` in `setup()` before the request is handled.
- Both E010 `prefill_http_v2/native_child.py` and E020 `external_spec_http/native_server.py` leave this default unchanged. E010 adds `benchmark_delivery` to every text event; E020 does not. Both emit HTTP headers, individual SSE records, and a separate DONE write.
- E010 also has per-step timers, allocator observations, prefill-call observation, trace serialization and a 128-entry listener backlog; E020 removes those observers and uses the standard backlog. Model settings agree at C1, but these adapters are not byte-for-byte equivalent.

Local HTTP client `httpcore/_backends/sync.py:196–215` already applies `TCP_NODELAY=1` to its outbound socket. The client setting does not establish the server's sending policy. CPython `asyncio/selector_events.py:948` calls `base_events._set_nodelay`; the inspected Uvicorn auto-loop code chooses asyncio when uvloop is absent. Runtime accepted-socket readback is still needed; a source path is not a packet capture.

Nagle/delayed-ACK interaction is a hypothesis. Neither source inspection nor a roughly 40 ms difference proves it. The new experiment will not claim a fixed macOS ACK timer, a measured packet trace, GPU acceleration, or a kernel speedup.

## Small candidate

`candidate.py` creates a request-handler subclass with only the standard `disable_nagle_algorithm` flag, plus actual `getsockopt` validation on every accepted connection. OFF preserves the standard default and asserts 0; ON asks the standard setup to set 1 and asserts 1. No global socket or stdlib class is patched, no extra SSE field or padding is added, and buffering remains unchanged.

`serve.py` loads an explicitly SHA-pinned, frozen copy of the E020 native shim. It substitutes only that module's `ThreadingHTTPServer` factory so the inner handler gets the flag. The server remains the directly owned child PID. A receipt from the first readiness connection records the actual option before warm/formal requests; later connections validate it without writing receipts. There is no per-token observer. The wrapper is a prototype seam, not a production API proposal.

The CPU mock tests make no real socket and load no MLX: OFF preserves the default, ON sets the option, raw writes remain identical, mismatched socket options fail closed, invalid configuration fails early, and the base class stays unchanged. Additional tests cover the real `ThreadingMixIn.process_request_thread` exception-swallowing path, receipt-write failure, permanent failure overriding an earlier success, fixed pairing and all gate boundaries. `serve.py --help` exits before loading the source shim.

`ThreadingMixIn` normally catches handler exceptions and closes only that connection. Therefore setup/readback/receipt failures call a mandatory production-wrapper fatal callback: write `socket-failure.json` if possible, then `os._exit(70)` in `finally`, even when writing the receipt fails. This ends the whole directly owned model server. The reused controller detects its exit and reaps it, with no replacement process. A later failure can never be waived because an earlier `socket.json` succeeded; the E022 controller rejects any permanent failure file.

## Fixed next GPU plan, before launch

The owner must first freeze the experiment sources, the exact E020 native shim, its native core snapshot/manifest, model/tokenizer identity, client and lease/resource helpers. A machine-readable plan with those pins and the following schedule must be committed before launching.

- One M5 Ultra / 256 GiB, one owned GPU server at a time. Same interpreter and four MLX environment settings as E020. No external baseline rerun or core edit in this experiment.
- Same E020 native C1 engine configuration, 19 prompt IDs, at most 32 greedy output IDs, normal EOS, target-only, 2 GiB pool, compile/fuse/gates ON, norm reuse OFF. Same client, prompt, usage handling and SSE payload schema in both arms.
- Three independent fresh process pairs: OFF→ON, ON→OFF, OFF→ON. Six fresh servers total. Each has exactly three warm then three formal serial requests, with the same two-second cooldown and resource samples outside request timers. No extra tuning, replacement run, or outlier removal.
- Every request must pass the unchanged native E020 contract: raw IDs equal the pinned native oracle and canonical visible IDs, 19/32 usage, normal `length` finish, cache count explicitly zero, DONE, correct timestamps and first-chunk prefix. Both socket arms must have the expected actual option for every accepted connection. Any transport/resource/lifecycle failure stops the matrix and preserves partial evidence.
- Each process reports the median of its three formal POST→DONE times, first-text TTFTs and actual usage/window throughput. Retain all 36 warm/formal rows and every process. Pair effects use OFF/ON latency and TTFT, and ON/OFF throughput; report all three ratios and their geometric means. E010 and E020 measurements are historical context, never pooled with these six servers.
- Advance the transport-only option only if every correctness/resource/reap check passes, all three paired E2E ratios exceed 1, E2E geometric-mean speedup is at least 1.05, and every paired TTFT speedup is at least 0.97. These thresholds are frozen before any E022 observation. A failure remains a failure; no revised gate based on which point looks favorable.
- Direct child ownership, unique listener PID, bounded readiness/request/resource probes, cancellation cleanup, wait/reap/PID absence/listener absence, and outer service T→S identity restoration follow the final E020 lifecycle contract. Final restoration is audited after the controller exits.

An HTTP benefit would support this transport setting for the benchmark shim. It would not establish that removed metadata caused the historical gap, nor that model execution became faster. Identifying the old/new payload contribution would require a separately registered payload factor; it is deliberately not mixed into this one-factor test.

CPU entrypoints (no real listener):

```sh
.venv/bin/python -B -m unittest discover -s bench/m5_ultra_256gb/native_transport -p 'test_*.py' -v
python3 -B bench/m5_ultra_256gb/native_transport/serve.py --help
```

## CPU preparation and scheduled controller

`prepare.py` checks the reviewed E020 plan SHA and its completed contract, verifies and reuses the already frozen E020 `run.py`, `client.py`, `contract.py`, `launch.py`, native shim and resource/lease helpers, and pins this directory plus the actual socketserver standard-library sources. All E020 weights, packs, runtime and core pins are inherited by reference, without copying or hashing their large payloads during CPU preparation. The scheduled controller fully verifies inherited pins before/after the six-server matrix, with per-arm source/stat checks. No model bytes are rehashed within a request timer.

`run.py` calls the unmodified frozen E020 `run_one` for each native server, replacing only its launch specification in the isolated imported module. The same client, native oracle, three warm/formal requests, fixed cooldown, absolute request/probe deadlines, listener ownership, cancellation mask, reap proof and resource guards execute. Raw files reside under `off|on/pairN-native-none/`; `transport-result.json` adds the socket proof. No code in E020 is changed. The outer lease remains the only owner of service pause/resume, and its restoration must be independently audited after the controller exits.

`test_lifecycle_cpu.py` imports the actual frozen E020 controller/client/contract/launch sources and drives both launch arms through `run_one`, replacing all socket, process, resource, tokenizer and HTTP operations with mocks. It checks six requests per arm and the final cleanup path, without a listener or model. CPU receipts, preparation results and later execution status are written outside this pinned README.

Prepare using the same interpreter as the inherited controller (new directory required):

```sh
.venv/bin/python -B bench/m5_ultra_256gb/native_transport/prepare.py \
  --e020-plan-sha256 e3f0c3e24d4ea841d5e6c0b0094652e65e463c02367836e765ef6d9e3c6e2bdc \
  --out .artifacts/m5-ultra-256gb/native-transport-v1-inputs
```

After reviewing and committing that plan, the outer lease's sole job is this argv (substitute the newly prepared E022 plan SHA; no guarded wrapper):

```text
/Users/ryanli/code/minivllm/.venv/bin/python -B
/Users/ryanli/code/minivllm/bench/m5_ultra_256gb/native_transport/run.py
--plan /Users/ryanli/code/minivllm/.artifacts/m5-ultra-256gb/native-transport-v1-inputs/plan.json
--plan-sha256 <reviewed E022 plan SHA>
--out /Users/ryanli/code/minivllm/.artifacts/m5-ultra-256gb/native-transport-v1
--lease /Users/ryanli/code/minivllm/.artifacts/m5-ultra-256gb/native-transport-v1-service-lease/lease.json
```

Use the same four MLX environment variables and `suspended_service.py --pid 74384 --jobs <jobs.json> --out <new-lease-directory>` as E020. GPU execution must wait for owner review/commit and the sole-GPU queue. The current model download must also end before performance measurements. Independent archive/audit is pending. No E020 source, configuration, gate, or result is modified by these files.
