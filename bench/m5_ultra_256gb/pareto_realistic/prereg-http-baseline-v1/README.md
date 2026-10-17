# E023 actual HTTP baseline capture and replay v1

Goal: obtain real native HTTP scheduling and output evidence for the fixed
Qwen3.8-27B-nvfp4 D workload, before paired throughput–complete-request latency
measurements. E028's completed 37-request same-schedule qualification does not
supply this evidence. This run uses the existing native stack with E028 OFF.

## Fixed execution

Four sequential stages: C1 capture, C1 replay, C8 capture, C8 replay. C1 sends the
eight D fixtures once; C8 sends sixteen requests with the same fixtures twice.
Every request uses the unchanged prepared prompt, greedy sampling, normal EOS,
512-token cap and the existing native completion API. C1/C8 are real HTTP
concurrency levels. Preserve original raw SSE, actual UUID arrivals, schedules,
raw/committed IDs and every failure. Do not force recorded schedules or tokens
into the replay engines.

Each capture has one freshly started owned server. Before replay, the server
must exit, be waited/reaped and have no remaining PID/listener. Replay loads its
own independent eager reference and native candidate, checks full logits, all
state and delivered IDs on the captured arrival history, and then releases both.
The original 37-request qualification is separate. This experiment leaves the
long-context graph and E028 candidate disabled.

Model context2048, batch token budget2048, allocator cache2GiB; compile-decode,
fuse-decode and gates-compile ON, norm-reuse OFF. The existing long-context guard
still applies. Explicit MLX environment: max ops/buffer50, max MB/buffer50,
BFS width20, FAST_SYNCH0. Local port18023; TCP_NODELAY ON; server IO timeout300s.
Capture bounds: readiness120s, request600s, group720s, overall900s, close10s,
TERM15s and KILL/reap15s; resource sampling interval5s. Each replay has the original
1800s child alarm with1830s owner bound. The complete four-stage work bound is5700s;
cleanup and source/model postflight have their separate finite bounds.

## Service and resources

The user has authorized the same cold-stop/experiment/restore cycle. V3 derives
the exact current original service identity from the immutable committed E028
RESTORED journal and its independently verified audit. It must match the live
process, fixed launchd job and port8001; it cannot discover and adopt another
service. Preparation is read-only. Stop occurs only after the final source/plan
and CPU review are committed.

The owner requires the exact bound STOPPED journal, pressure1, AC/thermal guards,
no new swap usage/Swapouts and no other recognized inference process. It verifies
runtime, source and model pins before and after. Each returned Popen child has
its own durable intent and release; unknown spawn is never called no-child.
A capture failure prevents replay. After failure or success, restore the original
oMLX model and verify fresh Lightning MTP/resident PLE plus the persisted-visible
contract. Preserve warm-KV and hidden-runtime limitations of a cold restart.

## Acceptance and interpretation

Both recorded workloads must complete with full numerical checks and strict
cleanup; retained records are independently inspected before any qualification
claim. Scope is the captured discrete histories. Actual early EOS is covered only
if it occurs, and output caps must not be reported as EOS tests.

This is an instrumented correctness experiment. No capture duration, replay
wall time, diagnostic memory peak or CPU test result becomes a Pareto point.
Subsequent formal runs measure actual committed output tokens divided by the
common request window against POST→DONE p50, accompanied by p95 and output-length
distributions. Same-run oMLX and Lithos comparisons retain the same model, precision
and workload contract. No universal-fastest claim is implied by this experiment.

Exact plan, source digests and CPU/read-only preparation receipts are saved in
this directory before execution. Raw runs use separately fresh directories under
`.artifacts/m5-ultra-256gb/http-baseline-v1-*`.
