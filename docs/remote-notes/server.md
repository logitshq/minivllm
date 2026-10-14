# A6 — OpenAI-compatible server, AsyncLLMEngine, bench script

Files: `minivllm/server.py`, `minivllm/engine/async_engine.py`, `minivllm/engine/llm_engine.py` (interface only),
`minivllm/sampling_params.py` (minimal, owned by the engine agent), `tests/fake_engine.py`, `tests/test_server.py`,
`bench/bench_serve.sh`. Everything here runs without a GPU: the engine is faked, the tokenizer is real.

Test command (box): `ssh surf@surf-gpu 'cd ~/minivllm && .venv/bin/pytest tests/test_server.py -q -x -s'`

## 1. What was built

```
client ──HTTP──▶ uvicorn ──▶ FastAPI app (server.py)
                               │ tokenizer.encode / apply_chat_template      (server side, never the engine)
                               ▼
                         AsyncLLMEngine.generate(token_ids, SamplingParams)   async iterator of RequestOutput
                               │ pending adds/aborts ─┐
                               ▼                      ▼
                         step task:  drain pending → engine.step() → put RequestOutput on each request's asyncio.Queue
                               │ (step() runs in a 1-thread executor by default, inline with --step-inline)
                               ▼
                         LLMEngine (phase B) / FakeLLMEngine (tests): add_request / step / abort / has_unfinished_requests
```

* `POST /v1/completions`, `POST /v1/chat/completions`: non-streaming JSON or SSE. The SSE stream has exactly
  vLLM's shape because `vllm bench serve --backend vllm` parses it: `data: {json}\n\n` lines, `choices[0].text`
  (completions) / `choices[0].delta.content` (chat, first chunk carries `delta.role`), `finish_reason` on the last
  text chunk, then — only when `stream_options.include_usage` — a chunk with `choices: []` and `usage`, then
  `data: [DONE]`. One chunk per engine step per request, even when the text delta is empty (vLLM does the same;
  the bench counts ITL per chunk, so matching this keeps ITL/TPOT comparable).
* Honoured request fields: `max_tokens` (`max_completion_tokens` for chat), `temperature`, `top_p`, `top_k`,
  `stop` (strings, matched server side), `stop_token_ids`, `ignore_eos`, `stream`, `stream_options.include_usage`.
  `n != 1` → 400. Unknown fields (`logprobs`, `repetition_penalty`, …) are ignored, which is what the bench sends.
  Prompt may be a string, a list of token ids, or a one-element list of strings. Any `model` name is accepted and
  echoed back (the bench sends whatever `--model` it was given; a 404 here buys nothing).
* `GET /v1/models`, `GET /health`. Validation errors are 400 with `{"error": {...}}` (FastAPI's default 422 is
  remapped; vLLM also answers 400).
* The server owns tokenisation. The engine interface takes token ids; `usage.prompt_tokens` is simply
  `len(prompt_ids)`, `completion_tokens` counts every id the engine reported (including an EOS/stop token, as vLLM does).

## 2. Design decisions and why

### 2.1 LLMEngine interface (phase B fills it)
`add_request(prompt_or_token_ids, SamplingParams) -> req_id`, `step() -> list[RequestOutput]`, `abort(req_id)`,
`has_unfinished_requests()`; `RequestOutput(req_id, new_token_ids, finished, finish_reason)`. Semantics that the
async wrapper and the server rely on (also in the docstrings):
* `step()` reports only requests that sampled ≥ 1 token (a chunked-prefill step for a request produces nothing).
* `finished=True` exactly once, together with the last tokens. `finish_reason="stop"` means the LAST id of
  `new_token_ids` is the EOS/stop token — counted in `completion_tokens`, not detokenised. `"length"` = max_tokens.
* `abort()` is silent (no output) and ignores unknown ids; the async wrapper may abort an id whose last output
  is already in flight.
* The engine is single-threaded; the wrapper serialises every call.

### 2.2 AsyncLLMEngine: one step task, per-request queues, pending lists
`generate()` does not call the engine. It appends `(token_ids, sampling, queue, future)` to `_pending_adds` and
makes sure the step task is running; the step task drains the pending adds/aborts, calls `engine.step()`, and
puts each `RequestOutput` on its request's `asyncio.Queue`. The step task ends when `has_unfinished_requests()`
is False and is recreated by the next `generate()`, so an idle server burns no CPU.

Why pending lists instead of calling `engine.add_request` directly from `generate()`: with the step running in a
worker thread, a direct call would race with `step()`. Funnelling every engine call through the step task (which
runs on the event-loop thread between steps) gives the engine a strict single-caller guarantee without a lock —
and a lock would be worse: `add_request` would block the event loop until the running step finishes.

Client disconnects: Starlette cancels the response task, the `async for` inside `generate()` sees
`CancelledError`/`GeneratorExit`, and `finally` moves the request to `_pending_aborts`. Stop strings use the same
path: the server `return`s out of the `async for` when a stop string matched.

If `step()` raises, the exception object is put on every live queue and re-raised in each `generate()`, so a
phase-B bug shows up as a 500 on every stream instead of 32 hung connections.

### 2.3 step() inline vs in a thread — measured, thread is the default
Measurement: `tests/test_server.py::test_async_engine_event_loop_blocking` runs 32 streams × 50 tokens on
`FakeLLMEngine(step_time=5 ms)` (step = `time.sleep`, which releases the GIL like a CUDA sync does) while a
probe task does `await asyncio.sleep(1 ms)` in a loop and records how late it wakes up (= event-loop lag).

NUMBERS_LOOP_LAG

Pitfall found on the way (inline mode): the first version yielded with `await asyncio.sleep(0)` and showed
18 ms lag for 5 ms steps — three steps, not one. `sleep(0)` re-queues the task at the FRONT of the next loop
iteration; the socket events and timers that became due during the blocking step are appended BEHIND it in
`BaseEventLoop._run_once`, so each of them waits one more full step. Resuming through `loop.call_later(0, …)`
puts the step task behind the I/O callbacks and behind every timer that expired earlier (timers dequeue in `when`
order), and the lag dropped to one step. (`_yield_behind_io_and_timers` in async_engine.py.)

Decision: `step_in_thread=True` by default, `--step-inline` to switch. With the real engine the step holds the
GIL during Python-side scheduling and releases it inside CUDA waits, so the loop gets most of the step wall time
for HTTP work; the executor hop costs ~10–40 µs per step (the no-HTTP cost line below). Phase B must re-run
the lag test with the real engine; if pure-Python scheduler time dominates a step, the two modes converge and
inline saves the hop.

### 2.4 Incremental detokenisation (`IncrementalDetokenizer`)
Decoding each new token alone is wrong for a byte-level BPE vocabulary: a rare character arrives as 2–4 byte
tokens, each decoding to U+FFFD on its own. The classic HF/vLLM "prefix window" algorithm: keep offsets
`prefix_offset ≤ read_offset ≤ len(ids)`; decode `ids[prefix_offset:read_offset]` (already emitted) and
`ids[prefix_offset:]`; emit the difference only if it is longer and does not end with U+FFFD, then slide the
window. Otherwise the new tokens stay buffered; `flush()` at the end emits whatever is left (a genuinely
truncated character comes out as U+FFFD, which is the only honest answer).
Alternative rejected for now: `tokenizers.DecodeStream` (Rust, what vLLM V1 uses). It would cut the two
`decode()` calls per token to one Rust call; measured cost of the Python version is in §3 — it is not the
bottleneck at 32 streams, so the readable version stays.

### 2.5 Stop strings
Matched on the full output text each step and cut out (vLLM default `include_stop_str_in_output=False`).
While streaming, the last `max(len(stop)) − 1` characters are held back so a stop string spanning two chunks
can still be removed before the client sees any of it (vLLM V1 does the same, `stop_buffer_length`). A match
breaks out of `generate()` → abort in the engine.

### 2.6 FakeLLMEngine
Echoes the prompt ids cyclically, one per step, FCFS admission up to `max_num_seqs`, `time.sleep(step_time)`
per step. Echo makes everything checkable from the prompt alone: exact token counts, exact text, EOS handling
(the chat template contains `<|im_end|>` = eos, so a chat echo must finish with "stop" at that index), and
byte-fallback (pass the byte-token ids of 你好 as the prompt → the stream must be `["", "", "你", "", "", "好"]`).

### 2.7 bench/bench_serve.sh
Runs `~/vllm-baseline/.venv/bin/vllm bench serve` with the §10 protocol as defaults; every parameter is an
environment variable (`HOST PORT MODEL TOKENIZER INPUT_LEN OUTPUT_LEN NUM_PROMPTS MAX_CONCURRENCY BACKEND
VLLM_BIN RESULT_DIR`), the first positional is the result label, extra args pass through (`--ignore-eos`,
`--save-detailed`). JSON lands in `bench/results/<label>-<timestamp>.json` (gitignored).

## 3. Numbers

NUMBERS_SECTION

## 4. Pitfalls

PITFALLS_SECTION

## 5. What phase B has to do
* Fill `LLMEngine` with the semantics in §2.1; keep `abort(unknown)` a no-op.
* `server.py::main()` builds `EngineConfig(model=…, max_model_len, max_num_seqs, max_num_batched_tokens,
  gpu_memory_utilization, enforce_eager)` — align the keyword names with `config.py`.
* Re-run `test_async_engine_event_loop_blocking`'s probe idea against the real engine; keep the thread default
  unless inline measures better.
* Before benchmarking: the chat template's `enable_thinking` is left undefined → the template appends
  `<think>\n\n</think>\n\n` (thinking off). Expose `chat_template_kwargs` only if a client needs it.
