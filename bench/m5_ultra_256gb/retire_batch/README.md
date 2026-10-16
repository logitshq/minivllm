# Retire finished rows once per scheduler step

`ModelRunner.drop_many` collects a step's completed request IDs, retains the
surviving rows in their original order, and compacts their caches at most once.
When the entire batch finishes it clears the batch without gathering rows.
`LLMEngine.step` calls this once after building its output list. The single-ID
`drop` entry point remains available. Empty retirement preserves lookahead;
nonempty retirement invalidates it, as before. Drafter statistics are dropped
per request and its context resets once per retirement call.

The reference implementation repeatedly rebuilt the surviving KV/GDN rows for
each completed request. A simultaneous finish of B requests caused B-1
intermediate rebuilds, although none of those intermediate batches was decoded.
This change preserves arithmetic, request order, stopping and cache contents.

Run `test_retirement.py` for CPU lifecycle coverage. Before adoption,
`MINIVLLM_RETIREMENT_SOURCE` can point to the candidate source root; by default
the test exercises the repository implementation. It imports MLX but creates
no arrays and executes no model work.

`qualify.py --source SOURCE --out NEW_DIRECTORY` compares the candidate and a
frozen baseline using the real 27B model. It checks every output and unsigned
byte signature of every surviving cache array after each engine step. Cases
cover B2/8/32/64 simultaneous finishes and staggered finishes with late arrivals
and a stop-token request. It also counts rebuilds. These timings are diagnostic,
not a fair performance comparison: the candidate runs after the reference.

`frontier.py --source SOURCE --out NEW_DIRECTORY` uses the existing loopback
HTTP harness and three fresh processes at each of C8/32/64. Each process has
three warm and three measured groups. Earlier native and official-engine
measurements are retained as unpaired references. The plan and exact source
hashes are recorded before measurement. No result is discarded based on speed.

The source roots, qualification and raw measurements are retained under
`.artifacts/m5-ultra-256gb/retire-batch-*`. The main `decode_batch/replay.py`
accepts `--retirement-frontier` to independently include these measurements in
the final comparison while retaining the original implementation's points.
