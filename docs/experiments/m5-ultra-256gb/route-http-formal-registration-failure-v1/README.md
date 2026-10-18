# E024 route formal: failed CPU registration v1

This archive preserves the actual failed registration, its guarded wrapper, stdout/stderr, all source snapshots, the original inventory, and the complete partial preregistration directory. It does not repair or replace any original evidence.

The real owner-plan SHA is `d2f66bc9b744e5325476fb9d6c3734c4639d7f80d5d4848b86178c87016baeae`. The registrar reused its `digest` variable in the pin loop, then saved the final `test_execute_cpu.py` SHA `a594ab8885b6ba24dd24e644a037b06c6af3a47846d02ae22796a797bc4e96f8` into commands and preparation. The actual owner CPU preparation rejected `route formal plan SHA`. The original guard receipt remains **FAIL**.

The wrapper reports no GPU execution, service change or service PREPARED action; its backend/process/network/signal guards stayed intact. No v1 cold journal, owner, execution, CPU-prepare output or sealed receipt exists. These are registration-scope receipt/filesystem checks, not a fresh OS probe. Prior CPU tests passing does not make this registration valid.

A separate v2 registrar may fix this issue; it is outside this immutable failure archive. The compressed archive contains both entire raw trees, including all frozen-source bytes and empty stdout/stderr. `manifest.json` covers every regular tar member; `archive-verification.json` records reopened and safe-extracted per-member SHA/size plus original input rechecks. No model/runtime payload or inference ledger is included.
