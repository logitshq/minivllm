# M5 Ultra / 256 GiB qualification

Local migration of `tutorial-17-handoff-v1` (`559ced1`) for the actual Apple M5 Ultra (80 GPU cores, applegpu_g17d). Accepted engine: f976d6a; candidate engine: b158c0f. Engine sources and all numerical assertions remain unchanged. V19 remains default OFF.

`worker.py` retains the historical full-model numerical body and `fallback.py` is byte-identical. `test_migration.py` checks these boundaries. Changes are confined to machine identity, local runtime provenance, paths, desktop power detection, and ownership receipts. M5 retains the engine's existing NAX prefill fallback; this migration does not turn on the M4-only batched projection path. The original four MLX controls are used; MLX_ENABLE_TF32 is unset. Python is 3.13.16, differing from the source machine's 3.13.14; MLX/MLX-Metal remain 0.32.3.

The package in `docs/handoffs/256gb-mac` is unpacked into `.artifacts/m5-ultra-256gb/source`. Each new domain is frozen with `freeze.py`. `MINIVLLM_EXPERIMENT_DOMAIN` selects the domain; its frozen JSON pins the interpreter, installed MLX Python/native/header/Metal files, NumPy/tokenizer runtime files, all model files, exported engines, workload and harness. Hugging Face LFS hashes are verified against API metadata, not inferred from a resolved local cache filename.

Before a run, execute the CPU suite with the selected domain and save stdout/stderr. The suite includes byte/signed-zero differences, final-chunk nonfinite rejection, source identity, dispatch delegation, changed GO fields, pressure/swap stops, desktop power, and one-shot attempt consumption. `probe.py` runs a small BF16/NVFP4/Metal check separately from model qualification. Synthetic CPU fixtures are never model evidence.

`owner.py --frozen-sha256 HASH` consumes a single local attempt. The user requested these experiments; this local owner creates a fresh GO and never reuses M4 authorization. The driver loads accepted then candidate in separate, serial subprocesses. Four rounds of 32 outputs check 128 raw logits files, 16,384 state signatures, all cache boundaries, parameter identities, dispatch paths and 18 fallbacks. Processes are waited/reaped. The outer owner's exact PID absence must be checked after the execution tool returns. Process exit is the unload boundary; in-process allocator release is not asserted.

`replay.py OUTPUT` is a separate stdlib CPU verification of the recorded evidence, with explicit raw-logit byte comparisons in addition to signatures. A second human/agent review is not claimed. Performance is allowed only after the target's bounded correctness qualification and replay pass. Preserve the first failure and cleanup errors separately; do not overwrite or rerun an actual attempt to select a favorable result.

Resource inventory recognizes the known local inference CLIs and records its scope. It cannot prove that arbitrary GUI apps have no GPU activity. No user applications are closed and no system memory settings are changed.

## Target experiment revisions

Build1 stopped after the accepted 32-token pass: the inherited decode eligibility used the prefill backend predicate, so compiled gates (and V19) could not dispatch on M5. This is preserved as a STOP, not a numerical failure or a qualified run.

`prepare_m5_decode.py` creates independent build2 exports and a new frozen harness. Both accepted15 and V19 exports receive the same narrowly scoped M5 Ultra/MLX 0.32.3 decode predicate; the existing NAX batched-prefill exclusion remains unchanged. The accepted build2 round additionally compares every observation to build1's original, uncompiled 32-step output. This proves the gate-compilation change against the original path in this bounded workload before candidate OFF/ON/reset-OFF. Engine constructors still default both experiments OFF. `check_decode_gate.py` exercises six CPU backend eligibility cases.

The build2 target fullmodel run and separate stdlib replay PASS: 128 raw logits, 16,384 state signatures, 18 fallbacks. Four process PIDs are absent after tool return. All resource records are retained under `.artifacts/m5-ultra-256gb/fullqual-build2`.

E2E build1 stopped during preimport audit before model load because NumPy was not imported before its identity check. `prepare_e2e_v2.py` copies the unchanged protocol into a new immutable build2 directory and adds that import. Original failed source and artifacts remain intact. E2E build2 completed all six fresh processes and 36 exact raw-ID requests. Pair throughput changes: +1.5223%, +2.6309%, -2.7957%; aggregate median throughput +2.4593%, delivery proxy +2.2653%. Verdict: **DO_NOT_ADOPT**, because not all three throughput pairs are positive. No repeated trial was used to replace the negative pair.

`frontier/` compares the experimental V19-on point to official oMLX 0.7.1.dev1, both on MLX/MLX-Metal 0.32.3, at C1/C2/C4 with three fresh pairs per C. These are additional exploratory reference points, not an override of the internal DO_NOT_ADOPT result. Only the native engine exposes raw IDs; official output is checked via usage and canonical text tokens. HTTP concurrency and actual GPU batch are distinct.
