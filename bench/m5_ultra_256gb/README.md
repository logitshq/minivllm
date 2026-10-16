# M5 Ultra / 256 GiB qualification

Local migration of `tutorial-17-handoff-v1` (`559ced1`) for the actual Apple M5 Ultra (80 GPU cores, applegpu_g17d). Accepted engine: f976d6a; candidate engine: b158c0f. Engine sources and all numerical assertions remain unchanged. V19 remains default OFF.

`worker.py` retains the historical full-model numerical body and `fallback.py` is byte-identical. `test_migration.py` checks these boundaries. Changes are confined to machine identity, local runtime provenance, paths, desktop power detection, and ownership receipts. M5 retains the engine's existing NAX prefill fallback; this migration does not turn on the M4-only batched projection path. The original four MLX controls are used; MLX_ENABLE_TF32 is unset. Python is 3.13.16, differing from the source machine's 3.13.14; MLX/MLX-Metal remain 0.32.3.

The package in `docs/handoffs/256gb-mac` is unpacked into `.artifacts/m5-ultra-256gb/source`. Each new domain is frozen with `freeze.py`. `MINIVLLM_EXPERIMENT_DOMAIN` selects the domain; its frozen JSON pins the interpreter, installed MLX Python/native/header/Metal files, NumPy/tokenizer runtime files, all model files, exported engines, workload and harness. Hugging Face LFS hashes are verified against API metadata, not inferred from a resolved local cache filename.

Before a run, execute the CPU suite with the selected domain and save stdout/stderr. The suite includes byte/signed-zero differences, final-chunk nonfinite rejection, source identity, dispatch delegation, changed GO fields, pressure/swap stops, desktop power, and one-shot attempt consumption. `probe.py` runs a small BF16/NVFP4/Metal check separately from model qualification. Synthetic CPU fixtures are never model evidence.

`owner.py --frozen-sha256 HASH` consumes a single local attempt. The user requested these experiments; this local owner creates a fresh GO and never reuses M4 authorization. The driver loads accepted then candidate in separate, serial subprocesses. Four rounds of 32 outputs check 128 raw logits files, 16,384 state signatures, all cache boundaries, parameter identities, dispatch paths and 18 fallbacks. Processes are waited/reaped. The outer owner's exact PID absence must be checked after the execution tool returns. Process exit is the unload boundary; in-process allocator release is not asserted.

`replay.py OUTPUT` is a separate stdlib CPU verification of the recorded evidence, with explicit raw-logit byte comparisons in addition to signatures. A second human/agent review is not claimed. Performance is allowed only after the target's bounded correctness qualification and replay pass. Preserve the first failure and cleanup errors separately; do not overwrite or rerun an actual attempt to select a favorable result.

Resource inventory recognizes the known local inference CLIs and records its scope. It cannot prove that arbitrary GUI apps have no GPU activity. No user applications are closed and no system memory settings are changed.
