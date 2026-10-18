# GLM cold-cycle CPU preparation: retained failure and repaired fixture

The second fake-only CPU run passed all **33 tests**. The first run remains recorded as **32 PASS + 1 ERROR**; it is not relabelled or discarded. No GLM model was loaded, no service was changed, and these results do not qualify numerical correctness, physical capacity or performance.

| Attempt | Whole modules | Result | Original result SHA256 |
|---|---:|---|---|
| v1 | 4 / 33 tests | 32 PASS, 1 fixture ERROR | `89db7a2348d4dd5fdc751cb190e420f5fd2101e84aaa4fb874d581387b4dbd6d` |
| v2 | 4 / 33 tests | 33 PASS, no errors/failures/skips | `53637bfc4114386decca0aafa2eee8837f051aecd12584eb2568fda8eab41258` |

Both attempts kept source, formal-closure and GDN queue evidence unchanged. Both reported zero forbidden operations and zero backend imports. The v2 command returned 0 (tool chunk `de01da`). Production controller, owner, both auditors, input builder, registrar and execution driver were byte-identical between attempts.

## What failed and what changed

The early `NO_CHILD_STARTED` fixture named an external `owner-plan.json` but did not create it. The real controller correctly requires that immutable registered plan even when an owner has failed before saving its local copy. The fixture patched the module attribute `registered_trials`; the reused `reaped()` code has its own copied dependency dictionary, so it still called the real function and rejected the missing external file.

Only that fixture changed. It now creates the external plan, binds its SHA into the STOPPED snapshot and attempt, and exercises the real registration path. It still asserts the owner-local copy is absent. A negative check modifies the external plan and requires rejection. Expensive input validation remains fake; no production guard was weakened. See `fixture-fix.patch`.

The reusable lesson is to distinguish **registered input** from **runtime copies**, and to check the actual dependency dictionary when testing a function created with `FunctionType`. Mocking an unrelated module attribute does not establish the intended call path.

## Evidence and verification

`raw-evidence.tar.xz` retains the complete v1 CPU tree, its parent queue, the complete v2 CPU tree, both source-preparation manifests, original/repaired test sources via the captured source trees, logs, individual outcomes, runners and all recorded prerequisite bytes. The original local evidence directories remain unchanged. There are **678 files / 11,854,917 uncompressed bytes**; the compressed archive is **1,351,148 bytes**.

Archive SHA256: `b99b1ede1511196f76c94d61a9aaf6ebcc9f9dc7f50f38623c6fce4f85195ca8`.

Run the standalone stdlib verifier; it streams the tar without extracting or executing archived code, rejects unsafe member names and links, checks every member SHA/size and the two recorded outcomes, and compares unchanged production sources:

```sh
python3 -I -B docs/experiments/m5-ultra-256gb/glm-cycle-prepared-cpu-v2/verify.py
```

`archive-verification.json` records that verification. `manifest.json` records every original path, byte length and SHA. `summary.json` states the supported scope. CPU preparation remains separate from future explicit registration, service removal, actual direct-child GLM loading and independent restored-cycle audits.
