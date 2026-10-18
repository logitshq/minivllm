# oMLX D512: actual C1/C8 protocol qualification PASS

Two fresh official oMLX servers completed the original D512 workload: C1 eight requests and C8 sixteen requests, all with normal EOS and a 512-token cap. Independent route/count/text, resource and restoration audits all PASS. This instrumented run supplies **no performance point, cross-framework full-state equality or output-quality comparison**.

| Concurrency | Requests | Committed output tokens | Peak HTTP outstanding | Observed GPU batch peak | EOS completions |
| --- | ---: | ---: | ---: | ---: | ---: |
| C1 | 8 | 4,096 | 1 | 1 | 0 |
| C8 | 16 | 8,192 | 8 | 8 | 0 |

C8 records 881 batch-8 response events; its other batch sizes are preserved in the independent receipt. This establishes real multirow processing for this run. It does not establish the maximum supported concurrency. All 24 requests reached cap512, so early-EOS behavior is not exercised by these actual requests.

## Contract and evidence

The target is `mlx-community/Qwen3.8-27B-nvfp4@5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6`, using the same eight D coding fixtures as the native and Lithos D512 measurements. The observed official route is `VLMBatchedEngine` / `AsyncEngineCore`; prefix/hot/disk caches, MTP and speculative decoding are disabled. The actual runtime is oMLX0.7.1.dev1, MLX0.32.3, mlx-vlm0.7.7, mlx-lm0.31.4.dev132+g94cdcae13, Python3.13.16.

[Independent qualification](independent-qualification.json) reopens the raw requests and observer streams. It binds full prompt IDs, actual client/response/engine request IDs, sampling and loaded route, raw committed IDs, text, official usage, wire SHA/bytes, actual batch sizes and POST-to-DONE overlap. Official `completion_tokens` already excludes terminal EOS; it must not receive Lithos's usage adjustment. The request seal binds the actual request set and observer prefix; it is not generator EOF or a process-close record.

Both directly owned servers were terminated by their owner after their request seals, exited with the explained SIGTERM return code -15, and were strictly waited/reaped with absence/listener evidence. Closed stdout/stderr are retained and hash-checked; this is not semantic verification of every lifecycle log message. Both observer-close and generator-EOF proof flags remain false, explicitly.

Four supplied inventories were fully hashed by the owner before and after the run and matched: 5,245 oMLX runtime files, eight observed official source files, 4,309 outer runtime files, and 13 target files. The independent audit rehashed outer-runtime and tokenizer payloads, checked the recorded model inventories, and did not reread model weights. This inventory is not a proof of recursive dependency closure.

## Resource and restoration closure

[Independent resources](resource-summary.json): 41 recorded samples pass the raw pressure/AC/thermal/sustained-swap gates; pressure1 and swap growth0 MiB. Swapouts/Pageouts/Compressions growth0, Swapins+8 and Decompressions+373. The retained process inventory cannot rule out every unrecognized runtime or GUI GPU activity.

Owner93524 and direct servers93587/94094 are reaped. Original oMLX PID92152 was restored as PID94592, with the original model/configuration and fresh MTP/resident-PLE evidence. [Independent restoration](restoration-audit.json) PASS; 19 physical samples were retained. Warm KV and exact hidden runtime state were not restored. Execution closed at 2026-10-17 23:58:35 UTC with stop/owner/recovery return codes0.

## Archive and next measurement

[Evidence archive](evidence.tar.xz) has 414 regular members, 123,826,343 logical bytes and 3,267,180 compressed bytes; SHA256 `5d0f678837421c73062a83eb8ba43c17bdb8a65f491d7767a3d426483670365b`. [Verification](archive-verification.json) checks every member, original file and tree membership. Only the two exact model-directory links are omitted as filesystem objects; their targets/materialization receipts are preserved in [model-links.json](model-links.json), without opening or archiving weights. [Manifest](manifest.json) retains original absolute paths. Raw owner, journal, execution, independent analysis, preparation, preregistration and 47 registered source pins are included.

Owner-plan SHA `e5294522149dfccccab4e26189280d77ddb6f3eb0b54fe78034646c59a71d4ef`; sources and actual PREPARED journal were committed in `bccad9c` before the run. The driver reused the user's authorized cold-stop/restore contract. Prior CPU preparation is [57 tests PASS](../omlx-outer-prepared-cpu-v1/README.md).

Next, bind this completed qualification to observer-OFF official runs with the original warm8/formal64 request protocol and fresh-process repeats. Count committed tokens over the common first-POST to last-DONE window and report full POST-to-DONE p50/p95, with resource/restoration checks. Do not put these observed qualification timings into the Pareto graph. The existing native/Lithos D512 points remain unchanged until that formal run is complete.
