# GLM offline byte repack — actual preparation

Actual metadata CLI returned 0 on 2026-10-19 (tool d8b38f). Plan SHA256 `323ff8da5d2faf6bb774b1c03f3a52f49198cbb9898f21c91ac25aa546a4c50a`; candidate SHA256 `fd3c30de034a2eba2e3002eb61df08737b07652bd86a9edaeaba75fd35d99d5b`. The plan has 62 original shards, 203,976,457,080 payload bytes and 378 backbone expert stacks; 113,446 tensors become 4,960. It reads metadata only, not the original payload.

Independent source and plan review found no blocker after the actual MTP wrapper and per-FD Darwin F_NOCACHE fixes. Fifteen tiny guarded CPU checks passed, including real unaligned nocache IO and source-pinned fake-byte sanitizer/quantization aliases. All seven pinned loader sources and small metadata references matched at review. This does not prove actual EnginePool compatibility or memory fit.

The next explicit operation is the recorded execution-wrapper.py into fresh `.artifacts/models/GLM-5.3-Flash-4bit-stacked-v1`, after this plan is committed. It first rehashes every original shard, copies only raw bytes using bounded payload buffers, independently rereads output, and preserves partial failures. Available disk at root review was 888.824 GiB, versus roughly 190 GiB output plus the mandatory 64 GiB reserve. Original checkpoint and restored oMLX service are not changed. Output must have PASS_BYTES_ONLY complete.json before any future loader trial. F_NOCACHE reduces cache pollution; it is not a no-swap guarantee.

No actual payload copy or GPU load had run at preparation time. Wrapper logs live in `.artifacts/m5-ultra-256gb/glm-expert-repack-execution-v1`; completed evidence will be appended separately.
