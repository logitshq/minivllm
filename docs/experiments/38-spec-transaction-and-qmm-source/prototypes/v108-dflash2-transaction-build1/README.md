# V108: real prefix cache transaction, numerical only (GPU UNRUN)

This is an independent V44 candidate. The opt-in SpecRunner flag defaults to False. No old V44/V107 file, shader, native library or performance driver changes. The old target27B checkpoint stays fixed; a new explicit verify profile batches projections and norm/outproj, then runs the original V44 conv/SiLU and normreuse state helper once per token inside each GDN layer. This profile can differ from serial T1 projection math; all old38/44/89 greedy IDs remain hard gates.

The producer retains each actual returned S_t/conv_t and its B1 owner. Every literal history array is included as an output of the same mx.eval call before diagnostic hashing. A transaction commits the selected existing history entry and never replays chunk64/8 or freshly prefills a smaller shape. Attention histories copy the same invocation's projected K/V using separate array handles; for keep>0 the physical64 tail is zero, while keep0/abort restores the original physical buffer, including any old tail. Initial inputs and all history bytes are checked before/after commit.

The verify input count is bounded by remaining output and context before calling the drafter. With one output remaining it does not call the drafter. After EOS clipping, cache commit equals delivered count: the old anchor and all delivered drafts except the final delivered token. That final delivery remains the next uncomputed anchor, whether it is bonus or an EOS draft. Scheduler frontier advances by the same count; no hidden second trim is relied upon.

CPU checks use fake arrays and verify all prefixes, partial producer failure/abort, readonly source owners, tail rules, bounded delivery/EOS and AST restoration to V44. Tiny GPU qualification must run first: 21 actual cases (T1/3/5/8, every keep0..T), one real44 GDN producer and one attention cache, plus abort restoration. Scripted token-selection cases are labelled separately from model predictions.

The full numerical run requires an actual tiny receipt plus its root-supplied SHA and matching source pins. It loads the same target once, checks live serial A against all96 old IDs, then loads the actual local DFlash2 through its existing default affine4/g64 loader and checks B against all96 IDs. It checks all64 caches against the actual invocation prefix, all history/input bytes, 1349 target parameter name/object identity and raw hashes after each request, cleanup/reset, and extra budgets1/2/3/5. Every round records actual drafts/posterior, accepted/delivered counts, frontier and old44 helper call count. It preserves old128 cache-state SHA differences as diagnostics; it never calls those old132 strict equality or tolerances them into a PASS. Any ID/cache/readonly failure stops and preserves the partial receipt.

Raw DFlash2 source size is 3,848,817,896 bytes, not loaded memory. The S history bound for48layers×8tokens is1,207,959,552 bytes, plus23,592,960 bytes of conv history, initial references, attention copies, draft KV/features/activations and quantization transients. Full MLX budget34GiB/cache2GiB and tiny2GiB/cache64MiB are guard limits, not measured results. Root must own the GPU/resource/lifecycle window. No performance or adoption claim is prepared; future comparisons must allow qualified oMLX MTP/DFlash2 strong configurations, not only draftNone balanced.03.

CPU commands:

    python3 -B /private/tmp/minivllm-engine-optimization-20261014/v108-dflash2-transaction-build1/check_cpu.py
    python3 -B /private/tmp/minivllm-engine-optimization-20261014/v108-dflash2-transaction-build1/tiny.py --inert
    python3 -B /private/tmp/minivllm-engine-optimization-20261014/v108-dflash2-transaction-build1/fullqual.py --inert

Root tiny command (OUT must be new/absolute; no DYLD_INSERT_LIBRARIES):

    env MLX_MAX_OPS_PER_BUFFER=50 MLX_MAX_MB_PER_BUFFER=50 MLX_BFS_MAX_WIDTH=20 MLX_METAL_FAST_SYNCH=0 /Users/ryanli/code/minivllm/.venv/bin/python -B /private/tmp/minivllm-engine-optimization-20261014/v108-dflash2-transaction-build1/tiny.py --execute-gpu --out OUT

Root full command after tiny PASS and external release:

    env MLX_MAX_OPS_PER_BUFFER=50 MLX_MAX_MB_PER_BUFFER=50 MLX_BFS_MAX_WIDTH=20 MLX_METAL_FAST_SYNCH=0 /Users/ryanli/code/minivllm/.venv/bin/python -B /private/tmp/minivllm-engine-optimization-20261014/v108-dflash2-transaction-build1/fullqual.py --execute-gpu --out OUT --tiny-receipt ACTUAL_TINY_RECEIPT --tiny-sha256 ACTUAL_TINY_SHA256

The source pins bind every candidate source, drivers/proof/protocol, three old receipts, target/draft metadata, original44 helpers and runtime identities. Safetensors files have header-only identities and length guards; CPU preparation reads no tensor body. Copied MLX-derived source retains the parent's LICENSE-MLX-MIT. Actual imports must resolve inside this candidate and match pins.
