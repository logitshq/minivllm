V102 explicit CAP64 direct FIFO service/engine adapter. The default engine and candidate44 math remain unchanged; no92/95 profile is used. Frozen64→56→41 parents are imported by exact identity and unchanged. Inherit the qualified FIFO begin/run/drop/budget/depth2/two-slot/4B prediction behavior; no perstep state restore or future CPU token feedback.

Required startup sequence for BOTH qualification and future service entry:
1. Import this stdlib-only module; call `configure_physical_cap64(enabled=True)` BEFORE engine construction and ANY token. DefaultFalse changes nothing. It changes only AttnCache.step/DecodeBatch.step64 and returns original_steps; no live cache migration.
2. Construct old44 LLMEngine, allocator2GiB/C1max_seqs1/noDraft, gates+convnormreuseON. Ordinary warm remains ordinary engine path and is already physical64.
3. Explicitly `install(engine, lib101, lib101_SHA, diagnostic=bool)` while idle. Only this new library is DYLD-injected; no dual/native library mix. Installation does not imply numerical qualification.

Adapter requires startup class steps64; self.capacity64. Its build verifies true KV[1,4,64,256] BF16, then copies41.build with only final expectedcapacity literal replaced by self.capacity, and appends the original56 pipeline-enable checks. Inverting those narrow changes restores41 build AST. Model1349 owners and48S/83 staged counts are enforced by inherited build; no parameter arrays are added.

CPU checks (`python3 -B check_cpu.py`) prove strictbool/defaultOFF inert imports, inverse build AST, unchanged inherited run/begin/drop and metadata-only allocation setup. No MLX/NumPy/dylib/model/GPU execution. Actual tiny101 and full103 diagnostics/production must separately qualify before service/performance use. Source module cannot detect arbitrary external past token execution; its supported API contract is the startup-before-engine sequence above, which103 and the later service entry must follow visibly in source.
