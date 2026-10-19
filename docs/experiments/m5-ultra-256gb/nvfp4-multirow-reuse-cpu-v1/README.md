# NVFP4 multirow reuse — five CPU mechanism checks

Actual first run: **PASS 5/5**, no skips/errors/forbidden operations; executable
source and test-time README hashes were unchanged across the run. Tool `ce543f`
returned exit 0 after the separate C1 profiling driver completed and restored
its service. Harness elapsed time was 0.00398849998600781 s; this is CPU test
runtime, not an inference metric.

`actual/` is an exact copied receipt directory, including command argv/cwd,
raw stdout/stderr, result, source-before/source-after SHA maps, and frozen
candidate/test/README/guard source. `manifest.json` records each original path,
byte count and SHA256; `verification.json` records independent archive reread
and source/copy comparison. No failed attempt preceded this successful run.

Tests cover inert OFF/import behavior, version/device rejection, all three
widths and explicit MLX contiguous-staging option, shape/dtype rejection, and
error propagation without retry. An injected fake backend performs no Metal
work. The guard blocks backend imports, mmap, subprocess/process signals and
network. **No actual MLX projection, numerical equality, compiler behavior,
register/spill cost, full-model verifier, HTTP or Pareto point is qualified.**

The frozen test-time README says tests had not run yet; it is retained without
rewriting history. The live experiment README subsequently records this PASS.
Candidate SHA: `ee9c5eef0e1282125310ee6661be1ec5ce64dc93a82add48a6d6c17f8958816f`.
Test SHA: `f2cc9c296476fdc12185f23c0f005234e5e191843164d355dde5f4c4d7416058`.
