# E049 offline prefix audit

`audit.py` reuses the frozen independent HTTP and route auditors without changing
any per-step scheduler, metadata, complete-array, dtype, finite, or bit-equality
coverage check. Only the full-drain finish becomes the registered warm49 prefix
boundary. It imports neither the numerical qualifier nor MLX.

The actual saved-record audit passed: attempt2, exit 0, 0.672 seconds. It
rejoined both original eight-request raw SSE groups with the original trace,
verified the full unchanged 16-job / 1,030-step recording, then independently
checked all 49 observed eager↔graph steps against that recording.

- 380 delivered IDs; D06 has 49 IDs, including token 24959 at index 46.
- Actual decode B: 0, 2, 4, 6 each once; B8 for 45 steps.
- 65,005 original per-array bit-equality receipts and 130,010 finite receipts:
  103 full-vocabulary logits, 64,768 state arrays, 134 ahead-token arrays.
- 44 consumed-ahead events and three batch refills; no retirement in this prefix.
- 49 graph route calls: bound 1023 ×45, bound 1536 ×4. The captured route prefix
  ends after 28,556 bytes / 50 lines; its next unconsumed call belongs to step50.
- Zero completed requests, eight active, eight future: `INCOMPLETE_BY_DESIGN`.

Successful raw tensor payloads were not retained. The audit verifies complete
source-bound producer comparison receipts; it does **not** reread tensor bytes or
recompute GPU arithmetic. Recorded HTTP proves matching prefix IDs, not recorded
full states. No full-history, performance, resource, or restoration grant follows.
The saved 53,161,955,496-byte MLX peak is an allocator receipt, not physical memory.

Evidence under `.artifacts/m5-ultra-256gb/`:

- `c8-prefix-v1-independent-audit-attempt1`: retained audit FAIL; the new auditor
  initially compared the capture's entire shell environment with the replay's
  four MLX variables. Comparing the exact MLX subset corrected that audit bug.
- `c8-prefix-v1-independent-audit-attempt2`: actual PASS, original stdout/stderr,
  command, return code, source snapshot, and detailed result.
- `c8-prefix-audit-cpu-v1`: retained test error; the first-step fixture uses
  per-request `keys`/`values`, so the test's `/K` selector found no array.
- `c8-prefix-audit-cpu-v2`: seven tests pass, including missing recurrent-state
  receipt, either nonfinite arm, unequal bits, changed output, changed recording,
  invented completion, extended prefix, and malformed JSON rejection.

Both audit and test execution blocked MLX imports, subprocess launches and
process signals inside the child. No GPU retry or frozen experiment edit occurred.
