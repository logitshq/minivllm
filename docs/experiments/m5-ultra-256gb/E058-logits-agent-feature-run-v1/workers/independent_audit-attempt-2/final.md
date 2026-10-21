No MLX anywhere (grep exit 1 = no match). Everything is verified:

- `audit.py`: NumPy used only for vectorized uint16/uint32 finite bit-mask scans; parser, dtype/shape validation, byte comparisons remain stdlib and independent. `_load_member` now hashes the exact bytes it parses (single read, no changed-file gap). Manifest rejects extra fields (only `schema`/`pairs`). No MLX, no producer imports, inert on import.
- `test_audit.py`: new `ScanEquivalenceTests` (exhaustive 65536 BF16 + representative/random uint32 vs stdlib scalar reference), comparator tests restored to their own class, manifest extra-field/missing-field rejection tests added.

Done. Summary below.

---

## Summary

**Files:**
- `audit.py` — independent E058 raw safetensors byte auditor (stdlib parser/comparator; NumPy only for vectorized nonfinite scans).
- `test_audit.py` — 30 CPU tests.

**Changes this round (root-review follow-up):**
- Vectorized nonfinite scanning: `_bf16_nonfinite_numpy` (`<u2`, mask `0x7F80`) and `_f32_nonfinite_numpy` (`<u4`, mask `0x7F800000`) via `_np.frombuffer`; scalar stdlib versions kept as reference fallback when NumPy is absent. Parser, dtype/shape validation, and byte-exact compares untouched and producer-free.
- Equivalence tests: all 65536 BF16 bit patterns (bulk + per-word) and 15 representative + 4096 seeded-random uint32 patterns compared against the stdlib scalar reference; finite-only blobs asserted clean on both paths (`test_audit.py:160` `ScanEquivalenceTests`).
- `_load_member` now single-reads and hashes the exact bytes parsed (`audit.py:275`), closing the hash-then-parse changed-file gap while keeping path/symlink checks.
- Manifest rejects extra fields — accepts only `schema`/`pairs` (`audit.py:301`), with tests for extra and missing fields.

**Tests run (real exit status, no pipes):**
- `/Users/ryanli/code/minivllm/.venv/bin/python -m unittest test_audit -v` → **30 tests, OK (exit 0)**.
- Venv inert-import probe → imports cleanly, NumPy backend active, `NUM_LAYERS=64`; no MLX references anywhere.

**Contract preserved:** 24 exact labels `T{T}/k{k}/{phase}` (T∈4,8; k=1..T; selected/continuation), exactly 130 arrays, fixed BF16/F32 shapes per phase, signed-zero byte diffs fail, `audit()` raises `ValueError` on any invalid evidence and returns PASS only after all 24 pairs.

**Limitations:**
- The 64-layer/130-array shapes are validated via contract tests on tiny fixtures (overrides active during assertions); real ~12 GiB GPU-produced files will be audited later by Root.
- NumPy backend is exercised under the repo venv; the stdlib scalar path is covered only as the reference within equivalence tests (it dispatches live only on NumPy-less hosts).
- `\__main__` CLI (`--root --manifest --out`) is a convenience wrapper; covered by `test_cli_convenience` but not yet run against real evidence.
