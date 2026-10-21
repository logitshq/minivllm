Everything is complete and verified. All 26 tests pass, stdlib-only imports confirmed, module is inert on import, and the CLI works.

**Files Written**
- `audit.py` — stdlib-only independent raw safetensors byte auditor. Exports `audit(root, manifest) -> report` (manifest may be a path or dict), plus factored helpers `parse_safetensors`, `compare_arrays`, `expected_labels`, `parse_label`, `expected_arrays`, and a `main()` CLI (`--root --manifest [--out]`). Raises `ValueError` on any invalid evidence; `status: PASS` only after all 24 pairs verify. Imports: `argparse, hashlib, json, os, stat, struct, sys`. No repo helpers imported; inert on import.
- `test_audit.py` — 26 CPU tests across 4 suites (Parser, Comparator, Contract, Audit) using tiny fixtures plus exact full-scale contract assertions.

**Contract Enforced**
- Schema `E058-feature-pairs-v1`; exactly the 24 labels `T{T}/k{k}/{phase}` for T∈(4,8), k=1..T, phase∈(selected,continuation), in order.
- Paths: relative-only, no `..`/abs/backslash, per-prefix `lstat` symlink rejection, regular files, realpath confined to root; SHA256 exact 64-hex match.
- Parser: 8-byte LE length + UTF-8 JSON, duplicate-key rejection at any depth, metadata keys exactly `{dtype,shape,data_offsets}`, BF16/F32 only, byte-length = numel×width, offsets must tile payload exactly (no overlap/gap/shortfall), nonfinite BF16/F32 rejected via bit patterns.
- Arrays: exactly 130 names (logits, features, `state/{i}/K,V` for i%4==3 else `state/{i}/conv,S`); logits BF16[k,248320], features BF16[k,25600] (continuation rows=1), KV BF16[1,4,1536,256], conv BF16[1,3,10240], S F32[1,48,128,128] — cross-checked against `bench/m5_ultra_256gb/rowwise_mlp_shared_features/{candidate.py,round.py}`.
- Exact byte equality per array: signed-zero difference (0x0000 vs 0x8000) fails.

**Tests Run**
- `python3 -m unittest test_audit -v` → Ran 26 tests, OK (0.08s). Covers every required mutation: same-sized payload flip, dtype/shape mismatch, unsafe paths (abs/`..`/missing/dir/symlink), SHA malformed/mismatch, NaN/Inf bit patterns, duplicate header keys, incomplete/reordered/duplicated labels, signed-zero at both comparator and end-to-end audit level, extra metadata keys, offsets not covering payload end, and a PASS report count check (`python3 -m unittest test_audit` → OK).
- Import-time check: `python3 -c "import audit"` succeeds with no side effects; `python3 audit.py --help` works.

**Limitations**
- Full-scale fixtures (248320-col logits, 130 arrays/pair) are not exercised end-to-end on CPU due to size; tests use patched-constant tiny fixtures via the `AuditTree` context, while `ContractTests` assert the exact full-scale 130-name/shape/dtype set derived from the real constants.
- Real raw files don't exist yet, so currently only CPU test evidence exists; root can run `python3 audit.py --root <dir> --manifest <m.json> [--out report.json]` once artifacts are produced. Any contract violation raises `ValueError` (nonzero exit) rather than emitting a FAIL report, per the spec.
