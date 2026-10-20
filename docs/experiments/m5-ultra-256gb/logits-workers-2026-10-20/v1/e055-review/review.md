I have everything needed. One more quick read of the v1 prereg's latency/repeatability note is not necessary — the decision gate is confirmed. Writing the review.

**Verdict**

The v2 audit wrapper is a narrow, hash-pinned artifact resolver, not a semantic change to v1 math — but the three supplied C8 ON/OFF pairs **fail the preregistered 3% median-throughput gate**. Throughput gain is real but insufficient; no adoption or speed-gate claim passes.

**v2 Wrapper Review**
- `bench/m5_ultra_256gb/c8_batched_formal_analysis_v2/audit.py:13-15` freezes the v1 reducer by SHA (`7b054f…`) and refuses to load if it changes — so all paths, pairing math, medians, and gate arithmetic are literally v1's code (`c8_batched_formal_v1/audit.py:46-70`), confirmed against the preregistration `prereg-v1/plan.json:141-148`.
- The only behavioral override: `audit.py:18-26` allows the one literal `path+sha256` tokenizer reference to resolve to the exact hash-verified canonical HF blob (absolute, non-symlink target, size 19,989,325 bytes, matching SHA). Any other ref falls through to `value.pinned` (v1's strict resolver) unchanged (`:20`).
- Post-audit, `audit.py:49-56` rechecks the symlink→blob binding and flips the result to `FAIL` with `timing_valid=False` if it drifted — a sound defense against TOCTOU on the mutable HF cache.

**Concrete concerns**
1. `analysis_v2/audit.py:22-25` verifies the final path component is exactly `TOKENIZER` and resolves to `TARGET`, but does not inspect intermediate directory components of `TOKENIZER`; a hostile/mistaken rename of a parent directory would change `resolve()` and be caught only because `resolve(strict=True)` must equal the hardcoded absolute `TARGET`. Safe in practice; the real guarantee comes from the absolute-path equality, but the symlink invariant is shallower than the README wording suggests.
2. `audit.py:9-11` hardcodes `/Users/ryanli/.cache/...` absolute paths. Fine for this pinned artifact review, but the auditor is non-portable; any machine/user change silently fails rather than adapting. Deliberate, but should be documented as a constraint, since v1's failure (`.artifacts/…/c8-batched-formal-v1-independent.json:13-16`) shows the resolver is machine-state-sensitive.
3. Scope limitation of the reducer (a contract issue, not a v2 defect): `c8_batched_formal_v1/audit.py:63-66` has **no all-pair p95 gate** — only the median p95 ≤ 1.02. One pair could regress p95 sharply and still pass if the other two medians hold. This matches the preregistration, but flag it before treating `speed_gate_passed` as a per-pair guarantee.
4. The v1 README (`README.md:53`) is correct that `PASS_METRICS_AND_IDS` ≠ speed or Pareto; any v2 PASS inherits that limitation. The v2 failure cause (`independent.json:15`, "changed artifact … tokenizer.json") was environmental (HF symlink byte-equivalence), not numerical — v2's fix addresses exactly that, but until an independently inspected v2 output JSON exists, no v2 PASS can be claimed.

**Measured pairs interpretation**
Computed from the three supplied ON/OFF aggregate-output-tokens/sec values:

| Pair | ON | OFF | Ratio | Gain |
|---|---|---|---|---|
| 1 | 173.895 | 170.691 | 1.01877 | +1.877% |
| 2 | 174.067 | 170.466 | 1.02112 | +2.112% |
| 3 | 174.265 | 170.651 | 1.02118 | +2.118% |

- Median throughput ratio = **1.02112 (+2.11%)**, below the preregistered **≥ 1.03** (`plan.json:141`, `README.md:45`, `audit.py:17`). Throughput gate fails.
- Off-arm spread across the three OFF rates = **0.132%** of the median — well under the 2% variability cap, so the OFF baseline is stable; the miss is a genuine shortfall, not noise.
- All `throughput_ratio > 1` holds (3/3), so that sub-gate passes.

**What can and cannot be claimed**
- Can claim (measured): in this C8 six-process A/B, C8-ON showed a small, stable, reproducible throughput improvement (~1.9–2.1% per pair, median ≈2.11%), with OFF-arm variability under 0.2%.
- Cannot claim:
  - Speed-gate pass or adoption — the ≥3% median criterion is not met, and failing the win gate "rejects adoption" per `README.md:45`.
  - Any p50/p95 verdict — no latency ratios were supplied; `max_median_p50_ratio ≤ 0.97` and all-pair `p50 < 1` cannot be evaluated.
  - Correctness/token-identity qualification — that requires the independent v2 audit pass plus external lifecycle/resource review (`README.md:53`, `audit.py:136-143`); none is presented here.
  - Any oMLX/Lithos or Pareto frontier statement — these are same-workload pairs only, and a local throughput delta is not a full-engine Pareto gain.
  - GLM-5.3-Flash relevance — these pairs are the Qwen3.8-27B NVFP4 C8 path; nothing here generalizes across configs.

**Meaningful CPU regression cases** (hermetic, temp-dir copies only — never touching the live HF cache):
1. Happy path: exact symlink→canonical blob, matching size+SHA → resolves to `TARGET`.
2. Retargeted symlink to a different equal-sized valid tokenizer → must raise (`resolve != TARGET`).
3. Plain regular file at `TOKENIZER` (symlink replaced) → must raise (`is_symlink()` false).
4. `TARGET` itself a symlink → must raise (`not TARGET.is_symlink()`).
5. Byte corruption at same size (flip one byte, keep 19,989,325) → SHA mismatch must raise.
6. Truncated/extended blob (size ≠ 19,989,325) → must raise on size before hashing.
7. Ref dict with extra/missing key, wrong path, or wrong SHA → `audit.py:19-21` must raise before falling back.
8. Non-tokenizer `path+sha256` ref (e.g., `output_ids`) → must delegate to v1 strict resolver, hitting v1's failure path if absent.
9. Corrupted `c8_batched_formal_v1/audit.py` (any byte change) → `PREVIOUS_SHA` check at `audit.py:13` must refuse to load.
10. Drift between audit and recheck: blob swapped after `_original_audit` returns → `:49-56` flips result to FAIL with `timing_valid=False`; assert the error fields.

Cases 2–8 cover the actual workaround behavior; 1, 9, 10 are the regression anchors.

**Next experiment**

Run one hermetic CPU-only invocation of the v2 auditor against the already-closed six-process screen and registered plan (`analysis_v2/audit.py --screen <closed-v1-screen> --plan prereg-v1/plan.json --plan-sha256 <registered> --out <fresh>`), with noMLX/no-subprocess hook, and publish the resulting JSON. This resolves the single open integrity question — whether metrics + IDs independently validate — under unchanged v1 math, without touching any GPU or code; only after that PASS should the retained `points` be consulted for the full p50/p95 gate verdict on the measured pairs.
