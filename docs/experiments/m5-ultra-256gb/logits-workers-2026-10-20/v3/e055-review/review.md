**Corrected E055 Review: v2 Audit Complete and Contextualized**

**Retractions and Status Corrections**

1. **Retraction: Audit Execution Status.** I previously claimed the independent audit remained unrun; that was incorrect. The v2 audit is complete and documented in `independent-v2.json` (referenced in `README.md:13`).

2. **Retraction: Fallback Contract Mischaracterization.** I previously characterized the failure mode as "wrong path must raise before falling back." This misrepresents the v2 contract. `audit.py:19-25` defines a **whitelist** behavior: only the exact literal original tokenizer reference (path + SHA-256) resolves to the canonical blob; any deviation (wrong hash, wrong path, extra fields, mutated bytes, retargeted symlink) raises `ValueError` immediately. The `fallback` parameter (`audit.py:20`) is only invoked for non-matching paths, not for validation failures of the pinned reference. The 12 regression tests (`test_cpu.py:28-67`) verify this behavior: `test_reference_hash_cannot_change`, `test_extra_reference_fields_rejected`, and `test_retarget_to_identical_bytes_rejected` all confirm immediate exceptions without fallback invocation.

3. **Clarification: OFF Spread Formula.** The OFF throughput spread is calculated as `(max - min) / median`, or equivalently `max/min - 1` when normalized to the median. It is **not** range/median (which would be `(max-min)/median` without the normalization implied by the 2% gate). `audit.py:62` (v1) implements `spread=(max(rates)-min(rates))/statistics.median(rates)`, which is algebraically equivalent to `max/min - 1` when expressed as a fraction of the minimum, but as written uses median normalization and correctly passed the `<=0.02` gate (`README.md:11`).

**Verified Improvements vs. Population Claims**

The paired data demonstrates **observed improvement in exactly 3 pairs** at C8 concurrency:
- Pair 1: +2.1124% median throughput (170.69 → 173.90 tokens/s)
- Pairs 2 and 3: consistent positive deltas (`README.md:7-9`)

However, this is **not a proven population-level speedup**. The v1 audit correctly scopes this as "Three adjacent process pairs; no pooled-request statistical claim" (`c8_batched_formal_v1/audit.py:72`). The pre-registered gates (`DECISION` at `audit.py:17-18`) required:
- Throughput ratio ≥ 1.03 (actual: ~1.021, **failed**)
- p50 ratio ≤ 0.97 (actual: ~0.9797, **failed**)
- p95 ratio ≤ 1.02 (actual: ~0.980, passed but insufficient)

Thus, while `speed_gate_passed` is mathematically computed, the trial remains a **diagnostic observation below adoption threshold**, not a validated optimization.

**Remaining Verified Defects and Coverage Gaps**

*Defects:*
- **None blocking** in the v2 audit logic. The symlink resolution correctly validates the exact blob hash, size, and non-symlink status of the target (`audit.py:22-25`).

*Missing Coverage (Actionable):*
1. **Cross-Schedule Non-Determinism Unresolved.** `cross_schedule_token_equal=False` (`README.md:15`) indicates different greedy trajectories across schedules. While `numerical_error_inferred=False` (`audit.py:70`) correctly avoids false arithmetic claims, there is no root-cause analysis of why schedules diverge. This lacks coverage for scheduler-level determinism validation.
2. **Resource Qualification External.** `owner_resources_restore_qualified=False` and `performance_qualified=False` (`audit.py:86`) remain unclosed. The "285 physical/metadata samples" (`README.md:15`) are referenced but not integrated into this audit’s pass/fail logic.
3. **Numerical Prerequisite Scope.** The same-schedule prerequisite covers only the "E053 captured history" (`audit.py:137`), not all formal schedules. This is a scope limitation, not a defect, but blocks universal numerical claims.

**Is Another Audit Rerun Justified?**

**No.** The v2 audit is internally consistent, hash-bound, and mathematically unchanged from v1 except for the narrowly scoped tokenizer resolver. All six processes closed, 432 request UUIDs are unique, and original evidence is hash-pinned (`audit.py:98,135`).

Unless new evidence emerges (e.g., a fourth independent repeat at C8, or a concurrency sweep beyond C8 to establish population-level significance), re-executing the same six-process audit would reproduce identical metrics and gates. The appropriate next step is not re-audit, but rather:
- A **population-level study** (more than 3 pairs, varying concurrency) to test whether the +2.1% trend achieves statistical significance and meets the 3% threshold, or
- **Root-cause analysis** of the cross-schedule greedy divergence (`README.md:15`) to determine if scheduler non-determinism masks true numerical equivalence.

The current evidence is sufficient to classify E055 as a **diagnostic gain below adoption gate**, and no further audit cycles are required to support that classification.
