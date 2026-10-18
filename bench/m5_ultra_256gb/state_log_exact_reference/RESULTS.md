# First CPU attempt

The first explicit run passed on 2026-10-17. It retained the exact commands,
stdout/stderr, source copies and immutable JSON under
`.artifacts/m5-ultra-256gb/state-log-exact-cpu-v1/`. No test or experiment retry
was needed; no backend/model/service was invoked.

- 8 mechanism and negative CPU tests passed, no failures/errors/skips.
- 24 toy cases: three banks × 64/128 steps × merge 1/2/4/8.
- 2,304 updates; 4,608 full-state/output comparisons plus 120 lifecycle
  comparisons; 516,672 bytes compared exactly.
- All four merge intervals produce identical trajectory hashes for each of the
  six bank/length combinations.
- Algebraically combining the two recorded delta factors gives FP32 1, while
  the specified sequential recurrence gives FP32 0. This is an expected negative
  result, retained in `toy-result.json`, not a failed test hidden by tolerance.

For one 64-update 8×6 case, including the midstream reset segmentation but
excluding reset's own writes, the **analytical persistent-write counts** are:

| Merge interval | Dense oracle words | Deferred base + log words | Extra words if final dense flush is required |
|---|---:|---:|---:|
| 1 | 3,072 | 3,072 | 0 |
| 2 | 3,072 | 2,016 | 48 |
| 4 | 3,072 | 1,504 | 48 |
| 8 | 3,072 | 1,248 | 48 |

These are not Python allocation counts, hardware counters or speed estimates.
The deferred implementation does more reconstruction/replay work. Native Metal
FMA/exp/reduction/normalization equivalence, real traffic, full-model state,
HTTP behavior and performance remain unqualified.

Reproduction: the retained `runner.py --run` records this fixed attempt to a
fresh directory and refuses overwrite. For subsequent experiments use the
README's `toy.py --run --out NEW_JSON` command and retain a new attempt directory.

Frozen first-run sources:

- `toy.py`: `15f4bc23c0d413da8adeab857d8ed4c757c8c877aedcf728f402ffc241fd60c1`
- `test_toy_cpu.py`: `84b9544879ca01e0f46ba4b0b88a023277b3e4c7d5defd80b92e6f9054eccd8e`
- Runner: `bf6a5759fb911c9820d30f79084e6d88cdcf780a496c2d40eeeb8236d1898432`
- Toy result: `19b152e2e682b048e947c4d0c71430264eaa00da34a817f1db68eb61d1d32124`
- Attempt result: `13e64e931bf5caf6b0ea073f9b5768f61a87ff6142b63db9cebba42d80467aa1`
