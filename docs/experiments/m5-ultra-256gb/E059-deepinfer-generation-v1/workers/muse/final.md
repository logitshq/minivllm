**Changes**
- `policy.py:1` `choose_width` and `survival_from_full_width` per spec
- `test_policy.py:1` 8 tests covering short/long wins, cap, tie, cumulative distinction, empty/NaN/bool/censored
- `design.md:1` calibration-only/full-width, frozen eval, DFlash block8, cost vs sunk, future C1/C8

**Tests**
- `python3 -m unittest test_policy.py -v` 8 OK

**Limits**
- No GPU/MLX, no external sources, simple unconditional averages only, no performance claims