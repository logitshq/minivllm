Fixed `timing.py`: rotation now `(group*REPEATS+repeat)%3` covering all 3 positions, authoritative `schedule()`/`SCHEDULE` compared in `reduce()`, `on_sample` propagation (no swallow), dead code/imports removed.

**Files written:** `timing.py`, `test_timing.py`

**Tests run:** `python3 -m unittest test_timing.py -v` — 8 tests OK (exact 60+12 counts, all-3 rotations, schedule tamper rejection, on_sample propagation with partial preservation, reduce math, bool/negative/missing rejection, failure propagation, no throughput field).

**Limitations:** CPU-only; no GPU/draft/HTTP/Pareto logic.
