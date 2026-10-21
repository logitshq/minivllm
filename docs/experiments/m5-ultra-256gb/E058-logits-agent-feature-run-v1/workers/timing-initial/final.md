**Files written:**
- `timing.py` – injectable serial collector/reducer (stdlib only, inert on import)
- `test_timing.py` – CPU tests with fake deterministic clock

**Tests run:**
- `python3 -m unittest test_timing.py -v` – 6 tests OK (exactly 60 timed calls +12 warmups, order rotation, pairing math, rejection missing/duplicate/negative/bool rows, failure propagation, no misleading throughput fields)

**Limitations:**
- Warmup failure propagation covers warmup phase; formal-call failure during collect will propagate as exception (partial rows preserved via `on_sample` if provided)
- `reduce` validates exact 60-row schedule, positive int elapsed_ns (bool rejected), ordinal 0..59, per-width overhead computed as median(shared_with_features)/median(shared_without_features); no draft/acceptance/HTTP/Pareto logic
- No GPU/device timing – local verifier/capture only
