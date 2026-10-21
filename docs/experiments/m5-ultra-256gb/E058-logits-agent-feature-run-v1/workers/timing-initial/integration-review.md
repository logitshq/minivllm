# Integration Review — E058 timing vs root (CPU-only, corrected)

**Scope:** Static read of `/private/tmp/e058-root/{screen.py,runner.py,execute.py,terminal_hold.py}` and local `timing.py`. No GPU/MLX/network/service. No speedup claimed. No edits to root or timing.py/tests.

## Retractions (previous revision)
1. `fresh()` position guidance retracted: `fresh()` must preserve seed position 1022 / lens 1023; teacher-forced refers to supplied tokens, not position reset.
2. File attributions corrected: cleanup (`engine.runner.*`, `gc.collect`, `mx.clear_cache`) is in `screen.py`, not `runner.py`; `close_child` is in `runner.py`, not `execute.py`.
3. Recommendation for extra final `timings-partial.json` write on failure withdrawn: `on_sample` already writes each completed row immediately; extra write unnecessary and could mask original failure.
4. E057 SHA / snap-count claim retracted: `snap()` 130 entries = 2 outputs + 128 state arrays; E057 snapshot was 129, so E057 SHA does not establish E058 130 contract. No claim made.

## Substantiated observations (no new actionable issues invented)
- `timing.py` `schedule()`/`collect()` 60-row contract matches `screen.py` expectation `completed_pairs==24` and `len(timings)==60`; JSON round-trip check `json.loads(json.dumps(...))` handles int→string keys 4/8 (owner-fixed).
- `screen.py:callback()` creates fresh `c=fresh()` per invoke with `mx.eval`+`synchronize` inside measured region — preserves seed positions and isolates arms; `project` reuse for `shared_without_features` baseline is intentional overhead diagnostic.
- `screen.py` timing scope string explicitly local verifier+capture only (token-major with supplied tokens, materialized logits/state+features), no draft/acceptance/HTTP/Pareto — consistent with `runner.py:validate_complete()` handling.
- `terminal_hold.py` pinned argv `command[:4]` and `len==11` is strict by design; `execute.py`/`runner.py` observe loop uses `time.monotonic()` deadlines and strict absence reap.

**Result:** No remaining actionable issues substantiated from CPU-only review under stated facts.
