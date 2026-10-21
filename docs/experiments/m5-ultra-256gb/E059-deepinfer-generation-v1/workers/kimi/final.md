All good. The work is complete.

**Changes**
- `work/selection.py` — `select(drafts, posterior, remaining, eos_ids) -> Selection` (frozen dataclass: `produced` tuple, `accepted_drafts`, `computed_count`, `next_anchor`, `finished`). Greedy longest-matching-prefix + correction/bonus token, trimmed at earliest EOS (inclusive) or `remaining` budget; `accepted_drafts = min(a, len(produced))`, `computed_count = len(produced)` (retained block state = `[anchor] + produced[:-1]`, pending last token), `next_anchor = produced[-1]` unless finished; finished on EOS or exhausted budget. Strict validation: nonnegative strict-int IDs (bool/float/str/negative rejected), widths 4/8 only, `posterior` length = block width, positive strict-int `remaining` (0 raises ValueError), `eos_ids` accepts tuple/list/set/frozenset; no input mutation.
- `work/test_selection.py` — 13 unittest cases: exhaustive enum of width 4/8 × every mismatch length × every remaining (1..width+2) × every EOS position × all 4 EOS container forms (2928 cases) checked against an independent reference; multi-round causal simulation against a literal scripted target stream asserting `committed + [pending] == emitted` each round, retained block = `[anchor] + produced[:-1]`, and emitted history = target stream prefix; validation/no-mutation/frozen-result tests.

**Tests**
- `/usr/bin/python3 test_selection.py -v` → 13/13 OK (0.02s), plus manual edge-case assertions (full-accept bonus, budget truncation inside the accepted prefix with pending draft counted, EOS at first draft slot).

**Limits**
- Stdlib only, CPU-only, runs under Python 3.9.6 (`unittest`, no pytest/network/installs); no GPU/MLX imports, no repo edits — both files live only in the isolated `work/` dir of my cwd.