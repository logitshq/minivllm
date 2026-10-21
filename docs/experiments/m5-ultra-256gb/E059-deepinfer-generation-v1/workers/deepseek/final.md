Created both files in the isolated cwd:

- `research.md`: 33 lines. Covers official measured gains versus Minivllm hypotheses, exact pinned source links, missing confidence head, causal censoring, DFlash width 8 with verification prefixes 4/8, full-round costs, and three prioritized Apple GPU optimizations.
- `review_checklist.json`: 17 lines. Contains 12 concrete multi-round greedy cache/feature commit failure cases; validated with `python3 -m json.tool`.

Changes are limited to these two files. No GPU experiments or Minivllm performance claims were made. Validation was static source review and JSON parsing only.