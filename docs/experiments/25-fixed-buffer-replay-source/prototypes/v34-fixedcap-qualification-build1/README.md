# V34 direct fixed-capacity qualification

Root CLI: `.venv/bin/python fullqual.py --out NEW_ABSOLUTE_DIRECTORY --tokens 32`, controls50/50/20/0, no native DYLD injection. Same accepted25 weights loaded once; all48GDN gatesON; conv/whole-functional OFF. A uses exact L20..50; B differs only in ForwardContext.max_len=256. KV capacity256, boolean lens mask, independent K[:]/V[:] context handles, complete128-array states.

One shared19-token prefill seeds both independent greedy31-decode chains (32outputs). Every step compares complete logits and all128full-capacity caches for shape/dtype/finite/strict raw bytes. Retained cache arrays and explicit token/position/lens inputs are checked with raw bytes. First failure stops. Original resources raw precede guard/derived data; beforeload, afterload before any token, each4decode, finally. Cache pool2GiB; root owns timeout/process monitoring/cleanup.

CPU source eligibility: official MLX0.32.3 selector stays one-pass below1024; installed sdpa_vector.h bool-mask false skips arithmetic, preserving valid-key partition and reduction tree. This is a qualification prerequisite, not actual numeric/performance proof. No native runtime, service integration, timing/adoption claim, changed-prompt or reset-request coverage.
