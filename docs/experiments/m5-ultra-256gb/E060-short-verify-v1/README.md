# E060: short verification after E059's mixed result

GPU not run at source checkpoint. E059 found fixed4 faster on math but slower on Chinese, while fixed8 was worse on both. Test T2 to reduce rejected-suffix work; this is an exploratory follow-up on the same two synthetic1024-token contexts, not a new held-out benchmark.

Frozen draft block8 is unchanged; verify one anchor+one proposal at T2. Candidate arithmetic already admitted T2, but E058/E059 did not qualify it on device. E060 must first compare real generated IDs, last logits, every target feature and all128 final states, plus a forced-first-rejection trajectory alternating T2/T4/T8. EOS/cap selection now covers widths2/4/8 in3072 CPU cases.

Four timed arms native/fixed2/fixed4/fixed8, one warmup each and3groups×2contexts×4arms=24 samples. No calibration/adaptive policy. Same M1 host-driven native runner; no HTTP/prefill timing/serving lookahead. Fresh outputs and inherited bounded pause/restore owner. A win here is not an oMLX/Lithos Pareto win.

Pre-run failure preserved: first CPU run passed29/30; the expanded enumeration correctly produced3072 cases but the copied assertion still expected2928. Only that count/comment corrected, not selection behavior.
