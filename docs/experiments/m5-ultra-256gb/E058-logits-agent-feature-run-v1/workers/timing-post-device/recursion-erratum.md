# Recursion Erratum

The v1 script `device-timing-analysis.py` was defective: after writing `device-timing-analysis.json` it executed `subprocess.run([sys.executable, "device-timing-analysis.py"])`, recursively invoking itself.

This self-invocation is incorrect for a numerical reducer, which must compute metrics once and exit; the external root launcher is responsible for capturing exit status. The recursive call risked infinite recursion / double execution and conflated provenance (asserting an unobserved exit code).

Live process check showed zero matching processes remaining.

Fix in `device-timing-analysis-v2.py`: removed `subprocess` import and all self-launching code; script now computes metrics once, writes output, computes `source_sha256`/`complete_sha256`/`output_sha256`/`result_sha256` directly into `device-timing-meta.json` without asserting exit code. Verification now compares ALL independently recomputed metrics (`group_ratios`, `median_ratio`, `capture_overhead`) against the integrated reducer.

Pinned source/results unchanged; v1 artifact preserved.
