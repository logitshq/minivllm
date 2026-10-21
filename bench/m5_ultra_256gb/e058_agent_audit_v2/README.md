# E058 independent auditor v2

Kimi-authored metadata compatibility correction. The frozen v1 rejected actual MLX `__metadata__: null`. V2 treats only the reserved key as metadata (absent/null/string map), preserving the exact130-tensor contract and byte/finite/hash checks. Root reran40 CPU tests; Kimi audited all48 actual raw files /3120tensor pairs. The E058 GPU source and raw outputs remain unchanged. This is an evidence reader, not a serving engine path.
