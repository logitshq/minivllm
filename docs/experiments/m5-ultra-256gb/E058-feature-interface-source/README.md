# E058 feature-interface source checkpoint

New default-OFF verifier prototype captures complete decoder-layer outputs and
aligns accepted features with a private cache prefix. It is not integrated into
the production engine and has no GPU or performance result.

CPU attempt1: 7/7 PASS (`f402fe`), subprocess stdout/stderr/source hashes retained.
CPU attempt2 after adding the shared-head case: 8/8 PASS (`605bf7`), actual tool
result reference and source hashes retained. No failed CPU attempt occurred.
Tests are scalar control-flow fingerprints, not BF16 numerical qualification.

[Prototype and next device gate](../../../../bench/m5_ultra_256gb/rowwise_mlp_shared_features/README.md).
