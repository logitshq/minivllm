# GLM5.3Flash input preparation, first actual CPU run

The first explicit CPU input preparation exited 0. No model was loaded, no backend was imported, and no service was changed. This is not numerical, fit, resource-safety, or performance qualification.

The archive retains all three original trees: input plans/inventory, the fixed plain coding prompt, and execution attempt/result/logs/source copies. Every tar member was compared byte for byte and by SHA-256 with its original file after compression. `manifest.json` records the original paths, sizes, member names and SHA-256 values.

Preparation hashed 24 explicitly supplied runtime files (43,842,459 bytes) and retained source/metadata pins. It checked new file statistics for all 62 shards (203,992,076,296 bytes), using the previously retained download SHA-256 values; it did not reopen or rehash shard payloads. The inventory is a supplied set, not a recursive dependency closure. Model identity is `glm5_next`, revision `fa04cbc3622b16aa7475404060b05e506d81bd4e`.

The fixed plain prompt has no chat template. Actual tokenization and the <=128-token gate remain for the future probe. The unchanged probe uses normal EOS and output caps 8/32/32/32. A separate actual cold-service registration and physical admission are still required.

Replay: inspect `manifest.json`, ensure the tar contains exactly its listed regular-file members (no absolute paths, links or `..`), then extract into a new directory. All members live under `raw/glm-flash-inputs-v1*`. The files contain historical absolute paths; extraction does not authorize or run those commands.
