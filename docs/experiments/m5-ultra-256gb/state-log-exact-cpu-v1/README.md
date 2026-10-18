# Ordered state-log: first CPU evidence

The first stdlib-only attempt passed all 8 tests and 24 toy trajectories. It compared full FP32 states and outputs after 2,304 updates, plus clone/reset/retire behavior. The deliberately algebraic delta merge produces 1 where ordered replay produces 0; this expected counterexample is retained, not tolerated away.

The archive contains every original attempt file: runner, commands, stdout/stderr, exact source snapshots, results and summary. The original source snapshot includes the read-only native kernel used for contract comparison. Source paths and every member SHA/size are in manifest.json. archive-verification.json records decompressed-member and original-tree rechecks. No test was rerun during archiving.

This is a CPU arithmetic toy, not native Metal equivalence, model/HTTP qualification or performance evidence. Persistent-write counts are analytical only.
