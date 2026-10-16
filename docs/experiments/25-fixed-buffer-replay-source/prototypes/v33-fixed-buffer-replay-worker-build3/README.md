# V33 fixed L20 native replay qualification, worker build3

The worker differs from frozen worker build2 only in the native library default (native build4). Root must inject this exact library before process startup with DYLD_INSERT_LIBRARIES. Native4 handles captured same-queue fence/event ordering; only root tiny4 then model qualification can establish runtime support.

CLI: `/Users/ryanli/code/minivllm/.venv/bin/python /private/tmp/minivllm-engine-optimization-20261014/v33-fixed-buffer-replay-worker-build3/fullqual.py --out NEW_ABSOLUTE_DIRECTORY --tokens32`. Controls50/50/20/0; root owns GPU, timeout, monitor, and process cleanup.

One accepted25 model load, all48GDN gatesON, conv/functionalOFF. A32-token warm request then fresh exactL20/cap256 seed. All parameter owners, BF16bitviews, memoryviews, and NumPy frombuffer views remain pinned; no weight byte snapshot/copy. Strict raw snapshots cover131seed inputs and129outputs (logits plus all128states). Native restores the same seed before each of two teacher-forced replays; all129outputs must match raw reference bytes and retained inputs/reference arrays remain unchanged. Resources use the original guard at each boundary.

This is a correctness probe, with no performance point, autoregressive replay, or ICB claim. `--inert` checks sources without importing MLX/model or loading a native library.
