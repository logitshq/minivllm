# V33 fixed L20 replay worker

CPU prepared; no model or MLX imported during preparation.
Root CLI: `/Users/ryanli/code/minivllm/.venv/bin/python fullqual.py --out NEW_ABSOLUTE_DIRECTORY --tokens32`. Inject the exact pinned native library with `DYLD_INSERT_LIBRARIES` before process startup, with controls 50/50/20/0. Default native library is build2; root tiny2 stopped on unsupported event signaling, so do not run this default against the model. Preserve this worker; a new worker directory will pin native3.

Accepted25 once, 48 gates ON, conv/functional OFF, 1x32 warm request followed by a fresh exact L20/cap256 seed. Shared parameter addresses and all zero-copy owners stay pinned; no weight byte snapshot/copy. Snapshots cover 131 seed inputs and 129 outputs (logits plus all128 states). Native restore/replay twice must strictly match all raw outputs and retained input bytes. Resources use the original guard; root owns external timeout/PID cleanup.

Qualification diagnostic only: no performance, autoregressive replay, or ICB claim. `--inert` checks pins without importing model/MLX or loading the native library.
