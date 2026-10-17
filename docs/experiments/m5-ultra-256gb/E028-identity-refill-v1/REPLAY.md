# Replay from a fresh checkout

The complete raw evidence is committed inside `evidence.tar.gz`; the unpacked
`raw/` copy is retained locally and ignored by Git to avoid duplicate storage.
The tar also includes the frozen verifier and manifest. Extract it into a fresh
empty directory and run its `verify.py` there with Python 3.13. The verifier needs
only the standard library and does not import MLX or execute experiment code.

Example, from the repository root (choose a new scratch directory):

```sh
mkdir /tmp/E028-v1-replay
tar -xzf docs/experiments/m5-ultra-256gb/E028-identity-refill-v1/evidence.tar.gz -C /tmp/E028-v1-replay
.venv/bin/python -I -B /tmp/E028-v1-replay/verify.py
```

`--runtime-pins` additionally checks the original installation paths and is
optional for offline replay. `root-review.json` independently repeats the
verifier with those pins and hashes every archive member. This replay note,
root review and `.gitignore` are later packaging additions outside the frozen
archive manifest; the experiment sources and original records are unchanged.
