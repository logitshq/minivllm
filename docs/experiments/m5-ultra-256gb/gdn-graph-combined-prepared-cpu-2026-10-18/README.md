# Combined GDN + route graph: CPU preparation evidence

The retained first attempts passed **20 combination tests + 6 observed-baseline bridge checks + 3 driver control-flow tests = 29 CPU checks**. These results grant no model numerical, HTTP, or performance qualification. The concurrent device experiment is outside this archive.

The two combination suites retain their guarded runner, original stdout/stderr, exact source copies and before/after source checks. The bridge retains its actual six-check preparation script, stdout, source identities and generated plans. The new driver suite retains its original three-test attempt; real process, socket, signal and backend operations were blocked and no denied operation occurred.

`execute_combined_v1.py` and its live execution remain unchanged. The separate v2 distinguishes a synchronous owner invocation that returned from an unknown launch outcome before selecting `--no-experiment`; it also requires the final journal to be `RESTORED`. Fake tests cover those branches. V2 still requires a real matching PREPARED journal and cannot resume the already-started v1 cycle.

`raw.tar.xz` contains the complete four closed CPU/preparation trees and explicitly listed small source snapshots. It excludes live owner/journal/execution data, developing combined `audit.py`, GLM monitor, weights and runtime binaries. The existing preparation plan is retained evidence, not permission to rerun it. Every regular tar member was compared by size and SHA-256 with its original, then the original file inventory was rechecked.

Verify the archived bytes without extracting or reading live originals:

```sh
python3 archive.py --verify .
```

The verifier requires exact manifest membership and only regular files under `raw/` or `source/`; it never calls `extractall`. `--originals` additionally rechecks the original closed inputs and current source files, so a later deliberate source change can make that optional check fail without invalidating the retained archive. `summary.json`, `manifest.json` and `archive-verification.json` are small sidecars. The included research note remains a proposal, with no ReaDy implementation or speed claim on this hardware.
