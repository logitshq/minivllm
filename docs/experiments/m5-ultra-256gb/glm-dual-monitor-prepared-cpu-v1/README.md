# GLM independent monitor lanes: CPU evidence

Five guarded, fake-only CPU attempts passed, with all original stdout/stderr,
runner files and source snapshots retained. No GPU, model, service or physical
probe ran. All suites reported zero skips/errors and forbidden operations;
source checks before and after each attempt were unchanged.

| Attempt | Tests actually run | Result |
| --- | ---: | --- |
| Original monitor helper | 13 | PASS |
| Incremental-history fix regression | 1 | PASS |
| v3 direct-child owner | 9 | PASS |
| v3 lifecycle / probe reap evidence | 6 | PASS |
| Independent dual-lane resource audit | 8 | PASS |

The first helper source is preserved separately from the later source. A report
field was clarified to avoid implying that `poll()` supplies the owner's full
wait/absence proof. The next change removed repeated deep copies of historical
command outputs: callbacks now inspect constant-size counters, and complete
history is copied only at exit. Only its new regression was rerun; this is **not
an assertion that all 14 current helper tests were rerun together**.

The tests cover physical observations continuing during slow metadata callbacks,
actual returned fake PID ownership, unknown launch rejection, cancellation and
bounded cleanup, incomplete monitoring, and independent sample/command ranges.
The new audit preserves the old physical parsing, request-token rules, runtime
inventory and restoration algorithms. It separately checks physical gap ≤1 s,
metadata gap ≤5 s, and callback duration ≤5 s; these are observed bounds, not
real-time scheduling guarantees. Original 220 GiB process / 240 GiB system and
zero-swap gates remain unchanged.

This archive grants **no device, load, correctness, HTTP or performance
qualification**. The v3 predecessor manifest and dependent auditor pins are
explicitly UNSEALED. The currently prepared single-child predecessor adapter
will need a reviewed binding if a different closed HTTP cycle runs first.

The archive has **96 regular members**, **1,004,650 uncompressed bytes**, and
**72,952 compressed bytes**. SHA256:
`8384f7cd4a722797f100f58479103cc77b4790142338b5a4feaadcbcd84ef270`.
Every member was manually extracted to a temporary directory and checked for
exact size/SHA; source bytes were also rechecked after packing. No tests were
rerun during archiving. See [manifest](manifest.json), [summary](summary.json)
and [verification](archive-verification.json). Run `python -I -B verify.py` for
an independent local recheck using only the standard library.

Implementation: [v3 README](../../../../bench/m5_ultra_256gb/glm_flash_cycle_v3/README.md).
