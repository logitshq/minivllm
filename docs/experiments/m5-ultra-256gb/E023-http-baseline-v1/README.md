# E023 HTTP baseline v1: readiness rejection

This attempt stopped before any inference HTTP request. The server startup trace
contains one startup record and no decode/prefill steps. The v1 listener check
rejected the actual macOS output `p81230`, `f4`, `n127.0.0.1:18023` because it
expected only PID and endpoint fields. This is a harness-format failure.
The installed `/usr/share/man/man8/lsof.8` documents that a file set begins
with `f` (lines 3201–3202) and that the file-descriptor field is always selected
(line 3256). The v2 parser admits that one numeric field while still requiring
exactly one owned PID and one exact localhost endpoint.

One server child, PID **81230**, was terminated, waited/reaped and confirmed absent.
No replay or C8 phase started. The original service was restored as PID **81337**;
the independent restoration-only audit is retained verbatim. The restoration
contract covers persisted settings and the visible loaded model. It does not claim
the same hidden runtime signature or warm KV state.

There is **no numerical qualification or performance conclusion** from this run.
The v2 regression fixture is byte-identical to the failed run's raw listener receipt.

`evidence.tar.xz` contains all regular files from the owner, cold-journal and
execution directories, all v1 preregistration files and registered source, plus
the regression fixture and archive/restoration audit source. Runtime binaries and
weights were never copied into those evidence directories; their retained pin
receipts remain included. Nothing in the original evidence is overwritten.

`manifest.json` lists every archive member's SHA-256, size and original path/stat.
`archive-verification.json` records a full extraction and per-member verification,
followed by a second hash/stat check of every original input and directory member
list. `restored-journal.json` and `restoration-audit.json` are exact-byte sidecars
for the next recovery cycle, not rewritten summaries.
