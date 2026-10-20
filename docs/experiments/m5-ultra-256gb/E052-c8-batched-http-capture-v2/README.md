# E052 v2: actual batched GDN + graph HTTP capture

**Capture, raw join and saved resource/restore review PASS.** Two sequential,
fully drained waves of8 requests produce a new1,030-step/16-request recorded
history with8,192 raw/committed output IDs. This is an instrumented capture, not a
formal performance point. Its new history still requires complete numerical replay
and an independent ledger audit. E051 qualified the different original E047
history and does not automatically qualify this capture.

V2 corrects only the expected startup identity inherited by its specification:
the frozen combined server does not emit E047's `diagnostic_only` key. The actual
v1 server, candidate kernels and transport/cleanup remain pinned. The
[v1 FAIL archive](../E052-c8-batched-http-capture-v1/README.md) is unchanged. V1
sent no generation request, but its seven readiness GETs are explicitly retained.

## Actual capture and resources

The actual v2 cycle completed in54.879346166s with exit0 and original status
PASS_CAPTURE_AND_RESTORE. The server's trace and raw HTTP/SSE rows were joined only
after capture and cleanup completed. Both warm/screen groups contain8 requests;
the joined decode histogram is B0:2, B2:4, B4:4, B6:4, B8:1,016. HTTP concurrency8
and GPU decode batch size are separate quantities. Observer overhead and changed
arrivals/scheduling preclude treating these diagnostic clocks as a Pareto point.

Independent review reparsed20 physical samples plus20 metadata/checkpoint samples,
including the original admission baseline. Max observation gap3.186006667s;
baseline→first gap2.702540791s. Recorded maxima: system199,554,482,176B, child
current26,230,763,472B and child historical peak29,104,303,960B. These accounting
fields are not added together. All observed samples meet the original64GiB
child/240GiB system sampled thresholds, pressure1, AC power, thermal0 and zero swap
growth; Swapouts remains807478. Sampling does not impose allocation caps or bound
inter-sample peaks.

The saved direct server identity/PID/PPID, TERM if needed, wait/reap, strict
absence and port absence pass. The original service PID52818 is restored with its
same identity, resumed state and all three saved health contracts. The owner and
lease return0/closed. No new live health/process query was performed by this
review; unavailable full historical process inventories are not claimed reparsed.

## Evidence and preparation history

Parent plan SHA:
`5290ddc18239d422e846f76ff4fa222b48188c6fb57bae68ea5119ec3bd9066c`.
Spec SHA:
`e94ae435be73d39b060bafb7a24724011de48d03a485b1876ba0ecb07ee10a1d`.
Actual trial label is `c8-batched-mb50-two-waves-v2`; the original result and joined
receipt retain their v2 schemas. Source preparation CPU evidence is7/7 PASS.
The first metadata-prepare filename failure (root8f88b2) remains in its original
receipt directory; successful preparation45edb8 and all pending historical fields
are preserved. This is separate from the already archived v1 device failure.

`execution-provenance.json` records root15a76c/session12844 →dcb9e0 exit0.
The raw join is reported by root tool986843 exit0, with its exact preregistered
argv and actual joined-input receipt/recording retained. There were no separate
saved join subprocess stdout/stderr files: `join-tool-observation/receipt.json`
labels this distinction explicitly and does not fabricate logs. The independent
saved resource audit actually ran with9f1484 exit0. No raw join or GPU work was
repeated by this archiver.

`evidence.tar.gz` contains214 mapped members: original closed capture/SSE/trace and
join, resource/lease/execution records, source pins,7-test CPU evidence, failed
prepare receipt and independent resource review. All30,594,912 original bytes
compress to2,459,083B. Every member was independently reread and matched by
size/SHA to its original source and manifest; no requested directory is missing,
and no payload is excluded. No model weights or full numerical state tensors are
present. Integrity of captured bytes is not numerical model qualification.

Archive SHA256:
`0ebd6100b781e15b306cdac50c92c02e23957f4829483519dc55f424882480ba`.
Manifest SHA256:
`10d1020ea74b02f025f4967af63ca5f324524d3cce9c3f96e583349638f980ea`.
Outer side artifacts are bound by `document-integrity.json`, excluding itself.
The source-only archive/README drafts remain as preparation history. Actual
archive execution used `archive.py`; there were no failed v2 resource/archive
attempts. All numerical/full-state/performance/Pareto qualification flags remain
false here, and the formal v16 frontier is unchanged.
