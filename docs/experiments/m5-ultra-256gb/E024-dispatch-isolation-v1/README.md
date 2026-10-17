# E024 four-arm result: bound change reproduces the numerical difference

Completed 2026-10-17, M5 Ultra 256 GiB. Same fixed Qwen3.8-27B NVFP4 weights,
BF16, B1/used257 from the original 256-token D prefix. Gates/norm/fuse/core decode
compilation OFF. The original rejected E024 candidate remains unchanged.

| Contrast | Isolated change | Exact result, logits +128 state tensors |
| --- | --- | --- |
| A eager257 → B eager1536 | Attention length | 121/129 tensors differ |
| A eager257 → C compiled257 | Compilation at257 | All129 bit-equal |
| B eager1536 → D compiled1536 | Compilation at1536 | All129 bit-equal |
| A eager257 → D compiled1536 | Original combined candidate | 121/129 tensors differ |

Independent raw safetensors audit PASS, completed once without an audit retry.
A–B and A–D each differ in 35,960,404 original-dtype words /80,978,343 bytes:
logits plus120 state tensors (45/48 S,45/48 conv,15/16 K,15/16 V). The first four
layers' eight states agree; the first differing state is layer4. Logits alone
have228,516 BF16 words differing, first word403e→4037 at index1, matching the
preceding rejected attempt. All four greedy output IDs are1559, but token
agreement does not substitute for the original exact criterion.

All saved outputs are finite. Each arm's131 input arrays agree byte-for-byte
with the canonical input and its own after snapshot. Full1349 parameter
original-dtype value hashes/shape/finite receipts remain identical before/after
prefill and every arm; handle IDs are recorded separately. Weight payloads were
not saved/rehashed by the independent auditor. Successful initial independent
load/prefill checks are audited as complete device records, not recomputed.
Original source/model shard checks ran both before and after the diagnostic.
Root reviewed the implementation/auditor and rehashed all execution sources.

**Conclusion limited to this point:** changing the bound is sufficient to
reproduce the original difference, while compilation at a fixed bound adds
none. Matching official MLX source predicts a single-pass→two-pass SDPA switch
at K1024 on this device. No runtime kernel trace or layer-level kernel isolation
was performed, so the source mechanism is not itself a traced causal proof.
The result does not qualify a new1023/1536-bucket candidate, the full matrix,
parameter replacement, ahead lifecycle, actual D512 HTTP histories, or speed.

The diagnostic took47.761216s and its owner48.636947s, including loading, checks
and tensor saving; these are not inference latency measurements. Thirteen raw
resource samples passed: pressure1, swap growth0, Swapouts/Swapins/Pageouts0,
Decompressions+31. Known-process receipts identify only the owned child;
unrecognized runtimes/GUI GPU activity remain outside the observation scope.

Original oMLX PID832 was stopped16:42:07 UTC. Child11276 and owner11242 were
reaped/absent before recovery completed16:43:37 UTC as **PID11548**, with the
original model/configuration and fresh MTP/resident PLE log evidence. Independent
restoration audit PASS:19 physical samples,11,301 fresh log bytes; peak recorded
system footprint175,243,329,536 bytes. Warm KV and exact hidden runtime state
are not restored. The enclosing execution finished0.

Evidence storage: **689 small files are archived** (1,308,564 compressed bytes),
including all summaries, source/plan pins, resource/recovery logs, audits and
first-difference logits. **13 original full-state tensor files,3,312,040,693 bytes,
remain unchanged locally** at the exact source paths in the manifest; these
large tensors are not inside the Git archive. All local payload hashes and all
archived member hashes were checked. A remote-only checkout needs those original
files or a rerun before repeating the full raw tensor audit.

Archive SHA256 `ebee0341f2beb19c2adbb6ce74d4e5b3f5b0c8bc1852454d5ab8dcdcff7ad626`.
No numerical or performance qualification follows merely from packaging.

- [Independent full tensor audit](independent-numerical.json)
- [Recovery audit](restoration-audit.json) and [restored journal](restored-journal.json)
- [Exact manifest, including local tensor locations](manifest.json) and [verification](archive-verification.json)
- [Frozen preregistration](../../../../bench/m5_ultra_256gb/long_decode_isolation/prereg-v1/README.md)
- [First-principles tutorial](../../../optimization-guide/tutorials/024-decode-graph-dispatch.md)
- [Next route-preserving hypothesis](../../../../bench/reviews/2026-10-17-e024-route-preserving-bounds.md)
