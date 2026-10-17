# Lithos D512 original-message qualification passes

The separate retry preregistered in418281d passes all24 actual HTTP requests:
full prompt IDs, executed official route/pack, original raw output-token counts,
normal-EOS handling, original-token decoded text, and zero prompt-cache hits.
The frozen independent qualification, resource and restoration auditors each
ran once after restoration and all PASS. This is observed route/count/text/cache
qualification; observers are ON, so it adds no throughput/latency or Pareto point.
Full-state numerical equality across frameworks is not established here.

Same Qwen3.8-27B NVFP4 snapshot5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6,
existing official target-only Lithos pack, capacity2048, normal EOS and cap512.
The request now uses original fixture.messages instead of templating rendered
prompt_text a second time. [Failed v1](../E023-lithos-d512-qualification-v1/README.md)
remains unchanged and fully archived; no oracle or output tolerance was relaxed.

| HTTP concurrency | Requests | Raw tokens | Committed tokens | Early EOS | Peak outstanding |
| --- | ---: | ---: | ---: | ---: | ---: |
| C1 | 8 | 4096 | 4096 | 0 | 1 |
| C8 | 16 | 8192 | 8192 | 0 | 8 |

All24 requests reached the512 cap: actual early-EOS coverage is zero. C8 means
8 concurrent HTTP callers; the official GPU execution remains FIFO. Each phase
starts a fresh official server and preserves the original request/Generation/
wire binding. Full NDJSON, token arrays, prompt arrays and server logs are saved.

All70 raw resource samples passed the original pressure/AC/thermal/paging and
known-process guards. Pressure1, swap growth0MiB, Swapouts/Pageouts/Compressions0;
Swapins+12 and Decompressions+1634. Retained process inventory cannot exclude
unknown runtimes or GUI GPU activity. This qualification does not measure speed.

Owner26867 and direct servers26920/27564 were strictly reaped. Original oMLX
recovered asPID28531, started2026-10-17 18:44:51UTC, with original model/config
and fresh MTP/resident PLE evidence. Recovery completed18:45:27UTC and execution
returned0. Warm KV and exact hidden runtime state are not restored.

The owner verified all supplied model/pack/runtime/source hashes before and
after execution. The independent qualification audit separately rehashes its
CPU runtime/tokenizer and reparses raw requests; model/pack payloads themselves
are covered by the producer receipts, not a second independent payload rehash.

## Evidence and next measurement

- [Independent qualification](independent-qualification.json), SHA256 `df64cd94a1ec47dc4f5dd8f749e61066eba17d126abc0860ad929b1703f16189`.
- [Resource audit](resource-summary.json), SHA256 `f171bcbb8f0f709ececc7b9b35a1f25f5e0e4b3194ed5d542799a8fd5e38930e`.
- [Restoration audit](restoration-audit.json), SHA256 `269a9a85b50674077fee8dc464a0edc4a0531fb13e5386d95dbc8cc57ba66e43`.
- [Restored journal](restored-journal.json), SHA256 `302a54499f03d2cac02981a5ad06f9369a255aa5d78be90243736205529be6af`.
- [Evidence archive](evidence.tar.xz):603 members,1,562,748 bytes, SHA256 `7e7f8ad0b1af248b3e289f2697c0d91535004750af5df41fca5cd6686a90aed4`.
- [Manifest](manifest.json), SHA256 `05b06c47fb0c236d1c972784bf65a0eb94333d2b48f0ae4d28e2d3214db5d758`; [full byte/tree verification](archive-verification.json); [archive scope](ARCHIVE.md).

The next stage is a separately frozen observer-OFF formal run:8warm+64formal
requests per fresh process, C1/C8, repeated processes, committed output tokens
per common POST→DONE window versus complete request latency. The
[formal implementation plan](../../../../bench/reviews/lithos-d512-formal-plan.md)
is unexecuted. Existing native D512 points remain45.372/167.399 outputtok/s at
p50 11.283/24.432s, one process each. No new Lithos/oMLX D512 performance point
or competitor dominance is claimed from this qualification.
