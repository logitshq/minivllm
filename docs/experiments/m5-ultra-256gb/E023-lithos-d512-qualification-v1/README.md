# Lithos D512 qualification v1: double-templated input rejected

The actual preregistered run completed eight C1 requests, then rejected the full
prompt-ID gate. Our request builder sent `fixture.prompt_text`, already rendered
by the chat template, as `user.content`. Official Lithos applied its template again.
C8 was skipped because C1 did not qualify. There is no performance/Pareto point.

All eight captured inputs have eleven extra tokens; the first difference is index3,
actual248045 (`<|im_start|>`) versus expected60856 (`Implement`). The official
CPU tokenizer reproduces every captured input from the old double-templated body,
and reproduces every frozen prompt ID from the original `fixture.messages`.
The separately tested retry sends those original messages and retains the exact
original prompt-ID oracle. This is a harness request bug, not a numerical failure
inside the inference engine. The failed run is not edited or retroactively qualified.

| Fixture | Frozen prompt tokens | Actual v1 tokens |
| --- | ---: | ---: |
| D01 TTL LRU | 978 | 989 |
| D02 DAG scheduler | 886 | 897 |
| D03 event rollup | 926 | 937 |
| D04 frame parser | 921 | 932 |
| D05 retry budget | 954 | 965 |
| D06 interval booking | 935 | 946 |
| D07 versioned store | 903 | 914 |
| D08 sliding limit | 958 | 969 |

The eight observed outputs contain4096 raw/committed tokens, no early EOS.
These counts describe rejected requests and do not confer benchmark eligibility.
The source-pinned qualification/resource auditors were each run once after
restoration and reject the incomplete two-phase owner before their full audits.
Their FAIL envelopes do not indicate a measured hardware-resource violation.
Original raw resource records are retained.

Owner24167 and directly owned server24265 were strictly reaped. Original oMLX
was restored asPID25221, started2026-10-17 18:31:11UTC, with the original model
and configuration plus fresh MTP/resident PLE evidence. Independent restoration
PASS; warm KV and exact hidden runtime state are not restored. Execution ends
18:31:47UTC with owner return1 and recovery return0.

## Evidence

Preregistration was pushed as47ba0fd before execution. All original v1 sources
remain unchanged. The new retry lives in a separate directory.

- [CPU diagnosis and exact scope](prompt-diagnosis.md), [script](prompt-diagnosis.py), [complete receipt](prompt-diagnosis.json).
- [Qualification rejection](independent-qualification.json), [resource audit rejection](resource-summary.json), [restoration PASS](restoration-audit.json).
- [Evidence archive](evidence.tar.xz):490 members,1,301,588 bytes, SHA256 `3fc34079e6fcd74c943a6b2a11b57b6d441ea590af7075facb147db0e54e66fc`.
- [Manifest](manifest.json), SHA256 `39fdeef9866cd2865632fc1d96a7197cc3f7c0ac964d94cc95cb7257b982aec2`; [full archive verification](archive-verification.json).
- [Restored predecessor journal](restored-journal.json), SHA256 `1798b9d3ce54bd43bebdd3b99e8851e3fba01f04fca101999dce730070cfd775`.
- Restoration audit SHA256 `8365647c09a4df945079055aa6e88492a62b9d43a6e9bb117b5cad6ce89ffd73`.

The archive contains complete raw HTTP/observer/server/resource/cold-cycle records,
registered sources and221 pinned official Lithos source files. Model/pack weights
and runtime binaries remain external; the archive preserves their supplied pin
inventories and producer receipts, without independently rehashing these payloads.
See [archive scope](ARCHIVE.md). Archive PASS means preservation, not qualification.
