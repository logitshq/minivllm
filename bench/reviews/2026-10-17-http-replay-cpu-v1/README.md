# Recorded HTTP correctness replay — CPU preparation

The native HTTP join and replay now connect retained SSE bytes, actual scheduler traces and two independently executing engines. They never force recorded tokens or schedules. The wrapper retains the original numerical comparator and model/runtime pins; the recorded C1 and C8 cases remain separate from the 37-request E028 model qualification.

Root validation: 28 CPU tests (join 8, replay 11, wrapper 7, pipeline 2). The pipeline uses the frozen HTTP handlers, scheduler and CPU toy model with MockTransport; it is not a live network or full-model test. Actual 24 planned D requests were parsed with the pinned server and tokenizer. See `actual-inputs-cpu-v2.json` and its reproducer `verify_inputs_cpu.py`.

Root found that an integer timeout in the expected identity differed from the server argparse float. The before-fix sources remain in `identity-timeout-before-fix/`; `final-correction-tests.json` records the repair and seven affected tests rerun. Its final source hashes supersede the wrapper hashes in `cpu-tests.json`; other source/tests were unchanged. The original actual-input receipt is retained, with v2 superseding its startup identity.

Peer review: draft_graph reviewed replay; draft_window reviewed join; official_mtp reviewed wrapper and provided the pipeline tests. These are preparation checks only: no real HTTP capture, full-model qualification, or new performance point. Received-tail validation cannot prove bytes the frozen client never received after DONE.

Before live HTTP admission, implement a capture/cleanup owner and version the native client cleanup repair: cancelled response/client closers must fail the run, as already enforced in the Lithos adapter. Do not silently mutate the frozen client or claim its close lifecycle is qualified.

On 2026-10-17 the user explicitly authorized the separate prepared oMLX cold stop and recovery: “允许停止、实验后恢复”. Root will now execute frozen E028 model v1 and always recover the original persisted-visible MTP/resident-PLE service afterward, including on experiment failure. No approval is pending for that transaction.
