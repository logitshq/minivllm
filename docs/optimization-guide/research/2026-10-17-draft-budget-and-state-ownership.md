# Draft budgets, committed state, and public implementation boundaries

Read 2026-10-17 while the paired E024 HTTP experiment runs. This is source research, not new device evidence. No model, binary, runtime change or performance test was executed for this note.

## October 14: bound the drafter, keep the target context

[Long-Context Drafting Does Not Require Long-Context Drafters, v1](https://arxiv.org/pdf/2610.20865) was read at §§4–6.2 and the measurement controls. It bounds a frozen target-conditioned drafter's attention window and assigns stable rotary indices when keys enter the cache. The target retains its full context. Common-shift equivalence assumes exact arithmetic and a dependency graph wholly inside one index band; crossing boundaries is explicitly outside that guarantee. This does not prove BF16 bit equality.

Its main ladder uses an MI355X node, batch one, ignored EOS, 4096 generated tokens and decode-only timing. Those are different from our normal-EOS D512 HTTP contract. Most useful measurement lesson: asymmetric graph capture can overwhelm the apparent speculative effect. We must compare matching target runtime settings when testing a speculative candidate; E024's intentionally different OFF/ON graph setting is itself the current experimental variable.

Our decision: retain bounded draft history as a later option, after the already-recorded E014 results and actual verification-cost table. Current D prompts do not exercise million-token rotary boundaries. Do not alter target KV or claim this paper gives an immediate D512 gain.

## October 6: fewer verifications only help if total work falls

[DLoop, v1](https://arxiv.org/pdf/2610.07659) was read at §§3.2–4.2 and Tables2/4/A4. A confidence gate can append complete draft stages before one verification. This trades away intermediate bonus-token opportunities. The strong results combine the loop with additional drafter training; the untrained loop sometimes increases acceptance without improving speed. Its experiments use H100/BF16 and different models; a component timing table also excludes the gate cost.

Our inference: longer accepted blocks are not the objective. For one unchanged workload, record total draft, verify, commit/rollback, host gate and synchronization time, divided by actually committed tokens. Do not select a confidence threshold using the final evaluation requests. The existing short T1/T2/T3/T4/T5 verification measurements must precede an adaptive policy; no new loop is implemented here.

## September 21: a token tree is also a state tree

[DeepSeek-V4 tree speculation, v1](https://arxiv.org/html/2609.24698v1) was read at §§3–5.3. The implementation isolates branch-dependent compressed states, then refreshes only the accepted path. Higher acceptance eventually stops translating into throughput because state and verification overhead grow. Its eight-NVIDIA-GPU setup and CSA/HCA representation are not our GDN implementation.

Our inference: a future Qwen proposal module needs one committed prefix owner and temporary branch state. Acceptance must commit matching token, attention, convolution and recurrent state together. Rejection must not leave the next step using state from a discarded branch. Keep the initial proposal linear and bounded until measured cost justifies branching; a tree mask alone is insufficient.

## Pinned BaseRT public interface, not an engine source audit

GitHub API resolved and read commit `24badc8dd173e4111b64996a718d0e056405ea62`. The [README](https://github.com/basecompute/baseRT/blob/24badc8dd173e4111b64996a718d0e056405ea62/README.md) and [engine-release documentation](https://github.com/basecompute/baseRT/blob/24badc8dd173e4111b64996a718d0e056405ea62/docs/reference/engine-releases.md) distinguish the Apache-licensed CLI/format/headers/bindings from the separately distributed proprietary engine. The public recursive tree contains no Metal engine implementation. Earlier unpinned interface reading is now supplemented by this pinned inspection; no kernel was copied or benchmarked.

Selected portions of [baseRT_tick.h](https://github.com/basecompute/baseRT/blob/24badc8dd173e4111b64996a718d0e056405ea62/include/baseRT/baseRT_tick.h) expose generation-tagged references, caller-owned buffers loaned until terminal completion, authoritative terminal emission separate from streaming deltas, and whole-sequence state adoption including recurrent state. These are declared interface contracts, not proof that an unseen runtime implements them correctly.

For our small engine, the useful rule is local: the request's committed state has one owner, pending GPU work retains its inputs, and the output counter advances only for committed tokens. Do not import a general daemon, tiering system or artifact registry merely to obtain these invariants. Current E024/GDN candidates remain separate default-OFF modules; combined correctness and performance have not been established.
