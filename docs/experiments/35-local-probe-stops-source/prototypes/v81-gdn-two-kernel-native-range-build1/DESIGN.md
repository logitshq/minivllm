# V81 CPU preparation checkpoint

A44 and B78 each execute 15 fixed saved banks, prework then state, with the same externally materialized BF16 activation and FP32 g/beta/S per bank. No state feedback. Capture order A0.prework,A0.state,...A14.state,B0.prework,...B14.state. One 60-command ICB exposes [0,30) and [30,60). Each state has barrier-before for its prework outputs. Each executed 30-command group ends with an external Buffers barrier before slots are reused.

The frozen V72 implementation cannot be used unchanged: its 16 independent QMV dispatch contract rejects these producer/consumer edges. New V81 code will include frozen native40 capture/arena source, preserve its existing exports, and add strict GDN-only registration/reflection/range exports. No donor files are edited.

Every original activated/g/beta/S allocation must be exact readonly pinned input. All ready/scales/output allocations are separately registered; no original allocation or packed arena range may alias any distinct allocation. The only allowed output-to-input edges are the four ready/scales allocations of one bank's prework into that exact bank's state (scales may be inactive). S must be reflection ReadOnly and never source-licensed writable. Exact source-proven store-only output parameters may retain reflection ReadWrite only through a narrow per-function/index/name policy; other RW or unknown buffers STOP. Reflection WriteOnly is reported distinctly from inferred store-only.

Numeric-only preparation, actual reflection, dispatch/geometry, mapping, terminal o/S raw bytes, readonly hashes, and group-loop parity gate performance. CPU compilation is not Metal qualification. Timing, once separately authorized by an actual numeric receipt plus SHA, will follow the V72 six AB/BA plus four AA/BB schedule and 5%/5-of-6 gate with each control within 5% and each-arm control median within 2%.

CPU source/contract preparation is complete. The new library is compiled on CPU, never loaded here. GPU numeric qualification is pending; performance is deliberately unavailable in this build. See README.md, source-access-proof.json and CPU-READY.json for the final frozen boundary.
