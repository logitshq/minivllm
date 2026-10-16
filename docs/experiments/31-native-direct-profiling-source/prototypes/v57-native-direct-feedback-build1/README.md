V57 CPU READY; actual GPU unqualified.

Default OFF preserves native40build2. v57_enable_direct_feedback() after seal+ICB/before reset enables sync-only direct feedback.
Both banks must prove ALL132 output ranges disjoint from ALL131 input ranges; ANY alias, even identity, rejects. No scratch/cycle/mode feature.
83 non-S links copy arena→arena once. No capture-stage encoder or stage copies. Old stage allocation remains retained but unused.
All132 outputs remain readable after successful CB completion: non-S directly from arena; S from existing current bank. Valid until next reset/replay.
Actual44 evidence:34584 pair checks,0aliases,39,949,336→19,726,348 copied B/step. This is payload evidence, not a speed claim.
Root tiny: DYLD_INSERT_LIBRARIES=ABS/libv57replay.dylib +50/50/20/0; venvpython ABS/smoke.py NEW_ABS_OUT.
New tiny/full numerical qualification required before performance/adoption. Author did not import MLX, load dylib or use GPU.
