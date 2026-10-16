# E009: reuse the lane-local GDN state

The original kernel reads each FP32 state element in the `k * S` reduction and
again in the updated-state/output reduction. This prototype saves each lane's
32 elements in a local array between those loops. Arithmetic, reduction order,
thread geometry, original projections and independent output state remain.

This is a source-level memory-traffic hypothesis. Metal may already optimize
the second read, cache it, or spill the new local array; no register occupancy
or actual DRAM traffic is claimed without a hardware trace. B9..64 only; the
B8 control retains the original function. The runtime never imports this code.

`screen.py` uses E007's complete qualification and timing procedure, changing
only the candidate. It corrects the earlier `cold_ms` label to
`first_timed_ms_after_qualification`. A 5% full-forward gain at both B32/B64 is
required before separate HTTP qualification. Source and plan are pinned in
the new result directory. Use the owned guard and the same environment as E007,
replacing both path/output prefixes with `gdn_state_reuse` / `gdn-state-reuse`.
