# V40 CPU-ready local candidate
- Helper API: `gated_delta_step_norm_reuse_metal(u,g,beta,S)` in `gdn_norm_reuse_cached_metal.py`; exact V19 fixed shapes and lazy import.
- V19 header, compact activation/prework, serial norm, safe compile policy, scales, arithmetic/reduction/barriers are preserved. Only two recurrent loops become ascending unrolled held[32] loops from V30.
- Register reuse is intent; actual register allocation, spills and removal of a physical reload are unmeasured.
- Numeric reference is accepted25 native recurrence after original MLX BF16 SiLU/repeat; candidate consumes the same saved BF16 u and FP32 g/beta/S.
- Nine historical real banks (L0/32/62 × D1/16/31) plus six saved synthetic banks: strict finite FP32 o/S_new bytes and all external inputs readonly. No new real capture, model load, timing or fullmodel qualification.
- CPU: `python3 check_cpu.py`; `python3 numeric.py` validates saved input provenance without MLX/NumPy import.
- Root-only actual child: `/Users/ryanli/code/minivllm/.venv/bin/python numeric.py --execute-gpu --out NEW_ABSOLUTE_DIR` under environment 50/50/20/0 and root-owned resource/timeout/cleanup wrapper.
- First byte mismatch stops; first-failure-A/B.bin are retained. No engine default or frozen source was changed.
- Native38 currently recognizes only the old GDN PSO for S ping-pong; V40 integration requires a separate pinned-function eligibility update and fresh full qualification.
