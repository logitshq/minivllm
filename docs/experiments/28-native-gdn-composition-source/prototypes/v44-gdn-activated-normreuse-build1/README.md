# V44 already-activated normreuse/held32 helper (CPU-ready)
API: gated_delta_step_activated_norm_reuse_metal(activated,g,beta,S), activated BF16[1,10240], g/beta FP32[1,48], canonical S FP32[1,48,128,128].
Only prework's three SiLU reads become direct activated BF16 loads. Serial norms and safe compile policy stay exact. V40 state body, state factory name, header, reduction order/barriers and held32 are unchanged.
The state PSO name remains V40, so native40build2 can be reused; actual new combined capture must still prove eligibility and full numerical correctness.
Local reference consumes historical RAWu via original nn.silu→split/repeat→accepted25 GDN. Candidate receives that same nn.silu result directly; it performs no activation. External RAWu/activated/qkv/g/beta/S readonly checked.
CPU: python3 check_cpu.py; python3 numeric.py (inert saved15bank SHA verification).
Root-only child: venv/bin/python numeric.py --execute-gpu --out NEW_ABSOLUTE_DIR, standard50/50/20/0 with root resource/timeout/cleanup ownership. Strict30 FP32 o/S_new comparisons; first mismatch stops.
No model/engine/conv integration, new real capture, GPU execution or performance claim here. Planned combined caller must feed true conv_silu23 activation and its tail exactly once, with defaultOFF and accepted25 A reference.
