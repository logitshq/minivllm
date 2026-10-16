# V76 narrow BA + FP32 gates + activated norm prework probe

Independent extension, CPU prepared only; no engine/native/timing/performance claim. Actual oMLX Q35 decode lines2054–2100 fuse conv/SiLU/L2 prework but leave gates/recurrence separate. Qwen4 lines300–322 use different BF16 gates/outerexp; that math is not copied.

Production prework has8 original inputs (activated,X,WB,SB,WA,SA,Alog,dtbias),6 FP32 outputs (Q,K,V,scales,g,beta),14 declared buffers. A qualification-only probe adds2 BF16 BA outputs,16 declared buffers. Actual Metal reflection remains unobserved. No packing/reorder/weight concat; ensure_row_contiguous=False and actual stride probe must prove canonical input mapping before either prework call.

QK16groups/serialnorm are original44 byte-exact. Each of48 Vgroups copies V using128threads; its first32lanes compute two independent QMV rows:16values/lane×10 K512steps, exact66 qdot/fp4/fp8 source, original ascendingK additions/simd_sum/BF16cast. Lane0 expands BA toFP32, applies exact installed Sigmoid/Exp and Add/Negative/Multiply/LogAddExp/log1p float functors. Softplus is logaddexp(x,0), not a replacement precise-exp formula. State40 full body/name/HEADER/options and S index6 remain exact44; no recurrence math change.

Contract/JIT context equivalence is unproven: prework retains44 math_mode=safe while original QMV/compiled gates use their native context. CPU source identity cannot qualify Metal arithmetic. Any strict mismatch stops; no widening, auto-fallback, performance preparation or model adoption follows a failure.

Fixture: only18 checkpoint ranges for layers0/32/62,830016B total. WB/WA U32 and scalesU8 remain original; stored A_log/dt_bias BF16 are cast toFP32 exactly as model loader. CPU preparation reads safetensors headers only. Root runtime detaches each selected memmap to owned NumPy host memory before MLX; no full shard/model load.

45 cases=3layers×15 historical saved banks (9real,6synthetic). Activated=nn.silu(savedRAWu), materialized outside candidate; S unchanged. Eight predeclared synthetic generic X vectors map bankindex%8. GenericX is not the original hidden state matching those saved activations; no coherent full-model tuple claim.

720 strict shape/dtype/finite/raw checks=45×(QMM BA vs diagnostic2 + originalQMM/compiledgates/44prework vs production6 + production vs diagnostic6 + o/S2). Reference gates compile original _gates_explicit AST with explicit current weights. All71 original arrays/addresses/SHA checked readonly after each layer; both o/S and all prework outputs are consumed. Fresh outputs discarded eachcase;512MiB allocator/64MiB cache cap, actual memory/resources recorded. No temporal state feedback.

Root-only CLI through sole-GPU timeout/resource/PID owner,50/50/20/0, no DYLD injection:
```
/Users/ryanli/code/minivllm/.venv/bin/python -B /private/tmp/minivllm-engine-optimization-20261014/v76-gdn-gates-prework-build1/numeric.py --run-gpu --out /private/tmp/NEW-V76-NUMERIC --power-domain 'AC Power'
```
Default AC; explicit Battery Power is a separate numeric resource domain. Inner guards require declared domain/thermal0/pressure1/no swap growth; root owns timeout/reap/release. CPU author checks: python3 -B check_cpu.py; numeric.py --inert. No MLX/NumPy/GPU imported by author. Old negative results and frozen sources remain unchanged.
