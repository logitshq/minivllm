# V49 SG2→SG4, CPU ready only
Installed native NVFP4 R4 math is retained byte-for-byte except num_simdgroups2→4 and a whole-SIMD padded-row early return. Same FP4/FP8 decode,16values/lane,K512,qdot/ascending accumulation/simd_sum/cast,weight+scales layout and defaultcompile policy. NewTG128threads/grid(128,ceil(N/16),1); eligibility N%4=0,K%512 and nine actualcheckpoint(N,K) shapes. Every declared current shape N%16=0; guard's padded ownership is CPU-proved, not GPU-tested. No weightcopy/reorder/factorfold/SG8 variant.
Initial actual screen only L0gate N17408/K5120: nativeU32+U8 ranges50,135,040B,8synthetic+1historicalsavedrealBF16x. No model/fullshard load. Strict36 consists of9unchangedSG2R4-vs-directnativeQMM,9compiledA-vs-directnativeQMM,9compiledSG4B-vs-directnativeQMM,9compiledB-vs-compiledA. Readonly before/after; anyfailureSTOPbeforetiming. CompiledA calls mx.quantized_matmul directly.
Timer derives control-PASS V32: prime2048/repeat2048,8rotatinginputs,6balancedAB/BA+4AA/BBcontrols; controlmedian[.98,1.02]/each[.95,1.05];local>=5%/5of6 unchanged. Everyoutputeval; hashesafterbothblocks; no retracing on compiledhits. Qualifiedgeometry on other sourceeligible shapes remains UNKNOWN. Old26/28/31/32STOPs unchanged.
Root owns GPU and external resources/deadline/cleanup. Internal exactdirectchild170s cleanup/raw-firstresourceguards,2GiB allocator and50/50/20/0 preserved. DefaultAC; explicit --power-domain 'Battery Power' produces separate exploratorydomain, neverACfrontier. --qualify-only exits0 after36strict+readonly without timer.

```sh
# CPU only, no MLX/GPU:
python3 /private/tmp/minivllm-engine-optimization-20261014/v49-nvfp4-sg4-geometry-build1/check_cpu.py
python3 /private/tmp/minivllm-engine-optimization-20261014/v49-nvfp4-sg4-geometry-build1/micro.py cpu --out NEW_CPU_DIR
# Root only, under external owner; defaultsAC. Setstandard50/50/20/0 and usefreshout:
/Users/ryanli/code/minivllm/.venv/bin/python -B /private/tmp/minivllm-engine-optimization-20261014/v49-nvfp4-sg4-geometry-build1/micro.py run --out NEW_NUMERIC_DIR --qualify-only --power-domain 'AC Power'
# Samecommand omit--qualify-only for declaredstationarymicro; Batteryrequiresliteral 'Battery Power'.
```
No author MLXimport/GPU execution; localnumeric/performance/model/E2Equalification allUNKNOWN untilrootactual.
