# V34 native fixed-capacity feedback numerical qualification

Root CLI: `.venv/bin/python fullqual.py --out NEW_ABSOLUTE_DIRECTORY --tokens 32`, controls50/50/20/0 and exact native1 DYLD injection at process startup. One accepted25load, 48gatesON, conv/whole-functionalOFF,2GiB allocator. Root owns soleGPU, monitor, timeout, and strict process cleanup.

After one ordinary32warm, capture one max_len256 decode graph including132outputs: logits+128states+MLX argmax(logits).astype(int32)+positions+1+lens+1. Register131inputs and131GPU feedbacklinks. All source/NumPy/memoryview/bitview owners stay pinned; no weight byte snapshots or copies.

Three predeclared requests: base19prompt/32outputs; reset-same19/32outputs; changed20prompt obtained by inserting base[3] before base[10], up to32outputs honoring actualEOS. A runs original model exactL and feeds its own previous greedy prediction. B receives all131seed ranges once per request, then consumes its own capturedGPUargmax/state/position/lens feedback; no perstep writes or savedfuture token input. CanonicalIDs only assert base/reset. Changed-prefill state inputs must actually differ.

Every step compares all132outputs (shape/dtype, finite reference, Cmemcmp raw equality). Cache/logits outputs never make full CPUcopies; only4Bnativeprediction is copied for observation. Original131captureseed,132captureoutputs, and currentrequestseed are checked by Cmemcmp against once-only immutable raw snapshots. Reference retainedstate readonly uses streaming SHA over pinnedmemoryview without full byte copies. Native counters prove exact reset/step/upload/compare counts and no perstepCPUrestore.

Original resourceguard raw precedesderived data: beforeload, afterload-before-token, warm/capture/request boundaries, each4decode,finally. This is an instrumented correctness qualification; no speed,service,ICB,or adoption claim. Native has safe stagedGPUstate feedback; eliminating this GPUcopy is future work requiring alias/lifetime proof. `--inert` checks pins without loading MLX/model/native library.
