# V27 micro interpretation — CPU only
- Actual micro status remains CONTROL_DRIFT_STOP; no rerun or loosened controls.
- Descriptive A142.213562us / B122.484975us gives13.872507% local host-wall reduction,6/6wins. The19.728587us difference×48layers is0.946972ms/token before whole-model interactions, not≥2ms.
- AA ratios0.970083866/0.972558050 and BB0.983116192/0.972059761 show a systematic faster second slot; their medians violate fixed0.98..1.02 controls.
- All blocks used the same128prime/2048repeat. First-slot prime wall was about30–38ms; second-slot15–19ms. First slots followed pair-outside resource/hash work; second slots followed a timed block. Thermal sampling cannot distinguish GPU frequency, caches, residency or host scheduling within those windows.
- Per-call wall includes Python shape/branch checks, Metal encode/dispatch, mx.eval and synchronization. Receipt has no GPU timestamps or empty-dispatch control; it cannot prove hostAPI dominates or quantify GPU kernel savings.
- Candidate reads canonical S with key-lane stride128floats on first step, but persistent physical Dv×Dk with adjacent key entries thereafter. The micro intentionally measured the latter. S_strides metadata lookup is not itself proof of a slow memory access.
- Register-state kernel removes source-level second state load, explicit barriers, qk repeat/FP32 materialization and redundant full norm scans; compiler reuse, DRAM traffic, occupancy and actual timing remain unmeasured.
- Root explicitly authorized independent defaultOFF fullmodel numeric exploration despite local control STOP. Actual full32greedy/finite/readonly/fixed-error gates and eventual fresh whole-model measurements decide further work; this document does not promote it.
