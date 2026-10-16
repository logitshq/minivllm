# V19 fullqual build3 independent result review: FULLMODEL_QUAL_PASS
Final SHA256: 3d8ed2124007e2ee62356b94f18d74dce8f6be834d916858e082964aad697093
Release SHA256: 3daa0d8bc3f7666afc121db3d323da3de5ba28a311b27a0bbeba90bfdecbf47b
One original stdlib CPU replay: exit0, stderr literal empty; stdout SHA256 a705594fa5b2ae131ca75e7da1d31e3f5f214af2cbf4c1e21b829fe99215d9ab.
Actual accepted/OFF/ON/reset-OFF: four arms ×32 expected IDs; all 128 retained FP32 logits files compared byte-for-byte.
All 16,384 state signatures and complete cache/observation signatures equal; full raw states were not permanently retained.
ON: six production routes each 48 layers ×31 =1,488 calls; prework/state each created one real kernel; legacy absent.
OFF/reset-OFF use legacy; 18 distinct fallbacks strictly equal with no production fusion dispatch.
1,349 fingerprints per worker unchanged at required boundaries; crossworker stable fields equal; Python object IDs are process local.
Original replay validated all 1,431 frozen bytehash pins; accepted25/candidate26 actual source bytehashes separately checked.
Lifecycle: one load per worker, default OFF, fresh/idle caches, reset factories and hooks restoration verified.
All 22 actual resources: fixed AC, pressure1, thermal0, swap7571.62MB; Swapouts delta0.
Pageouts rose63979→64543 (564); original guard uses pressure/ownership and sustained swap/Swapouts, not a zero-Pageouts rule.
Raw written before guard; load+synchronize→modelready guard→token0 and serial worker ordering verified from pinned code/raw.
Actual firstfailure null and cleanup arrays empty; prior build2 STOP/pressure2/cleanup warning and actual stderr warnings preserved.
Session99715 tool return exit0; driver8236 and workers8249/8407 waited/reaped exit0.
Posttool actual ps8222/8236/8249/8407: rc1, stdout="", stderr=""; raw ps and every referenced SHA link checked.
Raw owner/driver GPU_released=false preserved; external posttool GPU_RELEASED is a separate boundary.
Scope: bounded fullmodel correctness only; no E2E/performance/default-on/adoption/promotion/reload/spec-inference/overall-win qualification.
Blocks: []
