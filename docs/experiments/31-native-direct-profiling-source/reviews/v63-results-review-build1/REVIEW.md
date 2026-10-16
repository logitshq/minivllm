V63 saved-evidence independent CPU audit PASS; no native load, MLX import or GPU recomputation.
Receipt a4459b88…: all69 source pins,93×132=12276 output records exactly equal actual44 and V38 A metadata/SHA;279 original-readonly records,93 reference-readonly flags.
Three requests have32 matching IDs; changed-prefill alters129 seed arrays; once/request131 uploads, no per-step CPU token/state feeding;93 ICB executions/[48,45],48 S pingpong/83 staged.
All3 raw files replayed: identical1625 ordinal/PSO/binding maps; sampled reset/changed each1625 valid exact uint64 deltas, no error sentinel/filtering; unsampled control1625 explicit null deltas.
SDK MTLCounters.h:73–76 provides uint64 timestamp without unit; MTLComputePass.h:33–49 samples encoder processing. Capture has no timestamp calibration/frequency query. Keep raw units.
MTLCommandBuffer.h:286–294 explicitly gives host seconds: control CB68.945833ms, sampled101.538667/124.894750ms. Perturbed diagnostics; no production exclusive fraction or saved-ms inference.
All497 QMV/profile decoded with fp_quantized.h:1248–1255 arg5=K,arg6=N; arg0/1 actual buffer roles/full parameter ranges confirm names and exact U32/U8 payload sizes; unknown0.
| K→N | commands/profile | reset sum(rawdelta) | changed sum(rawdelta) |
|---|---:|---:|---:|
|5120→17408 (MLP gate/up)|128|14296624|14551036|
|17408→5120 (MLP down)|64|7198248|7174170|
|5120→10240 (GDN qkv)|48|3311206|3313249|
|6144→5120 (GDN out + attention o)|64|2918623|2941328|
|5120→6144 (GDN z)|48|2120089|2113629|
|5120→248320 (lm_head)|1|1427667|1416541|
|5120→48 (GDN b/a)|96|1398082|1406585|
|5120→12288 (attention q)|16|1274040|1275126|
|5120→1024 (attention k/v)|32|510291|743669|
53 saved resources: AC/thermal0/pressure1,minfree46%,SwapMB/SO growth0,Pageouts+230 (overlapping owner/child timelines, do not add). Owner79046/child79055 reaped0+external strict absent.
Details: receipt.json + qmv-bindings.csv (all1491 QMV mappings, including unsampled control). Saved evidence replay only; no HTTP/Pareto/adoption claim.
