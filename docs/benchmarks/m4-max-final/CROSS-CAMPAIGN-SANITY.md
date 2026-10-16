# M4 cross-campaign sanity: arithmetic/batching valid; campaigns not normalized
Read-onlyCPU audit; no GPU/MLX/probe/benchmark/source changes; all21new-process raw retained.
New/old C2 ratios: native x0.503952/y0.517135; oMLX x0.475539/y0.502184. C1 medians remain near1x.
Hard denominator check: each C2group has2requests/64actualtokens; C4 has4/128; POSTmin→DONEmax recomputes storedy exactly.
Mature request/group source byte-exact; C2 starts within0.287ms; first chunks1token; no barrier serialization/missing-factor2 found.
Native max_seqs4 and real warm B2/B4 confirmed; onlymodel/gdn.py changed fromoldV7, other24/25 inclrunner/scheduler/cache/sampler identical.
Native e.step/pipeline loop unchanged; warm hooks restored beforemeasure; no new pertoken eval/synchronize hook found.
oMLX pinned source/native tree unchanged; actual relevant settings match: prefill1/completionC, balanced default.03, cachefalse, chunkedprefillfalse.
New controls explicitly50/50/20/MLX_METAL_FAST_SYNCH0 for both engines; old client inherited env and did not record these actual values.
Old MLXcontrols are UNKNOWN, not proven unset/default/same; cannot attribute observed .5x tobuffers or pool old/new points.
OldC1/C2 thermal0; newC1 first2blocks thermal0; newC1block3OM and allC2/C4 thermal1; bothpressure1/AC, per-arm swapstable.
NewC1 block3 nativey19.597/OM15.016 vsprior~25; medians conceal shared late-run variation, so show replicaranges/allsigns.
NativeC2 measured stepwall~69–83ms versusold~36.96ms while retainingbatch2/64tokens: genuine execution-wall slowdown, not denominator artifact.
Additional context differs: old1freshscreen vsnew3replicas, resource5→17samples/process and group-idle gaps; probes are outside timedwindows.
No GPUclock/power/temperature telemetry: quantitative thermal/power/buffer/systemactivity cause remains UNKNOWN; fair1 was allowed by originalguard.
No concrete client arithmetic/scheduler/cache/pipeline bug requiring code repair found; this is a bounded audit, not elimination of allconfounders.
Required reporting fix: label common explicitruntimecontrols and thermal0→1, retainallreplicas; do not claim matched historical regression or stock-runtime speedup.
details.json SHA256 ea79b4b931df76b66f24b80e31e671d3b399aaeb13086bc2b7df996ee06e4ab7; source/report/reference hashes and rawgroup/step evidence included.
