# V49/V52 timer and control CPU diagnosis
1. Replayed all actual10pairs×2=20 timed envelopes and20 prime envelopes per run; no16-envelope subset was selected.
2. Old timer is v22-nvfp4-qmv-r2-build1/micro.py:168–186; V49/V52 override its private PRIME/REPEATS globals to2048/2048 at micro.py:32–34.
3. Each block invokes the same arm2048 times, synchronizes, then times2048 calls plus final synchronize; prime duration is outside the timed wall.
4. invoke at both micro.py:211–214 calls the compiled function on pool[j%8],mx.eval(value),then retains its final result; both prime/repeat counts divide8.
5. Cold compile/first eval is outside timing(:176–185);9-input qualification follows(:186–193); traces remain A1/B1 through timing. No evidence of Python retracing in a timed block.
6. Both arms share immutable weight/scales handles; actual36 strict byte/finite comparisons and all pre/post readonly pass. A_fn/invoke AST and all9 input fingerprints match between49/52.
7. Pair probes precede both blocks; hashes/probes/file writes follow both(:215–228),so no between-slot hash/resource asymmetry was found.
8. Exact gate source v22:101–103 AND both protocol.json files: each ratio[.95,1.05],each arm’s two-control median[.98,1.02]. There is no .975 individual bound.
9. V49 B controls .896073419/.996988852:median.946531135 and first ratio<.95,correct CONTROL_DRIFT_STOP; A median1.000960646 passes.
10. V52 A controls .974165982/.985597845:median.979881913<.98,correct CONTROL_DRIFT_STOP; individual ratios pass. B median1.000797331 passes.
11. Descriptive medians49:A217.760803/B215.392130us(+1.087741%,3wins);52:A232.758596/B220.343211us(+5.334018%,5wins). Neither is a qualified performance win.
12. The identical A baseline is6.887278% slower in52; B is2.298636% slower. Between-run subtraction cannot establish an SG8 benefit or its cause.
13. Concrete drift example49 firstBB:prime240.601/timed225.712us then prime200.326/timed202.254us;52 finalAA:prime234.614/timed208.603 then205.615/205.599us.
14. Every timed block still had its2048-call prime; no fixed-count warmup stationarity test or subwindow timings were recorded. These trends motivate measurement exploration but do not prove insufficient warmup.
15. Source/raw timestamps show pair gaps~.19–.215s containing the two external probes,versus between-slot gaps~17–31us. Probe-gap causality,clock state or GC interference is unmeasured.
16. Host wall includes Python/compiled API/allocation/eval/wait. There are no GPU-only timestamps,frequency telemetry or intra-block samples; no concrete math/compile/consumption/timing bug explains the physical drift.
17. One metadata typo:V52 protocol timing.B still says SG4; actual geometry.py/runtime/qualified threadgroup are SG8/256. It does not change executed kernel or timing.
18. Old sources/results remain untouched. A new53 could test longer fixed prime with the same gates and schedule; it would be a new protocol exploration,not a repair/reclassification of49/52.
