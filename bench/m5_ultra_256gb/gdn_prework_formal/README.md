# GDN-only paired formal timing

Prepared and source-bound to the real GDN HTTP interfaces. The full guarded CPU suite actually passed 93/93; see [retained first-run evidence](../../../docs/experiments/m5-ultra-256gb/gdn-http-formal-cpu-v1/README.md). No formal timing result exists at this checkpoint.

Actual execution requires the completed GDN C1/C8 capture/replay, separate independent numerical/resource/restoration PASS, and the frozen archive. OFF uses the native entry. ON selects GDN prework only; trace and dispatch observers are disabled, with E024 and E028 OFF. The original 8 warm +64 formal requests, three fresh pairs for each of C1/C8, normal EOS and fixed D fixtures remain.

See [lifecycle instructions](LIFECYCLE.md) for exact preparation, registration, supervised execution and closure. Successful preparation cannot enter Pareto.
