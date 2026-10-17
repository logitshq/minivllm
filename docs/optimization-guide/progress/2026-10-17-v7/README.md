# Route-preserving graph: forward-only progress

The figure compares our own eager and compiled graph at eight preregistered
configurations on M5 Ultra 256 GiB, Qwen3.8-27B NVFP4. Both panels share a zero-based
millisecond scale. Bars are medians; whiskers are observed min–max across six
formal samples per arm. These ranges are **not confidence intervals**. Three ACCA
blocks follow four excluded warm samples per configuration, all within one process.

This measures forward + evaluation of all129 outputs + synchronization, with
copies/observers/checks/teardown excluded. It does not measure committed output
tokens or complete request latency; **do not add these values to the HTTP Pareto**.
The rate-gain table and exact correctness/restoration limits are in the
[result](../../../experiments/m5-ultra-256gb/E024-route-timing-v1/README.md).

[PNG](route-graph-forward.png) · [SVG](route-graph-forward.svg) ·
[Chart data with all formal samples](chartdata.json) · [Renderer](render.py).
Renderer verifies the committed audit hash and recomputes each median from samples.
Run with a Python environment containing matplotlib; on this machine:
`.artifacts/m5-ultra-256gb/progress-render-venv/bin/python docs/optimization-guide/progress/2026-10-17-v7/render.py`.

[Actual D512 throughput–latency chart](../2026-10-17-v6/README.md) remains the two
native baseline observations. Lithos/oMLX same-workload D512 points are still pending.
