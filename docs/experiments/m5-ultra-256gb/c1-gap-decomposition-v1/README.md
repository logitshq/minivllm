# C1 D512 gap: closed HTTP timestamp decomposition

Read-only diagnostic of the existing, audited Qwen3.8-27B NVFP4 D512 measurements. No new model execution, service change, performance point, or change to v13 occurred.

Each engine contributes three processes with 64 formal requests each, all capped at 512 output tokens. The additive decomposition uses **arithmetic request means**, not the process-median p50/p95 values in the Pareto chart.

| Request mean (seconds) | Native E024 ON | oMLX | Native minus oMLX |
|---|---:|---:|---:|
| Request start to DONE | 11.174634 | 10.083155 | 1.091480 |
| Request start to first nonempty output | 0.635892 | 0.563829 | 0.072062 |
| First nonempty output to DONE | 10.538743 | 9.519325 | 1.019417 |

The last interval accounts for **93.3977% of the mean end-to-end gap**. It includes model computation, scheduling, evaluation waits, text decoding, SSE encoding/transport and client receipt of DONE. It is not GPU decode time, a kernel attribution, token ITL, or a causal paired profile. The two engines ran in separate cycles. First output is the client receipt of a nonempty text or reasoning string; all 384 observed requests used the text channel. Streaming aggregation differs, so event counts are not model token counts.

`reproduce.py` reads the six original local raw groups, checks their SHA against the closed PASS audits, and writes compact request/process/engine summaries. The existing source archives retain the large raw payloads. Run from the repository root, directing output to a new directory:

```sh
python3 docs/experiments/m5-ultra-256gb/c1-gap-decomposition-v1/reproduce.py --repo "$PWD" --output /tmp/c1-gap-recheck
```

[Interpretation and next falsifiable experiment](../../../optimization-guide/tutorials/030-c1-gap-diagnosis.md). [Unchanged Pareto](../../../optimization-guide/progress/2026-10-17-v13/README.md).
