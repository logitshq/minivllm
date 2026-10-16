# M5 Ultra Lithos adaptation

Local Lithos source commit `223658090ae549e59220bdec459957cbdad7926a`
adds an exact `Apple M5 Ultra / Apple10 / 80-core` registration and independent
native backend. The source-built runtime and editable installation live at
`/Users/ryanli/code/lithos-metal`. No device check is bypassed and no M5 Max
measured recipe or cost table is copied. The profile remains unmeasured;
the experiment explicitly uses `MONOLITH_ACCELERATOR=on`.

`qualify.py` passed 64 upstream GPU kernel cases and exact 19-token chat-template
matching, after 48 CPU profile/backend contract tests. This is not full-model
bitwise equivalence to MLX. `guarded.py` records resources and owns/reaps only
its qualification child.

`run.py` froze new inputs in `.artifacts/m5-ultra-256gb/lithos-inputs-v1` and
ran nine fresh C1 processes: Lithos/native/oMLX, native/oMLX/Lithos,
oMLX/Lithos/native. Each ran three warm and three measured requests. Native
is the existing qualified V19-on experimental source; its prior internal
DO_NOT_ADOPT decision and default-off setting are unchanged. All engines use
the same checkpoint and 19/32 input/output workload. Native/oMLX use
Completions; Lithos uses Chat Completions with independently checked identical
prompt IDs. The request timing and group-aggregation function ASTs are unchanged
from the original frontier client.

Results and raw SSE timing records remain in
`.artifacts/m5-ultra-256gb/lithos-frontier-v1`. Do not overwrite/retry that attempt.
`replay.py` reconstructs all 54 requests (27 measured) using stdlib only. All
nine services were reaped; the owner and services were absent after tool return.
Native pair2 measured request2 had a system Pageouts increase of 44; it is
retained, not silently excluded or rerun. Pressure remained 1, thermal 0,
swap 0. Another chat was downloading models. Treat the comparison as observed
C1 performance, not a completely paging-stable isolated qualification.

This Lithos OpenAI endpoint rejects concurrent requests with HTTP 429, so no
C2/C4 figures are invented. The local adapter is not an upstream optimized
M5 Ultra release. Full reports and a startup script are archived under
`docs/experiments/m5-ultra-256gb/lithos-adaptation-v1`.
