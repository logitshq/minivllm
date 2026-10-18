# Combined GDN + graph: formal CPU preparation

The factory's 12 checks passed. The lifecycle and independent-auditor suite's second full attempt passed all 58 checks (19 owner, 15 controller, 11 restoration and 13 formal-audit). These are 70 distinct CPU checks, not numerical, actual HTTP-history or performance qualification.

The first 58-check attempt remains intact: 53 passed and five failed because its runner rejected asyncio's SIGINT-handler installation before reaching the mocked HTTP body. The v2 runner suppresses only asyncio runner SIGINT handler changes without touching the OS; it retains the other process, network, signal and backend-import guards. All production and test sources stayed unchanged between these two attempts. Both runners, source snapshots, commands, results and full stdout/stderr are retained.

The initial factory12 ran with an UNSEALED cycle-auditor pin. A separate recorded seal replaced that one literal with the actual frozen auditor SHA, then verified the 42-source closure. It did not change factory arithmetic, protocol or lifecycle logic. The final factory source was also included in the subsequent 58-check source-before/source-after snapshots; the original 12-check source and outcome have not been relabeled as a rerun.

Both formal arms use the same combined server with route graph ON; GDN prework alone changes OFF/ON. Each fresh process has eight warm and 64 formal requests, three paired repetitions at C1 and C8 (12 processes; 864 total requests, 768 formal). The factory requires the independently qualified HTTP histories; these CPU receipts do not satisfy that gate. No formal timing or observed compiled-call claim is made here.

`execute.py` is retained as an additional source snapshot only. It was not in these suites' tested source registry, and this archive does not claim its execution was tested. Actual owners, journals, model weights and runtime payloads are excluded.

`raw.tar.xz` retains the complete four CPU/seal evidence trees and one packer-preflight failure tree and the explicit small source closure. `manifest.json` maps every regular member to its original path, size and SHA-256. Archive verification streams and checks all members without extracting, then checks the original tree memberships and bytes again. Symlinks, special members, duplicates, absolute paths and traversal are rejected. No original evidence was modified or tests rerun.

```sh
python3 archive.py --verify .
```

Optional `--originals` additionally rechecks the source/CPU files at their original absolute paths. Later intentional changes can invalidate that optional comparison without invalidating the archived bytes. The manifest, verification, summary, README and packer copy remain alongside the tar.
