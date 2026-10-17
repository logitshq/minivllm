# E028 v3: unchanged input prefix and independent concat reconstruction

**PASS.** CPU-only inspection confirms that v3's sequential synthetic-input
materialization preserved every available v2 tensor-file byte. It also confirms
B2→4 and B4→6 cache reconstruction directly from saved source bytes. This is a
supplement to the main v3 numerical audit, not a full-model or performance result.

| Check | Coverage | Result |
|---|---:|---|
| Identical relative paths in v2 and v3 | 14 complete safetensors files | Exact bytes |
| v2 failure snapshots versus the corresponding v3 pre-write boundary | 2 complete files | Exact bytes |
| Independent B2/B4 output reconstruction | 40 full BF16/F32 tensors | Exact bytes and metadata |

All 16 v2 safetensors files are accounted for. The two error-only filenames are
`cases/identity-4-to-6/{OFF,ON}-inputs-on-error.safetensors`; their semantic
counterparts are v3's same-arm `inputs-after-rebuild.safetensors`. They were not
dropped because v3 completed successfully. Total cross-version comparison was
**671,577,268 B per side**, with zero differing bytes. V3 has 108 tensor-file
paths; the 94 paths lacking a same-name v2 counterpart are explicitly listed in
the receipt and are not described as cross-version matches.

The independent reconstruction uses only safetensors headers and original
payload bytes, without importing MLX, the candidate or the screen:

- Keep each old K/V row's complete `[4,1536,256]` BF16 payload in identity order.
- For each incoming K/V row, copy each head's first 935 or 954 positions from
  its `[1,4,1024,256]` source, then append zero bytes to reach L1536.
- Append GDN conv rows as BF16 bytes and S rows as FP32 bytes without arithmetic.
- Check request IDs and lengths against the registered row order.

Both arms' first-evaluation and after-source-write outputs match this expected
K/V/conv/S reconstruction. For B4→6, both arms' after-output-write snapshots also
match the reconstruction with exactly the first scalar of each array replaced
by −32.0. These are **40 full tensor comparisons / 493,928,448 B per side**, with
zero differing bytes. No element sampling, tolerance, conversion or omitted tail
was used. The comparator writes every differing byte to JSONL if a mismatch is
encountered; no difference files were needed for this run.

[Machine receipt](result.json) · [source manifest](inputs-manifest.json) ·
[reproducible stdlib script](compare.py). The manifest hashes 40 actual input
files, including the run/case metadata. Their logical sizes total 1,580,813,994 B;
the source files remain in the primary experiment evidence rather than being
copied into this supplementary directory. Hardlinked paths retain their own
provenance entries.

Result SHA256:
`985c8541eb50b54501b7b2acab1ec26142a5236d1ef400fc808383500ad26ac6`.

Input-manifest SHA256:
`91e3613d12550ca4b55093a793185fa22e8af3bca82e4a387e0924f22ba5b1cc`.

With the retained `.artifacts/m5-ultra-256gb/identity-refill-v{2,3}-owner`
directories present, rerun without overwriting this receipt:

```sh
python3 -B docs/experiments/m5-ultra-256gb/E028-identity-refill-v3-cross-version/compare.py --out /tmp/e028-cross-version-new
```

V2 remains a resource failure. This supplement does not replace the main v3
audit of all ten cases, source identities, resource observations or service
lifecycle. It supplies no latency/throughput sample or new Pareto point.
