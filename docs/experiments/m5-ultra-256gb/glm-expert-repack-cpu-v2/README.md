# GLM expert repack: CPU and header-plan evidence

The original guarded CPU attempt passed 11/11 before the MTP-wrapper and Darwin
F_NOCACHE revision. The revised source passed 15/15, including exact source-pinned
sanitizer/quant-alias AST fixtures on fake packed bytes and real tiny uncached
Darwin IO with unaligned offsets and short final blocks. Both attempts report
zero failures, errors, skips or forbidden operations and unchanged sources. There
are **no failed CPU attempts** among these two runs. Earlier source-edit syntax
errors are documented separately in the archived review; no test ran in them.

The actual metadata preparation exited 0 with HEADER_PLAN_ONLY: 113,446 source
tensors become 4,960 output tensors, including 378 backbone expert stacks, while
203,976,457,080 payload bytes are conserved. Plan SHA256 `323ff8da5d2faf6bb774b1c03f3a52f49198cbb9898f21c91ac25aa546a4c50a`. This retains
headers/metadata only; no actual checkpoint payload was copied or evaluated.
MTP layer 45 retains its original bytes and still stacks through the loader.

The archive has 31 regular files, 488,590 raw bytes
and 112,024 compressed bytes. SHA256
`9860434049c2f42a760d80c53a6f8929f95f27dc5732fb26545232237a88442d`. Every archive member, original file and original
directory membership was independently rechecked. Original PASS statuses remain
separate. This establishes archive integrity and tiny byte/layout contracts, not
an official loader run, physical-memory fit, numerical inference or Pareto point.
