# Profile CPU evidence: two actual runs, no rerun

Both original whole-module fake CPU runs passed 11/11 with no errors, failures or
skips (0.056 s and 0.053 s). The exact guard commands and complete tool-combined
output were recovered from the original local session records. They are preserved
under attempt-1/ and attempt-2/. This directory was assembled later; no tests ran
during assembly. UTC timestamps are October 18; local PDT execution was October 17.

The tool merged stdout/stderr, so separate streams were not retained and are not
reconstructed. The second command printed three production/test source SHA256s;
all match final-sources/. The first run preceded small source-identity checks and
had no contemporaneous source snapshot/hash. This evidence does not retroactively
pin its source to the final version. The final README copy is documentation written
after the tests. Full exact command bodies preserve each import/effect guard.

First extraction stopped at a metadata-format assertion (tool a3fd2e): unlike
first chunk c842ec, the second tool result exposed plain output without a chunk
ID. Its original CommandExecution ID/output and exact tool text are retained;
receipt.json uses null for the unavailable chunk. This was an assembly error,
not another test attempt. No tests were reexecuted to manufacture evidence.

These results prove only the explicitly fake CPU tests. They do not run MLX, a
model, a service or HTTP device qualification, nor measure diagnostic overhead or
produce formal performance evidence. manifest.json hashes every other retained
file. Original transcript reasoning and unrelated session content are excluded.
