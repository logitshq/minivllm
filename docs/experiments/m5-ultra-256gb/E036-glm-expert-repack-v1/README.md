# E036: offline GLM expert repack

The actual offline copy exited 0 in **502.305588 s**, with producer status
**PASS_BYTES_ONLY**. It converted 62 source shards into 58 output shards:
113,446 tensors become 4,960, including 378 stacks for backbone MoE layers 3–44.
The 203,976,457,080 payload bytes (about 190 GiB) are preserved without numerical
conversion. Original MTP layer 45 tensors remain unchanged and still need the
official loader's MTP stacking path.

The separate, candidate-independent validator reports
**PASS_HEADERS_INDEX_MANIFEST_ONLY**. It read the 62 original and 58 output
headers, both indices, the complete group/part manifest and sidecars. It checked
exact keys, shapes, dtypes, byte ranges, original expert order 0–287, exactly-once
source coverage, full payload-byte accounting and the preregistered layout SHA.
Config and tokenizer files are byte-identical; mixed 4/5/6-bit quantization maps
are unchanged. All original source stat identities and header hashes still match
the plan. It did not reread tensor payloads or import the candidate/model runtime.

The producer's source-full-SHA check, copy-time group/part hashes and independent
output reread are retained in the original receipts. Those establish its byte
copy result; the metadata validator does not independently re-prove payload
digests. Neither result establishes official-loader compatibility, physical
memory fit, inference correctness or speed. **No Pareto point was produced.**

`evidence.tar.xz` contains only explicit metadata members, including execution
stdout/stderr and final/invocation records, exact candidate and wrapper, source
pins, plan, original/output headers, indices, group/part SHA manifest and small
sidecars. Header members end in `.safetensors.header` and contain only the
8-byte length plus JSON. Tokenizer JSON is retained as a checked hash reference.
No weight payload is archived. Every one of 164 members was reread and
SHA-verified; the original metadata was checked again after archiving.

The original RUNNING invocation and preparation-time text are preserved as
historical evidence; final.json and complete.json establish closure. This named
copy has no failed predecessor attempt. See RESULTS.md for global memory
endpoints and the limits of that observation. The offline operation contains no
service stop, reload or configuration change; service identity/health was not
independently sampled by this wrapper.

Archive SHA256: `1cbefb1a4be08ca27a2ada84fb19ba5650f9e3f7cfdfa11b712907a633226c4a`.
Plan SHA256: `323ff8da5d2faf6bb774b1c03f3a52f49198cbb9898f21c91ac25aa546a4c50a`.
