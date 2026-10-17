# E024 fixed long-decode graph diagnostic v1

Run only after the current E023 GPU owner has finished and its service lease has restored the external service. This is a separate numerical diagnostic, not an HTTP benchmark or Pareto point.

Two independent Qwen NVFP4 BF16 engines retain gates/norm/packed fusion OFF. The candidate alone uses the opt-in1536 graph. Fixed direct points are B1 through8 at used257/1024/1280/1536; inputs explicitly repeat/truncate the frozen D token IDs. Actual compiled callable invocation is required at every diagnostic point. Full logits and64-layer128 cache arrays must match exact bits; fallback cannot pass.

Changed and restored norm/packed-weight/scale values run B1→B8→B1 through the same compiled functions. A separate8-request length1533/cap4 normal-EOS lifecycle must show actual B8,1534–1536 and ahead production/consumption. Its backing rounds to1792. Insufficient EOS coverage fails instead of forcing tokens. This is not the complete D512, gates-ON, mixed-length or nonzero-padding qualification.

The fixed qualifier plan and a separate owner plan reuse real E023 source/runtime/model identity evidence. No synthetic inherited plan is substituted. The owner retains the original deadline, resource checks, direct child reap and separate service-lease restoration. Absolute budgets are child1800s and owner1830s plus bounded cleanup. Sources and CPU tests are frozen before dispatch.
