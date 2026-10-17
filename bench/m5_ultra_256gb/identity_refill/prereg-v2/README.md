# E028 v2 preregistration: input-generator operator repair

V1 stopped before any rebuild/comparison because MLX has no reverse bitwise OR.
V2 only swaps the two OR expressions to array-first and updates version bindings.
Candidate, finite bit pattern, 10 cases / 88 checks / 53 phases, observation limits
and default-OFF scope are unchanged. The failed v1 sources and evidence remain.

23 CPU tests pass; new mocks omit reverse OR and reproduce both original errors.
The complete fixed inputs are in plan.json; receipt.json records source checks.
Plan SHA256: `26ebdcf2ee6fb09b0c86a15b3bb87a7d221ba67811a4ccf382ae150d3b495b42`.
Jobs SHA256: `9f27f71b5986b9a88e1c3ddfc5ed20a0439ce90da01ac43a6ebdbea97a4572a7`.
573 source/runtime pins. No device result exists at preregistration.
