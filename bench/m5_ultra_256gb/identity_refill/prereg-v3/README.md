# E028 v3 preregistration: bound synthetic input construction

v2 completed B2→4 and 11 checks before its historical MLX peak reached
1,131,872,976 bytes at B4→6, exceeding the unchanged 1 GiB observation gate.
No mismatch was observed in those completed checks; remaining coverage is absent.

v3 evaluates each generated input parent separately, before returning the plain
array or constructing the identical offset/strided view. All input data were
already evaluated before rebuild in v2; rebuild/output evaluation remains unchanged.
No peak reset, contiguous conversion, shape reduction or budget relaxation.
Candidate, 10 cases / 88 checks / 53 phases and mutation contract stay frozen.

26 CPU tests pass, including actual value() chronology and unchanged pattern/run_case AST.
Independent review found no static blocker. Compare all overlapping v2/v3 saved
input bytes after execution. No device result exists at registration.

Plan SHA256: `fbcf114be8507484d30b2ceb367e5864f2a2928c475423d3a30192c23cfc72a7`.
Jobs SHA256: `1e9aade04e219aea9b1dfad21cdb61d65d7f33a6751208fb00df48cca43d495f`.
576 source/runtime pins. Full-model and performance qualification remain false.
