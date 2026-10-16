V63 perturbed stage-counter diagnostic qualification

Run only through root sole-GPU resource/timeout/PID owner. The worker has no independent supervisor.
CPU preparation: AST/source69 and --inert passed; no MLX/native/GPU execution by author.

```sh
MLX_MAX_OPS_PER_BUFFER=50 MLX_MAX_MB_PER_BUFFER=50 MLX_BFS_MAX_WIDTH=20 MLX_METAL_FAST_SYNCH=0 \
DYLD_INSERT_LIBRARIES=/private/tmp/minivllm-engine-optimization-20261014/v63-native-stage-profile-build1/libv63replay.dylib \
/Users/ryanli/code/minivllm/.venv/bin/python -B \
/private/tmp/minivllm-engine-optimization-20261014/v63-native-stage-profile-build1/fullqual.py \
--out /private/tmp/minivllm-engine-optimization-20261014/ROOT-CHOOSE-NEW-V63-OUTPUT --tokens 32
```

Three once-seeded requests: base19, reset-base19, changed-prefill20. Step1 uses segmented API sampled0/1/1 respectively; other90 steps use original40 v34_replay_step. All93 steps compare logits,128states,prediction,next-position,next-length (12276 strict checks), preserving reference input/seed/captured-reference readonly checks. Every A hash is cross-checked to actual44 and original38.

raw-profile-{case}.json retains all1625 ordinal rows and exact native JSON bytes. Sampled0 has no timestamps. Sampled1 must have every row valid, finite CB interval, exact command ordinal/PSO/category/bindings map; any invalid sample stops, without filtering or retry. No local timing threshold is imposed. Stage start/end samples include perturbation from one original ICB command per concurrent encoder plus chain fences; raw units are not converted and these are not exclusive production shares. No HTTP/Pareto/adoption claim.
