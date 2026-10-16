# V53 fixed longer-prime SG8 experiment (CPU preparation)
Exact V52 SG8 geometry.py/Metal math/launch/input handles are retained. Only per-slot prime changes2048→8192, recorded as4×2048 contiguous loops. The measured loop remains2048 calls. Each output is evaluated; no input/output/compile policy changes. Original36 finite/shape/dtype/raw-byte numeric checks and readonly qualification precede timing.

This tests measurement stability under longer fixed sustained work. The physical source of V49/V52 drift is unknown. It neither fixes/relabels old STOPs nor assumes longer warmup will pass. Chunk timestamps are in memory during prime and only written in pairs.json after both full blocks finish. They cannot affect parameter selection: no adaptive duration/block filtering/retry.

Same10-pair schedule:AA,BB,AB,BA,AB,BA,AB,BA,AA,BB;8 rotating fixed inputs;each slotprime8192/repeat2048. Controls each[.95,1.05],per-arm median[.98,1.02];local reduction>=5% AND>=5/6wins. Resource/power2GiB cache/50-50-20-0/owned170s child cleanup remain unchanged. V52’s old SG4 metadata typo is corrected only in this new protocol; its old files remain exact.

CPU: python3 -B check_cpu.py; native-python -B micro.py cpu --out NEW_CPU_DIR. Root-only actual command, under existing external owner and declared AC domain:
```sh
/Users/ryanli/code/minivllm/.venv/bin/python -B /private/tmp/minivllm-engine-optimization-20261014/v53-nvfp4-sg8-longprime-build1/micro.py run --out NEW_ABSOLUTE_DIR --power-domain 'AC Power'
```
It performs36 strict comparisons before one micro run. Expected~45–55s at saved wall times,not guaranteed;170s timeout retained. No model/fullmodel/E2E/Pareto/adoption qualification; no author MLX import or GPU execution.
