"""Actual-device and upstream small-kernel checks; no full-model equivalence claim."""
import json
import sys
from pathlib import Path

import pytest
from monolith.runtime import _native as nt
from monolith.backends.metal import config_for_device, get_backend
from transformers import AutoTokenizer

root = Path('/Users/ryanli/code/lithos-metal')
out = Path(sys.argv[1])
assert not out.exists()
info = nt.Device().info()
identity = {k: getattr(info, k) for k in ['name', 'gpu_cores', 'apple_family', 'recommended_working_set']}
assert (info.name, info.gpu_cores, info.apple_family) == ('Apple M5 Ultra', 80, 10)
profile = config_for_device(info.gpu_cores, info.apple_family, info.name)
get_backend(profile.backend).validate_device(profile, info)
assert profile.name == 'apple-m5-ultra-80c'
targets = [
    'tests/kernels/test_nvfp4_scale_decode.py',
    'tests/kernels/test_gemm_tile.py::test_cooperative_layout_matches_the_fill_formula',
    'tests/kernels/test_gemm_tile.py::test_gemm_tile_matches_oracle',
    'tests/kernels/test_gemm_tile.py::test_gemm_tile_bf16_output_and_k2048',
    'tests/kernels/test_gemm_tile.py::test_gemm_tile_block_scale_placement',
]
code = pytest.main(['-q'] + [str(root / p) for p in targets])
assert code == 0, code
workload = json.loads(Path('/Users/ryanli/code/minivllm/.artifacts/m5-ultra-256gb/fullqual-build2/workload.json').read_text())
tokenizer = AutoTokenizer.from_pretrained(workload['model_dir'], local_files_only=True, trust_remote_code=False)
ids = tokenizer.apply_chat_template([{'role': 'user', 'content': '用一句话解释什么是 KV cache。'}],
    tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)
assert ids == workload['prompt_ids']
out.write_text(json.dumps({'status': 'PASS_DEVICE_KERNELS_TEMPLATE', 'device': identity,
    'profile': profile.raw, 'native_module': nt.__file__, 'pytest_exit': int(code), 'tests': targets,
    'prompt_ids': ids, 'scope': 'Upstream finite-scale, Apple10 tensor layout, quantized GEMM oracle and BF16/block-scale checks. Full-model bitwise equivalence to other engines is not claimed.'}, indent=2) + '\n')
print('PASS_DEVICE_KERNELS_TEMPLATE', flush=True)
