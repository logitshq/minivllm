"""V23 fixed-Y4 conv/SiLU/tail. Default-off engine experiment; MIT helpers."""
import importlib.metadata
from functools import lru_cache
import mlx.core as mx

HEADER = '// Typed bodies pinned to MLX 0.32.3 unary_ops.h and binary_ops.h.\nstruct V18NativeSigmoid {\n  template <typename T>\n  T operator()(T x) thread {\n    auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));\n    return (x < 0) ? y : 1 - y;\n  }\n};\nstruct V18NativeMultiply {\n  template <typename T>\n  T operator()(T x, T y) thread {\n    return x * y;\n  }\n};\nstruct V18BF16Activation {\n    bfloat16_t sigmoid_value;\n    bfloat16_t activated;\n};\ninline V18BF16Activation v18_bf16_activation(bfloat16_t u) {\n    bfloat16_t sigmoid_value = V18NativeSigmoid()(u);\n    bfloat16_t activated = V18NativeMultiply()(u, sigmoid_value);\n    return {sigmoid_value, activated};\n}\n'
SOURCE = "// V23 prototype: MLX metal_kernel body, not a standalone Metal translation unit.\n// SPDX-License-Identifier: MIT\n// See LICENSE-MLX-MIT for the MLX-derived typed arithmetic helpers.\n// SUM_Y is the measured reference reduction's logical y count, not this grid's y.\nuint channel = thread_position_in_grid.x;\nif (channel >= 10240) {\n    return;\n}\n// One physical owner evaluates the declared logical y-local folds in order.\n// No FP32 accumulator, atomics, input writes, or cross-channel sharing.\nbfloat16_t partial[SUM_Y];\nfor (uint y = 0; y < SUM_Y; ++y) {\n    bfloat16_t total = bfloat16_t(0);\n    for (uint tap = y; tap < 4; tap += SUM_Y) {\n        bfloat16_t input_value = (tap < 3)\n            ? old_tail[tap * 10240 + channel]\n            : raw[channel];\n        bfloat16_t product = V18NativeMultiply()(input_value, w[channel * 4 + tap]);\n        // Match Sum<bfloat16_t>::operator()(product, total), including +zero.\n        total = bfloat16_t(product + total);\n    }\n    partial[y] = total;\n}\nbfloat16_t conv = partial[0];\nfor (uint y = 1; y < SUM_Y; ++y) {\n    conv = bfloat16_t(partial[y] + conv);\n}\nact[channel] = v18_bf16_activation(conv).activated;\n// New output storage only; old_tail and raw remain immutable.\nnew_tail[channel] = old_tail[10240 + channel];\nnew_tail[10240 + channel] = old_tail[20480 + channel];\nnew_tail[20480 + channel] = raw[channel];\n"

@lru_cache(maxsize=1)
def _identity_supported():
    try:
        if any(importlib.metadata.version(n) != "0.32.3" for n in ("mlx", "mlx-metal")):
            return False
        info = mx.device_info()
        return (info.get("device_name") == "Apple M4 Max"
                and info.get("architecture") == "applegpu_g16s")
    except Exception:
        return False

def backend_supported():
    return mx.default_device() == mx.gpu and _identity_supported()

@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(name="v23_gdn_conv_silu_raw_tail",
        input_names=["raw", "old_tail", "w"], output_names=["act", "new_tail"],
        source=SOURCE, header=HEADER, ensure_row_contiguous=True,
        compile_options={"math_mode": "safe"})

def conv_silu_raw_tail(raw, old_tail, w):
    if not (raw.shape == (1,10240) and old_tail.shape == (1,3,10240)
            and w.shape == (10240,4)
            and raw.dtype == old_tail.dtype == w.dtype == mx.bfloat16):
        raise ValueError("V23 fixed BF16 conv metadata")
    return tuple(_kernel()(inputs=[raw, old_tail, w], template=[("SUM_Y",4)],
        grid=(10240,1,1), threadgroup=(256,1,1),
        output_shapes=[(1,10240),(1,3,10240)], output_dtypes=[mx.bfloat16]*2))
