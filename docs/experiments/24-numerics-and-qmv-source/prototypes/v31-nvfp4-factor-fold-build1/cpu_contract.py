"""Exhaustive scalar value/sign math and exact two-edit source contract; no MLX."""
import ast
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
from fractions import Fraction

BASE = Path(__file__).resolve().parent


def f32(value):
    try: return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError: return math.copysign(math.inf, value)


def bits32(value): return struct.pack("<f", value).hex()


def fp8(code):
    if code & 127 == 127: raise ValueError("nonfinite E4M3 code")
    exponent, mantissa = (code >> 3) & 15, code & 7
    value = Fraction(mantissa, 512) if exponent == 0 else Fraction(8 + mantissa, 8) * Fraction(2) ** (exponent - 7)
    return -value if code & 128 else value


def main():
    checked = 0
    signed_zero = 0
    magnitudes = (Fraction(0), Fraction(1, 2), Fraction(1), Fraction(3, 2), Fraction(2), Fraction(3), Fraction(4), Fraction(6))
    for nibble in range(16):
        raw_bits = ((nibble & 7) << 9) | ((nibble & 8) << 12)
        raw = struct.unpack("<e", struct.pack("<H", raw_bits))[0]
        native = struct.unpack("<e", struct.pack("<e", raw * 16384))[0]
        expected = -magnitudes[nibble & 7] if nibble & 8 else magnitudes[nibble & 7]
        assert Fraction.from_float(native) == expected and Fraction.from_float(raw) * 16384 == expected
        for scale_code in range(256):
            if scale_code in (127, 255): continue
            value = fp8(scale_code)
            scale = math.copysign(float(value), -1 if scale_code & 128 else 1)
            folded_scale = f32(scale * 16384.0)
            assert math.isfinite(folded_scale) and Fraction.from_float(folded_scale) == value * 16384
            left = f32(native * scale)
            right = f32(raw * folded_scale)
            assert bits32(left) == bits32(right), (nibble, scale_code)
            signed_zero += int(left == 0)
            checked += 1
    source = BASE.parent / "v22-nvfp4-qmv-r2-build1/kernel_header.metal"
    native_header = source.read_text()
    expected_header = native_header.replace("    converted *= 16384.0;\n", "").replace(
        "    return T(*(thread fp8_e4m3*)(&s));", "    return T(float(*(thread fp8_e4m3*)(&s)) * 16384.0f);")
    assert (BASE / "folded_header.metal").read_text() == expected_header
    assert (BASE / "kernel_body.metal").read_bytes() == source.with_name("kernel_body.metal").read_bytes()
    for path in BASE.glob("*.py"): compile(ast.parse(path.read_text()), str(path), "exec")
    # Scalar weight equality does not imply equivalence through every possible
    # FP32 overflow. This explicit finite-BF16 activation is a counterexample.
    extreme_x = struct.unpack("<f", struct.pack("<I", 0x7f7f0000))[0]
    native_intermediate = f32(extreme_x * 2.0)
    native_final = f32(native_intermediate * float(fp8(1)))
    folded_intermediate = f32(extreme_x * (2.0 / 16384))
    folded_final = f32(folded_intermediate * (float(fp8(1)) * 16384))
    assert math.isinf(native_final) and math.isfinite(folded_final)
    assert not any(n == "mlx" or n.startswith("mlx.") for n in sys.modules)
    report = {"status": "PASS_CPU_SCALAR_AND_TWO_EDIT_CONTRACT", "nibble_scale_pairs": checked,
              "finite_scale_codes": 254, "excluded_nonfinite_scale_codes": [127, 255],
              "zero_pairs_with_sign_preserved": signed_zero, "raw_fp4_min_nonzero": 2 ** -15,
              "folded_scale_min_nonzero": 32, "folded_scale_max_abs": 7340032,
              "FP32_scale_required": True, "header_only_two_edits": True, "R4_body_unchanged": True,
              "generic_all_finite_activation_lossless_claim": False,
              "overflow_counterexample": {"activation_BF16_hex": "7f7f", "fp4_nibble": 4, "scale_code": 1,
                  "native_intermediate": "Infinity", "native_final": "Infinity", "folded_final_FP32_hex_le": bits32(folded_final)},
              "half_denormal_gpu_conversion_and_FTZ": "UNKNOWN until actual strict GPU gate",
              "MLX_imported": False, "GPU_used": False, "numeric_gpu_and_performance": "UNKNOWN"}
    print(json.dumps(report))


if __name__ == "__main__": main()
