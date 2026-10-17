"""E025 isolated same-split NVFP4 partial prototype; OFF by default.

No runtime import, device compilation or model loading occurs on import.
The official 0.32.3 SIMD and NAX helpers live in separate Metal libraries.
Only the NAX helper's name, K_eff parameter and loop bound are changed.
"""
import hashlib
from importlib.metadata import distribution, version
import json
from pathlib import Path
import re


SHAPES = (("gdn_z", 5120, 6144), ("mlp_down", 17408, 5120), ("out", 6144, 5120))
NORMAL_M = (19,)
FALLBACK_M = (1, 11, 12, 32, 33, 128)
PINS = Path(__file__).with_name("installed-sources.json")


def geometry(M, K, N):
    if any(type(v) is not int for v in (M, K, N)):
        return None
    if M not in NORMAL_M or (K, N) not in {(k, n) for _, k, n in SHAPES}:
        return None
    split = min(max(1, 512 // (((N + 31) // 32) * ((M + 31) // 32))), K // 32)
    while split > 1 and K % (split * 32):
        split -= 1
    if split <= 1 or K % split or (K // split) % 64:
        return None
    result = dict(M=M, K=K, N=N, split_k=split, K_eff=K // split)
    for kind, bn, bk in (("reference", 32, 32), ("candidate", 64, 64)):
        # fast.metal_kernel takes THREAD counts, unlike native dispatch_threadgroups.
        result[kind] = dict(bm=32, bn=bn, bk=bk, threadgroup=(32, 2, 2),
            grid=(((N + bn - 1) // bn) * 32, ((M + 31) // 32) * 2, split * 2))
    return result


def source_prefix(text):
    """Retain Apple's notices/helpers, stopping before unrelated kernel wrappers."""
    start = text.index("METAL_FUNC void fp_qmm_t_impl(")
    first = text.index("{", start)
    # Strip comments/string contents only for brace matching; keep exact offsets.
    blanked = re.sub(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"',
                     lambda m: " " * len(m.group()), text, flags=re.S)
    depth = 0
    for i in range(first, len(text)):
        depth += (blanked[i] == "{") - (blanked[i] == "}")
        if depth == 0:
            return text[:i + 1] + "\n"
    raise ValueError("unterminated official helper")


def nax_partial_helper(prefix):
    start = prefix.index("METAL_FUNC void fp_qmm_t_impl(")
    before, helper = prefix[:start], prefix[start:]
    changes = (
        ("METAL_FUNC void fp_qmm_t_impl(", "METAL_FUNC void e025_fp_qmm_t_partial_nax("),
        ("    const constant int& M,\n", "    const constant int& M,\n    const constant int& K_eff,\n"),
        ("for (int k = 0; k < K; k += BK)", "for (int k = 0; k < K_eff; k += BK)"),
    )
    for old, new in changes:
        if helper.count(old) != 1:
            raise ValueError("official helper seam changed: " + old)
        helper = helper.replace(old, new)
    return before + helper


def installed_root():
    if version("mlx") != "0.32.3":
        raise ValueError("exact MLX 0.32.3 distribution required")
    return Path(distribution("mlx").locate_file("mlx/include")).resolve()


def build_source(kind, g, include_root=None):
    if kind not in ("reference", "candidate") or geometry(g['M'], g['K'], g['N']) != g:
        raise ValueError("fixed geometry and known arm required")
    root = Path(include_root).resolve() if include_root is not None else installed_root()
    expected = json.loads(PINS.read_text())["files"]
    observed, expanded = {}, set()
    def read(relative):
        if relative not in expected:
            raise ValueError("unregistered official header: " + relative)
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("header escaped include root")
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if sha != expected[relative]:
            raise ValueError("installed official header SHA mismatch: " + relative)
        observed[relative] = sha
        return raw.decode()
    def expand(text):
        def include(match):
            relative = match.group(1)
            if relative in expanded:
                return ""
            expanded.add(relative)
            return expand(read(relative))
        return re.sub(r'^\s*#include "([^"]+)"\s*$', include, text, flags=re.M).replace("#pragma once", "")
    base = "mlx/backend/metal/kernels/"
    gemm = "steel/gemm/gemm.h" if kind == "reference" else "steel/gemm/nax.h"
    fp = "fp_quantized.h" if kind == "reference" else "fp_quantized_nax.h"
    header = expand(read(base + gemm)) + "\n" + expand(read(base + "quantized_utils.h"))
    prefix = source_prefix(read(base + fp))
    if kind == "candidate":
        prefix = nax_partial_helper(prefix)
    header += "\n" + expand(prefix)
    header += (f"\nconstant int E025_K = {g['K']};\nconstant int E025_N = {g['N']};\n"
               f"constant int E025_M = {g['M']};\nconstant int E025_K_eff = {g['K_eff']};\n")
    bk, bn = g[kind]['bk'], g[kind]['bn']
    source = f"""
uint3 tid = threadgroup_position_in_grid;
uint lid = thread_index_in_threadgroup;
uint simd_gid = simdgroup_index_in_threadgroup;
uint simd_lid = thread_index_in_simdgroup;
const int k_start = tid.z * E025_K_eff;
// Preserve full K in all matrix row strides. Only contraction length is K_eff.
const device uint32_t* wp = weight + k_start / 8;
const device uint8_t* sp = scales + k_start / 16;
const device bfloat* xp = activation + k_start;
device bfloat* yp = partial + tid.z * E025_M * E025_N;
threadgroup bfloat Ws[{bn} * ({bk} + 8)];
"""
    if kind == "reference":
        source += """
threadgroup bfloat Xs[32 * (32 + 8)];
fp_qmm_t_impl<bfloat, 16, 4, true, false, 32, 32, 32>(
    wp, sp, nullptr, xp, yp, Xs, Ws,
    E025_K, E025_N, E025_M, E025_K_eff, tid, lid, simd_gid, simd_lid);
"""
    else:
        source += """
e025_fp_qmm_t_partial_nax<bfloat, 16, 4, true, false, 32, 64, 64, 2, 2, bfloat>(
    wp, sp, nullptr, xp, yp, Ws,
    E025_K, E025_N, E025_M, E025_K_eff, tid, lid, simd_gid, simd_lid);
"""
    return dict(header=header, source=source, pins=observed)


class SameSplitPartial:
    """One isolated projection. No global patching; caller owns arrays and runtime."""
    def __init__(self, mx, *, enabled=False, include_root=None):
        if type(enabled) is not bool:
            raise TypeError("enabled must be bool")
        self.mx, self.enabled, self.include_root = mx, enabled, include_root
        self.kernels, self.receipts = {}, []

    def _geometry(self, x, weight, scales):
        mx = self.mx
        if (x.ndim != 2 or weight.ndim != 2 or scales.ndim != 2
                or x.dtype != mx.bfloat16 or weight.dtype != mx.uint32 or scales.dtype != mx.uint8):
            return None
        M, K = x.shape
        N = weight.shape[0]
        if weight.shape != (N, K // 8) or scales.shape != (N, K // 16):
            return None
        return geometry(M, K, N)

    def _hardware(self):
        mx = self.mx
        return (version('mlx') == '0.32.3' and mx.default_device() == mx.gpu
                and mx.device_info().get('device_name') == 'Apple M5 Ultra'
                and mx.metal.device_info().get('architecture') == 'applegpu_g17d')

    def original(self, x, weight, scales):
        return self.mx.quantized_matmul(x, weight, scales, transpose=True,
                                       group_size=16, bits=4, mode='nvfp4')

    def partials(self, x, weight, scales, kind='candidate'):
        g = self._geometry(x, weight, scales)
        if g is None or not self._hardware():
            raise ValueError('partial diagnostic outside fixed shape/backend domain')
        key = (kind, g['M'], g['K'], g['N'])
        if key not in self.kernels:
            material = build_source(kind, g, self.include_root)
            name = f'e025_{kind}_m{g["M"]}_k{g["K"]}_n{g["N"]}'
            self.kernels[key] = self.mx.fast.metal_kernel(name=name,
                input_names=['activation', 'weight', 'scales'], output_names=['partial'],
                source=material['source'], header=material['header'],
                ensure_row_contiguous=True, compile_options={'math_mode': 'safe'})
            self.receipts.append(dict(kind=kind, name=name, geometry=g, pins=material['pins'],
                header_sha256=hashlib.sha256(material['header'].encode()).hexdigest(),
                body_sha256=hashlib.sha256(material['source'].encode()).hexdigest()))
        return self.kernels[key](inputs=[x, weight, scales], grid=g[kind]['grid'],
            threadgroup=g[kind]['threadgroup'], output_shapes=[(g['split_k'], g['M'], g['N'])],
            output_dtypes=[self.mx.bfloat16])[0]

    def __call__(self, x, weight, scales):
        if not self.enabled or self._geometry(x, weight, scales) is None or not self._hardware():
            return self.original(x, weight, scales)
        # No mx.compile surrounds this: preserve contiguous [s,M,N], axis0 sum.
        return self.mx.sum(self.partials(x, weight, scales), axis=0)
