#!/usr/bin/env python3
"""CPU structural/address checks only. No floating numerical GPU surrogate."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
D = Path(__file__).resolve().parent
P = D.parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    for p in D.glob('*.py'):
        compile(p.read_text(), str(p), 'exec')
    spec = importlib.util.spec_from_file_location('v124_source_helper', D/'shared_m1.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    assert not any(n == 'numpy' or n.startswith(('numpy.', 'mlx')) for n in sys.modules)
    pinned = P/'v113-broadcast-m1-diagnostic-build1/source/mlx__backend__metal__kernels__fp_quantized.h'
    assert sha(pinned) == '359a1ea9008f74cc15c61e9d39c6300dcffb9cfb9cc0ebd59ec7c00b789d719f'
    assert (D/'kernel_header.metal').read_bytes() == (P/'v66-gdn-four-projection-build2/kernel_header.metal').read_bytes()
    assert (D/'LICENSE-MLX.txt').read_bytes() == (P/'v66-gdn-four-projection-build2/LICENSE-MLX.txt').read_bytes()
    old = pinned.read_text().split('inline U qdot(',1)[1].split('qdot_safe(',1)[0]
    new = (D/'shared_impl.metal').read_text()
    expr = lambda text: re.sub(r'\s+', '', re.search(r'accum\s*\+=.*?;', text, re.S).group())
    assert expr(old) == expr(new), 'original four-term arithmetic expression changed'
    assert 'return scale * accum;' in old and 'return scale * accum;' in new
    fast = pinned.read_text().split('METAL_FUNC void fp_qmv_fast_impl(',1)[1].split('METAL_FUNC void fp_qmv_impl(',1)[0]
    for fragment in ('packs_per_thread = 2;', 'num_simdgroups = 2;', 'ws += block_size * bytes_per_pack / pack_factor;', 'scales += block_size / group_size;', 'result[row] = simd_sum(result[row]);'):
        assert fragment in fast
    for fragment in ('k += 512', 'ws += 256;', 'sc += 32;', 'int(lane) * 8;', 'int(lane) * 16'):
        assert fragment in new
    assert 'result[token][row] += v124_qdot_cached(cached, x_thread, scale);' in new
    assert 'result[token][row] = simd_sum(result[token][row]);' in new
    assert 'static_cast<T>(result[token][row])' in new
    assert new.index('cached[i] = wl[i]') < new.index('for (int token = 0; token < VERIFY_T; token++)')
    for token in ('simd_shuffle', 'fma(', 'dot(', 'half2', 'fast_math', 'simdgroup_matrix', 'threadgroup_barrier'):
        assert token not in new
    guards = 0
    for n,k in sorted(helper.SHAPES):
        helper.validate_geometry(n,k,(k//8,1),(k//16,1));guards += 1
        for t in helper.ROWS:
            helper.validate_input((t,k),(k,1),k);guards += 1
    for shape,stride,k in [((1,5120),(5120,1),5120),((3,5120),(5120,1),5120),
            ((6,5120),(5120,1),5120),((2,1,5120),(5120,1),5120),
            ((2,5120),(0,1),5120),((2,5120),(5120,2),5120),
            ((2,5120),(10240,1),5120),((2,6144),(6144,1),5120)]:
        try: helper.validate_input(shape,stride,k)
        except ValueError: guards += 1
        else: raise AssertionError(('accepted unsupported X',shape,stride,k))
    for n,k,ws,ss in [(48,5120,(641,1),(320,1)),(48,5120,(640,1),(0,1)),
            (47,5120,(640,1),(320,1)),(48,5121,(640,1),(320,1))]:
        try: helper.validate_geometry(n,k,ws,ss)
        except ValueError: guards += 1
        else: raise AssertionError('accepted unsupported W/scales')
    address_cases = 0
    for n,k in sorted(helper.SHAPES):
        # Every lane retains exactly its old increasing K sequence.
        for lane in range(32):
            stripe = [block+lane*16+i for block in range(0,k,512) for i in range(16)]
            assert len(stripe) == k//32 and stripe == sorted(set(stripe))
            assert min(stripe) >= 0 and max(stripe) < k
        assert sorted(i for lane in range(32) for block in range(0,k,512)
            for i in range(block+lane*16,block+lane*16+16)) == list(range(k))
        for t in helper.ROWS:
            for tile in set((0,n//8-1)):
                owners = set()
                for sg in range(2):
                    for row in range(4):
                        outrow = tile*8+sg*4+row
                        for token in range(t):
                            owner = token*n+outrow
                            assert owner not in owners and 0 <= owner < t*n
                            owners.add(owner)
                        for lane in range(32):
                            for block in range(0,k,512):
                                base = outrow*(k//2)+lane*8+block//2
                                group = outrow*(k//16)+lane+block//16
                                assert 0 <= base <= n*k//2-8 and 0 <= group < n*k//16
                                for i in range(4):
                                    old_offset = outrow*(k//2)+(block+lane*16)//2+2*i
                                    assert base+2*i == old_offset
                                for token in range(t):
                                    lo=token*k+block+lane*16
                                    assert token*k <= lo and lo+15 < (token+1)*k
                assert len(owners) == 8*t
                address_cases += 1
    result = dict(status='PASS_CPU_SOURCE_AND_INTEGER_ADDRESS_CHECKS_ONLY',
        guard_cases=guards,address_tile_cases=address_cases,shapes=len(helper.SHAPES),
        rows=list(helper.ROWS),qdot_four_term_expression_matches_pinned_M1=True,
        header_and_MIT_license_verbatim=True,MLX_imported=False,GPU_used=False,
        floating_numerical_claim=False,metal_compiler='xcrun metal unavailable; no compilation claim',
        required_next='root-only real Metal strict numeric against concat-M1 and V113; no engine/adoption/performance qualification')
    print(json.dumps(result,indent=2))
    return result
if __name__ == '__main__': main()
