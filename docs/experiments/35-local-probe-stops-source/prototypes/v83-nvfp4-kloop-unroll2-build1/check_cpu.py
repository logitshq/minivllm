"""CPU-only reverse-source proof: exact original MLX header plus one outer K unroll pragma."""
from pathlib import Path
import ast,json,sys,hashlib
D=Path(__file__).resolve().parent;P=D.parent
header=(D/'kernel_header.metal').read_text();pragma='#pragma clang loop unroll_count(2)\n'
assert header.count(pragma)==1 and pragma+'  for (int k = 0; k < in_vec_size; k += block_size)' in header
assert header.replace(pragma,'')==(P/'v66-gdn-four-projection-build2/kernel_header.metal').read_text()
assert 'fp4pair' not in header
for p in D.glob('*.py'):ast.parse(p.read_text())
assert not any(k.startswith(('mlx','numpy')) for k in sys.modules)
print(json.dumps({'status':'PASS_CPU_EXACT_ORIGINAL_HEADER_PLUS_K_UNROLL2','GPU_used':False,'old_qdot_scale_association_K_order_R4_SG2_unchanged':True,'compiler_contract_equivalence_unproven':True}))
