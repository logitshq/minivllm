"""Source proof only; the expected B symbol is not an actual reflection observation."""
from pathlib import Path
import hashlib,json,re,ast
D=Path(__file__).resolve().parent;P=D.parent;H=P/'v79-nvfp4-half2-decode-build1'
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def function(text,signature):
 a=text.index(signature);brace=text.index('{',a);depth=1;i=brace+1
 while depth:
  if text[i]=='{':depth+=1
  elif text[i]=='}':depth-=1
  i+=1
 return text[a:i],text[brace:i],text[:a].count('\n')+1,text[:i].count('\n')+1
installed=Path('/Users/ryanli/code/minivllm/.venv/lib/python3.13/site-packages/mlx/include/mlx/backend/metal/kernels/fp_quantized.h')
a,ab,al,ae=function(installed.read_text(),'METAL_FUNC void fp_qmv_fast_impl(')
b,bb,bl,be=function((H/'kernel_header.metal').read_text(),'METAL_FUNC void fp_qmv_fast_impl(')
assert ab==bb,'79 recurrence/output-store body must equal original native R4 impl; paired decode is confined to qdot'
ylines=[l.strip() for l in b.splitlines() if re.search(r'(?<![.\w])y\b',l)]
assert ylines==['device T* y,','y += tid.x * out_vec_size + out_row;','y[row] = static_cast<T>(result[row] * inv_scale_enc);','y[row] = static_cast<T>(result[row]);']
body=(H/'kernel_body.metal').read_text();assert body=='''uint local = thread_position_in_threadgroup.x;
uint3 tile = uint3(0, threadgroup_position_in_grid.y, 0);
fp_qmv_fast_impl<T, 16, 4, false, R>(weight, scales, nullptr, x, output,
    v79_in_vec_size, v79_out_vec_size, tile, local / 32, local % 32);
'''
g=(H/'geometry.py').read_text();ast.parse(g)
for piece in ['ROWS, SIMD_GROUPS, THREADS = 4, 2, 64','name=f"v79_nvfp4_half2_r4_sg2_n{self.N}_k{self.K}"','input_names=["weight", "scales", "x"], output_names=["output"]','template=[("T", mx.bfloat16), ("R", ROWS)]','weight.dtype != mx.uint32','scales.dtype != mx.uint8','x.dtype != mx.bfloat16','output_dtypes=[mx.bfloat16]']:assert piece in g,piece
old=json.loads((P/'v72-native-range-icb-build2/source-write-proof.json').read_text());assert old['status']=='PASS_CPU_PINNED_SOURCE_STORE_ONLY_OUTPUT_PROOF'
observed=old['observed'][:8];assert all(x['arm']==0 and x['function']=='nvfp4_qmv_fast_bfloat16_t_gs_16_b_4_batch_0' and x['output_index']==4 and x['output_name']=='y' for x in observed)
expected='custom_kernel_v79_nvfp4_half2_r4_sg2_n17408_k5120__bfloat16_t_4_uint32_t_uint8_t_bfloat16_t_bfloat16_t'
proof={'status':'PASS_CPU_SOURCE_DERIVED_EXACT_OUTPUT_POLICY_EXPECTED_NOT_OBSERVED','actual_B_function_observed_by_author':False,'expected_B_function':expected,'expected_name_basis':'frozen79 literal name with N17408/K5120, template T=bfloat16/R=4, source input/output typed pointers; exact MLX0.32.3 naming pattern previously observed in72','B_output':{'index':3,'name':'output','source_pointer_dtype':'bfloat16_t','reflection_RW_remains_RW':True,'source_y_reads':0,'impl_lines':[bl,be],'impl_y_uses':ylines},'B_sources':{str(H/n):h(H/n) for n in ('geometry.py','kernel_header.metal','kernel_body.metal','LICENSE-MLX.txt','LICENSE-Lithos-Apache-2.0.txt')},'impl_body_exact_original_native':True,'installed_impl':{'path':str(installed),'sha256':h(installed),'lines':[al,ae]},'A_observed_provenance':{'proof':str(P/'v72-native-range-icb-build2/source-write-proof.json'),'sha256':h(P/'v72-native-range-icb-build2/source-write-proof.json'),'8_records':observed},'all_other_RW_or_unknown':'REJECT','runtime_required':'all16 exact functions + exact active buffer name/index/access/binding-kind schema + original full pin role ranges + exclusive output allocations + whole-allocation disjointness; actual mismatch STOP','element_dtype_limit':'source typed symbol plus runtime array metadata; no independent Metal buffer element-dtype reflection claim','GPU_used':False,'native_loaded':False}
(D/'source-write-proof.json').write_text(json.dumps(proof,indent=2)+'\n');print('PASS_CPU_SOURCE_DERIVED_EXPECTED_B_NOT_OBSERVED')
