"""Pinned local C++/Metal source proof only; no Metal load/MLX import."""
from pathlib import Path
import hashlib,json,re
D=Path(__file__).resolve().parent;P=D.parent
installed=Path('/Users/ryanli/code/minivllm/.venv/lib/python3.13/site-packages/mlx/include/mlx/backend/metal/kernels/fp_quantized.h')
header=P/'v69-nvfp4-r8-reuse-build1/kernel_header.metal';body=P/'v69-nvfp4-r8-reuse-build1/kernel_body.metal'
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def function(text,signature):
 a=text.index(signature);brace=text.index('{',a);depth=1;i=brace+1
 while depth:
  if text[i]=='{':depth+=1
  elif text[i]=='}':depth-=1
  i+=1
 return text[a:i],text[brace:i],text[:a].count('\n')+1,text[:i].count('\n')+1
x,xb,a,b=function(installed.read_text(),'METAL_FUNC void fp_qmv_fast_impl(')
y,yb,c,d=function(header.read_text(),'METAL_FUNC void fp_qmv_fast_impl(')
assert xb==yb, 'frozen original69 implementation body differs from installed native source'
ylines=[l.strip() for l in x.splitlines() if re.search(r'(?<![.\w])y\b',l)]
assert ylines==['device T* y,','y += tid.x * out_vec_size + out_row;','y[row] = static_cast<T>(result[row] * inv_scale_enc);','y[row] = static_cast<T>(result[row]);']
wrap,wb,wa,wbline=function(installed.read_text(),'[[kernel]] void fp_qmv_fast(')
assert 'if (batched) {' in wb and wb.count('fp_qmv_fast_impl<')==1
assert re.findall(r'(?<![.\w])y\b',wrap)==['y','y','y'], 'unexpected wrapper y use'
# Actual A symbol is batch_0; its observed inactive global/batch args are not bindings.
frozen_body=body.read_text()
assert frozen_body=='''uint local = thread_position_in_threadgroup.x;
uint3 tile = uint3(0, threadgroup_position_in_grid.y, 0);
fp_qmv_fast_impl<T, 16, 4, false, R>(weight, scales, nullptr, x, output,
    v69_in_vec_size, v69_out_vec_size, tile, local / 32, local % 32);
'''
actual=P/'v23-root-control-build1/fullmodel-v72num-attempt1/child-output/native-failure.json';trace=json.loads(actual.read_text())
assert len(trace['slot_records'])==16
observed=[]
for i,s in enumerate(trace['slot_records']):
 arm=i//8;fn='nvfp4_qmv_fast_bfloat16_t_gs_16_b_4_batch_0' if arm==0 else 'custom_kernel_v69_nvfp4_r8_sg2_n17408_k5120__bfloat16_t_8_uint32_t_uint8_t_bfloat16_t_bfloat16_t'
 assert s['arm']==arm and s['slot']==i%8 and s['function']==fn
 rw=[r for r in s['actual_active_arguments'] if r['access']==1]
 assert len(rw)==1 and rw[0]['index']==(4 if arm==0 else 3) and rw[0]['name']==('y' if arm==0 else 'output')
 assert rw[0]['original_offset']==0 and rw[0]['allocation_bytes']==49152
 observed.append({'arm':arm,'slot':i%8,'function':fn,'output_index':rw[0]['index'],'output_name':rw[0]['name'],'reflection_access':1,'reflection_interpretation':'ReadWrite','source_inferred_access':'store-only output','full_allocation_bytes':49152,'logical_output_bytes':34816})
proof={'status':'PASS_CPU_PINNED_SOURCE_STORE_ONLY_OUTPUT_PROOF','reflection_policy':'RW retained; source-only write inference for exact two observed output args','installed_source':{'path':str(installed),'sha256':h(installed),'impl_lines':[a,b],'wrapper_lines':[wa,wbline]},'frozen69_header':{'path':str(header),'sha256':h(header),'impl_lines':[c,d],'body_exact_installed':True},'frozen69_body':{'path':str(body),'sha256':h(body)},'impl_y_uses':ylines,'impl_y_reads':0,'wrapper_A_batched':False,'actual_failed_trace':{'path':str(actual),'sha256':h(actual)},'observed':observed,'limitations':'Source inference is distinct from reflection; no binary-disassembly claim. New fresh actual numeric qualification is mandatory.','GPU_used':False,'MLX_imported':False}
(D/'source-write-proof.json').write_text(json.dumps(proof,indent=2)+'\n');print('PASS_SOURCE_STORE_ONLY_16_OBSERVED_RW_OUTPUTS')
