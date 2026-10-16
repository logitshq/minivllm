"""CPU stdlib detached small checkpoint ranges; no MLX/NumPy/model/native library."""
from pathlib import Path
import json,struct,hashlib,math
P=Path('/private/tmp/minivllm-engine-optimization-20261014');D=Path(__file__).resolve().parent
C=Path('/Users/ryanli/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-nvfp4/snapshots/5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6')
def sha(b):return hashlib.sha256(b).hexdigest()
def f32(v):return struct.unpack('<f',struct.pack('<f',v))[0]
def frombf(b):return struct.unpack('<f',struct.pack('<I',b<<16))[0]
def tobf(v):
 u=struct.unpack('<I',struct.pack('<f',v))[0]
 if (u&0x7f800000)==0x7f800000 and (u&0x007fffff):return ((u>>16)|0x40)&0xffff
 return ((u+0x7fff+((u>>16)&1))>>16)&0xffff
idx=json.loads((C/'model.safetensors.index.json').read_text())['weight_map'];cache={};rows=[];gated=[];convs=[]
def meta(shard):
 if shard not in cache:
  p=C/shard
  with p.open('rb') as f:ln=struct.unpack('<Q',f.read(8))[0];blob=f.read(ln)
  cache[shard]=(8+ln,json.loads(blob),sha(blob))
 return cache[shard]
for k,shard in idx.items():
 if not k.startswith('language_model.model.'):continue
 base,header,hh=meta(shard);m=header[k]
 if k.endswith('conv1d.weight'):convs.append({'key':k,'shape':m['shape'],'dtype':m['dtype']})
 centered=k.endswith(('input_layernorm.weight','post_attention_layernorm.weight','q_norm.weight','k_norm.weight')) or k=='language_model.model.norm.weight'
 is_gated='.linear_attn.norm.weight' in k
 if not (centered or is_gated):continue
 assert m['dtype']=='BF16';a,b=m['data_offsets'];p=C/shard
 with p.open('rb') as f:f.seek(base+a);raw=f.read(b-a)
 assert len(raw)==b-a
 vals=struct.unpack('<'+'H'*(len(raw)//2),raw);src=[frombf(u) for u in vals];assert all(map(math.isfinite,src))
 if centered:
  delta=[tobf(f32(x-1.0)) for x in src];effective=[tobf(f32(1.0+frombf(u))) for u in delta]
 else:delta=[];effective=list(vals)
 out=[frombf(u) for u in effective];diff=[i for i,(u,v) in enumerate(zip(vals,effective)) if u!=v]
 record={'key':k,'class':'zero_centered_loader_roundtrip' if centered else 'GDN_gated_plain_weight_NO_SUBTRACT_ADD','dtype':m['dtype'],'shape':m['shape'],'tensor_header':m,'shard':str(p),'shard_header_sha256':hh,'absolute_byte_range':[base+a,base+b],'source_range_bytes':len(raw),'source_raw_sha256':sha(raw),'source_min':min(src),'source_max':max(src),'elements':len(vals),'effective_gamma_roundtrip_differences':len(diff),'max_absolute_deviation':max(abs(x-y) for x,y in zip(src,out)),'effective_min':min(out),'effective_max':max(out),'stored_delta_min':min(map(frombf,delta)) if delta else None,'stored_delta_max':max(map(frombf,delta)) if delta else None,'first_difference_indices':diff[:8],'effective_raw_sha256':sha(struct.pack('<'+'H'*len(effective),*effective))}
 (rows if centered else gated).append(record)
assert len(rows)==161 and len(gated)==48 and len(convs)==48 and all(x['shape']==[10240,4,1] for x in convs)
assert sum(x['elements'] for x in rows)==668672
out={'status':'CPU_STDLIB_BIT_EXACT_BF16_RNE_SOURCE_MODEL_NO_GPU_VALIDATION','checkpoint_index_sha256':sha((C/'model.safetensors.index.json').read_bytes()),'source_sequence':'checkpoint BF16 gamma -> FP32(gamma-1.0) -> BF16_RNE loader storage -> FP32(1.0+delta) -> BF16_RNE norm weight','zero_centered_arrays':len(rows),'elements':sum(x['elements'] for x in rows),'differences':sum(x['effective_gamma_roundtrip_differences'] for x in rows),'affected_arrays':sum(x['effective_gamma_roundtrip_differences']>0 for x in rows),'max_absolute_deviation':max(x['max_absolute_deviation'] for x in rows),'source_min':min(x['source_min'] for x in rows),'source_max':max(x['source_max'] for x in rows),'small_range_payload_bytes':sum(x['source_range_bytes'] for x in rows+gated),'gated_plain_arrays_not_centered':len(gated),'sanitize_actual_has_unsanitized_conv1d':False,'conv_header_evidence':convs,'selected_arrays':rows,'gated_plain_arrays':gated,'limitations':'RNE/FP32 CPU arithmetic models explicit loader + norm source order, not a GPU load/weight readback. No full shard/model loaded; no semantic permission to change loader or old gold.','GPU_used':False,'MLX_imported':False}
(D/'norm-gamma-checkpoint-roundtrip.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ('status','zero_centered_arrays','elements','differences','affected_arrays','max_absolute_deviation','source_min','source_max','small_range_payload_bytes','gated_plain_arrays_not_centered')},indent=2))
