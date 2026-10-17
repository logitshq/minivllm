from pathlib import Path
import json,hashlib,struct,math
R=Path('/Users/ryanli/code/minivllm'); O=R/'.artifacts/m5-ultra-256gb/shared-w-m1-v1-owner'; S=O/'child/screen'
d=json.loads((S/'complete.json').read_text()); seen={}; desc={}; pairs=0; compared=0
sha=lambda b:hashlib.sha256(b).hexdigest()
def raw(rec):
 p=(S/rec['file']).resolve(); assert p.is_relative_to(S.resolve())
 b=p.read_bytes(); assert len(b)==rec['bytes'] and sha(b)==rec['sha256']
 width={'bfloat16':2,'uint32':4,'uint8':1,'uint64':8}[rec['dtype']]
 assert len(b)==math.prod(rec['shape'])*width
 if rec['dtype']=='bfloat16': assert rec['finite'] and all((v[0]&0x7f80)!=0x7f80 for v in struct.iter_unpack('<H',b))
 seen[rec['file']]={'bytes':len(b),'sha256':sha(b)}
 return b
def layout(rec,owner,shape):
 b=raw(rec['descriptor']); values=list(struct.unpack('<4Q',b)); assert values==shape+[shape[1],1]
 assert rec['shape']==shape and rec['actual_strides_elements']==[shape[1],1] and rec['input_owner_id']==owner
 assert rec['ensure_row_contiguous'] is False
 desc[rec['descriptor']['file']]=True
for c in d['cases']:
 assert c['status']=='PASS' and c['inputs_unchanged']
 for k,before in c['inputs_before'].items():
  b=raw(before); after=c['inputs_after'][k]
  assert {k:v for k,v in before.items() if k!='file'}=={k:v for k,v in after.items() if k!='file'}
  if 'file' in after:assert raw(after)==b
 shape=[c['T'],c['K']]
 for k in ('layout_before','layout_after'):layout(c[k],c['inputs_before']['x']['owner_id'],shape)
 expected=[c['T'],c['N']]
 outputs={k:raw(v) for k,v in c['outputs'].items()}
 assert set(outputs)=={'native','control','shared'}
 for v in c['outputs'].values(): assert v['shape']==expected and v['dtype']=='bfloat16'
 assert outputs['native']==outputs['control']==outputs['shared']
 pairs+=2;compared+=2*len(outputs['native'])
for bank in d['bank_receipts']:
 assert bank['launch_count']==3 and bank['kernel_build_count']==1
 for k,before in bank['inputs_before'].items():
  after=bank['inputs_after'][k];assert raw(before)==raw(after)
  for name in ('layouts_before','layouts_after'):layout(bank[name][k],before['owner_id'],before['shape'])
assert len(d['cases'])==33 and pairs==66 and len(desc)==110
result={'status':'PASS_LOCAL_RAW','scope':'Root full saved matrix bytes, original finite BF16, read-only bytes and actual descriptor consistency; no performance/full-model conclusion','cases':33,'comparisons':pairs,'compared_output_bytes':compared,'gpu_descriptors':len(desc),'unique_raw_files':len(seen),'unique_raw_file_bytes':sum(v['bytes'] for v in seen.values()),'screen_complete_sha256':sha((S/'complete.json').read_bytes()),'raw_files':seen}
path=R/'.artifacts/m5-ultra-256gb/shared-w-m1-v1-root-raw-audit.json';path.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='raw_files'},indent=2))
