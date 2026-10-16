"""CPU-only independent replay of saved BF16 raw outputs and diagnostics; no model/GPU."""
from pathlib import Path
import array,hashlib,itertools,json,math,struct,sys
P=Path('/private/tmp/minivllm-engine-optimization-20261014');O=P/'v23-root-control-build1/numeric-v91-attempt1';C=O/'child-output';D=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
r=read(C/'receipt.json');assert r['status']=='DIAGNOSTIC_COMPLETED_NO_ADOPTION' and r['all_finite'] and r['all15_originals_readonly']
assert (r['cases'],r['outputs'],r['pairwise_diagnostic_comparisons'],r['readonly_case_checkpoints'])==(27,108,162,27)
for p,h in r['source_pins'].items():assert sha(p)==h
owner=read(O/'owner-receipt.json');rel=read(O/'external-release.json');assert owner['returncode']==0 and owner['waited_reaped'] and rel['GPU_released'] and rel['tool_exit']==0
for x in rel['strict_absence']:assert x['returncode']==1 and x['stdout']==x['stderr']==''
rows=read(C/'cases.json');assert len(rows)==27
stats={a+'-'+b:{'unequal_cases':0,'different_elements':0,'max_BF16_ULP_distance':0,'max_absolute_difference':0.0,'signed_zero_bit_differences':0} for a,b in itertools.combinations('ABCD',2)}
raw_count=0
for case in rows:
 values={}
 for name,out in case['outputs'].items():
  path=C/out['raw'];data=path.read_bytes();assert sha(path)==out['sha256'] and len(data)==out['bytes']==10240
  bits=array.array('H',data)
  if sys.byteorder!='little':bits.byteswap()
  assert len(bits)==5120 and all((v&0x7f80)!=0x7f80 for v in bits)
  floats=[struct.unpack('<f',struct.pack('<I',v<<16))[0] for v in bits];values[name]=(data,bits,floats);raw_count+=1
 for a,b in itertools.combinations('ABCD',2):
  aa,bb=values[a],values[b];key=a+'-'+b;declared=case['comparisons'][key]
  ordered=lambda u:((~u)&0xffff) if u&0x8000 else u|0x8000
  m={'strict_bytes_equal':aa[0]==bb[0],'different_elements':sum(x!=y for x,y in zip(aa[1],bb[1])),'max_BF16_ULP_distance':max(abs(ordered(x)-ordered(y)) for x,y in zip(aa[1],bb[1])),'max_absolute_difference':max(abs(x-y) for x,y in zip(aa[2],bb[2])),'signed_zero_bit_differences':sum(x!=y and (x&0x7fff)==(y&0x7fff)==0 for x,y in zip(aa[1],bb[1]))}
  assert all(declared[k]==v for k,v in m.items()),(case['gamma'],case['input'],key,m,declared)
  q=stats[key];q['unequal_cases']+=not m['strict_bytes_equal'];q['different_elements']+=m['different_elements'];q['signed_zero_bit_differences']+=m['signed_zero_bit_differences']
  for field in ('max_BF16_ULP_distance','max_absolute_difference'):q[field]=max(q[field],m[field])
assert {k:v['unequal_cases'] for k,v in stats.items()}==r['unequal_pair_counts']
assert stats['A-C']['unequal_cases']==stats['B-D']['unequal_cases']==0 and stats['A-B']['unequal_cases']==21
out={'status':'PASS_CPU_SAVED_RAW_DIAGNOSTIC_REPLAY','source_pins_verified':len(r['source_pins']),'raw_outputs_verified':raw_count,'pairwise_metrics_recomputed':162,'stats':stats,'scope':'Only27cases/3gamma/9X. A-C and B-D equality supports two local rounding paths; no universal/kernel-cause/performance/adoption claim.','actual_receipt_sha256':sha(C/'receipt.json')}
(D/'audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
