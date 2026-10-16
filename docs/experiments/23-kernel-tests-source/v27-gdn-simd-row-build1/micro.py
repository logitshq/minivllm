"""V27 stationary local host-wall micro; import/default preflight is GPU inert."""
import argparse,json,os,statistics,time,traceback
from pathlib import Path
from common import *
PRIME=128;REPEATS=2048
SCHEDULE=(('A','A'),('B','B'))+(('A','B'),('B','A'))*3+(('A','A'),('B','B'))
NAMES=('real-L0-D1','real-L0-D16','real-L0-D31','real-L32-D1','real-L32-D16','real-L32-D31','real-L62-D1','real-L62-D16')
def micro_identity():
 numeric_sources=pins();files=json.loads((D/'micro-pins.json').read_text())
 for name,expected in files.items():
  if sha(name)!=expected:raise ValueError('micro source changed: '+name)
 return {'numeric_source_pins':numeric_sources,'micro_source_pins':files,'plan':{'prime':PRIME,'repeats':REPEATS,'schedule':SCHEDULE,'banks':NAMES,'minimum_gain':.05,'minimum_wins':5,'control_median_bounds':[.98,1.02],'control_each_bounds':[.95,1.05],'boundary':'compact postactivation BF16 q/k/v + g/beta/S -> FP32 o and full logical S_new; A includes repeated qk and FP32 casts; B uses persistent transposed storage','timing_scope':'local host wall only; no model/E2E claim','repeat_reason':'2048 fixed heavy-state calls per block, same128 prime; all blocks identical before any timing'}}
def eligible(path,expected):
 if sha(path)!=expected:raise ValueError('numeric receipt hash changed')
 r=json.loads(Path(path).read_text())
 if r['status']!='PASS_V27_LOCAL_NUMERIC_BOUNDED' or r['source_pins']!=pins() or len(r['cases'])!=15 or len(r['chain'])!=CHAIN_STEPS:raise ValueError('matching actual local numeric PASS required')
 if not all(c['input_readonly'] for c in r['cases']+r['chain']):raise ValueError('numeric readonly missing')
 return {'path':str(Path(path).resolve()),'sha256':expected,'status':r['status']}
def run(a,ident):
 proof=eligible(a.numeric_receipt,a.numeric_sha256)
 if {k:os.environ.get(k) for k in CONTROLS}!=CONTROLS:raise ValueError('requires fixed50/50/20/0 controls')
 a.out.mkdir(exist_ok=False);r={'status':'STARTED','identity':ident,'numeric_eligibility':proof,'pairs':[],'resources':[],'model_qualified':False,'E2E_run':False};write(a.out/'receipt.json',r);started=time.perf_counter()
 try:
  mx,nn,np,rt=runtime();r['runtime']=rt;base=module('v27_micro_base',BASE);cand=module('v27_micro_candidate',D/'gdn_simd_row.py');timing=module('v27_micro_timing',D/'timing_block.py');resource=module('v27_micro_resource',P/'resource_snapshot_power_v1.py');timing.PRIME=PRIME;timing.REPEATS=REPEATS
  pool=[]
  for name in NAMES:
   bank=load_bank(BANKROOT/name,mx,np);q,k,v=compact(bank['u'],mx,nn);pool.append((name,q,k,v,bank['g'],bank['beta'],bank['S'],transposed_storage(bank['S'],mx)))
  mx.eval(*[x for bank in pool for x in bank[1:]])
  def inputs_hash():return {bank[0]:hashes(bank[1:],mx,np) for bank in pool}
  def values(variant,j):
   _,q,k,v,g,beta,canonical,persistent=pool[j%8]
   return reference(q,k,v,g,beta,canonical,mx,base) if variant=='A' else cand.gated_delta_step_simd(q,k,v,g,beta,persistent,backend=mx)
  def invoke(variant,j):
   outputs=values(variant,j);mx.eval(*outputs);return outputs
  def qualify():
   before=inputs_hash();checks=[]
   if len({tuple(v) for v in before.values()})!=8:raise ValueError('timing banks are not8distinct inputs')
   for j,bank in enumerate(pool):
    left,right=invoke('A',j),invoke('B',j);checks.append({'bank':bank[0],'o':compare(left[0],right[0],ONE_STEP,mx,np),'S':compare(left[1],right[1],ONE_STEP,mx,np),'per_variant_output_sha256':{'A':hashes(left,mx,np),'B':hashes(right,mx,np)}})
   after=inputs_hash()
   if before!=after:raise ValueError('input readonly failed')
   return {'checks':checks,'input_sha256':before,'input_readonly':True}
  def probe(label):
   mx.synchronize();s=resource.snapshot(label,os.getpid());write(a.out/('resource-%02d-raw.json'%len(r['resources'])),s)
   s['mlx_memory']={'active':mx.get_active_memory(),'cache':mx.get_cache_memory(),'peak':mx.get_peak_memory()};r['resources'].append(s);resource.deltas(r['resources']);write(a.out/'resources.json',r['resources']);resource.guard(r['resources'])
   if s['power_metadata']['status']!='known' or s['power_metadata']['source']!='AC Power' or s['processinfo_thermal_metadata']['status']!='known' or s['processinfo_thermal_metadata']['code']!=0 or s['pressure_level_value']!=1:raise ValueError('AC/thermal0/pressure1 guard')
   if s['swap_used_mb']>r['resources'][0]['swap_used_mb'] or s['vm_counters']['Swapouts']!=r['resources'][0]['vm_counters']['Swapouts']:raise ValueError('swap growth')
   if s['known_inference_process_count']!=0 or s['exact_owned_pid_metadata']['status']!='known':raise ValueError('process inventory guard')
  probe('ready-before-qualify');r['checks_before']=qualify();probe('timing-ready')
  for i,order in enumerate(SCHEDULE):
   probe('pair%d-before'%i);before=inputs_hash()
   first,out1=timing.block(mx,invoke,order[0]);second,out2=timing.block(mx,invoke,order[1])
   # Full pair is complete before either output hash/resource probe/receipt write.
   first['output_sha256']=hashes(out1,mx,np);second['output_sha256']=hashes(out2,mx,np);after=inputs_hash()
   if before!=after:raise ValueError('timing input mutation')
   if order[0]==order[1] and first['output_sha256']!=second['output_sha256']:raise ValueError('same-variant deterministic outputs changed')
   row={'pair':i,'order':order,'kind':'control' if order[0]==order[1] else 'AB','blocks':[first,second],'slot2_div_slot1':second['ns_per_call']/first['ns_per_call'],'input_sha256_before':before,'input_sha256_after':after}
   if row['kind']=='AB':
    by={x['variant']:x for x in (first,second)};row['reduction']=1-by['B']['ns_per_call']/by['A']['ns_per_call']
   r['pairs'].append(row);write(a.out/'receipt.json',r);probe('pair%d-after'%i)
  r['checks_after']=qualify()
  if r['checks_before']!=r['checks_after']:raise ValueError('before/after deterministic numeric/hash check differs')
  probe('after-final-qualify');ab=[p for p in r['pairs'] if p['kind']=='AB'];controls={v:[p['slot2_div_slot1'] for p in r['pairs'] if p['kind']=='control' and p['order'][0]==v] for v in ('A','B')};control_ok=all(.98<=statistics.median(v)<=1.02 and all(.95<=x<=1.05 for x in v) for v in controls.values())
  medians={v:statistics.median(next(b['ns_per_call'] for b in p['blocks'] if b['variant']==v) for p in ab) for v in ('A','B')};gain=1-medians['B']/medians['A'];wins=sum(p['reduction']>0 for p in ab)
  r.update(status='CONTROL_DRIFT_STOP' if not control_ok else 'PASS_LOCAL_MICRO' if gain>=.05 and wins>=5 else 'STOP_MICRO_GAIN',medians_ns=medians,median_reduction=gain,wins=wins,control_passed=control_ok,control_slot2_div_slot1=controls,scope='stationary recurrence local host wall; no fullmodel/32greedy/E2E/Pareto qualification')
  return 0 if r['status']=='PASS_LOCAL_MICRO' else 2
 except BaseException as e:r.update(status='FAIL_STOP',error={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()});raise
 finally:r['wall_seconds']=time.perf_counter()-started;write(a.out/'receipt.json',r)
def main(argv=None):
 ap=argparse.ArgumentParser();ap.add_argument('--execute-gpu',action='store_true');ap.add_argument('--numeric-receipt',type=Path);ap.add_argument('--numeric-sha256');ap.add_argument('--out',type=Path);a=ap.parse_args(argv);ident=micro_identity()
 if not a.execute_gpu:print(json.dumps({'status':'CPU_MICRO_READY_ONLY','MLX_imported':False,'GPU_run':False,'identity':ident}));return 0
 if a.numeric_receipt is None or a.numeric_sha256 is None or a.out is None:raise ValueError('GPU run requires numeric-receipt/numeric-sha256/new-out; root owns external timeout/PID cleanup')
 return run(a,ident)
if __name__=='__main__':raise SystemExit(main())
