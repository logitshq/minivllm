"""V29 local numeric qualification; CPU default is inert, GPU only --execute-gpu."""
import argparse,json,time,traceback
from pathlib import Path
from common import *
def main(argv=None):
 ap=argparse.ArgumentParser();ap.add_argument('--execute-gpu',action='store_true');ap.add_argument('--out',type=Path);a=ap.parse_args(argv);identity=pins()
 if not a.execute_gpu:
  print(json.dumps({'status':'CPU_PREFLIGHT_ONLY','MLX_imported':False,'GPU_run':False,'source_files':len(identity),'one_step_bound':ONE_STEP,'chain_bound':CHAIN,'chain_steps':CHAIN_STEPS}));return 0
 if a.out is None:raise ValueError('new --out required; root owns timeout/PID/resources')
 a.out.mkdir(exist_ok=False);r={'status':'STARTED','source_pins':identity,'cases':[],'chain':[],'tolerances':{'one_step':ONE_STEP,'chain':CHAIN,'chain_steps':CHAIN_STEPS},'fullmodel_qualified':False,'engine_connected':False};write(a.out/'receipt.json',r);start=time.perf_counter()
 try:
  mx,nn,np,rt=runtime();r['runtime']=rt;base=module('v29_baseline',BASE);cand=module('v29_candidate',D/'gdn_simd_serial.py');descriptor=layout_probe(mx)
  names=sorted(p.parent.name for p in BANKROOT.glob('*/bank.json'));r['saved_banks']=names
  for name in names:
   bank=load_bank(BANKROOT/name,mx,np);q,k,v=compact(bank['u'],mx,nn);canonical=bank['S'];persistent=transposed_storage(canonical,mx);layouts={'canonical':descriptor(canonical),'persistent':descriptor(persistent)}
   if layouts['canonical']['strides_elements']!=[786432,16384,128,1] or layouts['persistent']['strides_elements']!=[786432,16384,1,128]:raise ValueError('unexpected actual state layout')
   values=[q,k,v,bank['g'],bank['beta'],canonical,persistent];before=hashes(values,mx,np);ref=reference(q,k,v,bank['g'],bank['beta'],canonical,mx,base)
   row={'bank':name,'provenance':bank['provenance'],'layouts':layouts,'checks':{}}
   for kind,S in [('canonical',canonical),('persistent',persistent)]:
    result=cand.gated_delta_step_simd_serial(q,k,v,bank['g'],bank['beta'],S,backend=mx)
    row['checks'][kind]={'o':compare(ref[0],result[0],ONE_STEP,mx,np),'S':compare(ref[1],result[1],ONE_STEP,mx,np),'output_descriptor':descriptor(result[1])}
    if row['checks'][kind]['output_descriptor']['strides_elements']!=[786432,16384,1,128]:raise ValueError('candidate output not persistently transposed')
   after=hashes(values,mx,np);row.update(input_readonly=before==after,input_sha256_before=before,input_sha256_after=after)
   if before!=after:raise ValueError('input mutated')
   r['cases'].append(row);write(a.out/'receipt.json',r)
  # Chained changing saved-real operands, with candidate state remaining a strided view.
  seed=load_bank(BANKROOT/'real-L0-D1',mx,np);state_a=seed['S'];state_b=transposed_storage(state_a,mx);original=hashes([state_a,state_b],mx,np);chain_pool=[load_bank(BANKROOT/f'real-L0-D{d}',mx,np) for d in (1,16,31)]
  for step in range(CHAIN_STEPS):
   bank=chain_pool[step%3];q,k,v=compact(bank['u'],mx,nn);old_a,old_b=state_a,state_b;inputs=[q,k,v,bank['g'],bank['beta'],old_a,old_b];before=hashes(inputs,mx,np)
   oa,state_a=reference(q,k,v,bank['g'],bank['beta'],old_a,mx,base);ob,state_b=cand.gated_delta_step_simd_serial(q,k,v,bank['g'],bank['beta'],old_b,backend=mx)
   row={'step':step,'bank':bank['name'],'o':compare(oa,ob,CHAIN,mx,np),'S':compare(state_a,state_b,CHAIN,mx,np),'input_readonly':before==hashes(inputs,mx,np),'output_descriptor':descriptor(state_b)}
   if not row['input_readonly'] or row['output_descriptor']['strides_elements']!=[786432,16384,1,128]:raise ValueError('chain mutation/layout')
   r['chain'].append(row)
  if original!=hashes([seed['S'],transposed_storage(seed['S'],mx)],mx,np):raise ValueError('initial state changed')
  r.update(status='PASS_V29_LOCAL_NUMERIC_BOUNDED',GPU_numeric_run=True,full32greedy_required_before_E2E=True);return 0
 except BaseException as e:
  r.update(status='FAIL_STOP',error={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()});raise
 finally:r['wall_seconds']=time.perf_counter()-start;write(a.out/'receipt.json',r)
if __name__=='__main__':raise SystemExit(main())
