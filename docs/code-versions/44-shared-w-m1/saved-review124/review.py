"""CPU-only replay of saved V124 numeric metadata; never imports inference runtimes."""
from pathlib import Path
import collections,hashlib,json
D=Path(__file__).resolve().parent;P=D.parent;R=P/'v23-root-control-build1/numeric-v124-attempt1';C=R/'child-output'
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,v):(D/n).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def main():
 r=load(C/'receipt.json');assert sha(C/'receipt.json')=='dac25c0507db682620b9d20e6b79b9bd73b90fce404d1c2806cfb2555efc35af'
 assert r['status']=='PASS_LOCAL_V124_SHARED_WEIGHT_M1_STRICT_SYNTHETIC' and r['error'] is r['final_resource_or_readonly_error'] is None
 cases=load(C/'cases.json');checks=load(C/'checks.json');ro=load(C/'readonly.json');samples=load(C/'resources.json');outer=load(R/'resources.json');runtime=load(C/'runtime.json')
 assert len(cases)==40 and len(checks)==80 and len(ro)==120 and sum(x['T'] for x in cases)==152
 expected=collections.Counter((n,k,t,b,False) for n,k in ((48,5120),(17408,5120),(5120,6144),(5120,17408)) for b in ('random','codes_signed_boundaries','extreme_cancellation') for t in (2,4,5))
 expected.update((n,k,5,'codes_signed_boundaries',True) for n,k in ((48,5120),(17408,5120),(5120,6144),(5120,17408)))
 assert collections.Counter((x['N'],x['K'],x['T'],x['bank'],x['row_permutation']) for x in cases)==expected
 for i,case in enumerate(cases):
  for item,label in zip(checks[2*i:2*i+2],('original-expression-control-vs-native-M1','shared-W-vs-native-M1')):
   assert item['case']=={k:case[k] for k in ('N','K','T','bank','seed','row_permutation')} and item['label']==label
   assert item['shape']==[case['T'],case['N']] and item['dtype']=='bfloat16' and item['nbytes']==case['T']*case['N']*2
   assert item['reference_finite'] is item['actual_finite'] is item['exact'] is True and item['reference_sha256']==item['actual_sha256']
   assert len(item['row_checks'])==case['T'] and all(v['row']==j and v['reference_sha256']==v['actual_sha256'] for j,v in enumerate(item['row_checks']))
  for item in ro[3*i:3*i+3]:
   assert item['case']==checks[2*i]['case'] and item['unchanged'] is True and item['before']==item['after']==case['input_metadata']
   assert set(item['before'])=={'weight','scales','x'}
   n,k,t=case['N'],case['K'],case['T']
   for name,shape,dtype,strides in [('weight',[n,k//8],'mlx.core.uint32',[k//8,1]),('scales',[n,k//16],'mlx.core.uint8',[k//16,1]),('x',[t,k],'mlx.core.bfloat16',[k,1])]:
    v=item['before'][name];assert (v['shape'],v['dtype'],v['strides'])==(shape,dtype,strides) and v['owner_id']>0
 assert runtime['model_loaded'] is runtime['checkpoint_read'] is runtime['performance_run'] is False
 assert runtime['mlx_version']=='0.32.3' and runtime['device']['device_name']=='Apple M4 Max' and runtime['memory_limit_bytes']==1<<30 and runtime['allocator_cache_limit_bytes']==64<<20
 assert len(samples)==14 and len(outer)==5
 for group in (samples,outer):
  for x in group:
   assert x['power_metadata']['source']=='AC Power' and x['processinfo_thermal_metadata']['code']==0 and x['pressure_level_value']==1
   assert x['swap_used_mb']<=group[0]['swap_used_mb'] and x['vm_counters']['Swapouts']==group[0]['vm_counters']['Swapouts']
   assert all(p['owned'] for p in x['known_inference_processes'])
 assert max(x['mlx_memory']['peak'] for x in samples)==51724488 and all(x['mlx_memory']['active']==0 for x in samples)
 manifest=load(P/'v124-shared-weight-m1-source-build1/manifest.json')
 for name,v in manifest['files'].items():assert sha(P/'v124-shared-weight-m1-source-build1'/name)==v['sha256']
 pins=load(P/'v124-shared-weight-m1-localqual-build1/pins.json')['files']
 for path,digest in pins.items():assert sha(path)==digest,path
 pre=load(C/'preimport-identity.json');assert pre['files']==pins and pre['local_pins_sha256']==sha(P/'v124-shared-weight-m1-localqual-build1/pins.json')
 owner=load(R/'owner-receipt.json');release=load(R/'external-release.json')
 assert owner['stage']=='reaped' and owner['returncode']==0 and owner['waited_reaped'] is True
 assert release['tool_session']==54508 and release['tool_exit_code']==0 and release['GPU_released'] is True and release['receipt_sha256']==sha(C/'receipt.json')
 assert {v['pid'] for v in release['strict_absence']}=={owner['owner_pid'],owner['driver_pid']}
 for v in [owner['absence'],*release['strict_absence']]:assert v['returncode']==1 and v['stdout']==v['stderr']=='' and v['absent'] is True
 result={'status':'PASS_SAVED_V124_SYNTHETIC_NUMERIC_REVIEW','receipt_sha256':sha(C/'receipt.json'),'cases':40,'matrix_checks':80,'rows_each_arm':152,'readonly_records':120,'complete_BF16_matrix_bytes':sum(x['nbytes'] for x in checks),'resources_worker':14,'resources_owner':5,'source_pins':len(pins),'recorded_peak_allocator_bytes':51724488,'active_sampling_scope':'All14 snapshots precede fixture allocation or follow fixture owner deletion. active0 is not zero GPU allocation/dispatch.','byte_scope':'Saved strict raw-equality and row hashes replayed; full matrix raw outputs not saved, so no fresh tensor recomputation.','cleanup_scope':'Reused root saved wait/reap and fresh external strictabsence; no liveps/socket/model/GPU execution by reviewer.','model_or_speed_or_adoption_qualified':False}
 write('verification.json',result)
 files=[p for p in R.rglob('*') if p.is_file()];write('raw-seals.json',{'files':{str(p):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(files)}})
 (D/'REVIEW.md').write_text('# V124 independent saved-results review\n\nPASS within the synthetic-only local scope. Replayed all 40 cases, 80 complete BF16 matrix equality/finite/shape/dtype records, 152 rows per arm, and 120 full W/scales/X owner-and-byte readonly records. Four of nine target geometries were actually exercised.\n\nSource manifest and all frozen local pins match. All 14 worker and 5 outer resource samples are AC/thermal0/pressure1; recorded Swapouts and swap use do not grow. Peak MLX allocator is 51,724,488 bytes. Active=0 is sampled outside live fixtures and does not imply no GPU allocation or dispatch.\n\nRoot tool54508 exit0, child waited/reaped; root saved strict two-PID external absence reused without liveps. Full raw matrices were not saved: this review replays saved exact checks/hashes, not GPU tensor recomputation. No real-weight, full-model, speed, or adoption conclusion.\n')
 write('receipt.json',{'status':result['status'],'actual_receipt_sha256':result['receipt_sha256'],'files':{n:sha(D/n) for n in ('review.py','verification.json','raw-seals.json','REVIEW.md')},'GPU_used':False})
 print(json.dumps({'status':result['status'],'review_sha256':sha(D/'REVIEW.md'),'receipt_sha256':sha(D/'receipt.json')}))
if __name__=='__main__':main()
