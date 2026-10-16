"""CPU-only review of saved V64 results; no framework/library execution."""
from pathlib import Path
import hashlib,json
P=Path('/private/tmp/minivllm-engine-optimization-20261014');D=Path(__file__).resolve().parent
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
goldpath=P/'v23-root-control-build1/fullmodel-v44native-attempt1/child-output/receipt.json';assert sha(goldpath)=='1dcfc41e7779c575602bd6c73bae4e4800664db42e1ee87e75414335bdc45b6d'
gold=json.loads(goldpath.read_text());golds={r['name']:r for r in gold['requests']};results={}
for mode,tag,digest in [('diagnostic','diag','86930ab760f361ae38204d3c79f4c101819289054720528765f62954e256276a'),('production','prod','7bb39bf79ac1f31c66a2cbb25bcbcd3e480e3be9737b3c2ffc1f886afe82bb63')]:
 q=P/f'v23-root-control-build1/fullmodel-v64{tag}-attempt1/child-output/receipt.json';e=q.parent.parent/'external-release.json';assert sha(q)==digest
 r=json.loads(q.read_text());released=json.loads(e.read_text());assert released['status']=='PASS_EXTERNAL_RELEASE' and released['GPU_released'] and released['tool_exit']==0 and all(x['returncode']==1 and x['stdout']==x['stderr']=='' for x in released['strict_absence'])
 assert r['mode']==mode and r['model_loads']==1 and r['build_input131_readonly'] and r['captured_originals_readonly'] and r['terminal_cleanup']
 assert (r['native_steps'],r['native_completed_steps'],r['total_submitted_steps'],r['total_discarded_steps'],r['request_count'],r['resets'])==(161,162,162,1,6,6)
 def compare(rows,reference):
  assert len(rows)==len(reference)==132
  for a,b in zip(rows,reference):assert a['label']==b['label'] and a['sha256']==a['qualified_A_sha256']==b['A_sha256']
 count=0
 for c in r['requests']:
  g=golds[c['name']];assert c['B_ids']==g['A_ids'] and len(c['steps'])==31 and c['captured_originals_readonly']
  if mode=='diagnostic':
   for row,step in zip(c['steps'],g['steps']):compare(row['checks'],step['outputs']);count+=132
  else:
   assert all(row['full_output_hash_checks']==0 and 'checks' not in row for row in c['steps'])
   compare(c['final132'],g['steps'][-1]['outputs']);count+=132
 for key in ('post_fallback_final132','post_early_stop_final132'):compare(r[key],gold['requests'][0]['steps'][-1]['outputs']);count+=132
 canonical=gold['requests'][0]['A_ids'];assert r['post_fallback_ids']==r['post_early_stop_ids']==[canonical]
 early=r['early_stop'];assert early['stop_token_id']==103725 and early['ids']==[canonical[:7]] and early['all132_output_checks']==0 and early['captured_originals_readonly']
 assert early['native_delivered_delta']==6 and early['native_completed_delta']==7 and early['discarded_delta']==1
 n=r['native'];f,p=n['feedback'],n['pipeline'];d=n['direct_pipeline']
 assert n['status']=='OK' and p['total_completed']==162 and p['total_delivered']==161 and p['total_discarded']==1 and p['total_failed']==p['inflight']==0
 assert f['resets']==6 and f['icb']['bank_executions']==[84,78] and f['pingpong']['eligible_count']==48 and f['pingpong']['staged_link_count']==83
 assert d['enabled'] and d['diagnostic_snapshot']==(mode=='diagnostic') and d['banks_checked']==2 and d['input_output_range_pairs_checked']==34584 and d['input_input_range_pairs_checked']==17030
 assert d['direct_feedback_bytes_per_step']==19726348 and d['direct_links']==83 and f['input_output_aliases']==[]
 assert count==r['strict_output_checks']==(12540 if mode=='diagnostic' else 660) and r['core_all132_steps']==(93 if mode=='diagnostic' else 3)
 assert d['quiescent_output_copies']==(0 if mode=='diagnostic' else 660) and d['legacy_F_stage_used']==(mode=='diagnostic') and d['legacy_F_stage_allocation_bytes_retained']==20223012
 results[mode]={'receipt':str(q),'sha256':digest,'release':str(e),'release_sha256':sha(e),'status':r['status'],'independently_recompared_output_hashes':count,'native_delivered_IDs':161,'native_completed':162,'discarded':1,'resets':6,'banks':[84,78],'reported_original_capture_owners_readonly':True,'production_perstep_allstate_claim':False,'GPU_used':False}
(D/'selected-fields.json').write_text(json.dumps({'status':'PASS_CPU_SAVED_V64_RESULTS_REVIEW','modes':results,'GPU_used':False},indent=2)+'\n');print(json.dumps(results,indent=2))
