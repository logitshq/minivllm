"""CPU source/actual saved-log rejection checks; no server/model/GPU."""
import ast
import hashlib
import json
from pathlib import Path
import runpy
import sys
from checks import inspect_log,summarize,owned_reap_state
D=Path(__file__).resolve().parent;P=D.parent

def main():
 count=0
 for f in D.glob('*.py'):compile(f.read_text(),str(f),'exec');count+=1
 def funcs(q):return {n.name:ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(q.read_text())) if isinstance(n,ast.FunctionDef)}
 new=funcs(D/'driver.py');old=funcs(P/'m4-final-native15-official-attempt2/client.py')
 for n in ['request','group']:assert new[n]==old[n];count+=1
 cfg=json.loads((D/'config.json').read_text());R=P/'v23-root-control-build1/omlx-mtp-v109-attempt1/child-output'
 none=(R/'none/server.log').read_text();mtp=(R/'mtp/server.log').read_text();alias=str(R/'mtp/models'/cfg['target_revision'])
 inspect_log(none,'none',3,cfg,'unused');count+=1
 p=inspect_log(mtp,'mtp',3,cfg,alias);assert p['decode_routes']==3 and len(p['terminal_stats'])==3 and not p['proposal_count_from_measurement_verified'];count+=1
 def reject(text,mode='mtp',config=cfg,modelalias=alias):
  try:inspect_log(text,mode,3,config,modelalias)
  except AssertionError:return
  raise AssertionError('log counterexample was accepted')
 reject(none+'\nvlm_mtp decode started: request=unexpected uid=-1 block_size=None','none');count+=1
 reject(none);count+=1
 for text in [mtp.replace(cfg['draft_model_dir'],'/wrongdraft'),mtp.replace('VLM MTP drafter attached to scheduler','missing scheduler attach'),mtp.replace('emitted=32','emitted=31'),mtp.replace('block_size=None','block_size=2'),mtp+'\nvlm_mtp routing skipped for badrequest',mtp.replace('VLM MTP drafter loaded:','MTP loader failed:')]:reject(text);count+=1
 # Six-request log fixture, preserving unique route/stat request identity correspondence.
 import re
 prefix='\n'.join(l for l in mtp.splitlines() if any(x in l for x in ['VLM MTP drafter loaded:','VLM MTP drafter attached to scheduler','VLM MTP drafter attached to engine:','VLM MTP enabled for ']))
 body='\n'.join(l for l in mtp.splitlines() if 'vlm_mtp decode started:' in l or 'vlm_mtp stats:' in l)
 second=re.sub(r'request=(\S+)',r'request=copy-\1',body)
 proof=inspect_log(prefix+'\n'+body+'\n'+second,'mtp',6,cfg,alias);assert proof['decode_routes']==6;count+=1
 arms=[{'pair':i,'mode':mode,'label':f'{i}{mode}','x':x,'y':y,'ttft_s':1,'E2E_s':2} for i,vals in enumerate([[(10,10),(9,9)],[(11,12),(10,10)],[(10,10),(11,11)]],1) for mode,(x,y) in zip(['none','mtp'],vals)]
 result=summarize(arms);assert len(result['pairs'])==3 and result['pairs'][0]['x_signed_MTP_pct']<0 and result['none_baseline_retained'] and result['all_signed_pairs_retained'] and not result['negative_or_slower_arms_discarded'];count+=1
 src=(D/'driver.py').read_text();assert "command=official.official_command('omlx',1" in src and "str(OLD/'official_entry.py') in command" in src
 assert 'time.sleep(20)' in src and 'if index<5:' in src and "vlm_mtp_draft_block_size':None" in src and 'install(' not in src and 'import observer' not in src;count+=1
 assert 'qualification_observers' in src and 'measurement_wrappers' in src and 'proposal_count_from_measurement_verified' in (D/'checks.py').read_text();count+=1
 import tempfile
 with tempfile.TemporaryDirectory() as tmp:
  arm=Path(tmp)/'failed-arm';arm.mkdir();(arm/'owned-pid.json').write_text(json.dumps({'pid':123}))
  assert owned_reap_state(tmp)['all_owned_servers_reaped'] is False;count+=1
  value={'pid':123,'wait_completed':True,'exact_pid_absence':{'pid':123,'returncode':1,'stdout':'','stderr':''}}
  (arm/'owned-process-exit.json').write_text(json.dumps(value));assert owned_reap_state(tmp)['all_owned_servers_reaped'];count+=1
  value['exact_pid_absence']['returncode']=0;(arm/'owned-process-exit.json').write_text(json.dumps(value));assert not owned_reap_state(tmp)['all_owned_servers_reaped'];count+=1
 assert not any(n=='mlx' or n.startswith(('mlx.','numpy')) for n in sys.modules)
 print(json.dumps({'status':'PASS_CPU_AST_SAVED_LOG_FAIL_CLOSED_AND_SIGNED_STATS','checks':count,'GPU_used':False,'server_started':False}));return count
if __name__=='__main__':main()
