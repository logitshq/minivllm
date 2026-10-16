"""Independent stdlib-only replay. State signatures are authenticated hashes, not retained raw state."""
import argparse,hashlib,json,math,struct,os
from pathlib import Path
CODE=Path(__file__).resolve().parent
D=Path(os.environ.get('MINIVLLM_EXPERIMENT_DOMAIN',str(CODE.parents[1]/'.artifacts/m5-ultra-256gb/fullqual-build1'))).resolve()

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  while b:=f.read(1<<20):h.update(b)
 return h.hexdigest()

def same_bytes(a,b):
 with Path(a).open('rb') as x,Path(b).open('rb') as y:
  while True:
   p,q=x.read(1<<20),y.read(1<<20)
   if p!=q:raise ValueError('strict byte difference')
   if not p:return True

def require(value,message):
 if not value:raise ValueError(message)

def signature(s):
 require(set(s)=={'shape','dtype','sha256','canonical_bytes'},'signature schema differs')
 require(s['canonical_bytes']==math.prod(s['shape'])*4,'signature shape/byte count differs')
 require(type(s['sha256']) is str and len(s['sha256'])==64 and all(c in '0123456789abcdef' for c in s['sha256']),'signature SHA malformed')
 require(s['dtype'] in ('mlx.core.bfloat16','mlx.core.float32'),'signature dtype differs')

def verify(output,include_owner=True):
 output=Path(output);f=json.loads((D/'frozen.json').read_text());freeze=digest(D/'frozen.json')
 for path,want in f['files'].items():require(digest(path)==want,'frozen pin changed '+path)
 expected=json.loads((D/'chapter15-smoke-expected-v1.json').read_text());layers=[str(i) for i in range(64) if i%4!=3];want={i:31 for i in layers}
 baseline=None;committed=None;raw_count=state_sigs=0;all_names=[];worker_resources=0;authorization=None
 for role,names in [('accepted',['accepted']),('candidate',['candidate-off','candidate-on','candidate-reset-off'])]:
  root=output/role;r=json.loads((root/'receipt.json').read_text())
  require(digest(root/'ROOT-GO.raw.json')==r['GO_sha256'] and json.loads((root/'ROOT-GO.raw.json').read_text())==r['GO'],'saved original GO bytes differ')
  if authorization is None:authorization=r['GO']
  else:require(r['GO']==authorization,'workers had different authorization')
  require(r['status']=='PASS_WORKER_BOUNDED_SCOPE' and r['frozen_sha256']==freeze,'worker status/freeze differs')
  require(r['model_load_attempts']==1 and r['default_off_checked'] and r['hooks_restored'] and not r['cleanup_errors'],'worker lifecycle differs')
  require([x['name'] for x in r['rounds']]==names,'fixed rounds differ')
  require(len(r['parameters']['initial'])==1349,'parameter count differs')
  require(all(name+'-'+side in r['parameters']['unchanged_at'] for name in names for side in ('before','after')),'parameter boundary missing')
  require(len(r['raw_resources'])==(5 if role=='accepted' else 11),'fixed resource sampling scope differs')
  for entry in r['raw_resources']:
   path=root/entry['file'];require(digest(path)==entry['sha256'],'raw resource changed');s=json.loads(path.read_text());worker_resources+=1
   require(s['power_metadata']['status']=='known' and s['power_metadata']['source']=={'AC':'AC Power','Battery':'Battery Power'}[authorization['expected_power']] and s['processinfo_thermal_metadata']['status']=='known' and s['processinfo_thermal_metadata']['code'] in (0,1) and s['pressure_level_value']==1,'actual raw resource guard differs')
  for row in r['rounds']:
   name=row['name'];all_names.append(name);obs=json.loads((root/row['observations_file']).read_text())
   require(row['output_ids']==expected['expected_output_ids'] and row['canonical_array_hashes']==4128 and row['fresh_cache_idle'],'IDs/hash count/cache scope differs')
   require(len(obs)==32 and [x['mode'] for x in obs]==['prefill']+['decode']*31 and [x['positions'][0] for x in obs]==list(range(18,50)),'model invocation protocol differs')
   require([x['greedy'][0] for x in obs]==row['output_ids'],'greedy logits token differs')
   on=name=='candidate-on';c=row['routes']
   require(all(c[k]==(want if on else {}) for k in ('norm_public','decode_norm','prepare','state','prework_dispatch','state_dispatch')) and c['legacy']==({} if on else want) and c['compiled_gates']==want and c['compile_creations']==48 and c['kernel_creations']==({'prework':1,'state':1} if on else {}),'actual N2 public/function/dispatch/gate counts differ')
   for step,x in enumerate(obs):
    signature(x['logits']);require(len(x['states'])==128,'missing cache signatures')
    for s in x['states'].values():signature(s);state_sigs+=1
   if baseline is None:baseline=obs;committed=row['committed_positions']
   else:require(obs==baseline and row['committed_positions']==committed,'strict complete signature/cache mismatch')
  require(len(r['raw_logits'])==32*len(names),'raw logits scope differs')
  for entry in r['raw_logits']:
   path=root/entry['file'];s=entry['signature'];signature(s)
   require(path.stat().st_size==s['canonical_bytes'] and digest(path)==s['sha256'],'raw logits bytes/hash differ')
   with path.open('rb') as stream:
    while block:=stream.read(1<<20):require(all(math.isfinite(x[0]) for x in struct.iter_unpack('<f',block)),'nonfinite raw logits')
   name=Path(entry['file']).parts[0];step=int(Path(entry['file']).name[:2]);same_bytes(path,output/'accepted'/'accepted'/('%02d-logits.float32le.bin'%step));require(s==json.loads((root/(name+'-observations.json')).read_text())[step]['logits'],'raw logits linked signature differs');raw_count+=1
  if role=='candidate':
   require(len(r['fallbacks'])==18 and len({x['name'] for x in r['fallbacks']})==18,'fallback scope incomplete')
   for guard in r['fallbacks']:
    require(guard['strict_equal'] and guard['signatures'][0]==guard['signatures'][1],'fallback signature mismatch')
    require(all(not route[k] for route in guard['routes'] for k in ('norm_public','decode_norm','prepare','state','prework_dispatch','state_dispatch','kernel_creations')),'fallback executed production fusion')
   require(r['pipeline_lifecycle']==[{'round':n,'prework_present':n!='candidate-off','state_present':n!='candidate-off','probe_absent':True,'reset_preserves_factories':n=='candidate-reset-off'} for n in names],'N2 lazy cache/reset lifecycle differs')
   require('final-reset-off' in r['parameters']['unchanged_at'],'final reset parameter check missing')
 if include_owner:
  owner=json.loads((output/'receipt.json').read_text());require(owner['status']=='PASS_FULLMODEL_BOUNDED_SCOPE' and not owner['cleanup_errors'],'owner result differs')
  require(digest(output/'ROOT-GO.raw.json')==owner['GO_sha256'] and owner['GO']==authorization,'owner authorization differs')
  require([r['role'] for r in owner['children']]==['accepted','candidate'],'serial children scope differs')
  for child in owner['children']:
   raw=json.loads((output/child['role']/'exact-absence.json').read_text())
   require(child['waited_reaped'] and child['returncode']==0 and raw==child['absence'] and raw['pid']==child['pid'] and raw['returncode']==1 and raw['stdout']=='' and raw['stderr']=='','child exact wait/reap/absence differs')
  require(len(owner['raw_resources'])==4,'owner resource sampling scope differs')
  for entry in owner['raw_resources']:require(digest(output/entry['file'])==entry['sha256'],'owner original raw resource changed')
 result={'verdict':'PASS_FULLMODEL_BOUNDED_SCOPE','frozen_sha256':freeze,'rounds':all_names,'raw_logits_files':raw_count,'state_signatures':state_sigs,'worker_raw_resources':worker_resources,
  'strict_full_state_hash_equality':True,'full_state_raw_permanently_retained':False,'all_actual_dispatch_counts_checked':True,
  'fallbacks':18,'performance_E2E_default_on_reload_spec_inference_qualified':False,'external_owner_reap_required':True}
 return result

if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output_dir',type=Path);args=ap.parse_args();print(json.dumps(verify(args.output_dir),indent=2))
