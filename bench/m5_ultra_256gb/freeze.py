"""Create a new local experiment domain, never overwrite an earlier attempt."""
import hashlib, importlib.metadata as md, json, os, platform, sys
from pathlib import Path
from runtime import digest
D=Path(__file__).resolve().parent;repo=D.parents[1]
P=Path(os.environ.get('MINIVLLM_EXPERIMENT_DOMAIN',str(repo/'.artifacts/m5-ultra-256gb/fullqual-build1'))).resolve()
P.mkdir(parents=True,exist_ok=True)
if (P/'frozen.json').exists():raise SystemExit('Domain already frozen; preserve it')
source=repo/'.artifacts/m5-ultra-256gb/source'
manifest=json.loads((source/'SOURCE-VERSIONS.json').read_text())
f={'schema':'m5-ultra-256gb-fullqual-build1','python_version':sys.version,'platform':platform.platform(),'selected_environment':{'MLX_MAX_OPS_PER_BUFFER':'50','MLX_MAX_MB_PER_BUFFER':'50','MLX_BFS_MAX_WIDTH':'20','MLX_METAL_FAST_SYNCH':'0'},'files':{},'distributions':{k:md.version(k) for k in ['mlx','mlx-metal','numpy','tokenizers','huggingface_hub','httpx']},'source_commits':{k:v['commit'] for k,v in manifest.items() if isinstance(v,dict) and 'commit' in v}}
for role,key,dirname in [('accepted','accepted15','accepted15'),('candidate','V19','v19')]:
 root=source/dirname/'tutorial_code/miniengine';actual={str(p.relative_to(root)):digest(p) for p in root.rglob('*.py')}
 if actual!=manifest[key]['files_sha256']:raise ValueError('export source identity mismatch')
 f[role+'_root']=str(root);f[role+'_sources']=actual
 f[('baseline' if role=='accepted' else role)+'_source_manifest_sha256']=hashlib.sha256(json.dumps(actual,sort_keys=True).encode()).hexdigest()
f['baseline_sources']=f['accepted_sources']
w=json.loads((source/'reference/workload.json').read_text());model=(repo/'models/Qwen3.8-27B-nvfp4').resolve()
if model.name!='5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6':raise ValueError('model revision mismatch')
w['model_dir']=str(model);(P/'workload.json').write_text(json.dumps(w,indent=2)+'\n')
expected={'prompt_ids':w['prompt_ids'],'expected_output_ids':w['continuation_ids'],'checkpoint':{'actual_path':str(model),'metadata_sha256':{p.name:digest(p) for p in model.iterdir() if p.name in ('config.json','tokenizer.json','generation_config.json') or p.name.endswith('.safetensors.index.json')}}}
(P/'chapter15-smoke-expected-v1.json').write_text(json.dumps(expected,indent=2)+'\n')
f['GO_schema']={'schema':'v19-fullmodel-qual-go-m5-ultra-build1','stage':'fullmodel_correctness_no_timing','frozen_manifest_sha256':None,'sole_gpu_owner':True,'actual_unsandboxed_execution':True,'attempt':1,'serial_model_loads':2,'baseline_gates_compile':True,'candidate_default_off':True,'no_performance':True,'baseline_source_manifest_sha256':f['baseline_source_manifest_sha256'],'candidate_source_manifest_sha256':f['candidate_source_manifest_sha256'],'expected_power':None,'authorizing_owner_pid':None,'output_dir':None}
paths=list(D.glob('*.py'))+[P/'workload.json',P/'chapter15-smoke-expected-v1.json',Path(sys.executable).resolve(),source/'SOURCE-VERSIONS.json']
for role in ('accepted','candidate'):paths.extend(Path(f[role+'_root']).rglob('*.py'))
for dist in ['mlx','mlx-metal','numpy','tokenizers']:
 distribution=md.distribution(dist)
 paths.extend(Path(distribution.locate_file(p)).resolve() for p in distribution.files or [] if str(p).endswith(('.py','.so','.dylib','.metallib','.h','METADATA','RECORD')))
audit=repo/'.artifacts/m5-ultra-256gb/model-hash-audit.json'
verified={r['name']:r for r in json.loads(audit.read_text())}
paths.append(audit)
paths.extend(p for p in model.iterdir() if p.is_file())
for p in sorted(set(paths)):
 if not p.is_file():continue
 h=digest(p);f['files'][str(p)]=h
 if p.parent==model and p.name in verified and h!=verified[p.name]['hub_lfs_sha256']:raise ValueError('HF LFS SHA mismatch '+str(p))
(P/'frozen.json').write_text(json.dumps(f,indent=2,sort_keys=True)+'\n');(P/'frozen.sha256').write_text(digest(P/'frozen.json')+'\n')
print(json.dumps({'domain':str(P),'pins':len(f['files']),'frozen_sha256':digest(P/'frozen.json'),'environment':f['selected_environment'],'distributions':f['distributions']},indent=2))
