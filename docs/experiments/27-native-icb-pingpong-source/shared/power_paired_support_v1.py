"""CPU-only identity, phase guard and failure preservation for power-paired runs."""
import ast,hashlib,json,os,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
P=Path(__file__).resolve().parent
EXPECTED={
 'bench_service_precise_v2.py':'b62315ba705dfcb7ea354095d9dbc95f380f538fd00f6304a85e19cca6debfb5',
 'core_http_precise_v1.py':'152e7ec2ef39498f3fe9fecf5a815d8e5e8478f54f85b666db6fae22c60998f4',
 'resource_snapshot_precise_v1.py':'d0aaa466e1cee70e109dbd4d02e4d457edda2a5e535dafab1b17d9dd6654538f',
 'resource_snapshot_power_v1.py':'f4624d7dd8e977759865e11c4b7d3ed0489d1fd55b496e60c29fc087b7d9eb42',
 'source-manifest-v7-accepted13-build1.json':'2a990456522fdd00d0f331011964be4e9b755e47b2b6eeccd293b1a8c435baad',
 'official-omlx-environment-recovery-v1.json':'4ffd1edada0f77d342bc3a7ccf54b1796de8ade454a876768c31ee3e958bc783',
 'freeze_official_environment_v1.py':'c871041168af74ecff3ab2877f3a80a4149f5aa114a29c68185e38f85de876d6'}
NATIVE_PYTHON=Path('/Users/ryanli/code/minivllm/.venv/bin/python')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_new(path,value):
 with Path(path).open('x') as f:f.write(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def verify_inputs():
 actual={k:sha(P/k) for k in EXPECTED};assert actual==EXPECTED,'pinned input changed'
 assert not [k for k in os.environ if k.startswith('OMLX_')],'inherited OMLX overrides forbidden'
 assert os.environ.get('PYTHONPATH') is None and os.environ.get('PYTHONHOME') is None,'inherited interpreter path override'
 m=json.loads((P/'source-manifest-v7-accepted13-build1.json').read_text());r=Path(m['source_root']);assert r==P/'candidate-v7-accepted13-build1-root/tutorial_code/miniengine'
 actual_native={str(f.relative_to(r)):sha(f) for f in r.rglob('*.py')};assert actual_native=={k:v['after_sha256'] for k,v in m['files'].items()},'frozen native source mismatch'
 base=(P/'bench_service_precise_v2.py').read_text();new=(P/'bench_service_power_pairs_v1.py').read_text();checks={}
 for name in ['request','group']:
  nodes=[next(n for n in ast.walk(ast.parse(s)) if isinstance(n,ast.FunctionDef) and n.name==name) for s in [base,new]]
  segs=[ast.get_source_segment(s,n) for s,n in zip([base,new],nodes)]
  assert segs[0]==segs[1];assert ast.dump(nodes[0],include_attributes=False)==ast.dump(nodes[1],include_attributes=False)
  checks[name]={'source_segment_byte_equal':True,'ast_equal':True,'sha256':hashlib.sha256(segs[0].encode()).hexdigest()}
 return {'pinned_inputs':actual,'native_source_root':str(r),'native_source_file_count':len(actual_native),'request_group':checks,'derived_client_sha256':sha(P/'bench_service_power_pairs_v1.py')}
def phase_guard(samples,expected_power='any',owned_pid=None):
 from resource_snapshot_power_v1 import guard
 guard(samples);s=samples[-1];power=s.get('power_metadata',{});thermal=s.get('processinfo_thermal_metadata',{})
 assert power.get('status')=='known','power metadata unknown'
 assert thermal.get('status')=='known','thermal metadata unknown'
 assert thermal.get('code') in [0,1],'thermal serious/critical: stop owned model'
 if expected_power!='any':assert power['source']=={'AC':'AC Power','Battery':'Battery Power'}[expected_power], 'unexpected power source: '+str(power['source'])
 if owned_pid is not None:
  e=s.get('exact_owned_pid_metadata',{});assert e.get('status')=='known' and e.get('pid')==owned_pid,'exact owned PID unknown/not alive'
 return s
def record_failure(directory,exc,phase,samples,pid=None):
 d=Path(directory);i=1
 while (d/f'failure-{i:03d}.json').exists():i+=1
 write_new(d/f'failure-{i:03d}.json',{'error_type':type(exc).__name__,'error':str(exc),'phase':phase,'owned_pid':pid,'recorded_at':datetime.now(timezone.utc).isoformat(),'resource_snapshots':samples})
def pid_absence(pid):
 r=subprocess.run(['/bin/ps','-p',str(pid),'-o','pid='],capture_output=True,text=True,timeout=10)
 absent=(r.returncode==1 and not r.stdout.strip())
 return {'pid':pid,'absent':True if absent else False if r.returncode==0 and r.stdout.strip()==str(pid) else None,'returncode':r.returncode,'stdout':r.stdout.strip(),'stderr':r.stderr.strip(),'scope':'exact supplied PID existence after owned wait; no inference about other processes'}
def native_identity():
 # find_spec on top-level package names and distribution metadata does not import MLX.
 probe="""import importlib.util,importlib.metadata as m,json,sys
out={'python_executable':sys.executable,'python_version':sys.version,'sys_path':sys.path,'module_resolution':{},'distributions':{}}
for n in ['mlx','mlx_lm','mlx_vlm','tokenizers','httpx']:
 s=importlib.util.find_spec(n);out['module_resolution'][n]={'origin':s.origin if s else None,'roots':list(s.submodule_search_locations or []) if s else [],'installed':s is not None}
for n in ['mlx','mlx-lm','mlx-vlm','tokenizers','httpx']:
 try:
  d=m.distribution(n);out['distributions'][n]={'version':d.version,'root':str(d.locate_file('')),'files':[str(d.locate_file(f)) for f in (d.files or [])]}
 except m.PackageNotFoundError:out['distributions'][n]={'status':'not installed','files':[]}
print(json.dumps(out))"""
 r=subprocess.run([str(NATIVE_PYTHON),'-c',probe],capture_output=True,text=True,check=True);resolved=json.loads(r.stdout);files={NATIVE_PYTHON.resolve()}
 for mod in resolved['module_resolution'].values():
  for root in mod['roots']:
   files.update(f.resolve() for f in Path(root).rglob('*') if f.is_file() and f.suffix in {'.py','.metal','.h','.hpp','.c','.cc','.cpp','.mm','.so','.dylib'})
 for dist in resolved['distributions'].values():
  for name in dist.pop('files'):
   f=Path(name)
   if f.is_file() and (f.suffix in {'.py','.metal','.h','.hpp','.c','.cc','.cpp','.mm','.so','.dylib'} or f.name in {'METADATA','direct_url.json'}):files.add(f.resolve())
 assert resolved['distributions']['mlx']['version']=='0.32.3'
 hashes={str(f):sha(f) for f in sorted(files)}
 return {'scope':'CPU-only actual native launch interpreter resolution, sources and native binaries; no model import','actual_environment':resolved,'source_and_native_file_sha256':hashes,'source_and_native_tree_sha256':hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),'source_contract':verify_inputs()}
