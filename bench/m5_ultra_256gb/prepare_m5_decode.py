"""Build a separately frozen M5 decode experiment without altering build1 evidence."""
from pathlib import Path
import hashlib,json,shutil
repo=Path(__file__).resolve().parents[2];base=repo/'.artifacts/m5-ultra-256gb';old=base/'source';source=base/'source-m5-decode-v2';harness=base/'build2-harness'
if source.exists() or harness.exists():raise SystemExit('Preserve existing build2; never overwrite')
source.mkdir();harness.mkdir();(source/'reference').mkdir()
shutil.copy2(old/'reference/workload.json',source/'reference/workload.json')
manifest=json.loads((old/'SOURCE-VERSIONS.json').read_text())
candidate=(repo/'tutorial_code/miniengine/model/gdn.py').read_text()
helper=candidate[candidate.index('@lru_cache'):candidate.index('def causal_conv1d')]
for dirname,key in [('accepted15','accepted15'),('v19','V19')]:
 shutil.copytree(old/dirname/'tutorial_code/miniengine',source/dirname/'tutorial_code/miniengine')
 p=source/dirname/'tutorial_code/miniengine/model/gdn.py'
 if dirname=='v19':p.write_text(candidate)
 else:
  s=p.read_text().replace('import mlx.nn as nn','import mlx.nn as nn\nfrom functools import lru_cache').replace('prefill_linear, prefill_batch_backend_supported','prefill_linear, prefill_batch_backend_supported, _prefill_backend_identity')
  s=s.replace('def causal_conv1d',helper+'def causal_conv1d').replace('and prefill_batch_backend_supported())','and gdn_decode_backend_supported())');p.write_text(s)
 files={str(p.relative_to(source/dirname/'tutorial_code/miniengine')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source/dirname/'tutorial_code/miniengine').rglob('*.py')}
 manifest[key]['upstream_files_sha256']=manifest[key]['files_sha256'];manifest[key]['files_sha256']=files
 manifest[key]['local_variant']='M5 decode eligibility only; original upstream commit retained as provenance, not exact variant identity'
(source/'SOURCE-VERSIONS.json').write_text(json.dumps(manifest,indent=2)+'\n')
for p in (repo/'bench/m5_ultra_256gb').glob('*.py'):
 if p.name in ('prepare_m5_decode.py',):continue
 s=p.read_text().replace('fullqual-build1','fullqual-build2').replace('go-m5-ultra-build1','go-m5-ultra-build2').replace('receipt-m5-ultra-build1','receipt-m5-ultra-build2')
 s=s.replace("D.parents[1]",repr(str(repo))) if False else s
 s=s.replace("D.parents[1]", "Path("+repr(str(repo))+")").replace("CODE.parents[1]", "Path("+repr(str(repo))+")")
 s=s.replace("repo/'.artifacts/m5-ultra-256gb/source'", "repo/'.artifacts/m5-ultra-256gb/source-m5-decode-v2'")
 if p.name=='worker.py':
  s=s.replace("catalog=[];result['raw_logits']=catalog", "catalog=[];result['raw_logits']=catalog")
  s=s.replace("result['rounds']=[];expected_obs=None", "result['rounds']=[];expected_obs=json.loads(Path(f['original_observations']).read_text()) if role=='accepted' else None")
 if p.name=='freeze.py':
  s=s.replace("paths.extend(p for p in model.iterdir() if p.is_file())", "paths.extend(p for p in model.iterdir() if p.is_file())\noriginal=repo/'.artifacts/m5-ultra-256gb/fullqual-build1/gpu-fullmodel-run1/accepted'\nf['original_observations']=str(original/'accepted-observations.json')\npaths.append(Path(f['original_observations']))\npaths.extend((original/'accepted').glob('*.bin'))")
 if p.name=='test_migration.py':
  # This extra initial observation comparison is the sole numerical-body addition.
  s=s.replace("new=(D/'worker.py').read_text();old=", "new=(D/'worker.py').read_text().replace(\"result['rounds']=[];expected_obs=json.loads(Path(f['original_observations']).read_text()) if role=='accepted' else None\",\"result['rounds']=[];expected_obs=None\");old=")
 (harness/p.name).write_text(s)
print(harness)
