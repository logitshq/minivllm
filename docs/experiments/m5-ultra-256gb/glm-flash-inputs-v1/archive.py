import hashlib,json,tarfile,io,shutil
from pathlib import Path
R=Path('/Users/ryanli/code/minivllm')
D=R/'docs/experiments/m5-ultra-256gb/glm-flash-inputs-v1'
S=[R/'.artifacts/m5-ultra-256gb'/n for n in ('glm-flash-inputs-v1','glm-flash-inputs-v1-request','glm-flash-inputs-v1-execution')]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
assert not D.exists()
assert json.loads((S[2]/'result.json').read_bytes())['returncode']==0
D.mkdir()
entries=[]
for root in S:
 for p in sorted(root.rglob('*')):
  assert not p.is_symlink()
  if p.is_file():
   entries.append(dict(member=f'raw/{root.name}/{p.relative_to(root)}',source=str(p),bytes=p.stat().st_size,sha256=sha(p)))
archive=D/'evidence.tar.xz'
with tarfile.open(archive,'x:xz',preset=6) as t:
 for row in entries:
  p=Path(row['source']);assert sha(p)==row['sha256'];t.add(p,arcname=row['member'],recursive=False)
with tarfile.open(archive,'r:xz') as t:
 members=t.getmembers();assert [m.name for m in members]==[r['member'] for r in entries]
 for m,row in zip(members,entries):
  assert m.isfile() and not m.name.startswith('/') and '..' not in Path(m.name).parts
  data=t.extractfile(m).read();original=Path(row['source']).read_bytes()
  assert data==original and len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
write(D/'manifest.json',dict(schema='GLM-inputs-CPU-archive-v1',files=entries,archive=dict(file=archive.name,bytes=archive.stat().st_size,sha256=sha(archive)),members=len(entries),raw_bytes=sum(r['bytes'] for r in entries)))
write(D/'archive-verification.json',dict(status='PASS',members=len(entries),all_member_bytes_equal_original=True,all_member_sha256_equal=True,original_tree_membership_unchanged=all(sorted(str(p) for p in root.rglob('*') if p.is_file())==sorted(r['source'] for r in entries if Path(r['source']).is_relative_to(root)) for root in S),model_loaded=False,service_prepared=False))
(D/'README.md').write_text('''# GLM5.3Flash input preparation, first actual CPU run\n\nThe first explicit CPU input preparation exited 0. No model was loaded, no backend was imported, and no service was changed. This is not numerical, fit, resource-safety, or performance qualification.\n\nThe archive retains all three original trees: input plans/inventory, the fixed plain coding prompt, and execution attempt/result/logs/source copies. Every tar member was compared byte for byte and by SHA-256 with its original file after compression. `manifest.json` records the original paths, sizes, member names and SHA-256 values.\n\nPreparation hashed 24 explicitly supplied runtime files (43,842,459 bytes) and retained source/metadata pins. It checked new file statistics for all 62 shards (203,992,076,296 bytes), using the previously retained download SHA-256 values; it did not reopen or rehash shard payloads. The inventory is a supplied set, not a recursive dependency closure. Model identity is `glm5_next`, revision `fa04cbc3622b16aa7475404060b05e506d81bd4e`.\n\nThe fixed plain prompt has no chat template. Actual tokenization and the <=128-token gate remain for the future probe. The unchanged probe uses normal EOS and output caps 8/32/32/32. A separate actual cold-service registration and physical admission are still required.\n\nReplay: inspect `manifest.json`, ensure the tar contains exactly its listed regular-file members (no absolute paths, links or `..`), then extract into a new directory. All members live under `raw/glm-flash-inputs-v1*`. The files contain historical absolute paths; extraction does not authorize or run those commands.\n''')
shutil.copyfile(__file__,D/'archive.py')
print(json.dumps(dict(directory=str(D),members=len(entries),raw_bytes=sum(x['bytes'] for x in entries),archive_bytes=archive.stat().st_size,archive_sha256=sha(archive),manifest_sha256=sha(D/'manifest.json'))))
