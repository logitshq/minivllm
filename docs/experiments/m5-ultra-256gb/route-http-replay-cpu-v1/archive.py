from pathlib import Path
import hashlib,json,tarfile,shutil
R=Path('/Users/ryanli/code/minivllm');A=R/'.artifacts/m5-ultra-256gb/route-http-replay-cpu-v1';D=R/'docs/experiments/m5-ultra-256gb/route-http-replay-cpu-v1'
assert not D.exists()
files=sorted(p for p in A.rglob('*') if p.is_file());assert all(not p.is_symlink() and p.stat().st_size<4*1024*1024 for p in files)
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest={str(p.relative_to(A)):{'sha256':h(p),'bytes':p.stat().st_size} for p in files}
result=json.loads((A/'result.json').read_bytes());assert result['status']=='PASS' and result['tests']==13 and result['failures']==result['errors']==result['skipped']==0
D.mkdir();target=D/'evidence.tar.gz'
with tarfile.open(target,'w:gz',dereference=True) as f:
 for p in files:f.add(p,arcname=str(p.relative_to(A)),recursive=False)
actual={}
with tarfile.open(target,'r:gz') as f:
 for m in f:
  assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts and m.name not in actual
  raw=f.extractfile(m).read();actual[m.name]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
assert actual==manifest and all(h(A/n)==v['sha256'] for n,v in manifest.items())
(D/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for name in ('result.json','post-review-doc-only.json'):shutil.copyfile(A/name,D/name)
shutil.copyfile(__file__,D/'archive.py')
summary={'status':'PASS','members':len(files),'tests':13,'archive_sha256':h(target),'archive_bytes':target.stat().st_size,'members_originals_verified':True,'actual_http_qualified':False,'performance_qualified':False}
(D/'archive-verification.json').write_text(json.dumps(summary,indent=2)+'\n')
(D/'README.md').write_text('''# Route HTTP replay adapter: 13 CPU tests PASS

These fake-backend tests cover new plan/source/capture/serving bindings, rejected absent reap and wrong route inputs, candidate installation/cleanup, and mismatched/extra/missing route rows. Real runtime/model imports, sockets, subprocesses and signals were prohibited. No GPU, actual HTTP capture/replay or performance qualification occurred.

The complete original command, stdout/stderr, tested source bytes and before/after hashes are preserved in evidence.tar.gz. After the run, one docstring phrase was corrected from “42-parameter” to “42-direct-call”; post-review-doc-only.json records that replacement and identical AST with the docstring excluded. The post-edit file was not falsely reported as rerun. Independent HTTP route audit and actual capture→reap→replay ownership remain required.

Archive member hashes and the unchanged original files were verified; [manifest](manifest.json), [result](result.json), [documentation-only edit](post-review-doc-only.json), [verification](archive-verification.json). No new numerical or timing point.
''')
print(json.dumps(summary))
