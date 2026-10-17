"""Read-only standard-library V117 source verification; no MLX/GPU imports."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

H=Path(__file__).resolve().parent;P=H.parent
D=P/'v117-verify-broadcast-m1-bounded-load-build1';OLD=P/'v114-verify-broadcast-m1-build1'

def load(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def verify():
 assert sha(D/'fullqual.py')=='8ff6d9599f002f7169dbbeba33b81e8085c039ca961d1495932e449fe1d47633'
 assert sha(D/'source-pins.json')=='57c4f2e30539ad83357b323e42bbe0b6d07b243b8b52b6d6d1ec0c04aab6c82c'
 assert sha(D/'CPU-READY.json')=='ec46a1e921984dc834c771f86f644b08892a753bc6d339f16c6ba7c9908e548b'
 pins=load(D/'source-pins.json');assert len(pins)==216
 for p,h in pins.items():assert sha(p)==h,p
 for n,h in load(D/'CPU-READY.json')['file_sha256'].items():assert sha(D/n)==h,n
 candidate=P/'candidate-v116-bounded-draftloader-build1-root'
 for n,h in load(P/'v116-bounded-draftloader-build1/candidate-manifest.json')['files'].items():assert sha(candidate/n)==h,n
 assert len(list(candidate.rglob('*.py')))==30
 s=(D/'fullqual.py').read_text()
 replacements=[
  ('V117 root-only full27B','V114 root-only full27B'),
  (",'V116_binding':json.loads((D/'v116-binding.json').read_text())['status']",''),
  ('    from loader_gate import require_v116\n    loader_prerequisite=require_v116()\n',''),
  (",'V116_loader_prerequisite':loader_prerequisite,'bounded_loader_stages':[]",''),
  ("        def bounded_loader_stage(label, detail):\n            r['bounded_loader_stages'].append({'label':label,'detail':detail,'allocator':memory(mx)})\n            save()\n",''),
  ('load_drafter(protocol[\'draft_snapshot\'],bounded_load=True,stage_observer=bounded_loader_stage)',"load_drafter(protocol['draft_snapshot'])"),
  ('PASS_V117_BOUNDED_LOAD_VERIFY_BROADCAST_M1_OLD96IDS_LITERAL_AND_ORDINARY_M1_COMMIT128_LASTLOGITS_STRICT','PASS_V114_VERIFY_BROADCAST_M1_OLD96IDS_LITERAL_AND_ORDINARY_M1_COMMIT128_LASTLOGITS_STRICT')]
 inverse=s
 for before,after in replacements:
  assert inverse.count(before)==1,before
  inverse=inverse.replace(before,after)
 assert ast.dump(ast.parse(inverse),include_attributes=False)==ast.dump(ast.parse((OLD/'fullqual.py').read_text()),include_attributes=False)
 for n in ('ordinary_reference.py','qualification_gate.py','v113-binding.json','assertion-gold.json','header-pins.json'):
  assert (D/n).read_bytes()==(OLD/n).read_bytes(),n
 support=(D/'runtime_support.py').read_text().replace('V117 numerical qualification','V114 numerical qualification').replace('candidate-v116-bounded-draftloader-build1-root','candidate-v114-verify-broadcast-m1-build1-root')
 assert support==(OLD/'runtime_support.py').read_text()
 gates={}
 for name,func in [('qualification_gate','require_v113'),('loader_gate','require_v116')]:
  spec=importlib.util.spec_from_file_location('v117_'+name,D/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  gates[name]=getattr(m,func)()
 assert gates['loader_gate']['receipt_sha256']=='fcfa4cc5b78bc197c20b1d1c6f1c0c2098ca8994d0694087e3882b1f10aad059'
 assert gates['loader_gate']['independent_review_sha256']=='8f56ce1f36acd3f679faa860472cfb21f6fbba557f4e9c903e580a06848c4b1a'
 assert s.index('loader_prerequisite=require_v116()')<s.index('import mlx.core as mx')
 callback=next(n for n in ast.walk(ast.parse(s)) if isinstance(n,ast.FunctionDef) and n.name=='bounded_loader_stage')
 assert [a.arg for a in callback.args.args]==['label','detail'] and len(callback.body)==2
 assert not any(isinstance(n,ast.Return) for n in ast.walk(callback))
 assert ast.unparse(callback.body[1])=='save()'
 assert "'allocator': memory(mx)" in ast.unparse(callback.body[0])
 assert 'mx.set_memory_limit(34<<30);mx.set_cache_limit(2<<30)' in s
 return dict(status='CPU_SOURCE_READY_V117_NO_CRITICAL_BLOCKER',GPU_used_by_reviewer=False,
             source_pins_verified=216,candidate116_py=30,candidate116_byte_identity=True,
             fullqual_inverse_AST_equal_114=True,inherited_hard_gates_unchanged=True,
             inherited_auxiliary_byte_identity=True,CPU_checks_independently_run=17,inert_independently_passed=True,
             prerequisite113=gates['qualification_gate']['receipt_sha256'],prerequisite116=gates['loader_gate']['receipt_sha256'],
             callback='metadata plus allocator integer counters and JSON save; no model/array arguments or output replacement',
             fullqual_sha256=sha(D/'fullqual.py'),source_pins_sha256=sha(D/'source-pins.json'),
             qualification_limit='new V117 full numerical run pending root; V114 resource STOP retained; no perf/physical memory/adoption claim')

if __name__=='__main__':
 assert not any(n=='mlx' or n.startswith(('mlx.','numpy')) for n in sys.modules)
 r=verify()
 if '--seal' in sys.argv:
  (H/'receipt.json').write_text(json.dumps(r,indent=2)+'\n')
  paths=set(load(D/'source-pins.json'))|{str(p) for p in D.iterdir() if p.is_file()}
  (H/'source-seals.json').write_text(json.dumps({p:dict(bytes=Path(p).stat().st_size,sha256=sha(p)) for p in sorted(paths)},indent=2)+'\n')
 else:
  assert r==load(H/'receipt.json')
  for p,e in load(H/'source-seals.json').items():assert Path(p).stat().st_size==e['bytes'] and sha(p)==e['sha256']
  for p,e in load(H/'manifest.json')['files'].items():assert (H/p).stat().st_size==e['bytes'] and sha(H/p)==e['sha256']
 print(json.dumps(r))
