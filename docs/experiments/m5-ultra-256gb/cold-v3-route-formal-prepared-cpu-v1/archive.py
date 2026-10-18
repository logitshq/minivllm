"""Archive three closed CPU attempts and the source-only v3 seal; never run tests."""
import argparse, ast, hashlib, json, sys, tarfile
from pathlib import Path
R=Path('/Users/ryanli/code/minivllm')
A=R/'.artifacts/m5-ultra-256gb'
OUT=R/'docs/experiments/m5-ultra-256gb/cold-v3-route-formal-prepared-cpu-v1'
EXPECTED={
 'cold-cycle-v3-source-seal':'8ce73874748d7e8075fb03a3a518dd7de560fb2900379c9bdd633ac5d96fd235',
 'cold-v3-sealed-cpu-v1':'8ab0604bc89fbe9b80f6a478f6a574293e0daacdf61cb4ede64548d3ff2f9ca3',
 'route-http-formal-cpu-v1':'5bfb3046c7b8a9cd84215c54e0373bd5a4ea30721cec1463063818df1298d311',
 'route-http-formal-cpu-v2':'23e53b4526b5b40d0165516c50bd75047dd995150e27b9ae94e823f93f896d43'}
COUNTS={'cold-v3-sealed-cpu-v1':('PASS',24,0,0),'route-http-formal-cpu-v1':('FAIL',12,5,1),'route-http-formal-cpu-v2':('PASS',12,0,0)}
def need(ok,msg):
 if not ok:raise ValueError(msg)
def info(p):
 b=Path(p).read_bytes();return {'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
def read(p):return json.loads(Path(p).read_bytes())
def write(p,obj):
 with Path(p).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')
def frozen(d,p):return d/'frozen-source'/str(p).lstrip('/')
def listing(d):
 allp=sorted(d.rglob('*'));need(not any(p.is_symlink() for p in allp),'no symlink')
 need(all(p.is_file() or p.is_dir() for p in allp),'regular files only')
 return [p for p in allp if p.is_file()]
def stable(source,keys):
 tree=ast.parse(source)
 for n in tree.body:
  if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in keys:n.value=ast.Constant('MIGRATION_SHA')
 return ast.dump(tree,include_attributes=False)
def run():
 need(not OUT.exists(),'fresh archive only'); entries={}; results={}; memberships={}
 for name,digest in EXPECTED.items():
  d=A/name;need(info(d/'result.json')['sha256']==digest,'exact result '+name); result=read(d/'result.json');results[name]=result
  files=listing(d);memberships[name]={str(p.relative_to(d)) for p in files}
  if name in COUNTS:
   status,tests,failures,errors=COUNTS[name]
   need((result['status'],result['tests'],result['failures'],result['errors'])==(status,tests,failures,errors),'actual counts '+name)
   need(result['expected_tests']==tests and len(result['test_outcomes'])==tests and result['skipped']==0 and result['expected_failures']==result['unexpected_successes']==0,'complete methods '+name)
   need(result['source_unchanged'] is True and result['source_before']==result['source_after'] and result['denied_operations']==[] and result['forbidden_modules_loaded']==[] and result['loader_errors']==[],'CPU-only source/guard '+name)
   if status=='PASS':need(all(x['status']=='PASS' for x in result['test_outcomes']),'all passing outcomes')
   inv=read(d/'artifact-manifest.json');need(inv['schema']=='cpu-artifact-inventory-v1' and set(inv['files'])==memberships[name]-{'artifact-manifest.json'},'exact inventory '+name)
   for member,expected in inv['files'].items():
    p=d/member;need(p.resolve().is_relative_to(d.resolve()) and info(p)==expected,'inventory bytes '+member)
   for source,expected in result['source_before'].items():need(info(frozen(d,source))==expected,'source snapshot '+source)
  for p in files:entries[f'evidence/{name}/{p.relative_to(d)}']={'source':str(p),**info(p)}
 seal=results['cold-cycle-v3-source-seal'];sd=A/'cold-cycle-v3-source-seal'
 need(seal['status']=='PASS_SOURCE_ONLY' and seal['algorithm_ast_unchanged'] is True and seal['service_operation'] is False and seal['tests_run'] is False,'source-only seal')
 need(stable((sd/'controller.py.before').read_text(),seal['values'])==stable((sd/'controller.py.after').read_text(),seal['values']),'algorithm AST retained')
 need(info(sd/'controller.py.before')['sha256']==seal['before_sources']['controller.py'] and read(sd/'lifecycle.json.after')==seal['after_manifest'],'seal exact originals')
 cold=A/'cold-v3-sealed-cpu-v1'
 for key,row in seal['after_manifest']['sources'].items():
  need(info(frozen(cold,row['path']))['sha256']==row['sha256'],'sealed tested source '+key)
 need(info(sd/'controller.py.after')['sha256']==seal['after_manifest']['sources']['controller']['sha256'],'sealed controller bytes')
 tests=R/'bench/m5_ultra_256gb/route_http_formal/test_formal_cpu.py';prod=tests.with_name('formal.py')
 before=frozen(A/'route-http-formal-cpu-v1',tests).read_bytes();after=frozen(A/'route-http-formal-cpu-v2',tests).read_bytes()
 old=b'self.root=Path(self.temp.name); self.base,self.binding,self.plan=fixture(self.root)'
 new=b'self.root=Path(self.temp.name).resolve(); self.base,self.binding,self.plan=fixture(self.root)'
 need(before.count(old)==1 and before.replace(old,new)==after,'only fixture canonicalization changed')
 need(info(frozen(A/'route-http-formal-cpu-v1',prod))==info(frozen(A/'route-http-formal-cpu-v2',prod)),'production formal unchanged')
 OUT.mkdir();(OUT/'archive.py').write_bytes(Path(__file__).read_bytes())
 write(OUT/'pack-command.json',{'argv':sys.argv,'scope':'stdlib archive only; no tests/service/backend operations'})
 summary={name:{k:r[k] for k in ('status','tests','failures','errors','skipped','argv','elapsed_seconds')} for name,r in results.items() if name in COUNTS}
 write(OUT/'attempts.json',summary)
 (OUT/'README.md').write_text('# Cold v3 sealing and route HTTP formal CPU evidence\n\n'
 'Three complete guarded CPU attempts are retained: sealed cold v3 24/24 PASS; route formal v1 ran all 12 methods and reported 5 assertion failures (including subtest failures) plus 1 error; the complete v2 rerun passed 12/12. All attempts recorded zero skips, forbidden imports or real backend operations. The original failed result, tracebacks, commands, runner and source snapshots remain intact.\n\n'
 'The only formal test repair canonicalized its temporary directory with Path.resolve(), covering macOS /var versus /private/var. The production formal source is byte-identical between attempts. Do not sum these reruns as 48 distinct tests.\n\n'
 'The source-only seal records exactly four actual archived route-predecessor SHA constants and the regenerated lifecycle manifest. The archived before/after controller AST is identical after replacing those four constant values; the sealed sources match the 24-test snapshot. The seal records no tests or service operation. Its original evidence does not contain a separate actual invocation argv; seal.py is retained without inventing one. CPU invocation argv are preserved verbatim in each result and attempts.json.\n\n'
 'evidence.tar.xz contains every file in all four original evidence trees, plus this archive script, command, summary and README. manifest.json maps source paths to member SHA256/size. archive-verification.json records every member reopened and matched to its original file. This is CPU wiring evidence only, not GPU, model numerical, HTTP runtime, performance or live restoration qualification.\n')
 for name in ('archive.py','pack-command.json','attempts.json','README.md'):entries[name]={'source':str(OUT/name),**info(OUT/name)}
 write(OUT/'manifest.json',{'schema':'cold-v3-route-formal-cpu-archive-manifest-v1','files':entries})
 archive=OUT/'evidence.tar.xz'
 with tarfile.open(archive,'w:xz',dereference=True,preset=1) as tar:
  for name,row in entries.items():tar.add(row['source'],arcname=name,recursive=False)
 with tarfile.open(archive,'r:xz') as tar:
  members=tar.getmembers();need(len(members)==len(entries) and {m.name for m in members}==set(entries),'exact tar membership')
  for m in members:
   need(m.isfile(),'regular member');raw=tar.extractfile(m).read();row=entries[m.name]
   need(len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'],'member bytes '+m.name)
   need(info(row['source'])=={k:row[k] for k in ('sha256','bytes')},'original bytes '+m.name)
 for name,expected in memberships.items():need({str(p.relative_to(A/name)) for p in listing(A/name)}==expected,'original tree membership')
 write(OUT/'archive-verification.json',{'schema':'cold-v3-route-formal-cpu-archive-verification-v1','status':'PASS','archive':info(archive),'members':len(entries),'raw_bytes':sum(r['bytes'] for r in entries.values()),'result_sha256':EXPECTED,'all_member_bytes_reopened':True,'all_original_files_rechecked':True,'fixture_only_resolve_repair':True,'formal_production_unchanged':True,'seal_only_four_constants_ast_verified':True,'numerical_qualified':False,'performance_qualified':False})
 print(json.dumps(read(OUT/'archive-verification.json')))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');a=p.parse_args()
 if a.run:run()
 else:print('DESCRIPTION_ONLY')
