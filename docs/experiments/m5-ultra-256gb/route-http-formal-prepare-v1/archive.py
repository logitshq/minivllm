"""Archive closed CPU prepare evidence and references to original CPU attempts."""
import argparse, hashlib, json, sys, tarfile
from pathlib import Path
R=Path('/Users/ryanli/code/minivllm');A=R/'.artifacts/m5-ultra-256gb/route-http-formal-prepare-v1'
OUT=R/'docs/experiments/m5-ultra-256gb/route-http-formal-prepare-v1'
CPU=OUT.parent/'cold-v3-route-formal-prepared-cpu-v1'
FORMAL='0e590191d32b12027bc8b5f4f76335ed42a633a4179d90431427aa42c6e18f30'
TEST='ea26053abdc4fbc1366d122bf09c79c8455cf8669973da31d42296defe07b771'
def need(ok,msg):
 if not ok:raise ValueError(msg)
def info(p):
 raw=Path(p).read_bytes();return dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
def read(p):return json.loads(Path(p).read_bytes())
def write(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def listing(root):
 paths=sorted(root.rglob('*'));need(all(not p.is_symlink() and (p.is_file() or p.is_dir()) for p in paths),'regular input tree only')
 return [p for p in paths if p.is_file()]
def run():
 need(not OUT.exists(),'new archive only')
 result=read(A/'result.json');files=listing(A);membership={str(p.relative_to(A)) for p in files}
 need(result['status']=='PASS' and result['prepared_specs']==12 and len(result['specs'])==12,'actual twelve-spec preparation PASS')
 need(result['device_run'] is result['execution_registered'] is result['performance_qualified'] is result['runtime_weights_verified'] is result['steps_ledger_opened'] is False,'CPU-only scope')
 need(result['source_unchanged'] is result['inputs_unchanged'] is True and result['source_before']==result['source_after'] and result['denied_operations']==result['forbidden_modules_loaded']==[],'closed source/guard checks')
 inv=read(A/'artifact-manifest.json');need(inv['schema']=='cpu-prepare-artifact-manifest-v1' and set(inv['files'])==membership-{'artifact-manifest.json'},'full prepare inventory')
 for name,expected in inv['files'].items():need(info(A/name)==expected,'actual artifact hash: '+name)
 for source,expected in result['source_before'].items():need(info(A/'frozen-source'/source.lstrip('/'))==expected,'source snapshot')
 for source,expected in result['inputs'].items():
  need(info(A/expected['copy'])=={k:expected[k] for k in ('sha256','bytes')},'actual input snapshot')
 need(result['source_before'][str(R/'bench/m5_ultra_256gb/route_http_formal/formal.py')]['sha256']==FORMAL,'frozen production formal')
 need(result['source_before'][str(R/'bench/m5_ultra_256gb/route_http_formal/test_formal_cpu.py')]['sha256']==TEST,'fixed fixture test source')
 order=[]
 for row in result['specs']:
  path=A/'specs'/(row['phase']+'.json');spec=read(path);order.append(row['phase'])
  need(row['status']=='PASS' and info(path)['sha256']==row['spec']['sha256'] and spec['phase']==row['phase'],'actual validated spec bytes')
  need(spec['observer_enabled'] is False and len(spec['jobs']['warm'])==8 and len(spec['jobs']['formal'])==64,'fixed observer-OFF queue')
 need(result['warm_requests']==96 and result['formal_requests']==768 and result['total_requests']==864,'planned request counts only')
 cpu=read(CPU/'archive-verification.json');attempts=read(CPU/'attempts.json')
 need(cpu['status']=='PASS' and cpu['archive']['sha256']=='98533c09acca95b763b548871f874ea1f49def9cc032ba3ab5103abcac821d0c','existing independent CPU archive reference')
 need(attempts['route-http-formal-cpu-v1']['status']=='FAIL' and attempts['route-http-formal-cpu-v2']['status']=='PASS','preserve first failure and full retry')
 OUT.mkdir();(OUT/'archive.py').write_bytes(Path(__file__).read_bytes())
 (OUT/'prepared-result.json').write_bytes((A/'result.json').read_bytes())
 write(OUT/'pack-command.json',dict(argv=sys.argv,scope='Closed stdlib CPU artifact archive only; no tests/device/service'))
 refs={str(CPU/name):info(CPU/name) for name in ('archive-verification.json','attempts.json','README.md')}
 write(OUT/'cpu-attempt-references.json',dict(schema='existing-cpu-archive-references-v1',files=refs,archive=cpu['archive'],
   first_failure_result_sha256=cpu['result_sha256']['route-http-formal-cpu-v1'],retry_result_sha256=cpu['result_sha256']['route-http-formal-cpu-v2'],
   payload_repacked=False,qualification=False))
 write(OUT/'summary.json',dict(schema='E024-route-http-formal-prepare-summary-v1',status='PASS',specs=12,processes_executed=0,
   planned_adjacent_pairs=6,planned_warm_requests=96,planned_formal_requests=768,planned_total_requests=864,
   phase_order=order,result_sha256=info(A/'result.json')['sha256'],formal_source_sha256=FORMAL,test_source_sha256=TEST,
   original_cpu_failures_preserved=True,execution_registered=False,performance_qualified=False))
 entries={'evidence/'+str(p.relative_to(A)):dict(source=str(p),**info(p)) for p in files}
 for name in ('archive.py','pack-command.json','summary.json','cpu-attempt-references.json'):
  entries[name]=dict(source=str(OUT/name),**info(OUT/name))
 for name in ('archive-verification.json','attempts.json','README.md'):
  entries['previous-cpu-archive/'+name]=dict(source=str(CPU/name),**info(CPU/name))
 write(OUT/'manifest.json',dict(schema='E024-route-http-formal-prepare-manifest-v1',files=entries))
 archive=OUT/'evidence.tar.xz'
 with tarfile.open(archive,'w:xz',dereference=True,preset=1) as t:
  for name,row in entries.items():t.add(row['source'],arcname=name,recursive=False)
 with tarfile.open(archive,'r:xz') as t:
  members=t.getmembers();need(len(members)==len(entries) and {m.name for m in members}==set(entries),'complete tar inventory')
  for m in members:
   need(m.isfile(),'regular tar member');raw=t.extractfile(m).read();e=entries[m.name]
   need(len(raw)==e['bytes'] and hashlib.sha256(raw).hexdigest()==e['sha256'],'actual reopened member bytes')
   need(info(e['source'])=={k:e[k] for k in ('sha256','bytes')},'original source unchanged')
 need({str(p.relative_to(A)) for p in listing(A)}==membership,'original membership unchanged')
 verification=dict(schema='E024-route-http-formal-prepare-archive-verification-v1',status='PASS',archive=info(archive),
  members=len(entries),raw_bytes=sum(r['bytes'] for r in entries.values()),manifest_sha256=info(OUT/'manifest.json')['sha256'],
  result_sha256=info(A/'result.json')['sha256'],all_member_bytes_reopened=True,all_original_files_rechecked=True,
  source_snapshot_verified=True,prior_cpu_archive=refs,device_run=False,performance_qualified=False)
 write(OUT/'archive-verification.json',verification)
 (OUT/'README.md').write_text('# E024 route HTTP formal: actual CPU preparation\n\n'
 'The real formal.prepare and validate functions successfully constructed all 12 OFF/ON specs from the original native formal registration and the closed, independently qualified E024 C1/C8 route histories. All request bodies, fixture identities, transport and count semantics remain bound to those inputs. No device or service operation was performed; these are planned points, not measured results or execution registration.\n\n'
 'The plan contains six adjacent OFF/ON pairs: three fresh processes per arm at C1 and C8. Each process plans 8 warm + 64 formal requests (96 + 768 = 864 overall). Both qualification observers are OFF. OFF retains the original native entry; ON selects the frozen route graph entry. Runtime/weight verification, the new direct-server owner/recovery binding and independent performance audit remain required.\n\n'
 'The original guarded CPU attempt (12 methods: 5 assertion failures including subtests and 1 error) and complete 12/12 retry remain in [the existing CPU archive](../cold-v3-route-formal-prepared-cpu-v1/README.md). Only the temporary fixture root gained Path.resolve(); production formal source remained byte-identical. This archive references that evidence and includes its small summaries, without repacking its tar or any inference tensors/ledgers.\n\n'
 f'All {len(entries)} tar members were reopened and matched to the manifest and original files. Compressed size: {verification["archive"]["bytes"]:,} bytes; SHA256 `{verification["archive"]["sha256"]}`. `prepared-result.json` is an exact sidecar copy; the full prepare source, inputs, 12 specs, stdout/stderr and original inventory are inside `evidence.tar.xz`.\n')
 print(json.dumps(verification))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');a=p.parse_args()
 if a.run:run()
 else:print('DESCRIPTION_ONLY')
