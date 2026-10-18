"""Verify the CPU evidence archive without extracting or executing its sources."""
import hashlib,json,tarfile
from pathlib import Path
D=Path(__file__).resolve().parent

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  while block:=f.read(1<<20):h.update(block)
 return h.hexdigest()

def verify():
 manifest=json.loads((D/'manifest.json').read_bytes());summary=json.loads((D/'summary.json').read_bytes())
 ref=manifest['archive'];archive=D/ref['filename']
 assert Path(ref['filename']).name==ref['filename'] and archive.stat().st_size==ref['bytes'] and digest(archive)==ref['sha256']
 expected=manifest['files'];seen=set();results={};source_hashes={}
 prefix='frozen-source/Users/ryanli/code/minivllm/bench/m5_ultra_256gb/glm_flash_cycle/'
 with tarfile.open(archive,'r|xz') as t:
  for member in t:
   n=member.name;p=Path(n)
   assert member.isfile() and not p.is_absolute() and '..' not in p.parts
   assert p.parts[0] in ('v1','v1-queue','v2','source-preparation') and n not in seen and n in expected
   row=expected[n];assert member.size==row['bytes'];stream=t.extractfile(member);h=hashlib.sha256();size=0;keep=[]
   retain=n in ('v1/result.json','v2/result.json','v1-queue/status.json')
   while block:=stream.read(1<<20):
    h.update(block);size+=len(block)
    if retain:keep.append(block)
   assert size==row['bytes'] and h.hexdigest()==row['sha256'];seen.add(n)
   if retain:results[n]=json.loads(b''.join(keep))
   for arm in ('v1','v2'):
    if n.startswith(arm+'/'+prefix):source_hashes[n]=h.hexdigest()
 assert seen==set(expected)
 old,new=results['v1/result.json'],results['v2/result.json'];queue=results['v1-queue/status.json']
 assert (old['status'],old['tests'],old['failures'],old['errors'],old['skipped'])==('FAIL',33,0,1,0)
 assert (new['status'],new['tests'],new['failures'],new['errors'],new['skipped'])==('PASS',33,0,0,0)
 assert old['errors_detail'][0]['test'].endswith('.test_preintent_empty_index_and_directory_without_plan_copy')
 assert 'FileNotFoundError' in old['errors_detail'][0]['traceback']
 assert queue['status']=='FAIL' and queue['returncode']==1 and queue['cpu_tests_completed'] is True
 assert queue['result_sha256']==expected['v1/result.json']['sha256'] and queue['child_ended_at']<=queue['ended_at']
 for row in (old,new):
  assert all(row[k] is True for k in ('source_unchanged','closure_unchanged','gdn_sequence_unchanged'))
  assert not row['denied_operations'] and not row['forbidden_modules_loaded']
  assert row['expected_tests']==row['discovered_tests']==33
  assert len(row['test_outcomes'])==len({v['id'] for v in row['test_outcomes']})==33
  assert row['device_executed'] is row['http_qualified'] is row['performance_qualified'] is False
 assert all(v['status']=='PASS' for v in new['test_outcomes'])
 assert sum(v['status']=='PASS' for v in old['test_outcomes'])==32
 for name,h in summary['production_source_unchanged'].items():
  assert source_hashes['v1/'+prefix+name]==source_hashes['v2/'+prefix+name]==h
 assert summary['archive']==ref and summary['archived_files']==len(seen)
 return dict(schema='GLM-cycle-cpu-archive-verification-v1',status='PASS',members=len(seen),
  raw_bytes=sum(v['bytes'] for v in expected.values()),archive=ref,
  original_failure_retained=True,production_sources_identical=True,all_members_sha256_verified=True,
  extraction_performed=False,sources_executed=False,device_executed=False,performance_qualified=False)

if __name__=='__main__':print(json.dumps(verify(),indent=2))
