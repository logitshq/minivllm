"""Actual E036 metadata and stat binding; never reads tensor payload bytes."""
import hashlib,json,struct
from pathlib import Path
HERE=Path(__file__).resolve().parent
MANIFEST_SHA='23db130f038bce2de201f89621f7818d9ae3d5164a6f2e23b81b241a37f1779b'
def need(ok,message):
 if not ok:raise ValueError(message)
def small(p):
 p=Path(p);need(p.suffix!='.safetensors' and p.stat().st_size<32<<20,'metadata only')
 return p.read_bytes()
def sha(p):return hashlib.sha256(small(p)).hexdigest()
def manifest():
 p=HERE/'checkpoint-manifest.json';need(sha(p)==MANIFEST_SHA,'sealed derived manifest changed')
 m=json.loads(small(p));need(m['schema']=='GLM-derived-stacked-checkpoint-v1' and len(m['shards'])==58,'real 58-shard checkpoint')
 return m
def evidence(plan):
 m=manifest()
 for key,status in (('repack_receipt','PASS_BYTES_ONLY'),('header_verification','PASS_HEADERS_INDEX_MANIFEST_ONLY')):
  r=m[key];need(plan[key]==r and sha(r['path'])==r['sha256'],'derived receipt binding')
  v=json.loads(small(r['path']));need(v['status']==status,'actual derived receipt scope')
 producer=json.loads(small(m['repack_receipt']['path']));headers=json.loads(small(m['header_verification']['path']))
 need(producer['source_plan_sha256']==headers['plan_sha256']==m['source_plan_sha256'] and
      producer['candidate_sha256']==headers['candidate_sha256']==m['candidate_sha256'],'exact original derivation')
 need(producer['source_payload_SHA_rechecked'] and producer['output_groups_reread'] and producer['original_unchanged_stat'] and
      producer['runtime_load_executed'] is False and producer['performance_qualified'] is False,'producer byte-only proof')
 need(producer['manifest_sha256']==headers['producer_manifest_sha256']==m['group_manifest_sha256'],'bound group manifest digest')
 need(producer['summary']==m['summary'] and headers['output_shards']==58 and headers['output_tensors']==4960,'exact derivation summary')
 for name,row in m['shards'].items():
  need({k:row[k] for k in ('identity','header_bytes','header_sha256','payload_bytes','metadata')}==headers['output_headers'][name] and
       dict(bytes=row['identity']['size'],sha256=row['sha256'])==producer['files'][name],'actual derived header/file receipt')
 return m
def metadata():
 m=manifest();evidence({k:m[k] for k in ('repack_receipt','header_verification')})
 root=Path(m['snapshot']);need(root.is_absolute() and root==root.resolve() and root.is_dir(),'real derived output directory')
 expected=set(m['shards'])|set(m['metadata'])|{'complete.json','input-plan.json','invocation.json','repack-manifest.jsonl'}
 need({p.name for p in root.iterdir()}==expected,'exact completed checkpoint membership')
 need(sha(root/'input-plan.json')==m['original_input_plan']['sha256'],'input plan unchanged')
 result=dict(metadata=m['metadata'],shard_stats={},weight_payload_SHA_rechecked=False,
             derived_manifest_sha256=MANIFEST_SHA,payload_scope=m['payload_scope'])
 for name,row in m['metadata'].items():
  raw=small(root/name);need(len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'],'derived metadata changed: '+name)
 for name,row in m['shards'].items():
  p=root/name;need(p.is_file() and not p.is_symlink(),'regular derived shard');s=p.stat()
  identity=dict(resolved=str(p.resolve()),size=s.st_size,dev=s.st_dev,ino=s.st_ino,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
  need(identity==row['identity'],'derived shard stat changed: '+name)
  with p.open('rb',buffering=0) as f:
   prefix=f.read(8);need(len(prefix)==8,'complete prefix');n=struct.unpack('<Q',prefix)[0]
   need(n+8==row['header_bytes'] and n<=16<<20,'bounded exact header');raw=f.read(n)
  need(hashlib.sha256(prefix+raw).hexdigest()==row['header_sha256'] and p.stat()==s,'derived header changed')
  result['shard_stats'][name]=dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev)
 need(not (root/'offload_index.json').exists(),'offload forbidden')
 return result
