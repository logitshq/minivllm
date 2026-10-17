"""Offline stdlib publication verifier. No inference imports, execution or writes."""
from pathlib import Path
import argparse,collections,gzip,hashlib,json
D=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--repo-root',type=Path);args=ap.parse_args()
 manifest=load(D/'manifest.json');actual={str(p.relative_to(D)) for p in D.rglob('*') if p.is_file() and p!=D/'manifest.json'}
 assert actual==set(manifest['files']),('extra/missing payload',actual^set(manifest['files']))
 for name,row in manifest['files'].items():
  p=D/name;assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(D)
  assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],name
 sources=load(D/'source-map.json');assert len(sources)==54 and len({r['published_path'] for r in sources})==54
 for r in sources:assert sha(D/r['published_path'])==r['published_sha256']==r['original_sha256'] and r['mode']=='exact_bytes'
 deps=load(D/'dependency-map.json')['dependencies'];assert len(deps)==77 and len({r['original_source'] for r in deps})==77
 modes=collections.Counter(r['mode'] for r in deps);assert modes=={'bundled_exact':33,'repository_exact':38,'repository_lossless_gzip':1,'original_identity_only':5}
 repo_count=0
 for r in deps:
  if r['mode']=='bundled_exact':assert sha(D/r['published_path'])==r['original_sha256']
  if args.repo_root is not None and r['mode'].startswith('repository_'):
   root=args.repo_root.resolve();p=(root/r['repository_path']).resolve();assert p.is_relative_to(root) and p.is_file()
   if r['mode']=='repository_exact':assert sha(p)==r['original_sha256'],r['repository_path']
   else:assert sha(p)==r['archive_sha256'] and hashlib.sha256(gzip.decompress(p.read_bytes())).hexdigest()==r['original_sha256']
   repo_count+=1
 p=load(D/'evidence/projection.json');raw=(D/p['published_gzip']).read_bytes();body=gzip.decompress(raw)
 assert hashlib.sha256(raw).hexdigest()==p['published_gzip_sha256'] and hashlib.sha256(body).hexdigest()==p['uncompressed_projection_sha256'] and len(body)==p['uncompressed_projection_bytes']
 q=json.loads(body);assert set(q)==set(p['field_whitelist'])|{'publication_scope','runtime_selected'}
 assert q['publication_scope']['original_sha256']==p['source_receipt_sha256']=='011ad4f71498ae2e4ab0ca6fc551b368242c0b2c1abc4591e63224ea6aad0d52'
 assert q['publication_scope']['mode']=='explicit_whitelist_projection_not_original_receipt_bytes' and not q['publication_scope']['matrix_raw_outputs_saved']
 rows=q['comparisons'];assert len(rows)==1491 and q['completed_matrix_checks']==q['strict_equal_matrix_checks']==1491
 expected_geometry={(48,5120):96,(1024,5120):32,(5120,6144):64,(5120,17408):64,(6144,5120):48,(10240,5120):48,(12288,5120):16,(17408,5120):128,(248320,5120):1}
 counts=collections.Counter()
 gold=load(D/'support/v113-broadcast-m1-diagnostic-build1/actual-linear-gold.json')['operators']
 for r in rows:
  a,b=r['A_native_M1'],r['B_shared_or_fallback'];t,n,k=r['T'],r['N'],r['K']
  assert a==b and a['shape']==[t,n] and a['dtype']=='mlx.core.bfloat16' and a['bytes']==t*n*2
  assert r['shape_dtype_finite_bytes_strict'] and r['route']=='shared124' and r['same_original_weight_and_scales']
  assert r['input_scope']==('actual_saved_B_verify_T5' if t==5 else 'derived_prefix_of_actual_T5_X')
  if t==5:assert a['sha256']==gold[r['name']]['row_M1_sha256'] and [r['input_signature']]==gold[r['name']]['input_signatures']
  counts[t,n,k]+=1
 assert counts==collections.Counter({(t,n,k):v for t in (2,4,5) for (n,k),v in expected_geometry.items()})
 assert len(q['original_hash_checks'])==len(q['readonly'])==497 and len(q['layouts'])==2485
 for r in q['original_hash_checks']:assert r['old112_and113_equal'] and r['batch_output']['sha256']==gold[r['name']]['batch_sha256']
 for r in q['readonly']:assert r['unchanged'] and r['before']==r['after'] and r['owner_ids_before']==r['owner_ids_after']
 assert len(q['parameters_before'])==1349 and q['parameters_before']==q['parameters_after'] and q['target1349_identity_readonly']
 assert q['keep0_restored_all128'] and q['initial_cache_matches_actual108_all128'] and q['fallback_calls']==0
 assert not q['fullmodel_qualified'] and not q['adopted'] and not q['timing_run'] and not q['live_drafting'] and q['draft_loads']==0
 resources=load(D/'evidence/resources-selected.json')['samples'];assert len(resources)==42 and collections.Counter(r['group'] for r in resources)=={'worker':21,'root_owner':20,'root_post_release':1}
 assert all(r['power_source']=='AC Power' and r['thermal_code']==0 and r['pressure_level']==1 and r['swap_used_MB']==5038.06 and r['Swapouts']==17519593 and r['only_owned_known_inference'] for r in resources)
 assert max(r['mlx_allocator']['get_peak_memory'] for r in resources if r['mlx_allocator'])==16952614436
 cleanup=load(D/'evidence/cleanup-selected.json');assert cleanup['tool_session']==82729 and cleanup['tool_exit_code']==0 and cleanup['waited_reaped'] and cleanup['GPU_released']
 assert {r['pid'] for r in cleanup['strict_absence']}=={50896,50911} and all(r['absent'] and r['returncode']==1 and r['stdout_empty'] and r['stderr_empty'] for r in cleanup['strict_absence'])
 review=load(D/'author-recheck127/receipt.json');assert review['status']=='PASS_AUTHOR_SAVED_EVIDENCE_RECHECK_V127_LOCAL_ONLY' and not review['source_author_independence_claimed']
 for n,h in review['files'].items():assert sha(D/'author-recheck127'/n)==h
 print(json.dumps({'status':'PASS_OFFLINE_SOURCE_AND_SELECTED_METADATA_ONLY','manifest_payloads':len(actual),'exact_source_and_audit_files':54,'original127_pins_mapped':77,'repository_dependencies_checked':repo_count,'historical_original_identity_only_unverified':5,'matrix_records':1491,'resource_samples':42,'GPU_import_or_execution':False,'raw_tensor_recomputation':False,'portable_GPU_replay_ready':False,'fullmodel_or_performance_qualified':False},ensure_ascii=False))
if __name__=='__main__':main()
