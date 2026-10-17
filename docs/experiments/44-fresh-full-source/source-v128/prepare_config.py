"""CPU-only V128 actual126/static-route and unchanged103-C1 source binding."""
import hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent;P=D.parent
BASE=P/'v125-fresh-c1-native-official-build1'
ENTRY=P/'v126-native-full-curve-prework-build1'
ACTUAL=P/'v23-root-control-build1/batch-entry-v126-attempt1'
RECEIPT_SHA='9c6cb9bacebfd7bf9d0d8f5a7306c0fd6ba92e0ee1fe20d53075bd76d3517376'
RELEASE_SHA='25b90bbe4a446970d7f09e19f6dd55f4e0f83a8973f58e6be658c392b71b4e9d'
ENTRY_PINS_SHA='59606283b28b8706af1465ebab156f08aedecb34f41b16a3991391c2530932bf'
REVIEW_SHA='0557f0f181c1f607a1658a7bbef49a022adda3b1461adf9128e50bd9d54e75ae'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 qpath=ACTUAL/'child-output/receipt.json';release=ACTUAL/'external-release.json'
 assert sha(qpath)==RECEIPT_SHA and sha(release)==RELEASE_SHA
 q=json.loads(qpath.read_text());r=json.loads(release.read_text())
 assert q['status']=='PASS_V126_STATIC_BATCH_ENTRY_C2_C4_GENUINE_BATCH_RAW_IDS_AND_IDLE'
 assert q['HTTP_requests']==24 and q['completion_tokens']==768 and [p['concurrency'] for p in q['points']]==[2,4]
 assert all(p['actual_requested_batch_GDN_calls']>0 and p['post_hook_reentry_actual_requested_batch'] and p['raw32_same_C_reference_and_visible_usage_exact'] and p['idle_source_assertions_completed'] for p in q['points'])
 assert r['status']=='PASS_EXTERNAL_RELEASE_V126_C2_C4_ENTRY' and r['tool_exit_code']==0 and r['GPU_released'] is True
 assert sha(ENTRY/'source-pins.json')==ENTRY_PINS_SHA
 for path,h in json.loads((ENTRY/'source-pins.json').read_text())['files'].items():assert sha(path)==h,path
 review=P/'v126-actual-independent-review-build1/receipt.json'
 assert sha(review)==REVIEW_SHA
 audit=json.loads(review.read_text())
 assert audit['status']=='PASS_INDEPENDENT_V126_ACTUAL_ENTRY_RESULTS'
 assert audit['actual_receipt_sha256']==RECEIPT_SHA and audit['actual_external_release_sha256']==RELEASE_SHA
 assert audit['source_pins_sha256']==ENTRY_PINS_SHA
 c=json.loads((BASE/'runtime-config.json').read_text());root=P/'candidate-v17-gdn-gates-build1-root/tutorial_code/miniengine'
 legacy=json.loads((P/'source-manifest-v17-gdn-gates-build1.json').read_text())['files']
 assert len(legacy)==25
 for name,h in legacy.items():assert sha(root/name)==h
 c.update(curve_entry=str(ENTRY/'server.py'),curve_profiles=str(ENTRY/'profiles.json'),
  curve_source_pins=str(ENTRY/'source-pins.json'),legacy_package_sources=legacy,
  same_C_ID_assertions=str(ENTRY/'old-same-C-ID-assertions.json'),
  curve_qualification={'receipt':str(qpath),'receipt_sha256':RECEIPT_SHA,'release':str(release),
   'release_sha256':RELEASE_SHA,'source_pins_sha256':ENTRY_PINS_SHA,
   'independent_review':str(review),'independent_review_sha256':REVIEW_SHA},
  native_configuration_set=['direct_c1:44+101+102/maxseq1/CAP64','legacy_batched:accepted25/maxseq4/CAP256'],
  dynamic_live_migration=False,old_results_pooled=False)
 for name,h in legacy.items():c['source_pins'][str(root/name)]=h
 for name in ('server.py','c1_server.py','batched_server.py','entry_policy.py','profiles.json','source-pins.json','old-same-C-ID-assertions.json'):
  c['source_pins'][str(ENTRY/name)]=sha(ENTRY/name)
 c['source_pins'][str(qpath)]=RECEIPT_SHA;c['source_pins'][str(release)]=RELEASE_SHA
 c['source_pins'][str(review)]=REVIEW_SHA
 (D/'runtime-config.json').write_text(json.dumps(c,indent=2)+'\n')
 controls={}
 for name in ('driver.py','owner.py','prepare_config.py','runtime-config.json','protocol.json','README.md','campaign-plan.json','check_cpu.py','source-map.json'):
  controls[str(D/name)]=sha(D/name)
 for name in ('common.py','client.py','official_entry.py','registry.json','pins.json'):controls[str(P/'m4-final-native15-official-attempt2'/name)]=sha(P/'m4-final-native15-official-attempt2'/name)
 for name in ('metal_buffer_bootstrap_v1.py','power_paired_support_v1.py','resource_snapshot_power_v1.py','resource_snapshot_precise_v1.py','workload.json'):
  controls[str(P/name)]=sha(P/name)
 (D/'pins.json').write_text(json.dumps({'files':controls,'runtime_config_sha256':sha(D/'runtime-config.json'),
  'scope':'V128 fresh21-arm same-campaign measured config-set Pareto, no old125/C4 pooling or dynamicmigration claim'},indent=2)+'\n')
 print(json.dumps({'status':'PASS_CPU_CONFIG_BOUND_ACTUAL126_AND103','runtime_sources':len(c['source_pins']),
  'controller_pins':len(controls),'pins_sha256':sha(D/'pins.json'),'GPU_used':False}))
if __name__=='__main__':main()
