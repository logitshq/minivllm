"""Short stdlib saved-result audit of V99; no GPU/MLX/native execution."""
import hashlib
import json
from pathlib import Path
import sys
P = Path('/private/tmp/minivllm-engine-optimization-20261014')
D = Path(__file__).resolve().parent
R = P / 'v23-root-control-build1/fullmodel-v99native-attempt1'
S = P / 'v99-simd4-cap64-fullqual-build1'
N = P / 'v99-native-simd4-cap64-build1'
refs = {}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    refs[str(p)] = sha(p)
    return json.loads(p.read_text())
def save(p,x): p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
    x=read(R/'child-output/receipt.json')
    assert refs[str(R/'child-output/receipt.json')]=='11654508130ba3d880a29075ae608db2d5aa97c33763d69ff5b69f1bbb1ae2f5'
    assert x['status']=='PASS_V99_SIMD4_SEGMENT_CAP64_NATIVE_ICB_BASE_RESET_CHANGED_OLD38_44_STRICT'
    assert not x['performance_point'] and not x['adopted']
    pins=read(S/'source-pins.json');assert x['source_pins']==pins and len(pins)==153
    for p,h in pins.items():
        assert sha(Path(p))==h,p
        refs[p]=h
    assert all(p in pins for p in x['actual_imported_sources'].values())
    old38=read(Path(x['A_saved_baseline_receipt']));assert refs[x['A_saved_baseline_receipt']]==x['A_saved_baseline_receipt_sha256']
    paths=[Path(p) for p,h in pins.items() if h==x['actual44_gold_sha256']];assert len(paths)==1
    old44=read(paths[0]);a38={r['name']:r for r in old38['requests']};a44={r['name']:r for r in old44['requests']}
    assert (x['fixed_cap'],x['canonical_capacity'])==(64,256)
    assert x['same_parameter_identity_count']==1349 and x['same_parameter_identity_preserved'] and x['source_pins_verified_final']
    assert x['SIMD4_segment_ON']==48 and x['A_baseline_restore_final'] and x['allocation_bucket_restored_finally']
    engagement=x['B_state95_capture_engagement']
    assert engagement=={'calls':48,'distinct_GDN_objects':48,'helper_calls':48,'all48enabled':True,'hooks_removed_before_replay':True}
    schema=read(N/'predicted-schema.json');observed=x['state95_actual_reflection']
    assert observed['status']=='OBSERVED_FULL_SCHEMA_MATCH' and observed['function']==schema['function']
    assert observed['active_arguments']==schema['active_arguments'] and len(observed['active_arguments'])==10
    capture=x['capture'];assert len(capture['inputs'])==131 and len(capture['outputs'])==132
    kv={r['label'] for r in capture['outputs'] if r['shape']==[1,4,64,256]};assert len(kv)==32
    assert all(r['shape']!=[1,4,256,256] for r in capture['inputs']+capture['outputs'])
    pso=[r for r in capture['native']['feedback']['icb']['PSO_records'] if r['function']==schema['function']]
    assert len(pso)==1 and pso[0]['argument_reflection_observed']
    assert sorted(pso[0]['active_arguments'],key=lambda r:r['index'])==schema['active_arguments']
    counts={'steps':0,'strict_outputs':0,'old38_hash_equal':0,'old44_hash_equal':0,'KV_tail_records':0,'readonly_array_checks':0,'readonly_input_steps':0,'B_step_upload_calls':0,'seed_upload_calls':0,'IDs_each_arm':0}
    requests=[]
    for i,q in enumerate(x['requests'],1):
        name=q['name'];assert len(q['steps'])==31
        assert q['A_ids']==q['B_ids']==a38[name]['A_ids']==a38[name]['B_ids']==a44[name]['A_ids']==a44[name]['B_ids']
        assert len(q['A_ids'])==32 and not (set(q['A_ids']) & set(x['eos_ids']))
        counts['IDs_each_arm']+=32
        assert q['seed_upload_calls']==q['seed_arrays']==131;counts['seed_upload_calls']+=131
        assert q['first_token']==q['A_ids'][0]
        assert not q['changed_seed_inputs_vs_capture'] if name in ('base','reset-base') else any(v>=3 for v in q['changed_seed_inputs_vs_capture'])
        n=q['native'];f=n['feedback'];pp=f['pingpong'];icb=f['icb']
        assert n['status']=='OK' and not n['error'] and n['parameter_ranges_registered']==1349
        assert f['sealed'] and f['capacity']==64 and f['request_steps']==31 and f['total_steps']==i*31 and f['resets']==i
        assert f['request_seed_uploaded_ranges']==131 and not f['CPU_full_state_restore_per_step']
        assert f['CPU_seed_upload_bytes_total']==158138380*i
        assert (pp['requested_count'],pp['eligible_count'],pp['staged_link_count'])==(48,48,83) and len(pp['pairs'])==48
        for pair in pp['pairs']:
            assert pair['eligible'] and len(pair['actual_binding_uses'])==2
            uses={r['argument']:r for r in pair['actual_binding_uses']};assert set(uses)=={'S','S_new'}
            assert uses['S']['index']==6 and uses['S']['access']==0 and uses['S_new']['index']==9 and uses['S_new']['access']==1
            assert all(r['PSO']==schema['function'] and r['offset']==0 for r in uses.values())
        assert icb['enabled'] and icb['banks']==2 and icb['execution_count']==i*31 and sum(icb['bank_executions'])==i*31
        assert icb['commands_per_bank']==n['dispatches']==1625
        for j,step in enumerate(q['steps'],1):
            assert step['step']==j and step['A_exact_L']==q['prompt_length']+j and step['B_fixed_L']==64
            assert step['A_token']==step['B_token']==q['A_ids'][j]
            assert step['seed_uploads_this_step']==0 and step['native_prediction_copied_bytes']==4
            assert step['reference_input_digest_readonly'];counts['readonly_input_steps']+=1
            for ro in step['originals_readonly'].values():
                assert ro['strict_bytes_equal'];counts['readonly_array_checks']+=ro['arrays']
            gold38={r['label']:r for r in a38[name]['steps'][j-1]['outputs']};gold44={r['label']:r for r in a44[name]['steps'][j-1]['outputs']}
            assert gold38==gold44 and len(step['outputs'])==132 and {r['label'] for r in step['outputs']}==set(gold38)
            for out in step['outputs']:
                assert out['strict_bytes_equal'] and out['finite'] and out['A_sha256']==out['canonical_B_sha256']
                old=gold38[out['label']]
                assert out['A_sha256']==old['A_sha256'] and (out['shape'],out['dtype'],out['bytes'])==(old['shape'],old['dtype'],old['bytes'])
                counts['strict_outputs']+=1;counts['old38_hash_equal']+=1;counts['old44_hash_equal']+=1
                if out['label'] in kv:
                    assert out['physical_shape']==[1,4,64,256] and out['shape']==[1,4,256,256]
                    assert out['physical_tail_zero_checked'] and out['canonical_CPU_zero_fill']
                    assert out['physical_bytes']==131072 and out['bytes']==524288;counts['KV_tail_records']+=1
                else:assert out['A_sha256']==out['physical_A_sha256'] and out['shape']==out['physical_shape']
            counts['steps']+=1
        requests.append({'name':name,'steps':31,'old38_44_A_B_all32_IDs_equal':True,'changed_seed_ranges':len(q['changed_seed_inputs_vs_capture'])})
    assert counts['steps']==93 and counts['strict_outputs']==counts['old38_hash_equal']==counts['old44_hash_equal']==12276
    assert counts['KV_tail_records']==2976 and counts['IDs_each_arm']==96
    assert x['strict_output_checks']==12276 and x['canonical_KV_tail_zero_checks']==2976 and x['native_steps']==93
    owner=read(R/'owner-receipt.json');rel=read(R/'external-release.json')
    assert owner['owner_pid']==4088 and owner['child_pid']==x['pid']==4103 and owner['returncode']==0 and owner['waited_reaped'] and owner['stage']=='reaped'
    assert owner['script_sha256']==sha(S/'fullqual.py') and owner['native_source_sha256']==sha(N/'replay.mm') and owner['native_library_sha256']==sha(N/'libv99replay.dylib')
    assert owner['absence']=={'returncode':1,'stdout':'','stderr':''}
    assert rel['status']=='PASS_EXTERNAL_RELEASE' and rel['GPU_released'] and rel['tool_exit']==0
    assert {r['pid'] for r in rel['strict_absence']}=={4088,4103} and all(r['returncode']==1 and r['stdout']==r['stderr']=='' for r in rel['strict_absence'])
    rawpaths=sorted(R.glob('resource-*-raw.json'))+sorted((R/'child-output').glob('resource-*-raw.json'));rows=[read(p) for p in rawpaths]
    for row in rows:
        assert row['process_inventory_ok'] and row['only_owned_known_inference_processes'] and row['pressure_level_value']==1
        assert row['power_metadata']['source']=='AC Power' and row['power_metadata']['percent']==100 and row['processinfo_thermal_metadata']['code']==0
    assert len({r['swap_used_mb'] for r in rows})==len({r['vm_counters']['Swapouts'] for r in rows})==1
    ext=read(R/'external-resource.json');assert len(ext)==1 and ext[0]['known_inference_process_count']==0
    assert read(R/'external-resource-00-raw.json')==ext[0]
    assert sha(R/'child-output'/x['afterload']['raw_file'])==x['afterload']['raw_sha256']
    ready=read(R/'child-output/model-ready.json');assert ready=={'pid':4103,'token_executed':False,'model_loads':1} and x['model_loads']==1
    assert not any(n=='mlx' or n.startswith(('mlx.','numpy')) for n in sys.modules)
    result={'status':'PASS_CPU_SAVED_V99_INTEGRATION_RESULTS_AUDIT','reviewer_relationship':'V95 author; independent99 integration savedresult review, not independent95 math implementation','actual_receipt_SHA256':refs[str(R/'child-output/receipt.json')],'pins_verified':153,'actual_imported_sources':len(x['actual_imported_sources']),'actual_full10_schema_observed':True,'S_input_access':'slot6 RO0','S_new_output_access':'slot9 RW1','helper_engagement':engagement,'counts_recomputed':counts,'requests':requests,'parameter_identity_preserved':1349,'ICB':{'banks':2,'GDN_S_pairs':48,'staged_links':83,'commands_per_bank':1625,'executions':93},'physical_CAP':64,'CPU_canonical_CAP':256,'resources':{'outer':22,'child':33,'external':1,'pressure':1,'swap_MB':rows[0]['swap_used_mb'],'Swapouts':rows[0]['vm_counters']['Swapouts'],'AC_percent':100,'thermal':'nominal'},'lifecycle':{'owner':4088,'child':4103,'rc':0,'reaped':True,'external_release':'PASS'},'GPU_used':False,'MLX_imported':False,'frozen_source_modified':False,'performance_or_adoption_qualified':False,'limitations':'saved hashes/counts/native memcmp assertions audited; full tensor raw corpus not retained; resource inventories and snapshots have their original scope'}
    save(D/'receipt.json',result);save(D/'refsSHA.json',refs)
    print(json.dumps({'status':result['status'],'counts':counts,'resources':result['resources'],'receipt_SHA256':sha(D/'receipt.json')}))
if __name__=='__main__':main()
