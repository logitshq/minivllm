"""Independent saved-result audit; CPU only, never launches models or probes."""
import ast
import errno
import hashlib
import json
import math
from pathlib import Path
import re
import statistics as st
import sys
import tarfile

P=Path('/private/tmp/minivllm-engine-optimization-20261014')
D=P/'v111-omlx-none-mtp-http-screen-build1'
R=P/'v23-root-control-build1/omlx-mtp-v111-attempt1'
C=R/'child-output'
O=Path(__file__).resolve().parent
seen={}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    path=Path(path);seen[str(path)]=sha(path);return path.read_text()
def load(path):return json.loads(read(path))
def same(a,b):assert math.isclose(a,b,rel_tol=2e-12,abs_tol=2e-12),(a,b)
def strict_absence(value,pid):
    assert value['pid']==pid and value['returncode']==1 and value['stdout']==value['stderr']==''
def hjson(value):return hashlib.sha256(json.dumps(value).encode()).hexdigest()


def main():
    assert not any(n=='mlx' or n.startswith(('mlx.','numpy')) for n in sys.modules)
    assert sha(D/'pins.json')=='bee1c5affc7e7c8f3a9d4d05e8f8793c22f3833d6bfaac79a77bc4ac7ec81b79'
    assert sha(D/'driver.py')=='a458f07e259c8ccb9e2099cf674b55bddae309af11f1ed6bdf3d6cbb67747c72'
    pins=load(D/'pins.json')['files'];assert len(pins)==68
    for path,digest in pins.items():assert sha(path)==digest,path
    cfg=load(D/'config.json');suite=load(C/'suite-complete.json')
    assert suite['status']=='PASS_FIXED_C1_NONE_VS_QUALIFIED_EXTERNAL_MTP_SCREEN'
    prior=load(cfg['qualification']['receipt']);release109=load(cfg['qualification']['release'])
    assert prior['status']=='PASS_EXTERNAL_MTP_GREEDY32_RAW_TARGET_CONTROL_RESOURCES'
    assert prior['source_pins_sha256']==cfg['qualification']['source_pins_sha256']
    assert prior['actual_fresh_servers']==2 and prior['actual_requests']==6 and prior['terminal_cleanup']
    assert prior['ordinary_control_raw32']==cfg['qualified_control_raw32']
    assert release109['status']=='PASS_EXTERNAL_RELEASE' and release109['GPU_released']
    # Actual target tokenizer is CPU-only; re-encode raw saved text independently.
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file(str(Path(cfg['target_model_dir'])/'tokenizer.json'))
    seen[str(Path(cfg['target_model_dir'])/'tokenizer.json')]=sha(Path(cfg['target_model_dir'])/'tokenizer.json')
    owner=load(R/'owner-receipt.json');assert owner['stage']=='reaped' and owner['returncode']==0 and owner['waited_reaped']
    owner_pid,driver_pid=owner['owner_pid'],owner['driver_pid']
    assert owner['deadline_seconds']==900 and owner['pins_sha256']==sha(D/'pins.json') and owner['driver_sha256']==sha(D/'driver.py')
    strict_absence(owner['absence'],driver_pid)
    command=owner['command'];assert command[command.index('--out')+1]==str(C) and command[command.index('--pins-sha256')+1]==sha(D/'pins.json')
    plan=cfg['plan'];assert [a['mode'] for a in plan]==['none','mtp','mtp','none','none','mtp']
    arms=[];windows=[];rawpaths=[];resourcegroups=[];serverpids=[];ports=[];stats_all=[];arm_details=[]
    for item in plan:
        arm=C/item['label'];mode=item['mode'];result=load(arm/'result.json');launch=load(arm/'launch.json');pre=load(arm/'official-preimport.json')
        owned=load(arm/'owned-pid.json');pid=owned['pid'];serverpids.append(pid);assert owned['ppid']==driver_pid
        assert pre['pid']==pid and pre['ppid']==driver_pid and pre['engine']=='omlx' and pre['mlx_not_imported']
        assert pre['source_native_tree_sha256']=='291516fcd115066a1c01b23e5753d2827cc95a010cca6ac78b2f6da6b27a9f0e'
        assert pre['files_checked']==2688 and pre['controls']==cfg['child_controls'] and not pre.get('qualification_observers_added',False)
        cmd=launch['command'];assert str(P/'m4-final-native15-official-attempt2/official_entry.py') in cmd
        assert not any('v109' in x or 'observer' in x for x in cmd)
        assert launch['official_cli_original_entry'] and not launch['qualification_observers'] and not launch['measurement_wrappers']
        settings=load(arm/'state/model_settings.json')['models'][cfg['target_revision']]
        assert settings['vlm_mtp_enabled']==(mode=='mtp') and settings['mtp_enabled'] is False and settings['dflash_enabled'] is False
        assert settings['vlm_mtp_draft_block_size'] is None and settings['vlm_mtp_draft_model']==(cfg['draft_model_dir'] if mode=='mtp' else None)
        launch_settings=launch['model_settings'];assert all(settings[k]==v for k,v in launch_settings.items())
        assert launch['environment']['knobs']==cfg['child_controls'] and not launch['environment']['other_mlx_key_value_sha256']
        assert (arm/'models'/cfg['target_revision']).resolve()==Path(cfg['target_model_dir']).resolve()
        birth=load(arm/'owned-birth.json')['identity'];ready=load(arm/'owned-ready-identity.json');end=load(arm/'owned-pre-cleanup-identity.json')
        for proof in (ready,end):
            assert proof['pid']==pid and proof['owner_pid']==driver_pid and proof['same_start'] and proof['direct_Popen_child_live']
            assert proof['initial_identity_words'][:5]==birth[:5] and proof['current_identity_words'][:5]==birth[:5]
        assert ready['parent_query']['ppid']==driver_pid
        listener=load(arm/'server-ready-exact-listener.json');port=listener['port'];ports.append(port)
        assert listener['owned_pid']==pid and listener['returncode']==0 and listener['stderr']==''
        assert {int(x[1:]) for x in listener['stdout'].splitlines() if x.startswith('p')}=={pid}
        exitrec=load(arm/'owned-process-exit.json');assert exitrec['pid']==pid and exitrec['wait_completed'] and exitrec['returncode']==-15
        strict_absence(exitrec['exact_pid_absence'],pid)
        assert exitrec['recorded_birth_identity_sha256']==hjson(end['current_identity_words'])
        assert exitrec['signal_identity_decisions']==[{'signal':15,'current_identity_sha256':hjson(end['current_identity_words']),'decision':'SAME_IDENTITY_SIGNAL'}]
        absent=load(arm/'port-absence.json');assert absent['port']==port and absent['absent'] and absent['checks'][-1]['connect_ex']==errno.ECONNREFUSED
        cleanup=load(arm/'cleanup.json');assert cleanup['first_error'] is None and cleanup['cleanup_errors']==[] and cleanup['owned_exit']==exitrec
        log=read(arm/'server.log');routes=re.findall(r'vlm_mtp decode started: request=(\S+) uid=(-?\d+) block_size=(\S+)',log)
        stats=re.findall(r'vlm_mtp stats: request=(\S+) finish=(\S+) rounds=(\d+) accepted=(\d+)/(\d+) \([^)]*\) tokens_per_round=([\d.]+) emitted=(\d+) block_size=(\d+)',log)
        if mode=='none':
            assert not routes and not stats and 'VLM MTP drafter loaded:' not in log and 'VLM MTP drafter attached to' not in log
        else:
            marker='VLM MTP drafter loaded: path='+cfg['draft_model_dir']+' kind=mtp model_type=qwen3_5_mtp'
            assert log.count(marker)==1 and log.count('VLM MTP drafter attached to scheduler')==1 and log.count('VLM MTP drafter attached to engine:')==1
            assert len(routes)==len(stats)==6 and [x[0] for x in routes]==[x[0] for x in stats] and len({x[0] for x in routes})==6
            assert all(int(x[1])<0 and x[2]=='None' for x in routes)
            assert 'vlm_mtp routing skipped for' not in log and 'VLM MTP toggle on for' not in log
            for row in stats:
                rid,finish,rounds,accepted,denominator,tpr,emitted,block=row
                assert (finish,int(rounds),int(accepted),int(denominator),int(emitted),int(block))==('length',17,15,34,32,3)
                stats_all.append({'arm':item['label'],'request_id':rid,'rounds':17,'accepted':15,'logged_fixed_denominator':34,'emitted':32,'block':3})
        for phase,n in [('warm',3),('measure',6)]:
            proof=load(arm/f'post-{phase}-log-proof.json')
            if mode=='mtp':
                assert proof['decode_routes']==n and len(proof['terminal_stats'])==n and proof['loaded_exact_path']
                assert [x['request_id'] for x in proof['terminal_stats']]==[x[0] for x in stats[:n]]
                assert proof['proposal_count_from_measurement_verified'] is False
            else:assert proof['operational_MTP_events']==0
        groups={}
        for phase in ('warm','measure'):
            groups[phase]=[]
            for index in range(3):
                g=load(arm/f'C1-{phase}-{index}.json');assert g['actual_request_count']==1 and len(g['requests'])==1
                q=g['requests'][0];start,first,done,closed=(q[k] for k in ['start_perf_counter_s','first_text_perf_counter_s','done_perf_counter_s','cleanup_complete_perf_counter_s'])
                assert start<first<done<=closed
                events=q['events'];assert events and events[0]['client_perf_counter_s']==first and all(e['text'] for e in events)
                assert all(start<e['client_perf_counter_s']<=done for e in events)
                assert [e['client_perf_counter_s'] for e in events]==sorted(e['client_perf_counter_s'] for e in events)
                assert ''.join(e['text'] for e in events)==q['text'] and events[0]['text']==q['first_chunk_text']
                visible=tokenizer.encode(q['text'],add_special_tokens=False).ids;prefix=tokenizer.encode(events[0]['text'],add_special_tokens=False).ids
                assert visible==q['visible_roundtrip_ids']==cfg['qualified_control_raw32'] and len(visible)==32
                assert q['output_ids_when_available'] is None and visible[:len(prefix)]==prefix and len(prefix)==q['first_chunk_reconstructed_tokens']
                assert q['first_chunk_prefix_verified'] and q['full_visible_roundtrip_count_matches_usage']
                assert q['finish_reason']=='length' and q['usage']['prompt_tokens']==19 and q['usage']['completion_tokens']==32 and q['usage']['prompt_tokens_details']['cached_tokens']==0
                x=(32-len(prefix))/(done-first);y=32/(done-start)
                for a,b in [(q['client_total_s'],done-start),(q['client_ttft_text_s'],first-start),(q['client_post_first_chunk_s'],done-first),(q['stream_cleanup_after_done_s'],closed-done),(q['x_client_estimated_interactivity_tok_s_user'],x),(g['group_window_s'],done-start),(g['x_client_estimated_interactivity_tok_s_user'],x),(g['y_total_output_throughput_tok_s_mac'],y)]:same(a,b)
                assert g['total_completion_tokens']==32
                windows.append((start,done));groups[phase].append({'x':x,'y':y,'ttft_s':first-start,'E2E_s':done-start})
        metrics={key:st.median(g[key] for g in groups['measure']) for key in ('x','y','ttft_s','E2E_s')}
        for key,value in metrics.items():same(value,result[key])
        arms.append({'label':item['label'],'pair':item['pair'],'mode':mode,**metrics})
        samples=load(arm/'resources.json');assert len(samples)==17
        raw=[arm/f'resources-{i:02d}-raw.json' for i in range(17)];rawpaths.extend(raw);resourcegroups.append(raw)
        bylabel={s['label']:s for s in samples}
        for phase in ('warm','measure'):
            for index in range(3):
                q=load(arm/f'C1-{phase}-{index}.json')['requests'][0]
                assert bylabel[f'before-C1-{phase}-{index}']['monotonic_s']<q['start_perf_counter_s']
                assert bylabel[f'after-C1-{phase}-{index}']['monotonic_s']>q['done_perf_counter_s']
        arm_details.append({'label':item['label'],'pid':pid,'port':port,'sampled_peak_RSS_MiB':max((s['exact_owned_pid_metadata'].get('rss_kib') or 0)/1024 for s in samples),
                            'swap_used_MB_range':[min(s['swap_used_mb'] for s in samples),max(s['swap_used_mb'] for s in samples)],
                            'Pageouts_delta':samples[-1]['vm_counters']['Pageouts']-samples[0]['vm_counters']['Pageouts']})
    assert len(set(serverpids))==len(set(ports))==6
    for a,b in zip(arms,suite['arms']):
        assert (a['label'],a['pair'],a['mode'])==(b['label'],b['pair'],b['mode'])
        for key in ('x','y','ttft_s','E2E_s'):same(a[key],b[key])
    aggregate={mode:{key:st.median(a[key] for a in arms if a['mode']==mode) for key in ('x','y','ttft_s','E2E_s')} for mode in ('none','mtp')}
    pairs=[]
    for pair in (1,2,3):
        sub=[a for a in arms if a['pair']==pair];none=next(a for a in sub if a['mode']=='none');mtp=next(a for a in sub if a['mode']=='mtp')
        row={'pair':pair,'order':[a['mode'] for a in sub],**{f'{key}_signed_MTP_pct':100*(mtp[key]/none[key]-1) for key in ('x','y')}}
        pairs.append(row)
        for key in ('x','y'):same(row[f'{key}_signed_MTP_pct'],suite['pairs'][pair-1][f'{key}_signed_MTP_pct'])
    paired={key:st.median(p[f'{key}_signed_MTP_pct'] for p in pairs) for key in ('x','y')}
    ratio_of_aggregate={key:100*(aggregate['mtp'][key]/aggregate['none'][key]-1) for key in ('x','y')}
    for mode,values in aggregate.items():
        for key,value in values.items():same(value,suite['arm_medians'][mode][key])
    for key,value in paired.items():same(value,suite['paired_signed_medians'][f'{key}_signed_MTP_pct'])
    assert all(p['x_signed_MTP_pct']<0 and p['y_signed_MTP_pct']<0 for p in pairs)
    assert suite['all_signed_pairs_retained'] and suite['none_baseline_retained'] and not suite['negative_or_slower_arms_discarded']
    assert not suite['adopted'] and not suite['global_best_claim'] and not suite['fullfrontier_win']
    assert (suite['HTTP_requests'],suite['HTTP_completion_tokens'],suite['measured_requests'],suite['measured_tokens'])==(36,1152,18,576)
    ownerraw=sorted(C.glob('owner-resources-*-raw.json'));assert len(ownerraw)==12;rawpaths.extend(ownerraw);resourcegroups.append(ownerraw)
    ext_raw=R/'external-resource-00-raw.json';rawpaths.append(ext_raw);resourcegroups.append([ext_raw])
    allraw=[]
    for path in rawpaths:
        s=load(path);allraw.append(s)
        assert s['power_metadata']['status']=='known' and s['power_metadata']['source']=='AC Power'
        assert "Now drawing from 'AC Power'" in s['power_metadata']['raw']['stdout'] and s['power_metadata']['raw']['returncode']==0
        assert s['processinfo_thermal_metadata']['code']==0 and s['processinfo_thermal_metadata']['raw']['stdout']=='0'
        assert s['pressure_level_value']==1 and s['pressure_level']['returncode']==0 and s['pressure_level']['text']=='1'
        assert s['process_inventory_ok'] and s['only_owned_known_inference_processes']
        assert all(p['owned'] and p['pid']==s['owned_model_pid'] for p in s['known_inference_processes'])
        assert not any(start<=s['monotonic_s']<=end for start,end in windows),'resource start lies inside a request clock'
        for key,value in s['vm_counters'].items():assert int(re.search(r'^'+key+r':\s*(\d+)',s['vm_stat']['text'],re.M)[1])==value
        same(float(re.search(r'used = ([\d.]+)M',s['swap']['text'])[1]),s['swap_used_mb'])
    assert len(allraw)==115 and len(windows)==36
    # Replay the exact original sustained-paging guards, independently from saved deltas.
    for group in resourcegroups:
        values=[load(path) for path in group]
        for i in range(2,len(values)):
            d=[values[j]['swap_used_mb']-values[j-1]['swap_used_mb'] for j in (i-1,i)]
            assert not all(x>64 for x in d)
            d=[values[j]['vm_counters']['Swapouts']-values[j-1]['vm_counters']['Swapouts'] for j in (i-1,i)]
            assert not all(x>0 for x in d)
    ownersamples=load(C/'owner-resources.json');bylabel={s['label']:s for s in ownersamples}
    for index,item in enumerate(plan[:-1],1):
        gap=load(C/f'cooling-{index}.json');assert gap['after_arm']==item['label'] and gap['fixed_seconds']==20 and gap['elapsed_seconds']>=20 and not gap['adaptive']
        before,after=(bylabel[f'cooling-{index}-{stage}']['monotonic_s'] for stage in ('before','after'))
        assert after-before>=20
        prior_port=load(C/item['label']/'port-absence.json')['checks'][-1]['monotonic_s']
        next_launch=load(C/plan[index]['label']/'resources-00-raw.json')['monotonic_s']
        assert prior_port<=before<after<=next_launch
    cleanup=load(C/'owner-cleanup.json');assert cleanup['pid']==driver_pid and cleanup['completed_arms']==6 and cleanup['all_owned_servers_reaped'] and cleanup['error'] is None
    assert set(cleanup['recorded_owned_server_arms'])=={a['label'] for a in arms}
    external=load(R/'external-release.json');assert external['status']=='PASS_EXTERNAL_RELEASE' and external['GPU_released'] and external['tool_exit']==0
    assert {x['pid'] for x in external['strict_absence']}=={owner_pid,driver_pid}
    for row in external['strict_absence']:strict_absence(row,row['pid'])
    summary={'status':'PASS_INDEPENDENT_SAVED_RESULT_REPLAY_MTP_SLOWER_NO_ADOPTION','scope':'source and saved-result replay, not independent GPU/HTTP recapture',
             'arms':arms,'arm_details':arm_details,'pairs':pairs,'aggregate_arm_medians':aggregate,'median_signed_pair_pct':paired,'ratio_of_aggregate_medians_pct':ratio_of_aggregate,
             'HTTP_requests':36,'HTTP_tokens':1152,'measured_requests':18,'measured_tokens':576,'raw_measurement_IDs_available':False,'independently_reencoded_visible32_requests':36,
             'MTP_logged_requests':len(stats_all),'MTP_logged_rounds':sum(r['rounds'] for r in stats_all),'MTP_logged_accepted':sum(r['accepted'] for r in stats_all),
             'MTP_logged_fixed_denominator':sum(r['logged_fixed_denominator'] for r in stats_all),'denominator_is_actual_proposals':False,'MTP_per_request_stats':stats_all,
             'prior109_actual_per_verify_qualification_sha256':cfg['qualification']['receipt_sha256'],
             'resources':{'raw_samples':len(allraw),'AC_Power_all':True,'thermal0_all':True,'pressure1_all':True,'resource_starts_outside_all36_requestwindows':True,
                          'free_percent_range':[min(s['free_percent'] for s in allraw),max(s['free_percent'] for s in allraw)],
                          'swap_used_MB_range':[min(s['swap_used_mb'] for s in allraw),max(s['swap_used_mb'] for s in allraw)],
                          'Swapouts_values':sorted({s['vm_counters']['Swapouts'] for s in allraw}),'sum_arm_Pageouts_delta':sum(a['Pageouts_delta'] for a in arm_details),
                          'original_sustained_paging_guards_pass':True,'zero_paging_claim':False,'RSS_is_sampled_process_RSS_not_total_GPU_memory':True},
             'ownership':{'root_wait_owner_pid':owner_pid,'driver_pid':driver_pid,'server_pids':serverpids,'ports':ports,'server_wait_returncodes':[-15]*6,'all8_saved_strict_absence':True,
                          'root_wait900s_reap0':True,'root_external_release':True,'independent_lease_artifact':False,'lease_scope':'serial execution convention per root; no separate saved lease file'},
             'performance_adoption_or_globalbest_or_fullfrontier_claim':False,'MLX_GPU_model_server_or_numpy_used_by_reviewer':False}
    (O/'verification.json').write_text(json.dumps(summary,indent=2)+'\n')
    # Preserve exact original runtime text bytes and small source/config/qualification evidence.
    archive=[]
    for f in R.rglob('*'):
        if f.is_file() and not f.is_symlink() and 'models' not in f.relative_to(R).parts and f.suffix in ('.json','.log'):
            archive.append((f,'runtime/'+str(f.relative_to(R))))
    for name in ['driver.py','checks.py','support.py','cpu_checks.py','config.json','protocol.json','pins.json','source-proof.json','CPU-READY.json','README.md']:
        archive.append((D/name,'source/'+name))
    for key in ('receipt','release','source_pins'):
        archive.append((Path(cfg['qualification'][key]),'prior109/'+key+'.json'))
    for i,review in enumerate(cfg['independent_reviews']):
        for key in ('report','receipt'):archive.append((Path(review[key]),f'prior109/review{i+1}-{key}'+('.md' if key=='report' else '.json')))
    sealed={name:{'original_path':str(f),'bytes':f.stat().st_size,'sha256':sha(f)} for f,name in archive}
    with tarfile.open(O/'saved-evidence.tar.gz','w:gz') as tar:
        for f,name in archive:tar.add(f,arcname=name,recursive=False)
    for f,name in archive:assert sha(f)==sealed[name]['sha256']
    (O/'evidence-manifest.json').write_text(json.dumps(sealed,indent=2)+'\n')
    seen[str(Path(__file__))]=sha(Path(__file__))
    (O/'source-map.json').write_text(json.dumps(seen,indent=2)+'\n')
    print(json.dumps({'status':summary['status'],'aggregate':aggregate,'paired':paired,'aggregate_ratio':ratio_of_aggregate,'resource':summary['resources'],'seal_files':len(sealed),'seal_bytes':(O/'saved-evidence.tar.gz').stat().st_size},indent=2))

if __name__=='__main__':main()
