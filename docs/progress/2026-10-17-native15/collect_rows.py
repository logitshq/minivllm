"""Read-only saved HTTP evidence -> safe aggregate snapshot and timing whitelist.
No raw response text, IDs, argv, settings, credential names/values are printed.
"""
from pathlib import Path
import csv,hashlib,json,math,re,shutil,statistics
D=Path(__file__).resolve().parent
P=D.parent
SOURCE=P/'m4-final-native15-official-attempt2'
OUT=SOURCE/'output'
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
def write(f,v):f.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
FORBIDDEN=re.compile(r'(^|_)(credential(s)?|api_?key|access_token|refresh_token|password|secret|authorization|cookie|argv|settings|raw|stdout|stderr)($|_)',re.I)
def safe(v):
    if isinstance(v,dict):
        for k,w in v.items():
            if FORBIDDEN.search(k) or k.lower() in ('text','rawtext','request_body','response_body'):raise ValueError('unsafe aggregate field; names/values suppressed')
            safe(w)
    elif isinstance(v,list):
        for w in v:safe(w)
    elif isinstance(v,str):
        if re.search(r'\b(?:Bearer\s+[A-Za-z0-9._~-]+|sk-[A-Za-z0-9]{12,})',v):raise ValueError('unsafe aggregate value; content suppressed')
def close(a,b):assert math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
def main():
    files=sorted(OUT.glob('*-validated.json'))+[OUT/'suite-complete.json'];assert len(files)==22
    loaded={f.name:json.loads(f.read_text()) for f in files}
    for q in loaded.values():safe(q)
    target=D/'aggregate-summary';target.mkdir(exist_ok=True)
    for f in files:shutil.copyfile(f,target/f.name);assert sha(f)==sha(target/f.name)
    suite=loaded['suite-complete.json'];assert suite['actual_requests']==270 and suite['actual_completion_tokens']==8640 and suite['measured_requests']==135 and suite['measured_tokens']==4320
    aggregates=[loaded[f.name] for f in files[:-1]];assert sorted(aggregates,key=lambda q:q['label'])==sorted(suite['configs'],key=lambda q:q['label'])
    arms=[];groups=[];requests=[];source_map=[];thermal={};power={};pressure={}
    for f in files:source_map.append({'role':'exact_aggregate_copy','source':str(f),'copy':str(target/f.name),'sha256':sha(f),'bytes':f.stat().st_size})
    for agg in aggregates:
        label,engine=agg['label'],agg['engine'];path=Path(agg['report']);assert sha(path)==agg['report_sha256']
        report=json.loads(path.read_text());batch=report['concurrency'];assert batch in (1,2,4) and report['engine']==engine and report['label']==label
        assert report['prefix_cache']==0 and report['output_cap']==32 and report['sampling']['temperature']==0 and report['sampling']['draft'] is None and report['sampling']['EOS']=='normal'
        assert len(report['prompt_ids'])==19 and report['activation_dtype']=='bfloat16'
        assert report['quantization']=={'group_size':16,'bits':4,'mode':'nvfp4'}
        assert report['checkpoint_revision']=='5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6'
        source_map.append({'role':'read_only_arm_report_not_copied','source':str(path),'sha256':sha(path),'safe_extraction':'whitelisted timing/count/labels only'})
        measured=[];armreq=[]
        for phase,values in [('warmup',report['warmups']),('measure',report['runs'])]:
            assert len(values)==3
            for index,g in enumerate(values):
                assert len(g['requests'])==batch and g['actual_request_count']==batch
                ds=[];xs=[];starts=[];dones=[];tokens=0
                for row in g['requests']:
                    start,end,first=row['start_perf_counter_s'],row['done_perf_counter_s'],row['first_text_perf_counter_s']
                    assert all(math.isfinite(x) for x in (start,end,first)) and start<=first<end
                    duration=end-start;close(duration,row['client_total_s']);close(end-first,row['client_post_first_chunk_s'])
                    assert row['usage']['prompt_tokens']==19 and row['usage']['completion_tokens']==32 and row['finish_reason']=='length'
                    assert row['usage'].get('prompt_tokens_details',{}).get('cached_tokens',0)==0
                    assert row['first_chunk_prefix_verified'] and row['full_visible_roundtrip_count_matches_usage']
                    if row['output_ids_when_available'] is not None:assert len(row['output_ids_when_available'])==32
                    count=row['first_chunk_reconstructed_tokens'];assert type(count)is int and 0<count<32
                    rate=(32-count)/(end-first);close(rate,row['x_client_estimated_interactivity_tok_s_user'])
                    record={'arm':label,'engine':engine,'concurrency':batch,'phase':phase,'group_index':index,'group_label':g['group_id'],'request_label':row['request_id'],'start_perf_counter_s':start,'done_perf_counter_s':end,'duration_s':duration,'first_text_perf_counter_s':first,'first_chunk_reconstructed_token_count':count,'prompt_token_count':19,'completion_token_count':32}
                    requests.append(record);armreq.append(record);ds.append(duration);xs.append(rate);starts.append(start);dones.append(end);tokens+=32
                window=max(dones)-min(starts);close(window,g['group_window_s']);y=tokens/window;close(y,g['y_total_output_throughput_tok_s_mac']);x=statistics.median(xs);close(x,g['x_client_estimated_interactivity_tok_s_user'])
                record={'arm':label,'engine':engine,'concurrency':batch,'phase':phase,'group_index':index,'group_label':g['group_id'],'request_count':batch,'request_p50_latency_ms':statistics.median(ds)*1000,'group_window_s':window,'completion_tokens':tokens,'y_total_output_throughput_tok_s':y,'old_x_interactivity_tok_s':x}
                groups.append(record)
                if phase=='measure':measured.append(record)
        lat=statistics.median(r['request_p50_latency_ms'] for r in measured);x=statistics.median(r['old_x_interactivity_tok_s'] for r in measured);y=statistics.median(r['y_total_output_throughput_tok_s'] for r in measured)
        close(x,agg['x']);close(y,agg['y']);assert agg['actual_completed_requests']==6*batch and agg['actual_completion_tokens']==192*batch
        allp50=statistics.median(r['duration_s'] for r in armreq if r['phase']=='measure')*1000
        arms.append({'arm':label,'engine':engine,'concurrency':batch,'latency_ms':lat,'throughput_tok_s':y,'old_x_interactivity_tok_s':x,'all_measured_request_p50_ms_appendix_only':allp50,'measure_groups':3,'measured_requests':3*batch,'measured_tokens':96*batch,'source_aggregate':f'{label}-validated.json','source_report_sha256':agg['report_sha256']})
        for sample in report['resource_snapshots']:
            code=sample.get('processinfo_thermal_metadata',{}).get('code','UNKNOWN');thermal[str(code)]=thermal.get(str(code),0)+1
            code=sample.get('power_metadata',{}).get('source','UNKNOWN');power[str(code)]=power.get(str(code),0)+1
            code=sample.get('pressure_level_value','UNKNOWN');pressure[str(code)]=pressure.get(str(code),0)+1
    configurations=[]
    for engine in ['native','omlx','lithos']:
        for batch in [1,2,4]:
            rs=[r for r in arms if r['engine']==engine and r['concurrency']==batch]
            if not rs:assert engine=='lithos' and batch in(2,4);continue
            assert len(rs)==3
            configurations.append({'engine':engine,'concurrency':batch,'latency_ms':statistics.median(r['latency_ms'] for r in rs),'throughput_tok_s':statistics.median(r['throughput_tok_s'] for r in rs),'old_x_interactivity_tok_s':statistics.median(r['old_x_interactivity_tok_s'] for r in rs),'fresh_arms':3,'measured_requests':sum(r['measured_requests'] for r in rs),'arm_labels':[r['arm'] for r in rs]})
    assert len(arms)==21 and len(configurations)==7 and len(groups)==126 and len(requests)==270
    assert sum(r['phase']=='measure' for r in requests)==135 and sum(r['completion_token_count'] for r in requests if r['phase']=='measure')==4320
    result={'snapshot_date':'2026-10-17','measurement_round':'historical m4-final-native15-official-attempt2','model':'Qwen3.8-27B NVFP4 g16/b4','device':'Apple M4 Max 48GiB','workload':'19 prompt tokens ->32 completion tokens, greedy/normal EOS, uncached','latency_definition':'per measure group p50(DONE-start), then median3groups perarm, median3fresh arms perconfiguration; milliseconds','throughput_definition':'per group B*32/(maxDONE-minPOSTstart), then median3measuregroups perarm, median3fresharms perconfiguration; token/s','old_x_definition':'per request (32-firstnonemptychunk reconstructedtokens)/(DONE-firstnonemptytext), then requestmedian/group, median3groups/arm, median3arms/config; token/s, not inverse latency','arms':arms,'configurations':configurations,'not_available':[{'engine':'lithos','concurrency':b,'status':'N/A','reason':'concurrency429 boundary; no measuredpoint inthis campaign, not zero'} for b in(2,4)],'resources_aggregate_only':{'thermal_state_sample_counts':thermal,'power_sample_counts':power,'pressure_level_sample_counts':pressure},'limits':['Historical accepted native15 source, not newest native candidate.','Three fresh repetitions and mixedthermal states; no significance/completeParetofrontier assertion.','No M5/GLM numbers mixed; Qwen3.8 native15 snapshot only.','Only aggregate copies and whitelisted timing/count/labels exported, not rawHTTP text/IDs/settings/credentials.','Existing suite GPU_released flag is historical completion-time metadata, not current liveGPU state; no currentresourcecheck or GPU action performed.']}
    write(D/'reviewed_rows.json',result);write(D/'group_rows.json',groups);write(D/'request_timing_whitelist.json',requests)
    for filename,rows in [('arm_rows.csv',arms),('configuration_medians.csv',configurations),('request_timing_whitelist.csv',requests)]:
        with (D/filename).open('w',newline='') as h:
            w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    source_map += [{'role':'definition_source_not_copied','source':str(SOURCE/'client.py'),'sha256':sha(SOURCE/'client.py'),'line_ranges':['80-112','142','164']}]
    write(D/'source-map.json',source_map)
    write(D/'verification.json',{'status':'PASS_CPU_SAVED_TIMING_AND_AGGREGATE_REPLAY','aggregate_recursive_scan_passed':True,'copied_aggregate_files_exact':22,'fresh_arms':21,'configuration_medians':7,'groups':126,'requests':270,'measured_requests':135,'measured_tokens':4320,'request_duration_from_actual_start_DONE':True,'saved_rate_and_groupthroughput_reproduced':True,'GPU_used':False,'raw_reports_copied':False,'W_Git_mutated':False})
    print('REVIEWED: 22 exact aggregate files; 21 arms; 7 medians; 270 whitelisted requests /135 measured; no unsafe field names or values printed.')
if __name__=='__main__':main()
