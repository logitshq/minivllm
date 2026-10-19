"""One-use offline C1 profile readout; no backend imports, service actions or Pareto point."""
import hashlib, importlib.util, json, math, statistics
from collections import Counter, defaultdict
from pathlib import Path
R=Path('/Users/ryanli/code/minivllm'); N=R/'bench/m5_ultra_256gb/c1_profile_cycle_v1'; O=R/'.artifacts/m5-ultra-256gb/c1-profile-cycle-v1-owner'
HERE=Path(__file__).resolve().parent; INPUTS={}

def need(ok,message):
    if not ok: raise ValueError(message)
def read(p):
    raw=p.read_bytes(); INPUTS[str(p)]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)); return json.loads(raw)
def merge(values):
    result=[]
    for start,end in sorted(values):
        if end<=start: continue
        if result and start<=result[-1][1]: result[-1]=(result[-1][0],max(end,result[-1][1]))
        else: result.append((start,end))
    return result
def duration(values): return sum(b-a for a,b in merge(values))/1e9
def intersection(a,b):
    a,b=merge(a),merge(b); result=[]; i=j=0
    while i<len(a) and j<len(b):
        lo=max(a[i][0],b[j][0]); hi=min(a[i][1],b[j][1])
        if hi>lo: result.append((lo,hi))
        if a[i][1]<=b[j][1]: i+=1
        else: j+=1
    return result
def stats(values):
    values=sorted(values)
    return dict(count=len(values),sum=sum(values),mean=statistics.mean(values),p50=statistics.median(values),p95=values[math.ceil(.95*len(values))-1],max=values[-1]) if values else dict(count=0)
def spans(events): return [(e['start_ns'],e['end_ns']) for e in events]

def main():
    audit=read(N/'analysis-v1/independent-diagnostic.json'); restore=read(N/'analysis-v1/restoration-audit.json')
    need(audit['status']==restore['status']=='PASS' and audit['diagnostic_complete'] and restore['qualified'],'closed independent audits required')
    jp=R/'bench/m5_ultra_256gb/pareto_realistic/http_join.py'; raw=jp.read_bytes()
    need(hashlib.sha256(raw).hexdigest()=='a362c321c2329d7ab53ab6a1a5a96981bb95b44733739fdbb56dd3d38bef7b4e','fixed raw UUID parser')
    INPUTS[str(jp)]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
    sp=importlib.util.spec_from_file_location('profile_raw_completion',jp); join=importlib.util.module_from_spec(sp); sp.loader.exec_module(join)
    result=dict(schema='C1-profile-readout-v1',status='PASS',performance_qualified=False,numerical_qualified=False,
        mode_order=['off','runner','full'],requests_per_mode=8,output_tokens_per_request=512,modes={},comparisons={},
        limitations=['One fixed-order process per mode, no repeat variance or causal observer-overhead estimate.',
            'All modes include shared admission/DONE/drain bookkeeping; none is a formal Pareto point.',
            'Python wall spans overlap and include driver/scheduler effects; existing_eval is not GPU-only time.',
            'Uninstrumented interval residual is not guaranteed removable CPU work.',
            'Local server clocks only are used for span unions; client clocks only for POST-to-DONE metrics.',
            'Driver observed_diagnostic_complete remains intentionally false; independent audit is the completion authority.'])
    original_bodies=None
    for mode in result['mode_order']:
        trial=O/'phases'/mode/'trial'; group=read(trial/'diagnostic-group.json'); spec=read(trial/'spec.json'); closure=read(trial/'server/closure.json')
        rows=group['requests']; bodies={r['fixture_id']:r['body'] for r in rows}; need(len(bodies)==len(rows)==8,'eight unique D8 requests')
        need(bodies=={j['fixture_id']:j['body'] for j in spec['jobs']['diagnostic']},'original diagnostic bodies')
        if original_bodies is None: original_bodies=bodies
        need(bodies==original_bodies,'same bodies across modes')
        closed={r['rid']:r for r in closure['drain']['requests'] if r['phase']=='diagnostic'}; requests=[]; windows={}
        for row in rows:
            rid,terminal,_=join.raw_completion(row); c=closed[rid]; need(terminal['usage']['completion_tokens']==512,'fixed output count')
            lo,hi=c['entered_ns'],c['done_flush_ns']; need(lo<hi<=c['handler_exit_ns'],'local closure clocks'); windows[rid]=[(lo,hi)]
            requests.append(dict(fixture_id=row['fixture_id'],rid=rid,e2e_s=row['done_perf_counter_s']-row['start_perf_counter_s'],ttft_s=row['ttft_s'],local_service_s=(hi-lo)/1e9,handler_exit_tail_us=(c['handler_exit_ns']-hi)/1e3))
        need(set(windows)==set(closed),'actual raw RID/closure join')
        metric=next(m for m in audit['modes'] if m['mode']==mode); need(metric['natural_exit'] is True,'natural exit')
        item=dict(audited_metrics=metric['metrics'],requests=requests,e2e_s=stats([r['e2e_s'] for r in requests]),ttft_s=stats([r['ttft_s'] for r in requests]))
        result['modes'][mode]=item
        if mode=='off': continue
        profile=read(trial/'server/profile.json'); events=profile['events']; need(profile['status']=='RECORDED' and not profile['overflow'] and len(events)==profile['event_count'],'complete profile')
        steps={e['step']:e for e in events if e['kind']=='runner_step' and set(e['rids']+e['prefill_rids']) & set(windows)}
        step_rid={}
        for step,e in steps.items():
            rids=set(e['rids']+e['prefill_rids']); need(len(rids)==1 and rids<=set(windows),'single actual RID step'); step_rid[step]=next(iter(rids))
        by_rid=defaultdict(list); graph_markers=[]; ignored=Counter()
        for e in events:
            need(type(e['start_ns']) is type(e['end_ns']) is int and e['end_ns']>=e['start_ns'] and not e.get('error'),'valid event interval')
            rid=e.get('rid') or step_rid.get(e.get('step'))
            if rid in windows:
                if e['kind']=='callable_family_created': graph_markers.append(e); continue
                by_rid[rid].append(e)
            elif e['kind'] in ('engine_step','http_handler'):
                for request_rid,window in windows.items():
                    if intersection(spans([e]),window): by_rid[request_rid].append(e)
            else: ignored[e['kind']]+=1
        kind_spans=defaultdict(list); kind_durations=defaultdict(list); residuals=Counter(); worker_intervals=[]; delivery_intervals=[]; request_stats=[]
        for request in requests:
            rid=request['rid']; window=windows[rid]; chosen=by_rid[rid]; kinds=defaultdict(list)
            for e in chosen:
                clipped=intersection(spans([e]),window)
                if clipped:
                    kinds[e['kind']]+=clipped; kind_spans[e['kind']]+=clipped; kind_durations[e['kind']].append(duration(clipped)*1000)
            runner=kinds['runner_step']; engine=kinds['engine_step']
            nested=sum((v for k,v in kinds.items() if k in ('decode_graph_submission','compiled_callable','sample_graph','existing_eval','existing_async_eval')),[])
            active=sum((v for k,v in kinds.items() if k in ('text_decode','http_write','http_flush')),[])
            residuals['runner_minus_nested_s']+=duration(runner)-duration(intersection(runner,nested))
            residuals['engine_minus_runner_s']+=duration(engine)-duration(intersection(engine,runner))
            residuals['service_outside_engine_s']+=duration(window)-duration(intersection(window,engine)) if mode=='full' else 0
            worker_intervals+=engine or runner; delivery_intervals+=active
            rs=[e for e in steps.values() if step_rid[e['step']]==rid]; ds=[e for e in rs if e['rids']]; ps=[e for e in rs if e['prefill_rids']]
            decode_windows=spans(ds); decode_children=intersection(nested,decode_windows)
            decode_counts=Counter(role for e in ds for role in e['actual_decode_calls'])
            request_stats.append(dict(fixture_id=request['fixture_id'],rid=rid,runner_steps=len(rs),prefill_steps=len(ps),decode_steps=len(ds),
                ahead_consumed=sum(e['ahead_consumed'] for e in ds),prior_ahead_key_matched=sum(e['prior_ahead_key_matched'] for e in ds),successor_retained=sum(e['successor_retained'] for e in ds),
                decode_calls=dict(decode_counts),runner_s=duration(runner),decode_runner_s=duration(decode_windows),prefill_runner_s=duration(spans(ps)),
                decode_runner_minus_nested_s=duration(decode_windows)-duration(decode_children)))
            if mode=='full':
                puts=[e for e in chosen if e['kind']=='delivery_queue_put']; gets=[e for e in chosen if e['kind']=='delivery_queue_get']
                put_map={e['ordinal']:e for e in puts}; get_map={e['ordinal']:e for e in gets}
                need(len(put_map)==len(puts)==len(get_map)==len(gets) and set(put_map)==set(get_map),'one put/get per actual message ordinal')
                request_stats[-1]['queue_messages']=len(put_map)
        all_windows=sum(windows.values(),[]); local_s=duration(all_windows)
        item['profile']=dict(total_recorded_events=len(events),selected_runner_steps=len(steps),local_service_s=local_s,
            per_kind={k:dict(interval_union_s=duration(v),service_coverage_pct=100*duration(v)/local_s,clipped_duration_ms=stats(kind_durations[k])) for k,v in sorted(kind_spans.items())},
            residual_wall_s=dict(residuals),delivery_active_union_s=duration(delivery_intervals),delivery_worker_overlap_s=duration(intersection(delivery_intervals,worker_intervals)),
            requests=request_stats,diagnostic_graph_creation_markers=len(graph_markers),diagnostic_first_callable_invocations=sum(e['kind']=='compiled_callable' and e['invocation']==1 for es in by_rid.values() for e in es),
            ignored_event_counts=dict(ignored),scope='Clipped server-clock wall unions; nested intervals and cross-thread overlaps are not additive or exclusive GPU costs.')
    for top,base in [('runner','off'),('full','off'),('full','runner')]:
        t,b=result['modes'][top],result['modes'][base]; bm={r['fixture_id']:r for r in b['requests']}
        result['comparisons'][top+'_vs_'+base]=dict(common_window_delta_pct=100*(t['audited_metrics']['diagnostic']['common_window_s']/b['audited_metrics']['diagnostic']['common_window_s']-1),
            throughput_delta_pct=100*(t['audited_metrics']['diagnostic']['aggregate_output_tok_s']/b['audited_metrics']['diagnostic']['aggregate_output_tok_s']-1),
            paired_e2e_delta_pct={r['fixture_id']:100*(r['e2e_s']/bm[r['fixture_id']]['e2e_s']-1) for r in t['requests']})
    result['input_files']=INPUTS
    with (HERE/'readout.json').open('x') as stream: json.dump(result,stream,indent=2,allow_nan=False); stream.write('\n')
    print(json.dumps({m:dict(metrics=v['audited_metrics']['diagnostic'],profile={k:v.get('profile',{}).get(k) for k in ['selected_runner_steps','local_service_s','residual_wall_s','delivery_active_union_s','delivery_worker_overlap_s','diagnostic_graph_creation_markers','diagnostic_first_callable_invocations']}) for m,v in result['modes'].items()},indent=2))
if __name__=='__main__': main()
