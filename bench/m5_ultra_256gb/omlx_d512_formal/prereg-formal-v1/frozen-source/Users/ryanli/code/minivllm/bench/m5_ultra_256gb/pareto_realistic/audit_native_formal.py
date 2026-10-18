"""Offline native formal cycle audit: raw tokens/latency plus ownership/resources.

No network, process probes, model execution or service action. Requires a complete
RESTORED cycle and all registered trials. Reuses pinned source/runtime and model
pre/post proof validation, including legitimate zero-byte Python files. Runtime
and copied sources are independently hashed; large model payloads are not hashed
again (full pre/post receipts and unchanged current file stats are checked).
Raw SSE is parsed again; actual committed tokens/common queue window and full
POST-to-DONE p50/p95 are independently recomputed. Original point files remain
performance_qualified=false. Only this final audited cycle can grant eligibility.
Representative captured histories supply numerical scope, not a proof of every
new formal arrival schedule or any claim about competitors or Pareto dominance.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
OWNER_SHA='e33faa8f211e905f40daabd00941f3fd016041c9705c6a7a4a2fe805fb0aceff'
FORMAL_SHA='ec9cdc924f7987ad46eac6820b6264495c95fb182c726545a6e3ce283df7e654'
RESTORE_SHA='af46699fcea1065c4fcf69a85225a916987fba484098d53ef2acb905eb4ae3a8'
LEGACY_SHA='8d8aee0372497fcf591d53e4bc02b3d1e7eac7eb1d2bf639168a72802e2395bd'
JOIN_SHA='a362c321c2329d7ab53ab6a1a5a96981bb95b44733739fdbb56dd3d38bef7b4e'
CAPTURE_SHA='536fe41d6c8e2bdffe83ba6f43fba87dd6265deb4c84f6204bc9c7220f8280e3'
CLIENT_SHA='ef64a114f4f7d90ec5044d6bdcf058ca05f98dea445d6639db706a4fc75217dc'
BASE_PLAN_SHA='21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb'
JOB='gui/501/local.omlx.server'


class Incomplete(Exception):pass
def need(ok,message):
    if not ok:raise ValueError(message)
def same(a,b,message):
    need(json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False),message)
def finite(x):return type(x) in (int,float) and math.isfinite(x)
def read(path):
    need(Path(path).suffix!='.jsonl','no trace/numerical ledger is opened by formal audit')
    return json.loads(Path(path).read_bytes())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while block:=f.read(1<<20):h.update(block)
    return h.hexdigest()
def load(path,digest,name):
    need(sha(path)==digest,'support source changed: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def pinned(item):
    need(set(item)=={'path','sha256'} and Path(item['path']).is_absolute(),'absolute pinned reference')
    need(sha(item['path'])==item['sha256'],'pinned receipt changed');return Path(item['path'])
def ref(path):return dict(path=str(path),sha256=sha(path))
def legacy():return load(HERE/'audit_http_owner_v2.py',LEGACY_SHA,'_formal_preparation_audit')
def preparation_pins(owner,wrapper,numeric):return legacy().preparation_pins(owner,wrapper,numeric)
def absence(row,pid):return legacy().absence(row,pid)


def resources(owner,complete,children,original_pid,created,phases):
    count=complete['samples'];need(type(count) is int and count>=4 and complete['absence_probes']==count,'resource sample count')
    for prefix in ('resource','absence'):
        need(sorted(p.name for p in owner.glob(prefix+'-*.json'))==[f'{prefix}-{i:03d}.json' for i in range(count)],'resource/absence file sequence')
    rows=[];deltas=[];known=Counter();labels={p:[] for p in phases};last_stage=-1
    for i in range(count):
        x=read(owner/f'resource-{i:03d}.json');absent=read(owner/f'absence-{i:03d}.json');label=x['label']
        need(absent['label']==label and absent['resource_index_if_sampled']==i,'absence/resource pairing');absence(absent,original_pid)
        wall=datetime.fromisoformat(x['recorded_at']).timestamp()
        need(finite(absent['time']) and created<=absent['time']<=wall and finite(x['monotonic_s']),'resource timestamp')
        expected_pid=None
        if i<2:need(label==('before-prepare','after-prepare')[i] and wall<=children[0]['spawned_at'],'initial preparation boundary')
        elif i==count-1:need(label=='postflight' and wall>=children[-1]['released_at'],'postflight after all children')
        else:
            stage,sub=label.split(':',1);need(stage in phases,'unknown resource phase')
            n=phases.index(stage);need(last_stage<=n<=last_stage+1,'phase sample order');last_stage=n
            child=children[n];labels[stage].append(sub)
            prior=created if n==0 else children[n-1]['released_at']
            need(wall>=prior,'sample before previous phase reaped')
            if sub=='pre-spawn':need(wall<=child['spawned_at'],'pre-spawn sample after Popen')
            elif sub in ('released','after-reap'):
                end=child['inner_released_at'] if sub=='after-reap' else child['released_at']
                need(wall>=end,'released sample before reap')
            else:
                need(child['spawned_at']<=wall<=child['released_at'],'owned sample outside phase child lifetime')
                expected_pid=child['pid']
        need(x['owned_model_pid']==expected_pid,'wrong per-phase owned PID target')
        need(x['pressure_level']['returncode']==0 and x['pressure_level']['text']=='1'
            and type(x['pressure_level_value']) is int and x['pressure_level_value']==1,'raw pressure1 gate')
        free=re.search(r'free percentage:\s*(\d+)',x['memory_pressure']['text']);swap=re.search(r'used = ([0-9.]+)M',x['swap']['text'])
        need(free and swap and x['memory_pressure']['returncode']==x['swap']['returncode']==x['vm_stat']['returncode']==0,'memory probes')
        need(type(x['free_percent']) is int and int(free[1])==x['free_percent'] and 0<=x['free_percent']<=100
            and finite(x['swap_used_mb']) and x['swap_used_mb']>=0 and float(swap[1])==x['swap_used_mb'],'raw memory parse')
        need({'Swapouts','Pageouts'}<=set(x['vm_counters']),'paging counters missing')
        for key,value in x['vm_counters'].items():
            match=re.search(r'^'+re.escape(key)+r':\s*(\d+)',x['vm_stat']['text'],re.M)
            need(type(value) is int and value>=0 and match and int(match[1])==value,'raw paging counters')
        power,thermal=x['power_metadata'],x['processinfo_thermal_metadata']
        need(power['source']=='AC Power' and power['raw']['returncode']==0 and 'AC Power' in power['raw']['stdout']
            and type(thermal['code']) is int and thermal['code'] in (0,1) and thermal['raw']['returncode']==0
            and thermal['raw']['stdout']==str(thermal['code']),'raw AC/thermal gate')
        need(all(x[k] is True for k in ('process_inventory_ok','native_process_inventory_ok','no_active_foreign_native_processes','only_owned_known_inference_processes'))
            and x['authorized_suspended_pids']==[],'inventory/empty suspended guard')
        for key in ('native_inference_processes','known_inference_processes'):
            for entry in x[key]:
                need(expected_pid is not None and entry['pid']==expected_pid and entry['owned'] is True,'foreign observed inference process')
                if key=='native_inference_processes':need(entry['authorized_suspended'] is False,'unexpected service exemption')
        exact=x['exact_owned_pid_metadata']
        if expected_pid is not None:
            if exact['status']=='known':
                fields=exact['raw']['stdout'].split(None,3)
                need(exact['pid']==expected_pid and exact['ppid']==complete['owner_pid'] and exact['raw']['returncode']==0
                    and len(fields)==4 and [int(fields[0]),int(fields[1])]==[expected_pid,complete['owner_pid']],'raw direct child identity')
                known[stage]+=1
            else:need(False,'live formal server identity disappeared before reap')
        else:need(exact['status']=='unknown' and exact['pid'] is None,'no-child sample identity')
        if rows:
            previous=rows[-1];need(x['monotonic_s']>previous['monotonic_s'],'resource monotonic order')
            delta=dict(swap=x['swap_used_mb']-previous['swap_used_mb'],Swapouts=x['vm_counters']['Swapouts']-previous['vm_counters']['Swapouts'])
            deltas.append(delta);need(not(len(deltas)>=2 and (all(d['swap']>64 for d in deltas[-2:]) or all(d['Swapouts']>0 for d in deltas[-2:]))),'original sustained swap gate')
        rows.append(x)
    for stage,parts in labels.items():
        order=['pre-spawn','spawned','starting','ready','warm','after-warm','formal','after-formal','after-reap','released']
        mandatory=[p for p in order if p!='starting']
        need(all(parts.count(p)==1 for p in mandatory if p not in ('warm','formal'))
            and all(p in parts for p in mandatory) and all(p in order for p in parts)
            and [order.index(p) for p in parts]==sorted(order.index(p) for p in parts) and known[stage]>0,
            'complete ordered formal trial observations')
        absence(read(owner/'phases'/stage/'pre-spawn-absence.json'),original_pid)
    aggregate=read(owner/'resources.json');need(len(aggregate)==count,'aggregate count')
    for i,(row,total) in enumerate(zip(rows,aggregate)):
        same({k:v for k,v in total.items() if k not in ('swap_used_mb_delta','vm_counter_delta')},row,'aggregate/raw equality')
        if i:
            need(total['swap_used_mb_delta']==row['swap_used_mb']-rows[i-1]['swap_used_mb'],'aggregate swap delta')
            same(total['vm_counter_delta'],{k:v-rows[i-1]['vm_counters'][k] for k,v in row['vm_counters'].items()},'aggregate VM deltas')
    return dict(samples=count,known_owned_samples=dict(known),pressure=1,authorized_suspended_pids=[],
        swap_delta_mb=rows[-1]['swap_used_mb']-rows[0]['swap_used_mb'],
        vm_counter_delta={k:v-rows[0]['vm_counters'][k] for k,v in rows[-1]['vm_counters'].items()},
        scope='Original raw pressure/AC/thermal/sustained-swap gates; retained process inventory cannot exclude unknown runtimes or GUI GPU activity.')


def raw_metrics(group):
    """Independent metric calculation from validated wire terminals and receive clocks."""
    parser=load(HERE/'http_join.py',JOIN_SHA,'_formal_raw_metric_parser')
    starts=[];ends=[];latencies=[];ids=[];tokens=0;events=[]
    for row in group['requests']:
        rid,terminal,_=parser.raw_completion(row)
        start,end=row['start_perf_counter_s'],row['done_perf_counter_s']
        need(finite(start) and finite(end) and start<end,'strict positive complete-request latency')
        deadline=row['absolute_deadline_perf_counter_s']
        need(finite(deadline) and deadline==start+group['request_timeout_s'] and end<=deadline,
             'actual per-request deadline')
        committed=terminal['benchmark_committed_output_ids']
        need(parser.ids(committed),'actual committed ID array')
        starts.append(start);ends.append(end);latencies.append(end-start);ids.append(rid);tokens+=len(committed)
        events.extend(((start,1,rid),(end,-1,rid)))
    need(starts and len(set(ids))==len(ids),'nonempty unique response UUIDs')
    active=peak=0
    for _,delta,_ in sorted(events):
        active+=delta;need(active>=0,'completion precedes active request');peak=max(peak,active)
    need(active==0 and peak==group['concurrency'],'actual full configured HTTP concurrency required')
    barrier=group['barrier_release_perf_counter_s'];deadline=group['group_deadline_perf_counter_s']
    close=group['cleanup']
    need(finite(barrier) and finite(deadline) and barrier<=min(starts) and max(ends)<=deadline
         and deadline==barrier+group['group_timeout_s'],'group absolute clock bounds')
    need(finite(close['start_perf_counter_s']) and finite(close['end_perf_counter_s'])
         and max(ends)<=close['start_perf_counter_s']<=close['end_perf_counter_s'], 'client close clock after all DONE')
    window=max(ends)-min(starts);ordered=sorted(latencies);n=len(ordered)
    median=ordered[n//2] if n%2 else (ordered[n//2-1]+ordered[n//2])/2
    return dict(requests=n,committed_output_tokens=tokens,common_window_s=window,
        aggregate_output_tok_s=tokens/window,e2e_p50_s=median,e2e_p95_s=ordered[math.ceil(.95*n)-1],
        first_post_s=min(starts),last_done_s=max(ends),max_http_outstanding=peak,
        client_close_end_s=close['end_perf_counter_s'],response_uuids=ids)


def audit_trials(owner,wrapper,specs,complete,decode_text):
    formal=load(HERE/'native_formal.py',FORMAL_SHA,'_audit_formal_pure')
    cap=load(HERE/'http_capture_v2.py',CAPTURE_SHA,'_audit_formal_identity')
    support=legacy();names=wrapper['trials']
    index=read(owner/'children.json');need(index['schema']=='cold-service-owner-children-v1','child index schema')
    rows=index['children'];need([r['phase'] for r in rows]==names,'complete registered direct children')
    same(read(owner/'formal-specs.json'),specs,'actual prepared formal specs')
    need([r['phase'] for r in complete['phases']]==names,'all trial stages completed')
    saved=read(owner/'points.json')
    need(saved['schema']=='E023-native-formal-points-v1' and saved['performance_qualified'] is False,'original unapproved point collection')
    same(saved['trials'],complete['phases'],'incremental saved points differ from final stages')
    children=[];points=[];all_uuids=set()
    for i,(row,spec,item) in enumerate(zip(rows,specs,complete['phases'])):
        stage=names[i];folder=owner/'phases'/stage;trial=folder/'trial'
        child=read(folder/'child.json');release=read(folder/'release.json')
        need(child['phase']==release['phase']==stage and child['pid']==row['pid']==release['pid']
             and child['ownership']=='direct-Popen-returned' and child['owner_pid']==complete['owner_pid']
             and child['identity'] is not None and child['identity']['pid']==child['pid']
             and child['identity']['ppid']==complete['owner_pid'],'actual owned trial PID')
        need(finite(child['spawned_at']) and finite(release['time']) and child['spawned_at']<=release['time']
             and release['waited_reaped'] is True and release['strict_absence'] is True,'owned trial reap/time')
        need(not children or children[-1]['released_at']<=child['spawned_at'],'trial processes overlap')
        result=read(trial/'complete.json')
        need(result['schema']=='E023-native-formal-result-v1' and result['status']=='PASS' and result['errors']==[]
             and result['concurrency']==spec['concurrency'] and result['repeat']==spec['repeat']
             and result['performance_qualified'] is False and all(result[k] is True for k in
                ('count_eligible','cleanup_complete','server_reaped','cleanup_task_resolved','client_task_resolved','small_pins_rechecked')),
             'closed valid formal trial required')
        need(finite(result['started_at']) and finite(result['ended_at'])
             and result['started_at']<=child['spawned_at']<=result['ended_at']<=release['time'],'inner/outer lifecycle ordering')
        need(item['status']=='PASS','completed trial stage status')
        same(item['result'],dict(path=str((trial/'complete.json').relative_to(owner)),sha256=sha(trial/'complete.json')),'trial result receipt')
        same(item['numerical_audit'],spec['qualification']['audit'],'trial numerical binding')
        same(read(trial/'spec.json'),spec,'executed formal spec')
        same(read(trial/'launch.json'),dict(command=spec['command'],environment=wrapper['http']['environment']),'actual formal CLI/environment')
        need('--qualification-trace' not in spec['command'] and not (trial/'trace.jsonl').exists()
             and spec['identity']['step_observer'] is False and spec['identity']['qualification_trace_enabled'] is False
             and spec['identity']['per_token_observer'] is False,'formal observer-OFF route')
        support.capture_artifacts(trial,result['artifacts'])
        same(read(trial/'child.json')['pid'],child['pid'],'inner returned Popen PID')
        for name in ('ready.json','ready-after-warm.json','ready-after-formal.json'):
            need(cap.ready_identity(read(trial/name),spec['identity']),'actual observer-OFF model/config identity')
        need(cap.owned_listener(read(trial/'listener.json'),child['pid'],wrapper['http']['transport']['port']),'actual localhost PID/listener')
        inner_release=read(trial/'release.json')
        need(inner_release['pid']==child['pid'] and inner_release['returncode']==release['returncode']
             and inner_release['waited_reaped'] is True and inner_release['strict_absence'] is True
             and inner_release['port_absent'] is True and finite(inner_release['time'])
             and child['spawned_at']<=inner_release['time']<=result['ended_at'],'inner wait/reap proof')
        fact=dict(phase=stage,pid=child['pid'],spawned_at=child['spawned_at'],released_at=release['time'],
                  returncode=release['returncode'],inner_released_at=inner_release['time'])
        metrics={}
        for phase,count in (('warm',8),('formal',64)):
            group=read(trial/(phase+'-group.json'));same(group,read(trial/phase/'group.json'),'full raw group duplicate binding')
            need(group['base_url']==f"http://127.0.0.1:{spec['transport']['port']}"
                 and group['request_timeout_s']==spec['deadlines']['request']
                 and group['group_timeout_s']==spec['deadlines'][phase]
                 and group['cleanup']['budget_s']==spec['deadlines']['close']
                 and group['cleanup']['frozen_client_sha256']==CLIENT_SHA
                 and group['cleanup']['client_close_started'] is True
                 and group['qualification_errors']==[], 'fixed local endpoint/client deadlines and cleanup')
            same(group['connection_policy'],dict(max_connections=2*spec['concurrency'],max_keepalive_connections=0,
                 retries=0,follow_redirects=False),'fixed actual connection policy')
            checked=formal.check_group(group,spec,phase,decode_text)
            same(read(trial/(phase+'-raw-check.json')),checked,'runner raw-check record differs from reparse')
            actual=raw_metrics(group);need(actual['requests']==count,'fixed warm/formal N')
            point=dict(concurrency=spec['concurrency'],repeat=spec['repeat'],**{k:actual[k] for k in
                ('requests','committed_output_tokens','common_window_s','aggregate_output_tok_s','e2e_p50_s','e2e_p95_s')})
            same(point,checked['point'],'independent wire count/window/p50/p95 calculation')
            need(not(all_uuids & set(actual['response_uuids'])),'UUID reused across trial/group requests')
            all_uuids.update(actual['response_uuids']);metrics[phase]=actual
        need(metrics['warm']['client_close_end_s']<=metrics['formal']['first_post_s'],'formal starts before warm client is closed')
        point=dict(concurrency=spec['concurrency'],repeat=spec['repeat'],**{k:metrics['formal'][k] for k in
            ('requests','committed_output_tokens','common_window_s','aggregate_output_tok_s','e2e_p50_s','e2e_p95_s')})
        same(result['point'],point,'terminal point differs from independent wire metrics')
        same(read(trial/'point.json'),dict(**point,count_eligible=True,performance_qualified=False),'raw point must remain unapproved')
        same(item['point'],dict(path=str((trial/'point.json').relative_to(owner)),sha256=sha(trial/'point.json')),'raw point digest')
        points.append(dict(phase=stage,**point,n_fresh_processes=1,max_http_outstanding=metrics['formal']['max_http_outstanding'],
            warm_requests_checked=8,formal_raw_group=ref(trial/'formal-group.json'),point_receipt=ref(trial/'point.json'),
            numerical_audit=spec['qualification']['audit']))
        children.append(fact)
    return children,points


def audit(owner,journal,plan_sha256):
    owner,journal=Path(owner).resolve(),Path(journal).resolve()
    result=dict(schema='E023-native-formal-independent-audit-v1',status='INCOMPLETE',qualified=False,
        performance_qualified=False,ledger_opened=False,http_raw_opened=False,audit_source_sha256=sha(__file__))
    try:
        data=read(journal/'journal.json');complete=read(owner/'complete.json')
        if data.get('state')!='RESTORED' or complete.get('status')=='INCOMPLETE':raise Incomplete('owner/restoration not terminal')
        if complete.get('status')!='PASS':
            result.update(status='FAIL',producer_errors=complete.get('errors'));return result
        start_hashes={str(p):sha(p) for p in (journal/'journal.json',owner/'complete.json',owner/'owner-plan.json')}
        need(data['owner_directory']==str(owner) and data['owner_plan']['sha256']==plan_sha256==sha(owner/'owner-plan.json'),'actual owner plan binding')
        need(data['owner_source']==dict(path=str(HERE/'native_formal_owner.py'),sha256=OWNER_SHA),'exact formal owner')
        restore=load(HERE.parent/'omlx_cold_restart/audit_restore_v4.py',RESTORE_SHA,'_formal_restore_audit')
        restoration=restore.audit(journal);result['restoration']=restoration
        need(restoration['status']=='PASS' and restoration['qualified'] is True
             and restoration['performance_qualified'] is False,'independent restoration/reap failed')
        owned=load(HERE/'native_formal_owner.py',OWNER_SHA,'_formal_owner_recipe')
        _,wrapper,_,numeric,specs=owned.validate_plan(SimpleNamespace(plan=owner/'owner-plan.json',plan_sha256=plan_sha256))
        need(complete['schema']=='E023-native-formal-owner-result-v1' and complete['errors']==[]
             and complete['first_failure'] is None and complete['formal_trials_complete'] is True
             and complete['all_children_reaped'] is True and complete['performance_qualified'] is False
             and complete['planned_trials']==wrapper['trials'],'complete registered cycle required')
        result['pins']=preparation_pins(owner,wrapper,numeric)
        # Pin the CPU decoder actually loaded here, using the independently audited runtime map.
        token_pins=read(pinned(wrapper['qualifications']['1']['audit']))['tokenizer_runtime_pins']
        same(token_pins,read(pinned(wrapper['qualifications']['8']['audit']))['tokenizer_runtime_pins'],'C1/C8 decoder runtime differs')
        import tokenizers
        need(str(Path(tokenizers.__file__).resolve()) in token_pins,'loaded exact CPU tokenizer package')
        for path,digest in token_pins.items():need(sha(path)==digest,'CPU decoder runtime changed')
        tokenizer=tokenizers.Tokenizer.from_file(str(pinned(specs[0]['tokenizer'])))
        result['http_raw_opened']=True
        children,points=audit_trials(owner,wrapper,specs,complete,lambda ids:tokenizer.decode(ids,skip_special_tokens=False))
        result['resources']=resources(owner,complete,children,data['original']['pid'],read(owner/'attempt.json')['created_at'],wrapper['trials'])
        need(all(sha(p)==h for p,h in start_hashes.items()) and sha(journal/'journal.json')==restoration['journal_sha256'],
             'terminal binding changed during audit')
        result.update(status='PASS',qualified=True,performance_qualified=True,owner_directory=str(owner),
            owner_plan_sha256=plan_sha256,owner_complete_sha256=sha(owner/'complete.json'),journal_sha256=restoration['journal_sha256'],
            points=points,children=children,support_source_pins={str(HERE/'native_formal.py'):FORMAL_SHA,
                str(HERE/'native_formal_owner.py'):OWNER_SHA,str(HERE/'audit_http_owner_v2.py'):LEGACY_SHA,
                str(HERE.parent/'omlx_cold_restart/audit_restore_v4.py'):RESTORE_SHA},
            scope='Representative captured histories '+__doc__.split('Representative captured histories',1)[1].strip(),
            raw_point_files_modified=False,model_payload_independently_rehashed=False)
    except (Incomplete,FileNotFoundError) as error:result.update(status='INCOMPLETE',reason=str(error))
    except Exception as error:result.update(status='FAIL',first_failure=dict(type=type(error).__name__,message=str(error)))
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('owner','journal','plan-sha256','out'):p.add_argument('--'+name,required=True)
    args=p.parse_args(argv);out=Path(args.out);need(not out.exists(),'new output required')
    result=audit(args.owner,args.journal,args.plan_sha256)
    with out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    return {'PASS':0,'FAIL':1,'INCOMPLETE':2}[result['status']]


if __name__=='__main__':raise SystemExit(main())
