"""Closed Lithos owner resource audit; original gates, qualification phase labels.

The raw resource parser below is copied from frozen audit_native_formal.resources;
only warm/formal label vocabulary becomes qualification. No service/model/GPU
operation. Process inventory still cannot exclude unrecognized runtimes/GUI GPU use.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
BASE_PATH=HERE.parent/'pareto_realistic/audit_native_formal.py'
BASE_SHA='b57e4e9657f3dddb04069290bdfbab4c99b0d6cbfb1427e454d0217d5463c870'
if hashlib.sha256(BASE_PATH.read_bytes()).hexdigest()!=BASE_SHA: raise ValueError('frozen resource support changed')
spec=importlib.util.spec_from_file_location('lithos_resource_helpers',BASE_PATH)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
need,same,finite,read,sha,absence=base.need,base.same,base.finite,base.read,base.sha,base.absence
JOB=base.JOB
PHASES=['lithos-c1','lithos-c8']


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
        order=['pre-spawn','spawned','starting','ready','qualification','after-qualification','after-reap','released']
        mandatory=[p for p in order if p!='starting']
        need(all(parts.count(p)==1 for p in mandatory if p not in ('qualification',))
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


def audit(owner):
    owner=Path(owner).resolve();complete=read(owner/'complete.json');attempt=read(owner/'attempt.json')
    need(complete['schema']=='E023-lithos-D512-owner-result-v1' and complete['status']=='PASS'
         and complete['planned_trials']==PHASES and complete['all_children_reaped'] is True,'closed successful two-phase owner required')
    plan=read(owner/'owner-plan.json');need(sha(owner/'owner-plan.json')==complete['owner_plan_sha256']==attempt['owner_plan_sha256'],'plan receipt')
    stopped=read(owner/'stopped-journal.json');need(sha(owner/'stopped-journal.json')==attempt['stopped_journal_sha256'],'stopped snapshot pin')
    index=read(owner/'children.json');release=read(owner/'release.json')
    need(index['schema']=='cold-service-owner-children-v1' and release['children_sha256']==sha(owner/'children.json')
         and index['children']==release['children'] and release['all_children_reaped'] is True,'closed children index')
    need([row['phase'] for row in index['children']]==PHASES,'exact child phase order')
    children=[]
    for row in index['children']:
        phase=row['phase'];folder=owner/'phases'/phase
        for name in ('intent','child','release'):
            need(row[name]['path']==f'phases/{phase}/{name}.json' and row[name]['sha256']==sha(folder/(name+'.json')),'phase receipt pin')
        child=read(folder/'child.json');outer=read(folder/'release.json');inner=read(folder/'trial/release.json')
        need(child['pid']==row['pid']==outer['pid']==inner['pid'] and child['owner_pid']==complete['owner_pid']
             and outer['waited_reaped'] is inner['waited_reaped'] is True
             and outer['strict_absence'] is inner['strict_absence'] is True and inner['port_absent'] is True,'actual child/reap binding')
        children.append(dict(phase=phase,pid=row['pid'],spawned_at=child['spawned_at'],released_at=outer['time'],inner_released_at=inner['time']))
    result=resources(owner,complete,children,stopped['original']['pid'],attempt['created_at'],PHASES)
    return dict(schema='Lithos-D512-raw-resource-audit-v1',status='PASS',independent_resource_verified=True,
        audit_source_sha256=sha(__file__),base_source_sha256=BASE_SHA,owner_directory=str(owner),
        owner_complete_sha256=sha(owner/'complete.json'),owner_plan_sha256=complete['owner_plan_sha256'],
        resource_summary=result,performance_qualified=False,restoration_verified=False,
        scope='Closed qualification-cycle raw resource and child-lifetime checks only; no timing/quality/restoration qualification.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--owner',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();need(not args.out.exists(),'fresh audit output')
    try: result=audit(args.owner)
    except Exception as exc: result=dict(schema='Lithos-D512-raw-resource-audit-v1',status='FAIL',error_type=type(exc).__name__,error=str(exc),independent_resource_verified=False,performance_qualified=False)
    with args.out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    raise SystemExit(0 if result['status']=='PASS' else 1)
