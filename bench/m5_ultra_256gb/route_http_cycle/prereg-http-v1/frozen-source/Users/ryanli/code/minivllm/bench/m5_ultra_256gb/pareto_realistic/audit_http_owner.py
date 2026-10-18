"""Offline four-phase HTTP owner provenance/resource/restoration audit.

Only terminal RESTORED/owner PASS runs are eligible. Consumes two separately
produced numerical audit files with explicit SHA pins; never opens their steps
ledgers. Rehashes runtime/source files, but not model shards. No service, HTTP,
process probe, GPU work, or fabricated legacy single-child evidence is used.
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

HERE=Path(__file__).resolve().parent
PHASES=('c1-capture','c1-replay','c8-capture','c8-replay')
RESTORE_SHA='5878d8a8bc19b38c4e8b2a255fc1b575536894a1430e8e98a2fb42a9e3836a16'
NUMERICAL_AUDIT_SHA='7cdcf5a2f02768ef75686b0ff0b0da9d21f5f9c82ee3866e24c1f869c709f031'
CAPTURE_SHA='536fe41d6c8e2bdffe83ba6f43fba87dd6265deb4c84f6204bc9c7220f8280e3'
OWNER_SHA='f645bc1a32bee8fdaf57d1626a84ebce1ad6d1c775ecdc9b6de3ebbc12105bf1'
BASE_PLAN_SHA='21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb'
JOB='gui/501/local.omlx.server'


class Incomplete(Exception):pass
def need(ok,message):
    if not ok:raise ValueError(message)
def same(a,b,message):
    need(json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False),message)
def finite(x):return type(x) in (int,float) and math.isfinite(x)
def read(path):
    need(Path(path).suffix!='.jsonl','no numerical/trace ledger is opened by owner audit')
    with Path(path).open() as f:return json.load(f)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while block:=f.read(1<<20):h.update(block)
    return h.hexdigest()
def load(path,digest,name):
    need(sha(path)==digest,'support source changed: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def pinned(item):
    need(set(item)=={'path','sha256'} and Path(item['path']).is_absolute(),'absolute pinned reference')
    need(sha(item['path'])==item['sha256'],'pinned receipt changed');return Path(item['path'])
def ref(path):return dict(path=str(path),sha256=sha(path))


def capture_artifacts(folder,artifacts):
    need(set(artifacts)=={p.name for p in folder.iterdir() if p.is_file() and p.name!='complete.json'},'complete capture artifact set')
    for name,item in artifacts.items():
        need(Path(name).name==name and set(item)=={'sha256','bytes'} and type(item['bytes']) is int and item['bytes']>=0,'capture artifact metadata')
        path=folder/name;need(sha(path)==item['sha256'] and path.stat().st_size==item['bytes'],'capture artifact SHA/size')


def preparation_pins(owner,wrapper,numeric):
    """Actual two preparation directories; no invented original child command."""
    expected={**numeric['execution']['pins'],wrapper['qualification']['path']:BASE_PLAN_SHA}
    need(len(numeric['execution']['pins'])==122,'original122 runtime/source pins')
    manifests={}
    for name in ('fixtures','source_manifest','model_manifest'):
        item=numeric[name];expected[item['path']]=item['sha256']
        path=owner/'preflight/frozen-source'/item['path'].lstrip('/')
        need(sha(path)==item['sha256'],'frozen preparation manifest');manifests[name]=read(path)
    core=Path(numeric['package_root'])/'tutorial_code/miniengine'
    expected.update({str(core/p):h for p,h in manifests['source_manifest']['files'].items()})
    need(len(expected)==157,'original exact157 provenance set')
    hashed={};model_sets=[]
    for phase in ('preflight','postflight'):
        folder=owner/phase;need(sha(folder/'plan.json')==BASE_PLAN_SHA,'original preparation plan SHA')
        verified=read(folder/'verified-pins.json');need(set(verified)==set(expected),'exact157 recorded entries')
        for source,h in expected.items():
            copied=Path(source).suffix in ('.py','.json','.md');row=verified[source]
            need(row['sha256']==h and row['source_copied'] is copied and type(row['bytes']) is int and row['bytes']>0,'recorded source pin')
            path=folder/'frozen-source'/source.lstrip('/') if copied else Path(source)
            need(path.stat().st_size==row['bytes'],'pinned file size')
            if path not in hashed:hashed[path]=sha(path)
            need(hashed[path]==h,'independent runtime/source hash')
        for source,h in wrapper['execution']['pins'].items():
            need(sha(folder/'frozen-adapters'/Path(source).name)==h,'frozen HTTP adapter SHA')
        model=manifests['model_manifest'];pins=read(folder/'verified-model-pins.json')
        wanted={str(Path(numeric['model'])/n):h for n,h in model['files'].items()}
        need(set(pins)==set(wanted) and model['schema']=='E023-model-v1' and model['model']==numeric['model'],'complete model manifest proof')
        for path,h in wanted.items():
            row=pins[path];need(set(row)=={'sha256','bytes','inode','mtime_ns'} and row['sha256']==h
                and all(type(row[k]) is int and row[k]>0 for k in ('bytes','inode','mtime_ns')),'model hash/stat receipt')
        model_sets.append(pins)
    same(model_sets[0],model_sets[1],'model hash/stat changed across experiment')
    # Stat is metadata-only. The actual source-pinned preparer performed full
    # model hashing before/after; this independent audit does not repeat it.
    for path,row in model_sets[-1].items():
        st=Path(path).stat();need((st.st_size,st.st_ino,st.st_mtime_ns)==(row['bytes'],row['inode'],row['mtime_ns']),'current model metadata changed')
    return dict(preparation_directories=2,verified_entries_each=157,execution_pins=122,
        independent_files_hashed=len(hashed),model_files=len(model_sets[0]),model_shards_rehashed=False,
        model_scope='Manifest plus source-pinned full-hash receipts before/after and unchanged current stat; not an independent payload rehash.')


def absence(row,pid):
    commands=[(['/bin/ps','-p',str(pid),'-o','pid='],1),(['/bin/launchctl','print',JOB],113),
        (['/usr/sbin/lsof','-nP','-a','-iTCP:8001','-sTCP:LISTEN','-Fpn'],1)]
    need(row['status']=='PASS' and len(row['checks'])==3,'original service absence coverage')
    for actual,(command,code) in zip(row['checks'],commands):
        same(actual,dict(command=command,returncode=code,stdout_sha256=hashlib.sha256(b'').hexdigest(),
            stdout_bytes=0,absent=True),'exact current-original absence code/output hash')


def resources(owner,complete,children,original_pid,created):
    count=complete['samples'];need(type(count) is int and count>=4 and complete['absence_probes']==count,'resource sample count')
    for prefix in ('resource','absence'):
        need(sorted(p.name for p in owner.glob(prefix+'-*.json'))==[f'{prefix}-{i:03d}.json' for i in range(count)],'resource/absence file sequence')
    rows=[];deltas=[];known=Counter();labels={p:[] for p in PHASES};last_stage=-1
    for i in range(count):
        x=read(owner/f'resource-{i:03d}.json');absent=read(owner/f'absence-{i:03d}.json');label=x['label']
        need(absent['label']==label and absent['resource_index_if_sampled']==i,'absence/resource pairing');absence(absent,original_pid)
        wall=datetime.fromisoformat(x['recorded_at']).timestamp()
        need(finite(absent['time']) and created<=absent['time']<=wall and finite(x['monotonic_s']),'resource timestamp')
        expected_pid=None
        if i<2:need(label==('before-prepare','after-prepare')[i] and wall<=children[0]['spawned_at'],'initial preparation boundary')
        elif i==count-1:need(label=='postflight' and wall>=children[-1]['released_at'],'postflight after all children')
        else:
            stage,sub=label.split(':',1);need(stage in PHASES,'unknown resource phase')
            n=PHASES.index(stage);need(last_stage<=n<=last_stage+1,'phase sample order');last_stage=n
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
            else:
                # Only the final running probe may observe a normally exited
                # replay before the owner has performed its immediate wait/reap.
                need(stage.endswith('replay') and sub=='running' and exact['pid']==expected_pid
                    and exact['raw']['returncode']==1 and not exact['raw']['stdout'].strip()
                    and read(owner/f'resource-{i+1:03d}.json')['label']==stage+':released','unexplained child disappearance')
        else:need(exact['status']=='unknown' and exact['pid'] is None,'no-child sample identity')
        if rows:
            previous=rows[-1];need(x['monotonic_s']>previous['monotonic_s'],'resource monotonic order')
            delta=dict(swap=x['swap_used_mb']-previous['swap_used_mb'],Swapouts=x['vm_counters']['Swapouts']-previous['vm_counters']['Swapouts'])
            deltas.append(delta);need(not(len(deltas)>=2 and (all(d['swap']>64 for d in deltas[-2:]) or all(d['Swapouts']>0 for d in deltas[-2:]))),'original sustained swap gate')
        rows.append(x)
    for stage,parts in labels.items():
        order=['pre-spawn','spawned','starting','ready','capturing','after-client','after-reap','released'] if stage.endswith('capture') else ['pre-spawn','running','released']
        mandatory=[p for p in order if p!='starting'];need(all(parts.count(p)==1 for p in mandatory if p not in ('capturing','running'))
            and all(p in parts for p in mandatory) and all(p in order for p in parts)
            and [order.index(p) for p in parts]==sorted(order.index(p) for p in parts) and known[stage]>0,'complete ordered phase observations')
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


def phases(owner,wrapper,numeric,complete,numerical):
    capture=load(HERE/'http_capture_v2.py',CAPTURE_SHA,'_audit_capture_pure')
    need(sha(HERE/'audit_http_replay_v2.py')==NUMERICAL_AUDIT_SHA,'numerical auditor source pin')
    index=read(owner/'children.json')['children'];need([r['phase'] for r in index]==list(PHASES),'all four observed phases')
    need([r['phase'] for r in complete['phases']]==list(PHASES),'four completed stages')
    specs=read(owner/'capture-specs.json');need(len(specs)==2,'C1/C8 capture specs')
    children=[];audits={}
    for i,row in enumerate(index):
        stage=row['phase'];folder=owner/'phases'/stage;C=1 if i<2 else 8
        child,release=read(folder/'child.json'),read(folder/'release.json')
        need(child['identity'] is not None and child['identity']['pid']==child['pid'] and child['identity']['ppid']==complete['owner_pid'],'actual direct child identity required')
        need(finite(child['spawned_at']) and finite(release['time']) and child['spawned_at']<=release['time'],'phase lifecycle times')
        need(not children or children[-1]['released_at']<=child['spawned_at'],'overlapping owned phases')
        fact=dict(phase=stage,pid=child['pid'],spawned_at=child['spawned_at'],released_at=release['time'],returncode=release['returncode'])
        item=complete['phases'][i];need(item['status']=='PASS','failed completed stage')
        result_path=folder/('capture/complete.json' if stage.endswith('capture') else 'qualification/complete.json')
        same(item['result'],dict(path=str(result_path.relative_to(owner)),sha256=sha(result_path)),'stage complete binding')
        result=read(result_path);need(result['status']=='PASS','phase result not PASS')
        if stage.endswith('capture'):
            spec=capture.prepare(wrapper['qualification']['path'],BASE_PLAN_SHA,C,python=numeric['execution']['python'],**wrapper['http'])
            same(specs[i//2],spec,'actual canonical capture spec');same(read(folder/'capture/spec.json'),spec,'saved capture spec')
            command=spec['command']+['--qualification-trace',str(folder/'capture/trace.jsonl')]
            same(read(folder/'capture/launch.json'),dict(command=command,environment=wrapper['http']['environment']),'capture actual CLI/environment')
            need(result['schema']=='E023-http-capture-result-v2' and result['errors']==[] and all(result[k] is True for k in
                ('capture_complete','cleanup_complete','server_reaped','cleanup_task_resolved','client_task_resolved','small_pins_rechecked')),'capture terminal contract')
            capture_artifacts(folder/'capture',result['artifacts'])
            for name in ('ready.json','ready-after-client.json'):
                need(capture.ready_identity(read(folder/'capture'/name),spec['identity']),'saved actual server identity')
            need(capture.owned_listener(read(folder/'capture/listener.json'),child['pid'],wrapper['http']['transport']['port']),'owned listener raw PID/fd route')
            fact['inner_released_at']=read(folder/'capture/release.json')['time']
        else:
            plan=read(folder/'plan.json');plan_sha=sha(folder/'plan.json');target=folder/'qualification'
            same(plan['inputs'],{key:ref(owner/'phases'/f'c{C}-capture/capture'/name) for key,name in (('group','group.json'),('trace','trace.jsonl'))},'actual captured raw input binding')
            command=[numeric['execution']['python'],'-I','-B',str(HERE/'http_qualification.py'),'--plan',str(folder/'plan.json'),
                '--plan-sha256',plan_sha,'--out',str(target)]
            same(read(folder/'launch.json'),dict(command=command,environment=wrapper['http']['environment']),'replay actual CLI/environment')
            need(release['returncode']==0,'replay exit0')
            reference=numerical[str(C)];audit=read(pinned(reference))
            need(audit['schema']=='E023-recorded-http-independent-audit-v2' and audit['status']=='PASS' and audit['qualified'] is True
                and audit['audit_source_sha256']==NUMERICAL_AUDIT_SHA and audit['concurrency']==C and audit['plan_sha256']==plan_sha
                and audit['base_plan_sha256']==BASE_PLAN_SHA and audit['child_directory']==str(target),'independent numerical audit binding')
            for field,name in [('child_complete_sha256','complete.json'),('base_complete_sha256','base/complete.json'),('recording_sha256','recording.json')]:
                need(audit[field]==sha(target/name),'numerical terminal digest')
            same(audit['sources'],plan['sources'],'numerical source binding');same(audit['raw_inputs'],plan['inputs'],'numerical raw inputs')
            need(audit['ledger']['requests']==(8 if C==1 else 16) and audit['performance_qualified'] is False
                and audit['owner_resources_restore_qualified'] is False and audit['original_37_request_gate_replaced'] is False,'numerical scope and count')
            audits[str(C)]={**reference,'requests':audit['ledger']['requests'],'ledger_sha256':audit['ledger']['sha256'],
                'ledger_binding_scope':'Streaming byte hash retained in independently pinned numerical audit; this wrapper does not reread the ledger.'}
        children.append(fact)
    return children,audits


def audit(owner,journal,plan_sha256,numerical):
    owner,journal=Path(owner).resolve(),Path(journal).resolve()
    result=dict(schema='E023-http-owner-independent-audit-v1',status='INCOMPLETE',qualified=False,ledger_opened=False,
                performance_qualified=False,audit_source_sha256=sha(__file__))
    try:
        data=read(journal/'journal.json');complete=read(owner/'complete.json')
        if data.get('state')!='RESTORED' or complete.get('status')=='INCOMPLETE':raise Incomplete('owner/restoration not terminal')
        if complete.get('status')!='PASS':
            result.update(status='FAIL',producer_errors=complete.get('errors'));return result
        need(data['owner_directory']==str(owner) and data['owner_plan']['sha256']==plan_sha256==sha(owner/'owner-plan.json'),'owner plan/runtime directory binding')
        need(data['owner_source']==dict(path=str(HERE/'http_owner_v2.py'),sha256=OWNER_SHA),'exact v2 owner only')
        restore=load(HERE.parent/'omlx_cold_restart/audit_restore_v3.py',RESTORE_SHA,'_http_restore_audit')
        restoration=restore.audit(journal);result['restoration']=restoration
        need(restoration['status']=='PASS' and restoration['qualified'] is True,'independent restoration/reap failed')
        wrapper=read(owner/'owner-plan.json');numeric=read(pinned(wrapper['qualification']))
        need(wrapper['qualification']['sha256']==BASE_PLAN_SHA,'fixed numerical source plan')
        need(complete['errors']==[] and complete['first_failure'] is None and complete['numerical_replay_complete'] is True
            and complete['all_children_reaped'] is True and complete['performance_qualified'] is False,'successful terminal owner contract')
        result['pins']=preparation_pins(owner,wrapper,numeric)
        children,audits=phases(owner,wrapper,numeric,complete,numerical)
        result['resources']=resources(owner,complete,children,data['original']['pid'],read(owner/'attempt.json')['created_at'])
        need(sha(journal/'journal.json')==restoration['journal_sha256'] and sha(owner/'owner-plan.json')==plan_sha256,'terminal binding changed')
        result.update(status='PASS',qualified=True,owner_directory=str(owner),owner_plan_sha256=plan_sha256,
            owner_complete_sha256=sha(owner/'complete.json'),journal_sha256=restoration['journal_sha256'],
            numerical_audits=audits,children=children,scope='This fixed C1/C8 captured-schedule numerical experiment, sources, observed resources and restoration only; no future formal timing or Pareto point.')
    except (Incomplete,FileNotFoundError) as error:result.update(status='INCOMPLETE',reason=str(error))
    except Exception as error:result.update(status='FAIL',first_failure=dict(type=type(error).__name__,message=str(error)))
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('owner','journal','plan-sha256','c1-audit','c1-sha256','c8-audit','c8-sha256','out'):p.add_argument('--'+name,required=True)
    args=p.parse_args(argv);out=Path(args.out);need(not out.exists(),'new output file required')
    numerical={str(C):dict(path=str(Path(getattr(args,f'c{C}_audit')).resolve()),sha256=getattr(args,f'c{C}_sha256')) for C in (1,8)}
    result=audit(args.owner,args.journal,args.plan_sha256,numerical)
    with out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    return {'PASS':0,'FAIL':1,'INCOMPLETE':2}[result['status']]


if __name__=='__main__':raise SystemExit(main())
