"""Closed GLM load diagnostic: retained raw physical samples + restoration.

Reparse actual Apple bytes and command outputs with the frozen pure physical
algorithms; independently bind actual direct-child time/PID, admission, source
inventory, observed token/count receipts and full cleanup. No backend imports.
No full tensor/logit reference, text retokenization, HTTP, quality, performance
or hard memory-safety qualification. Legacy argv inventory is intentionally
redacted by the owner and cannot be independently reparsed here.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
OWNER_SHA='37f47300c14d8da721d6c30fcc71f52325132bad3dc3ba902c1cc08af4326b40'
GUARD_SHA='908e0dc74812fd82b65c63a22e03ed8c5692e812bdbf3d6957e219aa52b14166'
RESTORE_SHA='732c60870b9400d1d7b6bfb9c303bce19b9c92032bc06bc9db193b66ace3bf6d'


def need(ok,message):
    if not ok:raise ValueError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while block:=f.read(1<<20):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def module(path,digest,name):
    need(sha(path)==digest,'fixed audit source changed: '+str(path))
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m

def raw_sample(path,pid,owner_pid,guard,limits):
    path=Path(path);row=read(path);raw_path=path.parent/'footprint.raw.json';ref=row['raw']
    need(raw_path.is_file() and not raw_path.is_symlink() and raw_path.stat().st_size<=(8<<20),'bounded original physical JSON')
    need(row['schema']=='GLM-physical-sample-v1' and row['pid']==pid and row['owner_pid']==owner_pid,
         'physical sample schema/exact process')
    need(ref==dict(path=str(raw_path),bytes=raw_path.stat().st_size,sha256=sha(raw_path)), 'original footprint byte binding')
    commands=[['/usr/bin/footprint','-p',str(pid),'--noCategories','--sysFootprint','-f','bytes','-j',str(raw_path)],
        ['/usr/sbin/sysctl','-n','kern.memorystatus_vm_pressure_level'],['/usr/sbin/sysctl','vm.swapusage'],
        ['/usr/sbin/sysctl','-n','hw.memsize'],['/usr/bin/vm_stat']]
    need(len(row['commands'])==5,'complete raw physical commands')
    for actual,argv in zip(row['commands'],commands):
        need(set(actual)=={'command','returncode','stdout','stderr'} and actual['command']==argv
             and type(actual['returncode']) is int and actual['returncode']==0
             and type(actual['stdout']) is type(actual['stderr']) is str and not actual['stderr'].strip(),
             'original successful physical command')
    parsed=guard.parse_raw(read(raw_path),pid,*(r['stdout'] for r in row['commands'][1:]))
    need(parsed==row['parsed'],'recorded physical parse differs from original bytes')
    guard.checked(row,limits)
    need(guard.number(row['started_at']) and guard.number(row['ended_at']) and row['ended_at']>=row['started_at'],
         'physical wall-clock ordering')
    return row


def checkpoints(owner,plan,attempt,child,release,guard):
    paths=sorted(owner.glob('checkpoint-*.json'),key=lambda p:int(p.stem.split('-')[-1]))
    count=read(owner/'complete.json')['checkpoints'];need(len(paths)==count and count>=6,'complete checkpoint inventory')
    expected_labels=['before-prepare','after-prepare','cold-0','cold-1','before-spawn']
    expected_labels+=['running']*(count-6)+['postflight'];previous=0
    old=read(owner/'stopped-journal.json')['original']['pid']
    native={'omlx-server','omlx','lithos-metal','llama-server','mlx_lm.server'}
    for i,(path,label) in enumerate(zip(paths,expected_labels)):
        need(path.name==f'checkpoint-{i:04d}.json','contiguous checkpoint indices')
        row=read(path);cpid=child['pid'] if label=='running' else None;cmd=row['commands']
        need(row['schema']=='GLM-owner-checkpoint-v1' and row['status']=='PASS' and row.get('error') is None
             and row['label']==label and row['old_pid']==old and row['child_pid']==cpid and row['owner_pid']==attempt['owner_pid']
             and row['authorized_suspended_pids']==[] and len(cmd)==7,'actual checkpoint contract')
        need(guard.number(row['time']) and attempt['created_at']<=row['time']<=release['time']
             and guard.number(row['monotonic']) and guard.number(row['ended_monotonic'])
             and previous<=row['monotonic']<=row['ended_monotonic']
             and row['ended_monotonic']-row['monotonic']<=plan['budgets']['checkpoint_seconds'],'checkpoint lifetime/deadline')
        previous=row['ended_monotonic']
        argv=[['/bin/ps','-p',str(old),'-o','pid='],['/bin/launchctl','print','gui/501/local.omlx.server'],
              ['/usr/sbin/lsof','-nP','-a','-iTCP:8001','-sTCP:LISTEN','-Fpn'],['/usr/bin/pmset','-g','batt'],
              ['/usr/bin/osascript','-l','JavaScript','-e','ObjC.import("Foundation"); String($.NSProcessInfo.processInfo.thermalState)'],
              ['/bin/ps','-axo','pid=,ppid=,comm='],['/bin/ps','-axo','pid=,ppid=,args=']]
        for n,(r,want) in enumerate(zip(cmd,argv)):
            need(r['command']==want and type(r['returncode']) is int and r['returncode']==(1 if n in (0,2) else 113 if n==1 else 0)
                 and type(r['stderr']) is str and (n==1 or not r['stderr'].strip()),'raw checkpoint command/code')
            if n<6:need(type(r['stdout']) is str,'raw checkpoint stdout')
            if n<3:need(not r['stdout'].strip(),'removed service returned')
        need(re.findall(r"Now drawing from '([^']+)'",cmd[3]['stdout'])==['AC Power'] and cmd[4]['stdout'].strip() in ('0','1'),
             'raw AC/thermal gate')
        seen=set();recognized=[]
        for line in cmd[5]['stdout'].splitlines():
            fields=line.split(None,2);need(len(fields)==3 and fields[0].isdigit() and fields[1].isdigit(),'native process row')
            pid,ppid=int(fields[0]),int(fields[1]);need(pid not in seen,'duplicate native PID');seen.add(pid)
            if Path(fields[2]).name in native or pid==cpid:
                need(pid==cpid and ppid==attempt['owner_pid'],'foreign native inference process')
                recognized.append(dict(kind='native',pid=pid,ppid=ppid))
        need(cpid is None or cpid in seen,'observed child missing')
        legacy=cmd[6];need('stdout' not in legacy and type(legacy['stdout_bytes']) is int and legacy['stdout_bytes']>=0
            and re.fullmatch('[0-9a-f]{64}',legacy['stdout_sha256']),'retained redacted legacy receipt')
        expected=recognized+([] if cpid is None else [dict(kind='legacy',pid=cpid,ppid=attempt['owner_pid'])])
        need(row['recognized']==expected,'native reconstruction/legacy producer receipt mismatch')
    return dict(count=count,native_inventory_reparsed=True,legacy_argv_reparsed=False,
                scope='Legacy argv bytes intentionally absent; only pinned owner matcher receipt/hash/size checked.')


STATE_KEYS={'running','waiting','prefilling','uid_to_request_id','request_id_to_uid',
            '_pending_async_removes','_inflight_store_futures'}


def empty_state(value):
    return type(value) is dict and set(value)==STATE_KEYS and all(type(x) is int and x==0 for x in value.values())


def request_counts(folder,complete,probe):
    rows=[read(folder/f'request-{i}.json') for i in range(4)]
    need(complete['completed_requests']==rows,'complete/request file equality')
    source=read(folder/'input.json');prompt=source['prompt_ids'];need(source['normal_eos']==probe.EOS
        and type(prompt) is list and 0<len(prompt)<=128 and all(type(t) is int and t>=0 for t in prompt),'actual prompt token receipt')
    before={i:[] for i in range(4)};final={i:[] for i in range(4)};reset=[]
    active=-1;pending=None;terminal=False;closed=True
    with (folder/'events.jsonl').open() as f:
        for line in f:
            need(line.endswith('\n'),'truncated probe event');event=json.loads(line)
            need(event['pid']==complete['pid'],'probe event PID')
            kind=event['kind']
            if kind=='reset_before':
                need(closed and pending is None and type(event['case']) is int and event['case']==active+1
                     and event['case']<4 and empty_state(event['counts']),'ordered exact drained reset')
                active=event['case'];reset.append(active);terminal=False;closed=False
            elif kind=='batch_before':
                need(not closed and pending is None and type(event['case']) is int and event['case']==active,
                     'batch before outside current request or unpaired before')
                probe.require_target_route(event['route']);pending=event['responses']
                need(type(pending) is list and (not terminal or not pending),'tokens after terminal')
                before[active].extend(pending)
            elif kind=='batch_after':
                need(not closed and pending is not None and type(event['case']) is int and event['case']==active,
                     'batch after outside matching before/current request')
                finals=event['finals'];need(type(finals) is list and len(finals)<=1,'one terminal per batch')
                endings=[r for r in pending if r['finish_reason'] is not None]
                need(len(endings)==len(finals) and len(endings)<=1 and (not finals or not terminal),
                     'terminal must pair actual finishing response once')
                if finals:
                    need(endings[0]['request_id']==finals[0]['request_id']
                         and endings[0]['finish_reason']==finals[0]['finish_reason'],'paired terminal identity')
                    terminal=True
                final[active].extend(finals);pending=None
            elif kind=='memory' and event.get('phase')=='request_after':
                need(not closed and pending is None and terminal and type(event.get('case')) is int
                     and event['case']==active,'request closes only after terminal pair')
                closed=True
    need(closed and terminal and pending is None,'complete final request event lifecycle')
    need(reset==list(range(4)),'four ordered drained requests');ids=set()
    for i,(row,cap) in enumerate(zip(rows,probe.CAPS)):
        feed=before[i];need(len(final[i])==1 and bool(feed),'one actual terminal per request');out=final[i][0]
        need(row['index']==i and row['cap']==cap and row['prompt_ids']==prompt and row['request_id']==out['request_id']
            and row['request_id'] not in ids and row['scheduler_output']==out,'unique complete request binding');ids.add(row['request_id'])
        need(all(r['request_id']==row['request_id'] and r['prompt_ids']==prompt and r['sampling']==dict(probe.SAMPLING,max_tokens=cap)
            and type(r['token']) is int and r['token']>=0 for r in feed),'actual target outputs/prompt/sampling')
        raw=[r['token'] for r in feed];committed=out['committed_ids'];finish=out['finish_reason']
        need(all(r['finish_reason'] is None for r in feed[:-1]) and feed[-1]['finish_reason']==finish,'terminal ordering')
        need(raw==row['raw_ids'] and committed==row['committed_ids'],'observed actual token stream binding')
        if finish=='stop':need(raw[-1] in probe.EOS and not any(t in probe.EOS for t in raw[:-1]) and committed==raw[:-1] and len(raw)<=cap,'EOS once exclusion')
        else:need(finish=='length' and len(raw)==cap and committed==raw and not any(t in probe.EOS for t in raw),'length token contract')
        need(out['error'] is None and type(out['cached_tokens']) is int and out['cached_tokens']==0
            and type(out['completion_tokens']) is int and out['completion_tokens']==len(committed)
            and type(out['prompt_tokens']) is int and out['prompt_tokens']==len(prompt)
            and empty_state(row['reset_after']) and row['performance_qualified'] is False,'usage/cache/final drain')
    need(rows[1]['raw_ids']==rows[2]['raw_ids']==rows[3]['raw_ids'],'three repeated greedy requests')
    return dict(requests=4,caps=probe.CAPS,raw_tokens=sum(len(r['raw_ids']) for r in rows),
        committed_tokens=sum(len(r['committed_ids']) for r in rows),raw_observed_ids_reparsed=True,
        independent_text_decode=False,full_state_reference=False)


def audit(journal,plan_sha256):
    result=dict(schema='GLM-closed-load-audit-v1',status='INCOMPLETE',observed_diagnostic_complete=False,
        full_model_correctness_qualified=False,chat_quality_qualified=False,performance_qualified=False,
        physical_resource_safety_qualified=False,audit_source_sha256=sha(__file__))
    try:
        restore=module(HERE/'audit_restore.py',RESTORE_SHA,'glm_closed_restore')
        restoration=restore.audit(journal);result['restoration']=restoration
        need(restoration['status']=='PASS' and restoration['qualified'] is True,'independent restoration required')
        data=read(Path(journal)/'journal.json');need(data['owner_plan']['sha256']==plan_sha256,'explicit owner plan binding')
        c,_=restore.sources();plan=c.owner_plan(data['owner_plan']['path'],plan_sha256,journal)
        owner=Path(data['owner_directory']);producer=module(HERE/'owner.py',OWNER_SHA,'glm_closed_owner_contract')
        probe=producer.helper('../glm_flash/load_probe.py');guard=module(HERE.parent/'glm_flash/physical_guard.py',GUARD_SHA,'glm_closed_raw_guard')
        attempt=read(owner/'attempt.json');complete=read(owner/'complete.json');phase=owner/'phases'/c.PHASE
        need(complete['status']=='PASS' and complete['errors']==[],'load diagnostic did not complete')
        child=read(phase/'child.json');release=read(owner/'release.json');limits=plan['physical_policy'];pid=child['pid']
        cold=[raw_sample(owner/f'cold-{i}/sample.json',attempt['owner_pid'],attempt['owner_pid'],guard,limits) for i in range(2)]
        admission=guard.cold_admission(cold,limits,owner_pid=attempt['owner_pid'])
        need(read(owner/'cold-physical.json')==dict(samples=[dict(path=str(owner/f'cold-{i}/sample.json'),sha256=sha(owner/f'cold-{i}/sample.json')) for i in range(2)],admission=admission), 'cold raw admission binding')
        paths=sorted(owner.glob('physical-*/sample.json'),key=lambda p:int(p.parent.name.split('-')[-1]))
        need(len(paths)==complete['live_physical_samples'] and bool(paths) and complete['checkpoints']==len(paths)+6,'complete nonempty live sample/checkpoint set')
        previous=None;maximum=0
        for i,p in enumerate(paths):
            need(p.parent.name==f'physical-{i:04d}','contiguous live sample index')
            row=raw_sample(p,pid,attempt['owner_pid'],guard,limits)
            need(child['spawned_at']<=row['started_at']<=row['ended_at']<=release['time'],'sample actual child lifetime')
            guard.observe(row,limits,admission,child_pid=pid,child_started_monotonic=child['spawned_monotonic'],previous=previous)
            previous=row;maximum=max(maximum,row['parsed']['aux_peak_bytes'])
        need(all(attempt['created_at']<=r['started_at']<=r['ended_at']<=child['spawned_at'] for r in cold),'cold samples in actual owner window')
        result['resources']=dict(status='PASS',cold_samples=2,live_samples=len(paths),maximum_observed_child_aux_peak_bytes=maximum,
            checkpoints=checkpoints(owner,plan,attempt,child,release,guard),no_final_child_peak_history=True,
            physical_observation_only=True,hard_allocation_cap=False)
        inventory=producer.verify_inventory(plan,probe,lambda:None)
        need(read(owner/'model-runtime.json')==read(owner/'model-runtime-after.json')==inventory,'original pre/post supplied inventory proof')
        before=read(owner/'source-pins.json');need(before==plan['execution']['pins'],'exact original owner source map')
        for p,h in before.items():need(sha(owner/'owner-source'/p.lstrip('/'))==h and sha(p)==h,'retained/current source')
        folder=phase/'probe';terminal=read(folder/'complete.json')
        producer.probe_terminal(terminal,plan,sha(owner/'admission.json'),pid,attempt['owner_pid'],probe)
        shutdown=read(folder/'shutdown.json')
        need(shutdown==terminal['shutdown'] and shutdown['runtime_work_unsettled'] is False
             and shutdown['outer_wait_reap_required'] is True and shutdown['hook_restoration']['errors']==[]
             and probe.shutdown_is_complete(returned=shutdown['returned'] and 'error' not in shutdown,
                retained=shutdown['retained_entries'],engine_cleared=shutdown['engine_references_cleared'],
                load_attempted=shutdown['load_attempted'],load_returned=shutdown['load_await_returned'],
                active_calls=shutdown['active_observed_calls'] or shutdown['hook_restoration']['active_at_restore'],
                hooks_restored=shutdown['hook_restoration']['complete']), 'actual cooperative shutdown receipt')
        expected_command=producer.child_command(plan,probe,owner/'admission.json',folder)
        need(read(owner/'launch.json')==dict(command=expected_command,environment=probe.ENVIRONMENT,probe_plan=plan['probe_plan'],
            admission=dict(path=str(owner/'admission.json'),sha256=sha(owner/'admission.json'))), 'real direct original probe argv')
        need(read(phase/'intent.json')['command']==read(owner/'spawn-intent.json')['command']==expected_command,'actual launch intent argv')
        need(sha(folder/'plan.json')==plan['probe_plan']['sha256'] and sha(folder/'admission.json')==sha(owner/'admission.json'),'child actual input bytes')
        original=read(plan['probe_plan']['path'])
        adm=read(owner/'admission.json')
        need(adm['schema']=='GLM-official-load-admission-v1' and adm['owner_pid']==attempt['owner_pid']
             and adm['plan_sha256']==plan['probe_plan']['sha256'] and adm['probe_sha256']==producer.FIXED['../glm_flash/load_probe.py']
             and adm['output_directory']==str(folder) and adm['snapshot']==str(probe.SNAPSHOT)
             and guard.number(adm['issued_at']) and guard.number(adm['expires_at'])
             and attempt['created_at']<=adm['issued_at']<=child['spawned_at']<=adm['expires_at']
             and adm['expires_at']-adm['issued_at']==1800
             and all(adm[k] is True for k in ('cold_service_absent','exclusive_resources','hard_reap_owner','model_runtime_verified')),
             'actual admission plan/owner/lifetime/assertions')
        expected_evidence={k:dict(path=str(owner/name),sha256=sha(owner/name)) for k,name in
            (('cold_release','cold-release.json'),('resources','cold-physical.json'),('model_runtime','model-runtime.json'))}
        need(adm['evidence']==expected_evidence,'actual admission evidence refs')
        cold_release=read(owner/'cold-release.json')
        need(cold_release==dict(stopped_journal=dict(path=str(owner/'stopped-journal.json'),sha256=sha(owner/'stopped-journal.json')),
            checkpoint=dict(path=str(owner/'checkpoint-0004.json'),sha256=sha(owner/'checkpoint-0004.json')),
            journal_id=data['id'],removed_at=data['removed_at']),'actual last pre-spawn absence evidence')
        need(read(folder/'source-pins-before.json')==read(folder/'source-pins-after.json')==original['source_pins'],'probe source before/after')
        need(read(folder/'model-metadata-before.json')==read(folder/'model-metadata-after.json')==inventory['metadata'],'same GLM metadata/stat')
        need(sha(folder/'prompt.utf8')==original['prompt']['sha256'] and read(folder/'input.json')['prompt']==(folder/'prompt.utf8').read_text(),'actual plain prompt bytes')
        result['requests']=request_counts(folder,terminal,probe)
        need(read(phase/'release.json')['returncode']==complete['child_returncode']==0,'normal successful direct-child exit')
        result.update(status='PASS',observed_diagnostic_complete=True,owner_plan_sha256=plan_sha256,
            original_pid=data['original']['pid'],replacement_pid=data['replacement']['pid'],
            weight_payload_SHA_rechecked=False,supplied_runtime_SHA_rechecked=True,
            scope='Closed four-request official load/count diagnostic plus raw observed resources; no full-model/reference, text, HTTP, quality, performance or safety qualification.')
    except FileNotFoundError as error:result.update(status='INCOMPLETE',error=str(error))
    except Exception as error:
        result.update(status='INCOMPLETE' if result.get('restoration',{}).get('status')=='INCOMPLETE' else 'FAIL',error_type=type(error).__name__,error=str(error))
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--journal',type=Path,required=True)
    p.add_argument('--plan-sha256',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args(argv)
    need(not a.out.exists(),'new audit receipt only');result=audit(a.journal.resolve(),a.plan_sha256)
    with a.out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    return {'PASS':0,'FAIL':1,'INCOMPLETE':2}[result['status']]


if __name__=='__main__':raise SystemExit(main())
