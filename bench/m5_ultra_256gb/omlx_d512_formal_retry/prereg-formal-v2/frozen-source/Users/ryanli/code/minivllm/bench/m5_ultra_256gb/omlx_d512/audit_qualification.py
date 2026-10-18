"""Closed oMLX C1/C8 raw qualification audit; no service/process/model actions.

Reuses frozen join_v2's raw SSE/actual output-tap join and Tokenizer.decode with
skip_special_tokens=False. Outer CPU runtime and tokenizer files are hashed;
model weight payloads are not independently read. Supplied pre/post hash receipts
are checked explicitly. Resource and restoration qualification remain separate.
Missing terminal owner files returns INCOMPLETE before opening request evidence.
Adapted from retry auditor a932583f5a5234b0aff434d9b967a004fcdea05ff9b1c03a1d5d34fe86f6df08.
Four explicit source inventories are not a recursive dependency-closure proof.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
OWNER_SHA='9c2df0b4e76bd2a6ef590c34bb7a6f9c89972c878f76f4b99f69d0b8e658019f'
TRIAL_SHA='18ceeadb0c47874308d58f87f93e61cc160d24bb551dd3e50c62f1084e4c0ec6'
JOIN_SHA='9a09959a840503240f387dc1a6e447311e08b1187b54ba3516e09a60c18cd8d1'
PROBE_SHA='d0498a5fd825adcca97b4eee420dd89dbec9b6ee986b9a604986e286d2ad8b9e'
PHASES=('omlx-c1','omlx-c8')
GROUPS=('omlx_runtime','omlx_source','target_model','outer_runtime')


class Incomplete(Exception): pass
def need(ok,message):
    if not ok: raise ValueError(message)
def same(a,b,message):
    need(json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False),message)
def finite(value): return type(value) in (int,float) and math.isfinite(value)
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        while block:=f.read(1<<20): digest.update(block)
    return digest.hexdigest()
def read(path):
    def pairs(items):
        out={}
        for k,v in items:
            need(k not in out,'duplicate JSON key');out[k]=v
        return out
    def bad(value): raise ValueError('nonfinite JSON: '+value)
    return json.loads(Path(path).read_bytes(),object_pairs_hook=pairs,parse_constant=bad)
def ref(path): return dict(path=str(path),sha256=sha(path))
def relative(root,item,expected=None):
    need(set(item)=={'path','sha256'},'receipt reference fields')
    path=Path(item['path']);need(not path.is_absolute() and '..' not in path.parts,'relative receipt path')
    if expected is not None: need(str(path)==expected,'exact receipt location')
    path=root/path
    need(path.is_file() and not path.is_symlink() and sha(path)==item['sha256'],'receipt SHA/path')
    return path
def support():
    for name,digest in (('owner.py',OWNER_SHA),('trial.py',TRIAL_SHA),('join_v2.py',JOIN_SHA),('serve_probe_v2.py',PROBE_SHA)):
        need(sha(HERE/name)==digest,'frozen oMLX source changed: '+name)
    path=HERE/'owner.py'
    spec=importlib.util.spec_from_file_location('_omlx_qualification_audit_owner',path)
    owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
    trial=owner.load('trial');return owner,trial,trial.load('join_v2.py'),trial.load('../pareto_realistic/http_capture_v2.py')


def preparations(root,plan):
    expected=plan['verification']['groups'];previous=None
    same(sorted(expected),sorted(GROUPS),'exact four supplied groups')
    for phase in ('preflight','postflight'):
        folder=root/phase;same(read(folder/'verification-plan.json'),plan['verification'],'preparation inventory')
        verified=read(folder/'verified-pins.json');same(sorted(verified),sorted(expected),'four pin groups')
        for group,pins in expected.items():
            same(sorted(verified[group]),sorted(pins),'complete supplied pin group')
            for path,want in pins.items():
                row=verified[group][path]
                need(set(row)=={'sha256','bytes','mtime_ns','inode','device'},'pin receipt fields')
                same({k:row[k] for k in want},want,'pin receipt hash/stat differs')
                need(type(row['bytes']) is int and row['bytes']>=0 and
                     all(type(row[k]) is int and row[k]>0 for k in ('inode','device')),'pin receipt types')
        if previous is not None: same(verified,previous,'pre/post pin identities differ')
        previous=verified
        done=read(folder/'complete.json')
        need(done['status']=='PASS' and done['groups']==list(GROUPS) and
             type(done['files']) is int and done['files']==sum(map(len,verified.values()))
             and done['recursive_runtime_coverage_verified'] is False,'complete supplied preparation receipt')
        for source,digest in plan['execution']['pins'].items():
            need(sha(folder/'frozen-sources'/source.lstrip('/'))==digest,'retained source copy')
    return dict(groups={g:len(p) for g,p in expected.items()},preflight_postflight_identical=True,
        model_weights_rehashed=False,recursive_runtime_coverage_verified=False,
        scope='Four supplied pre/post full-hash/stat receipts; no independent weight payload rehash or recursive dependency closure.')


def closed_bindings(root,complete,plan):
    attempt=read(root/'attempt.json');stopped=read(root/'stopped-journal.json')
    pid=complete['owner_pid'];plan_sha=sha(root/'owner-plan.json')
    need(type(pid) is int and pid>1 and attempt['owner_pid']==pid,'actual owner PID')
    need(attempt['schema']=='cold-service-owner-attempt-v1' and
         attempt['owner_plan_sha256']==complete['owner_plan_sha256']==plan_sha and
         attempt['owner_source']==plan['execution']['owner'],'owner attempt/plan/source')
    need(attempt['owner_identity']['pid']==pid and attempt['stopped_journal_sha256']==sha(root/'stopped-journal.json'),'owner identity/stopped snapshot')
    need(stopped['schema']=='E023-omlx-D512-cold-journal-v1' and stopped['state']=='STOPPED'
         and stopped['owner_directory']==str(root) and stopped['owner_plan']['sha256']==plan_sha
         and stopped['owner_source']==attempt['owner_source'] and stopped['id']==attempt['journal_id']
         and stopped['removed_at']==attempt['removed_at'] and stopped['trials']==list(PHASES)
         and stopped['predecessor']==plan['cold_restart']['predecessor'],'real STOPPED binding')
    need(attempt['journal_path']==plan['cold_restart']['journal_path']
         and attempt['controller_sha256']==stopped['controller_sha256']==plan['cold_restart']['controller']['sha256']
         and stopped['approved_contract']==plan['cold_restart']['approved_contract']
         and stopped['plan_sha256']==plan['cold_restart']['proposal_plan_sha256'], 'real cold controller binding')
    index=read(root/'children.json');final=read(root/'release.json');spawn=read(root/'spawn-intent.json')
    need(index['schema']=='cold-service-owner-children-v1' and final['schema']=='cold-service-owner-release-v2','two-phase release schema')
    children=index['children'];need([c['phase'] for c in children]==list(PHASES),'exact complete two-phase coverage')
    same(final['children'],children,'final release/index differs')
    need(final['children_sha256']==sha(root/'children.json') and final['all_children_reaped'] is True
         and final.get('no_popen_receipt') is None,'all actual children closed')
    bound={k:attempt[k] for k in ('owner_pid','owner_identity','owner_source','owner_plan_sha256','journal_id','removed_at')}
    bound['attempt_sha256']=sha(root/'attempt.json')
    need(all(final.get(k)==v for k,v in bound.items()),'final attempt binding')
    need(all(spawn.get(k)==bound[k] for k in ('owner_pid','owner_plan_sha256','journal_id','attempt_sha256')),'global intent binding')
    need(all(finite(t) for t in (attempt['removed_at'],attempt['created_at'],final['time']))
         and attempt['removed_at']<=attempt['created_at']<=final['time'],'terminal time order')
    previous=attempt['created_at'];seen=set()
    for child in children:
        phase=child['phase'];folder=root/'phases'/phase
        intent=read(relative(root,child['intent'],f'phases/{phase}/intent.json'))
        own=read(relative(root,child['child'],f'phases/{phase}/child.json'))
        release=read(relative(root,child['release'],f'phases/{phase}/release.json'))
        need(all(intent.get(k)==bound[k] for k in ('owner_pid','owner_plan_sha256','journal_id','attempt_sha256'))
             and intent['phase']==phase,'phase intent binding')
        cpid=child['pid'];identity=own['identity']
        need(type(cpid) is int and cpid>1 and cpid!=pid and cpid not in seen,'unique actual server PID');seen.add(cpid)
        need(own['phase']==phase and own['pid']==cpid and own['owner_pid']==pid and
             type(identity) is dict and identity['pid']==cpid and identity['ppid']==pid and
             child['identity']==identity and own['ownership']=='direct-Popen-returned','actual direct child identity')
        need(release['phase']==phase and release['pid']==cpid and type(release['returncode']) is int
             and release['waited_reaped'] is True and release['strict_absence'] is True,'actual child reap')
        need(all(finite(t) for t in (intent['time'],own['spawned_at'],release['time'])) and
             previous<=intent['time']<=own['spawned_at']<=release['time']<=final['time'],'sequential phase times')
        previous=release['time']
        absent=read(relative(root,release['absence']))
        same(absent['command'],['/bin/ps','-p',str(cpid),'-o','pid='],'actual PID absence command')
        need(type(absent['returncode']) is int and absent['returncode']==1 and absent['stdout']=='','actual reaped PID absence')
        ports=sorted(folder.glob('port-absence-*.json'));need(bool(ports),'outer port absence missing')
        for path in ports:
            raw=read(path);same(raw['command'],['/usr/sbin/lsof','-nP','-a',f"-iTCP:{plan['http']['port']}",'-sTCP:LISTEN','-Fpn'],'actual port absence command')
            need(type(raw['returncode']) is int and raw['returncode']==1 and raw['stdout']=='','actual port absence')
    expected={f'phases/{phase}/{name}' for phase in PHASES for name in
        ('intent.json','child.json','release.json','trial/spawn-intent.json','trial/child.json','trial/release.json')}
    found={str(p.relative_to(root)) for p in (root/'phases').rglob('*')
        if p.name in ('intent.json','child.json','release.json','spawn-intent.json','no-popen.json')}
    same(sorted(found),sorted(expected),'unresolved or omitted phase spawn evidence')
    return children


def explained_release(inner,outer):
    need(type(inner['returncode']) is type(outer['returncode']) is int
         and inner['returncode']==outer['returncode']
         and (inner['returncode']==0 or (inner['returncode']==-15 and inner['term_sent'] is True)),
         'only normal exit or actual trial-owned SIGTERM, with matching inner/outer exit')


def events_and_exit(folder,complete,release,cpid):
    """Bind own SIGTERM to retained trial events, not a free boolean assertion."""
    rows=[]
    with (folder/'events.jsonl').open('rb') as stream:
        for line in stream:
            need(line.endswith(b'\n'),'truncated trial event')
            row=json.loads(line);need(type(row) is dict and finite(row['time']) and finite(row['monotonic']),'event clocks')
            need(complete['started_at']<=row['time']<=complete['ended_at'],'event outside trial lifetime')
            if rows:need(row['monotonic']>=rows[-1]['monotonic'],'event monotonic order')
            rows.append(row)
    allowed={'checkpoint','readiness','server-term','server-term-timeout','server-kill'}
    need(rows and all(row['label'] in allowed for row in rows),'unknown/missing trial events')
    term=[r for r in rows if r['label']=='server-term']
    need(len(term)==int(release['term_sent']) and type(release['term_sent']) is bool
         and all(r['pid']==cpid and r['time']<=release['time'] for r in term),'actual own TERM event')
    need(not any(r['label']=='server-kill' for r in rows),'forced shutdown cannot qualify')
    return dict(server_term_events=len(term),server_returncode=release['returncode'])


def group_contract(group,spec):
    need(group['base_url']==f"http://127.0.0.1:{spec['port']}" and
         group['request_timeout_s']==spec['deadlines']['request'] and
         group['group_timeout_s']==spec['deadlines']['group'], 'actual HTTP origin/deadlines')
    same(group['connection_policy'],dict(max_connections=2*spec['concurrency'],max_keepalive_connections=0,
         retries=0,follow_redirects=False),'fixed client connection policy')
    clean=group['cleanup']
    need(clean['budget_s']==spec['deadlines']['close'] and clean['client_close_started'] is True
         and clean['frozen_client_sha256']=='ef64a114f4f7d90ec5044d6bdcf058ca05f98dea445d6639db706a4fc75217dc',
         'actual bounded fixed client cleanup')
    start,end=group['barrier_release_perf_counter_s'],group['group_deadline_perf_counter_s']
    need(finite(start) and finite(end) and end==start+spec['deadlines']['group'], 'actual group deadline')
    for row in group['requests']:
        begin,done=row['start_perf_counter_s'],row['done_perf_counter_s']
        need(finite(begin) and finite(done) and start<=begin<done<=end
             and done<=begin+spec['deadlines']['request'], 'request outside actual group/request deadline')


def trial_evidence(folder,spec,cpid,owner_pid,outer_release,trial,join,cap,decode):
    complete=read(folder/'complete.json')
    need(complete['schema']=='E023-omlx-D512-trial-result-v1' and complete['status']=='PASS'
         and complete['concurrency']==spec['concurrency'] and complete['errors']==[],'closed trial success')
    for name in ('route_count_text_qualified','server_reaped','cleanup_complete','cleanup_task_resolved','client_task_resolved','small_pins_rechecked','request_sealed','spawn_attempted','popen_returned'):
        need(complete[name] is True,'trial field: '+name)
    need(complete['performance_qualified'] is False and complete['full_state_qualified'] is False,'qualification-only trial')
    need(finite(complete['started_at']) and finite(complete['ended_at']) and complete['started_at']<=complete['ended_at'], 'closed trial clocks')
    actual={str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file() and p!=folder/'complete.json'}
    same(sorted(complete['artifacts']),sorted(actual),'complete retained artifact inventory')
    for name,row in complete['artifacts'].items():
        path=relative(folder,dict(path=name,sha256=row['sha256']))
        need(type(row['bytes']) is int and row['bytes']>=0 and path.stat().st_size==row['bytes'],'artifact size')
    same(read(folder/'spec.json'),spec,'registered full trial spec')
    same(read(folder/'launch.json'),{k:spec[k] for k in ('command','cwd','environment')},'actual server launch')
    intent=read(folder/'spawn-intent.json');same(intent['command'],spec['command'],'inner spawn command')
    need(intent['cwd']==spec['cwd'] and read(folder/'child.json')['pid']==cpid,'inner direct server identity')
    raw=read(folder/'child-identity.json')
    same(raw['command'],['/bin/ps','-p',str(cpid),'-o','pid=,ppid=,lstart=,comm='],'raw child identity command')
    need(type(raw['returncode']) is int and raw['returncode']==0 and not raw['stderr']
         and raw['stdout'].split()[:2]==[str(cpid),str(owner_pid)],'raw direct server PPID')
    for name,command in [('child-absence.json',['/bin/ps','-p',str(cpid),'-o','pid=']),
         ('port-before.json',['/usr/sbin/lsof','-nP','-a',f"-iTCP:{spec['port']}",'-sTCP:LISTEN','-Fpn']),
         ('port-after.json',['/usr/sbin/lsof','-nP','-a',f"-iTCP:{spec['port']}",'-sTCP:LISTEN','-Fpn'])]:
        raw=read(folder/name);same(raw['command'],command,'raw absence command');need(cap.no_process(raw),'raw PID/port absence')
    release=read(folder/'release.json')
    need(release['pid']==cpid and type(release['returncode']) is int and all(release[k] is True for k in ('waited_reaped','strict_absence','port_absent')),'inner actual release')
    need(outer_release['pid']==cpid and finite(release['time']) and finite(outer_release['time'])
         and release['time']<=outer_release['time'],'inner/outer actual reap ordering')
    explained_release(release,outer_release)
    exit_evidence=events_and_exit(folder,complete,release,cpid)
    listener=read(folder/'listener.json');need(cap.owned_listener(listener,cpid,spec['port']),'raw owned listener')
    same(listener['command'],['/usr/sbin/lsof','-nP','-a','-p',str(cpid),f"-iTCP:{spec['port']}",'-sTCP:LISTEN','-Fpn'],'actual owned listener command')
    need(trial.ready(read(folder/'ready.json'),spec) and trial.ready(read(folder/'ready-after.json'),spec),'actual served target/capacity')
    group=read(folder/'group.json')
    same(group,read(folder/'client/group.json'),'outer and actual client group bytes/value binding')
    group_contract(group,spec)
    with (folder/'probe/observer.jsonl').open('rb') as lines:
        computed=join.join(group,lines,spec['jobs'],{f['fixture_id']:f for f in spec['fixtures']},decode,
            PROBE_SHA,expected_official_args=spec['official_args'],expected_model_path=spec['model_directory'],expected_pid=cpid)
    need(computed['runtime']['ppid']==owner_pid,'observed process runtime PPID differs from direct server')
    same(computed,read(folder/'raw-join.json'),'retained raw join differs from independent recomputation')
    same(computed['seal'],read(folder/'request-seal.json'),'pre-TERM request seal differs from closed ledger')
    need(complete['observer_closed'] is computed['observer_closed'],'observer close scope receipt differs')
    for name in ('server.stdout','server.stderr'):need((folder/name).is_file(),'closed server log missing')
    return dict(concurrency=spec['concurrency'],requests=len(computed['requests']),pid=cpid,
        raw_join=ref(folder/'raw-join.json'),observer=ref(folder/'probe/observer.jsonl'),group=ref(folder/'group.json'),
        request_seal=ref(folder/'request-seal.json'),trial_complete=ref(folder/'complete.json'),
        server_stdout=ref(folder/'server.stdout'),server_stderr=ref(folder/'server.stderr'),exit_evidence=exit_evidence,
        observed_terminal_eos=computed['observed_terminal_eos'],raw_output_count=computed['raw_output_count'],
        committed_output_count=computed['committed_output_count'],max_http_outstanding=computed['max_http_outstanding'],
        actual_batch_histogram=computed['batch_histogram'],observer_closed=computed['observer_closed'],
        generator_eof_proven=False,count_rule=computed['count_rule'])


def decoder(plan,spec):
    # Same decode as trial.System.decoder; never retokenize text as a raw-ID oracle.
    for path,row in plan['verification']['groups']['outer_runtime'].items():
        need(sha(path)==row['sha256'],'outer tokenizer/HTTP runtime changed')
    for path,digest in spec['tokenizer_pins'].items():
        need(plan['verification']['groups']['target_model'][path]['sha256']==digest
             and sha(path)==digest,'actual tokenizer/config pin changed')
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file(str(Path(spec['model_directory'])/'tokenizer.json'))
    return lambda ids:tokenizer.decode(ids,skip_special_tokens=False)


def audit(directory):
    root=Path(directory).resolve()
    result=dict(schema='E023-omlx-D512-qualification-audit-v1',status='INCOMPLETE',qualified=False,
        route_count_text_qualified=False,full_state_qualified=False,performance_qualified=False,
        independent_resource_verified=False,restoration_verified=False,raw_requests_opened=False,
        owner_directory=str(root),audit_source_sha256=sha(__file__),owner_source_sha256=OWNER_SHA,
        recursive_runtime_coverage_verified=False,closed_server_logs_independently_verified=False,
        scope='Closed C1/C8 raw route/count/text/cache and direct server closure only. Logs are retained/hash-checked, not semantically interpreted as full lifespan health; separate resource/restoration audits required.')
    try:
        if not all((root/name).is_file() for name in ('complete.json','release.json','children.json','attempt.json')):
            raise Incomplete('terminal owner evidence missing; request evidence not opened')
        complete=read(root/'complete.json')
        need(complete['schema']=='E023-omlx-D512-owner-result-v1' and complete['status']=='PASS'
             and complete['errors']==[] and complete['qualification_trials_complete'] is True
             and complete['all_children_reaped'] is True and complete['performance_qualified'] is False
             and complete['full_state_qualified'] is False and complete['recursive_runtime_coverage_verified'] is False,
             'owner did not complete both closed trials')
        owner,trial,join,cap=support();plan_path=root/'owner-plan.json';plan=read(plan_path)
        args=SimpleNamespace(plan=plan_path,plan_sha256=sha(plan_path),out=root)
        _,validated,_,_,specs=owner.validate_plan(args);same(plan,validated,'plan validation')
        need(complete['planned_trials']==list(PHASES),'complete planned phases')
        children=closed_bindings(root,complete,plan);prep=preparations(root,plan)
        same(read(root/'trial-specs.json'),specs,'owner prepared both exact specs')
        published=read(root/'qualifications.json')
        need(published['schema']=='E023-omlx-D512-qualifications-v1' and
             published['performance_qualified'] is False and published['full_state_qualified'] is False,'published scope')
        same(published['trials'],complete['phases'],'published/terminal phase bindings')
        need([r['phase'] for r in published['trials']]==list(PHASES),'published exact phases')
        for phase,entry,c in zip(PHASES,published['trials'],(1,8)):
            need(entry['status']=='PASS' and entry['concurrency']==c,'published C1/C8 identity')
            for key,name in (('result','complete.json'),('raw_join','raw-join.json'),('request_seal','request-seal.json'),
                             ('group','group.json'),('stdout','server.stdout'),('stderr','server.stderr')):
                relative(root,entry[key],f'phases/{phase}/trial/{name}')
        decode=decoder(plan,specs[0]);rows=[];result['raw_requests_opened']=True
        for phase,child,spec in zip(PHASES,children,specs):
            rows.append(trial_evidence(root/'phases'/phase/'trial',spec,child['pid'],complete['owner_pid'],
                read(root/'phases'/phase/'release.json'),trial,join,cap,decode))
        need([r['requests'] for r in rows]==[8,16],'complete D8/D16 coverage')
        result.update(status='PASS',qualified=True,route_count_text_qualified=True,phases=rows,
            requests=24,preparation=prep,owner_plan_sha256=sha(plan_path),owner_complete_sha256=sha(root/'complete.json'),
            outer_runtime_rehashed=True,tokenizer_payloads_rehashed=True,
            model_weights_rehashed=False)
    except (Incomplete,FileNotFoundError) as exc: result['error']=dict(type=type(exc).__name__,message=str(exc))
    except Exception as exc:
        result.update(status='FAIL',error=dict(type=type(exc).__name__,message=str(exc)))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--owner',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args(argv)
    need(not args.out.exists() and args.out.parent.is_dir(),'new result file in existing directory')
    result=audit(args.owner)
    with args.out.open('x') as f: json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:result[k] for k in ('status','qualified','requests') if k in result}))
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__': raise SystemExit(main())
