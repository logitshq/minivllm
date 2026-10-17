"""Closed Lithos C1/C8 raw qualification audit; no service/process/model actions.

Reuses the frozen raw NDJSON/Generation join and exact official tokenizer decode
semantics. Outer CPU runtime and tokenizer files are hashed; model weights and
pack payloads are not independently read. Their supplied pre/post hash receipts
are checked explicitly. Resource and restoration qualification remain separate.
Missing terminal owner files returns INCOMPLETE before opening request evidence.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
OWNER_SHA='a6796b153d558954899a46a96cd8ead828928e8627aea1ba8a8b6453ab74bddd'
PHASES=('lithos-c1','lithos-c8')
GROUPS=('lithos_runtime','lithos_source','target_model','pack','outer_runtime')


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
    path=HERE/'owner.py';need(sha(path)==OWNER_SHA,'frozen Lithos owner changed')
    spec=importlib.util.spec_from_file_location('_lithos_qualification_audit_owner',path)
    owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
    trial=owner.load('trial');return owner,trial,trial.load('join.py'),trial.load('../pareto_realistic/http_capture_v2.py')


def preparations(root,plan):
    expected=plan['verification']['groups'];previous=None
    for phase in ('preflight','postflight'):
        folder=root/phase;same(read(folder/'verification-plan.json'),plan['verification'],'preparation inventory')
        verified=read(folder/'verified-pins.json');same(sorted(verified),sorted(expected),'five pin groups')
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
             done['files']==sum(map(len,verified.values())),'complete preparation receipt')
        for source,digest in plan['execution']['pins'].items():
            need(sha(folder/'frozen-sources'/source.lstrip('/'))==digest,'retained source copy')
    return dict(groups={g:len(p) for g,p in expected.items()},preflight_postflight_identical=True,
        model_weights_rehashed=False,pack_payload_rehashed=False,
        scope='Source-pinned producer full-hash/stat receipts; no independent model/pack payload rehash.')


def closed_bindings(root,complete,plan):
    attempt=read(root/'attempt.json');stopped=read(root/'stopped-journal.json')
    pid=complete['owner_pid'];plan_sha=sha(root/'owner-plan.json')
    need(type(pid) is int and pid>1 and attempt['owner_pid']==pid,'actual owner PID')
    need(attempt['schema']=='cold-service-owner-attempt-v1' and
         attempt['owner_plan_sha256']==complete['owner_plan_sha256']==plan_sha and
         attempt['owner_source']==plan['execution']['owner'],'owner attempt/plan/source')
    need(attempt['owner_identity']['pid']==pid and attempt['stopped_journal_sha256']==sha(root/'stopped-journal.json'),'owner identity/stopped snapshot')
    need(stopped['schema']=='E023-lithos-D512-cold-journal-v1' and stopped['state']=='STOPPED'
         and stopped['owner_directory']==str(root) and stopped['owner_plan']['sha256']==plan_sha
         and stopped['owner_source']==attempt['owner_source'] and stopped['id']==attempt['journal_id']
         and stopped['removed_at']==attempt['removed_at'] and stopped['trials']==list(PHASES)
         and stopped['predecessor']==plan['cold_restart']['predecessor'],'real STOPPED binding')
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
        need(absent['returncode']==1 and absent['stdout']=='','actual reaped PID absence')
        ports=sorted(folder.glob('port-absence-*.json'));need(bool(ports),'outer port absence missing')
        for path in ports:
            raw=read(path);same(raw['command'],['/usr/sbin/lsof','-nP','-a',f"-iTCP:{plan['http']['port']}",'-sTCP:LISTEN','-Fpn'],'actual port absence command')
            need(raw['returncode']==1 and raw['stdout']=='','actual port absence')
    expected={f'phases/{phase}/{name}' for phase in PHASES for name in
        ('intent.json','child.json','release.json','trial/spawn-intent.json','trial/child.json','trial/release.json')}
    found={str(p.relative_to(root)) for p in (root/'phases').rglob('*')
        if p.name in ('intent.json','child.json','release.json','spawn-intent.json','no-popen.json')}
    same(sorted(found),sorted(expected),'unresolved or omitted phase spawn evidence')
    return children


def trial_evidence(folder,spec,cpid,owner_pid,trial,join,cap,decode):
    complete=read(folder/'complete.json')
    need(complete['schema']=='E023-lithos-D512-trial-result-v1' and complete['status']=='PASS'
         and complete['concurrency']==spec['concurrency'] and complete['errors']==[],'closed trial success')
    for name in ('route_count_text_qualified','server_reaped','cleanup_complete','cleanup_task_resolved','client_task_resolved','small_pins_rechecked'):
        need(complete[name] is True,'trial field: '+name)
    need(complete['performance_qualified'] is False and complete['full_state_qualified'] is False,'qualification-only trial')
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
    need(raw['returncode']==0 and not raw['stderr'] and raw['stdout'].split()[:2]==[str(cpid),str(owner_pid)],'raw direct server PPID')
    for name,command in [('child-absence.json',['/bin/ps','-p',str(cpid),'-o','pid=']),
         ('port-before.json',['/usr/sbin/lsof','-nP','-a',f"-iTCP:{spec['port']}",'-sTCP:LISTEN','-Fpn']),
         ('port-after.json',['/usr/sbin/lsof','-nP','-a',f"-iTCP:{spec['port']}",'-sTCP:LISTEN','-Fpn'])]:
        raw=read(folder/name);same(raw['command'],command,'raw absence command');need(cap.no_process(raw),'raw PID/port absence')
    release=read(folder/'release.json')
    need(release['pid']==cpid and type(release['returncode']) is int and all(release[k] is True for k in ('waited_reaped','strict_absence','port_absent')),'inner actual release')
    listener=read(folder/'listener.json');need(cap.owned_listener(listener,cpid,spec['port']),'raw owned listener')
    same(listener['command'],['/usr/sbin/lsof','-nP','-a','-p',str(cpid),f"-iTCP:{spec['port']}",'-sTCP:LISTEN','-Fpn'],'actual owned listener command')
    need(trial.ready(read(folder/'ready.json'),spec) and trial.ready(read(folder/'ready-after.json'),spec),'actual served target/capacity')
    group=read(folder/'group.json')
    with (folder/'probe/observer.jsonl').open('rb') as lines:
        computed=join.join(group,lines,spec['jobs'],{f['fixture_id']:f for f in spec['fixtures']},decode,
            trial.FIXED['serve_probe.py'],expected_official_args=spec['official_args'],expected_pack_dir=spec['pack_dir'])
    same(computed,read(folder/'raw-join.json'),'retained raw join differs from independent recomputation')
    logs=join.generation_log_window((folder/'server.stderr').read_bytes(),len(spec['jobs']),
        require_setup_zero=False,expected_logger='monolith.serve')
    same(logs,read(folder/'generation-log-check.json'),'retained Generation log check differs')
    return dict(concurrency=spec['concurrency'],requests=len(computed['requests']),pid=cpid,
        raw_join=ref(folder/'raw-join.json'),observer=ref(folder/'probe/observer.jsonl'),group=ref(folder/'group.json'),
        trial_complete=ref(folder/'complete.json'),log_check=ref(folder/'generation-log-check.json'),
        observed_eos=computed['observed_eos'],raw_output_count=computed['raw_output_count'],
        committed_output_count=computed['committed_output_count'],max_http_outstanding=computed['max_http_outstanding'])


def decoder(plan,spec):
    # CPU tokenizer runtime is independently hashed; the large model/pack is not.
    for path,row in plan['verification']['groups']['outer_runtime'].items():
        need(sha(path)==row['sha256'],'outer tokenizer/HTTP runtime changed')
    for path,digest in spec['tokenizer_pins'].items():
        need(plan['verification']['groups']['target_model'][path]['sha256']==digest
             and sha(path)==digest,'actual tokenizer/config pin changed')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(spec['model_directory'],local_files_only=True,trust_remote_code=False)
    return lambda ids:tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)


def audit(directory):
    root=Path(directory).resolve()
    result=dict(schema='E023-lithos-D512-qualification-audit-v1',status='INCOMPLETE',qualified=False,
        route_count_text_qualified=False,full_state_qualified=False,performance_qualified=False,
        independent_resource_verified=False,restoration_verified=False,raw_requests_opened=False,
        owner_directory=str(root),audit_source_sha256=sha(__file__),owner_source_sha256=OWNER_SHA,
        scope='Closed observed C1/C8 raw route/count/text/cache only; separate resource/restoration audits required.')
    try:
        if not all((root/name).is_file() for name in ('complete.json','release.json','children.json','attempt.json')):
            raise Incomplete('terminal owner evidence missing; request evidence not opened')
        complete=read(root/'complete.json')
        need(complete['schema']=='E023-lithos-D512-owner-result-v1' and complete['status']=='PASS'
             and complete['errors']==[] and complete['qualification_trials_complete'] is True
             and complete['all_children_reaped'] is True and complete['performance_qualified'] is False,'owner did not complete both closed trials')
        owner,trial,join,cap=support();plan_path=root/'owner-plan.json';plan=read(plan_path)
        args=SimpleNamespace(plan=plan_path,plan_sha256=sha(plan_path),out=root)
        _,validated,_,_,specs=owner.validate_plan(args);same(plan,validated,'plan validation')
        need(complete['planned_trials']==list(PHASES),'complete planned phases')
        children=closed_bindings(root,complete,plan);prep=preparations(root,plan)
        same(read(root/'trial-specs.json'),specs,'owner prepared both exact specs')
        published=read(root/'qualifications.json')
        need(published['schema']=='E023-lithos-D512-qualifications-v1' and
             published['performance_qualified'] is False and published['full_state_qualified'] is False,'published scope')
        same(published['trials'],complete['phases'],'published/terminal phase bindings')
        need([r['phase'] for r in published['trials']]==list(PHASES),'published exact phases')
        for phase,entry,c in zip(PHASES,published['trials'],(1,8)):
            need(entry['status']=='PASS' and entry['concurrency']==c,'published C1/C8 identity')
            for key,name in (('result','complete.json'),('raw_join','raw-join.json'),('log_check','generation-log-check.json')):
                relative(root,entry[key],f'phases/{phase}/trial/{name}')
        decode=decoder(plan,specs[0]);rows=[];result['raw_requests_opened']=True
        for phase,child,spec in zip(PHASES,children,specs):
            rows.append(trial_evidence(root/'phases'/phase/'trial',spec,child['pid'],complete['owner_pid'],trial,join,cap,decode))
        need([r['requests'] for r in rows]==[8,16],'complete D8/D16 coverage')
        result.update(status='PASS',qualified=True,route_count_text_qualified=True,phases=rows,
            requests=24,preparation=prep,owner_plan_sha256=sha(plan_path),owner_complete_sha256=sha(root/'complete.json'),
            outer_runtime_rehashed=True,tokenizer_payloads_rehashed=True,
            model_weights_rehashed=False,pack_payload_rehashed=False)
    except Incomplete as exc: result['error']=dict(type=type(exc).__name__,message=str(exc))
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
