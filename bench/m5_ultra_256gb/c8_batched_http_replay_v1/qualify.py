"""Default-OFF full numerical replay of the actual E052 candidate HTTP history.

Retain the original full replay arithmetic, paired graph/route stream and restore.
Only new input/source bindings, batched trace policy and integration are supplied.
This process never owns a service; an external owner supplies admission and reap.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import traceback
from types import FunctionType, SimpleNamespace

HERE = Path(__file__).resolve().parent; B = HERE.parent
TRIAL = B/'c8_batched_http_v2/trial.py'
TRIAL_SHA = 'a45bd7db4a30c6d6e16b1b2b2ef0ce4c139f9cc8b6fc344ee92219c42e1c7a72'
SERVER = B/'c8_batched_http_v1/server.py'
SERVER_SHA = 'a3737f1b364e190e3fe45a1779e1be228c4f434701bf8875dbe18f1d519afc7b'
HTTP = B/'gdn_graph_combined/http_qualification.py'
HTTP_SHA = '33b86cb57bf3c4616762c1bd1cfea0abb8f7f063325e9dfe815642e8f591bc7d'
SCHEMA = 'c8-batched-http-replay-result-v1'
SCOPE = ('Only this actual candidate HTTP two-wave arrival history and complete recorded numerical replay; '
         'independent numerical/route audit and external owner resource/restoration checks remain required. '
         'No arbitrary-arrival or performance qualification.')


def need(ok, message):
    if not ok: raise ValueError(message)
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1 << 20): h.update(block)
    return h.hexdigest()
def parse(raw):
    def pairs(items):
        result = {}
        for k,v in items:
            need(k not in result, 'duplicate JSON field'); result[k] = v
        return result
    def bad(value): raise ValueError('nonfinite JSON constant: '+value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)
def read(path): return parse(Path(path).read_bytes())
def write(path, value): Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
def same(a,b,message): need(json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False),message)
def ref(path): return dict(path=str(Path(path).resolve()), sha256=sha(path))
def pinned(item):
    need(set(item)=={'path','sha256'}, 'exact path/SHA reference')
    path=Path(item['path'])
    need(path.is_absolute() and path.resolve()==path and path.is_file() and not path.is_symlink()
         and sha(path)==item['sha256'], 'pinned input changed: '+str(path)); return path

def load(path,digest,name):
    need(sha(path)==digest,'frozen helper changed: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def helpers():
    return (load(TRIAL,TRIAL_SHA,'_e053_trial'),load(SERVER,SERVER_SHA,'_e053_server'),
            load(HTTP,HTTP_SHA,'_e053_http'))

def source_pins(trial,q):
    pins={}
    for table in (trial.source_pins(),q.source_pins(),
                  {str(TRIAL):TRIAL_SHA,str(SERVER):SERVER_SHA,str(HTTP):HTTP_SHA,str(HERE/'qualify.py'):sha(__file__)}):
        for path,digest in table.items():
            need(path not in pins or pins[path]==digest,'conflicting source pin');pins[path]=digest
    need(all(sha(path)==digest for path,digest in pins.items()),'complete source closure changed')
    return pins


def terminal_ok(complete,child,release):
    need(complete['schema']=='c8-batched-http-result-v2' and complete['status']=='PASS'
         and all(complete.get(k) is True for k in ('capture_complete','server_reaped','cleanup_complete',
             'cleanup_task_resolved','client_task_resolved','small_pins_rechecked','route_events_retained'))
         and complete['errors']==[] and all(complete.get(k) is False for k in
             ('numerical_qualified','full_state_qualified','performance_qualified','replay_input_checked')),
         'closed successful unqualified E052 v2 capture required')
    pid=child['pid']
    need(type(pid) is int and pid>1 and pid!=os.getpid() and release['pid']==pid
         and type(release['returncode']) is int and all(release[k] is True for k in
             ('waited_reaped','strict_absence','port_absent')), 'separate captured server fully reaped')
    need(type(release['time']) in (int,float) and math.isfinite(release['time'])
         and child['time']<=release['time']<=complete['ended_at'],'capture/release time order')


def capture_gate(child,spec_sha,trial,server):
    spec=read(pinned(dict(path=str(child/'spec.json'),sha256=spec_sha)));trial.validate(spec)
    need(spec['concurrency']==8 and spec['wave_names']==['warm','screen'] and len(spec['jobs'])==16,
         'actual original two eight-request waves')
    complete,process,release=[read(child/n) for n in ('complete.json','child.json','release.json')]
    terminal_ok(complete,process,release)
    original=trial.old();capture=original.module(original.CAPTURE)
    need(capture.no_process(read(child/'child-absence.json')) and capture.no_process(read(child/'port-after.json')),
         'saved strict server and port absence')
    files={p.name:p for p in child.iterdir() if p.is_file() and p.name!='complete.json'}
    same(set_to_list(complete['artifacts']),set_to_list(files),'complete top-level capture inventory')
    for name,path in files.items():
        need(not path.is_symlink() and path.resolve().parent==child,'regular captured artifact')
        same(complete['artifacts'][name],dict(sha256=sha(path),bytes=path.stat().st_size),'captured artifact: '+name)
    same(read(child/'launch.json'),dict(command=spec['command']+['--qualification-trace',str(child/'trace.jsonl'),
         '--route-events',str(child/'route-events.jsonl')],environment=spec['environment']),'actual launch/environment')
    for name in ('ready.json','ready-after-client.json'):
        need(capture.ready_identity(read(child/name),spec['identity']),'saved actual ready identity')
    need(capture.no_process(read(child/'port-before.json'))
         and capture.owned_listener(read(child/'listener.json'),process['pid'],spec['transport']['port']),
         'saved initially free/owned capture listener')
    events=[parse(line) for line in (child/'events.jsonl').read_bytes().splitlines()]
    ownterm=any(r.get('label')=='server-term' and r.get('pid')==process['pid']
                and type(r.get('time')) in (int,float) and math.isfinite(r['time']) and r['time']<=release['time'] for r in events)
    need(not any(r.get('label')=='server-kill' for r in events)
         and (release['returncode']==0 or release['returncode']==-15 and ownterm),'normal or own explained TERM release')
    header=dict(event='route-startup',schema=server.ROUTE_SCHEMA,pid=process['pid'],phase='HTTP-C8',identity=spec['identity'])
    return spec,header


def set_to_list(value): return sorted(value)


def rejoin(child,spec,numeric,trial,q):
    """Original byte-level SSE/schedule join, with these actual jobs, never jobs_for()."""
    original=trial.old();groups=[];refs=[]
    for phase in ('warm','screen'):
        path=child/'client'/phase/'group.json';group=read(path)
        need(original.group_ok(group) and len(group['requests'])==8,'complete original eight-request wave')
        groups.append(group);refs.append(dict(phase=phase,**ref(path)))
    group=original.join_view(groups,refs);same(group,read(child/'group.json'),'derived view versus actual raw waves')
    fixtures=read(pinned(numeric['fixtures']))['fixtures'];same(spec['fixtures'],fixtures,'original full fixtures')
    model=read(pinned(numeric['model_manifest']));tokenizer_path=Path(numeric['model'])/'tokenizer.json'
    need(spec['tokenizer']==dict(path=str(tokenizer_path),sha256=model['files']['tokenizer.json'])
         and sha(tokenizer_path)==model['files']['tokenizer.json'],'exact original tokenizer')
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file(str(tokenizer_path))
    for f in fixtures:
        same(tokenizer.encode(f['prompt_text'],add_special_tokens=False).ids,f['prompt_ids'],'full prompt tokenizer IDs')
    join=q.load(q.NATIVE/'http_join.py','_e053_raw_join')
    with (child/'trace.jsonl').open('rb') as stream:
        recording=join.join(group,stream,spec['jobs'],{f['fixture_id']:f for f in fixtures},spec['identity'],
                            lambda ids:tokenizer.decode(ids,skip_special_tokens=False))
    same(recording,read(child/'joined-input/recording.json'),'complete raw rejoin versus registered input')
    receipt=read(child/'joined-input/receipt.json')
    need(receipt['schema']=='c8-batched-http-joined-input-v2' and receipt['inherited_join_schema']=='E047-joined-input-v1'
         and receipt['replay_numerical_checked'] is False and receipt['performance_qualified'] is False,'unqualified real join receipt')
    for key,value in dict(spec_sha256=sha(child/'spec.json'),source_groups=refs,trace_sha256=sha(child/'trace.jsonl'),
                          recording_sha256=sha(child/'joined-input/recording.json'),sources=spec['sources']).items():
        same(receipt[key],value,'actual join receipt binding: '+key)
    return recording,fixtures


def recording_ok(recording):
    need(recording['schema']=='E023-recorded-http-v1' and recording['concurrency']==8
         and type(recording['steps']) is list and len(recording['steps'])>0 and len(recording['jobs'])==16,
         'complete actual C8 recording')
    need(all(s['step']==i for i,s in enumerate(recording['steps'],1))
         and all(j['cap']==512 for j in recording['jobs'].values()),'contiguous actual steps and original caps')
    return dict(expected_steps=len(recording['steps']),expected_requests=len(recording['jobs']),
        expected_raw_outputs=sum(len(j['raw_ids']) for j in recording['jobs'].values()),
        expected_committed_outputs=sum(len(j['committed_ids']) for j in recording['jobs'].values()))


def route_expectations(path,header,server):
    """Plan counts from retained rows; numerical execution later consumes every exact row."""
    calls=hits=0;pending=None;seen=set();counters=[dict(candidate=0,fallback=0) for _ in range(48)]
    with path.open('rb') as stream:
        raw=stream.readline();need(raw.endswith(b'\n'),'complete route startup');same(parse(raw),header,'actual route startup')
        for raw in stream:
            need(raw.endswith(b'\n'),'truncated route row');row=parse(raw)
            if row['event']=='combined-graph-call':
                need(pending is None and row['status']=='SUBMITTED' and row['installation']==0
                     and row['phase']=='HTTP-C8' and row['direct_label'] is None,'complete paired combined trace')
                same(row['before'],counters,'continuous48 counters')
                candidate,fallback=server.batched_delta(counters,row['after'],row['B'])
                same([candidate,fallback],[row['per_layer_candidate_delta'],row['per_layer_fallback_delta']],'raw trace deltas')
                need(type(row['actual_invocations']) is int and row['actual_invocations'] in (0,1),'actual callable count')
                key=(row['bound'],row['B'])
                if row['actual_invocations']:
                    if candidate: seen.add(key);kind='compiled-tracing'
                    else: need(key in seen,'reuse without earlier actual trace');kind='compiled-reuse'
                else: need(candidate==1,'eager execution once');kind='eager-fallback'
                same(row['kind'],kind,'trace kind');counters=row['after'];pending=row
            else:
                need(row['event']=='route-call' and pending is not None and row['status']=='SUBMITTED','paired original route row')
                for k in ('phase','step','direct_label','used','actual_invocations','B'):same(row[k],pending[k],'pair context: '+k)
                calls+=1;hits+=row['actual_invocations'];pending=None
    need(pending is None and calls>0 and hits>0 and seen,'complete nonempty actual route stream')
    return dict(expected_route_calls=calls,expected_compiled_calls=hits,expected_route_fallbacks=calls-hits)


def prepare(child_capture_path,spec_sha256):
    child=Path(child_capture_path).resolve();trial,server,q=helpers()
    spec,header=capture_gate(child,spec_sha256,trial,server)
    need(spec['base_plan']['sha256']==q.BASE_PLAN_SHA,'original numeric model/workload plan')
    numeric=read(pinned(spec['base_plan']))
    need(numeric['configuration']['cache_limit_gib']==2 and numeric['configuration']['max_len']==2048
         and numeric['candidate']==dict(compile_decode=True,fuse_decode=True,gates_compile=True,norm_reuse=False),
         'unchanged original numeric settings/cache')
    recording,_=rejoin(child,spec,numeric,trial,q)
    files=[p for p in child.iterdir() if p.is_file()]
    files += [child/n for n in ('client/warm/group.json','client/screen/group.json','joined-input/receipt.json','joined-input/recording.json')]
    return dict(schema='c8-batched-http-replay-plan-v1',sources=source_pins(trial,q),qualification=spec['base_plan'],
        capture_directory=str(child),capture_spec=ref(child/'spec.json'),capture_inputs={str(p):sha(p) for p in sorted(files)},
        predecessor=spec['predecessor'],recording=ref(child/'joined-input/recording.json'),route_events=ref(child/'route-events.jsonl'),
        route_header=header,environment={k:v for k,v in spec['environment'].items() if k.startswith('MLX_')},
        switches=dict(gdn_prework=True,route_graph=True),allocator_cache_policy='unchanged-2gib',
        **recording_ok(recording),**route_expectations(child/'route-events.jsonl',header,server),
        numerical_qualified=False,http_qualified=False,performance_qualified=False,scope=SCOPE)


def validate(plan):
    same(plan,prepare(plan['capture_directory'],plan['capture_spec']['sha256']),'actual source/input replay plan changed')


def adapted_function(q,server):
    """Clone only the original Handle constructor and decorated adapter globals."""
    original_load=q.load
    def private_load(path,name):
        combined=original_load(path,name)
        need(path==q.HERE/'qualify.py','only the registered combined trace helper')
        namespace=dict(vars(combined),TraceGraph=server.trace_type(combined))
        class Handle(combined.Handle):
            __init__=FunctionType(combined.Handle.__init__.__code__,namespace,'__init__',combined.Handle.__init__.__defaults__)
        return SimpleNamespace(**dict(vars(combined),Handle=Handle))
    fn=q.adapted.__wrapped__
    private=FunctionType(fn.__code__,dict(fn.__globals__,load=private_load),fn.__name__,fn.__defaults__,fn.__closure__)
    return contextmanager(private)


def full_result_ok(inner,recording,coverage):
    expected=recording_ok(recording)
    need(inner['status']=='PASS' and inner['source_model_pins_rechecked'] is True
         and inner['total_requests']==16 and len(inner['cases'])==1,'complete original numerical result')
    case=inner['cases'][0]
    need(case['status']=='PASS' and case['name']=='HTTP-C8' and case['steps']==expected['expected_steps']
         and len(case['completed'])==len({r['rid'] for r in case['completed']})==16,'full actual history, never a prefix')
    targets={rid:dict(rid=rid,raw_ids=j['raw_ids'],committed_ids=j['committed_ids']) for rid,j in recording['jobs'].items()}
    same({r['rid']:r for r in case['completed']},targets,'all actual recorded output IDs')
    same(case['decode_batch_histogram'],recording['histogram'],'complete actual decode histogram')
    need(case['raw_output_count']==expected['expected_raw_outputs']
         and case['committed_output_count']==expected['expected_committed_outputs'],'actual raw/committed counts')
    need(coverage['installs']==1 and coverage['restores']==2,'fresh pair release')
    stats=coverage['combined']
    need(stats['installs']==stats['restores']==1 and stats['layer_installs']==stats['layer_restores']==48
         and stats['compiled-tracing']>0,'real48 batched installation/trace/restoration')
    return case


def run(plan,out):
    validate(plan)
    same({k:v for k,v in os.environ.items() if k.startswith('MLX_')},plan['environment'],'exact original MLX environment')
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False);write(out/'plan.json',plan)
    result=dict(schema=SCHEMA,status='INCOMPLETE',pid=os.getpid(),ppid=os.getppid(),plan_sha256=sha(out/'plan.json'),
        replay_numerical_checks_passed=False,source_input_pins_rechecked=False,independent_audit_required=True,
        numerical_qualified=False,http_qualified=False,performance_qualified=False,owner_resources_restore_qualified=False,scope=SCOPE)
    write(out/'complete.json',result)
    try:
        trial,server,q=helpers();recording=read(pinned(plan['recording']));numeric=read(pinned(plan['qualification']))
        fixtures=read(pinned(numeric['fixtures']))['fixtures'];write(out/'recording.json',recording)
        base=q.load(q.BASE,'_e053_private_base');replay=q.load(q.NATIVE/'http_replay.py','_e053_private_replay')
        integration=server.load(server.BATCHED,server.BATCHED_SHA,'_e053_actual_batched_integration')
        witness=q.load(q.ROUTE/'serving_qualification.py','_e053_private_witness')
        counter=q.load(q.OLD/'qualify.py','_e053_private_counter')
        with pinned(plan['route_events']).open('rb') as captured,(out/'replay-route-events.jsonl').open('x') as emitted:
            routes=q.CapturedRoutes(captured,plan['route_header'],emitted,plan['route_events'],'HTTP-C8')
            with adapted_function(q,server)(base,replay,integration,witness,counter,recording,fixtures,routes) as coverage:
                code=base.main(['--plan',plan['qualification']['path'],'--plan-sha256',plan['qualification']['sha256'],'--out',str(out/'base')])
            inner_path=out/'base/complete.json';inner=read(inner_path)
            result.update(base_result=dict(path=str(inner_path),sha256=sha(inner_path),returncode=code,status=inner['status']),
                coverage=coverage,peak_mlx_bytes=inner.get('peak_bytes'),elapsed_base_seconds=inner.get('elapsed_wall_s'))
            need(code==0,'original full replay failed; inspect retained base evidence')
            result['case']=full_result_ok(inner,recording,coverage)
            comparison=routes.finish(plan['route_events']['sha256'])
            for k,expected in (('calls','expected_route_calls'),('actual_compiled_invocations','expected_compiled_calls'),('fallbacks','expected_route_fallbacks')):
                same(comparison[k],plan[expected],'complete actual route counts: '+k)
            result.update(route_comparison=comparison,steps=plan['expected_steps'],total_requests=16)
        validate(plan)
        result.update(status='PASS',replay_numerical_checks_passed=True,source_input_pins_rechecked=True)
    except BaseException as exc:
        result.update(status='FAIL',error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
    finally:
        if (out/'recording.json').is_file():result['recording_sha256']=sha(out/'recording.json')
        if (out/'replay-route-events.jsonl').is_file():result['replay_route_events']=ref(out/'replay-route-events.jsonl')
        write(out/'complete.json',result)
    print(json.dumps(dict(status=result['status'],out=str(out))))
    return 0 if result['status']=='PASS' else 1


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);m=p.add_mutually_exclusive_group()
    m.add_argument('--prepare',action='store_true');m.add_argument('--run',action='store_true')
    p.add_argument('--capture-child',type=Path);p.add_argument('--spec-sha256');p.add_argument('--plan',type=Path)
    p.add_argument('--plan-sha256');p.add_argument('--out',type=Path);a=p.parse_args(argv)
    if a.prepare:
        need(a.capture_child and a.spec_sha256 and a.out and not a.plan,'actual closed capture/spec SHA/fresh output')
        plan=prepare(a.capture_child,a.spec_sha256);a.out.mkdir(parents=True,exist_ok=False);write(a.out/'plan.json',plan)
        print(json.dumps(dict(status='PREPARED_NOT_RUN',plan=str(a.out/'plan.json'),sha256=sha(a.out/'plan.json'))))
    elif a.run:
        need(a.plan and a.plan_sha256 and a.out and not a.capture_child and sha(a.plan)==a.plan_sha256,'explicit actual plan SHA/fresh output')
        return run(read(a.plan),a.out)
    else:
        need(not any((a.capture_child,a.spec_sha256,a.plan,a.plan_sha256,a.out)),'incomplete action arguments')
        print('DESCRIPTION ONLY: actual E052 captured history; complete original numerical replay; explicit prepare/run required.')
    return 0


if __name__=='__main__':raise SystemExit(main())
