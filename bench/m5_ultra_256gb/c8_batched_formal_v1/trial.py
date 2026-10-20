"""Default-OFF C8 paired timing; unchanged native 8-warm/64-formal lifecycle.

Only entry selection and actual E053 history binding are new. Observers are OFF.
Full IDs remain available; changed cross-process greedy paths are comparisons,
not automatically arithmetic failures or permission to claim a speed frontier.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import FunctionType, SimpleNamespace

HERE=Path(__file__).resolve().parent;B=HERE.parent
FORMAL=B/'gdn_graph_combined_formal/formal.py';NATIVE=B/'pareto_realistic/native_formal.py'
SERVER=B/'c8_batched_http_v1/server.py';AUDIT=B/'c8_batched_http_replay_analysis_v1/audit.py'
PRODUCER=B/'c8_batched_http_replay_v1/qualify.py'
FIXED={FORMAL:'4b567f8ed6090175f3639aa109584ca26c1426165c03de3167c3ebde74891a89',
 NATIVE:'ec9cdc924f7987ad46eac6820b6264495c95fb182c726545a6e3ce283df7e654',
 SERVER:'a3737f1b364e190e3fe45a1779e1be228c4f434701bf8875dbe18f1d519afc7b',
 AUDIT:'1576466a27c3e5463401ce19c14c9c4cb936a342cf920e90dc232c364699e4ec',
 PRODUCER:'c0a73e649718586365b42e6fcb67e4d0fbdb309f2238526b798952a9bfd18ad5'}
SCHEMA='E055-c8-batched-formal-spec-v1'
MLX={'MLX_MAX_OPS_PER_BUFFER':'50','MLX_MAX_MB_PER_BUFFER':'50','MLX_BFS_MAX_WIDTH':'20','MLX_METAL_FAST_SYNCH':'0'}
TRIALS=('r1-c8-on','r1-c8-off','r2-c8-off','r2-c8-on','r3-c8-on','r3-c8-off')
SCOPE=('Same frozen C8 entry, graph ON, GDN OFF/ON; original8warm/64formal. E053 qualifies only its '
       'captured history. New output comparisons, independent timing/resources/restoration remain required; '
       'different cross-schedule greedy IDs alone do not establish a candidate arithmetic error.')


def need(ok,message):
    if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return module(FORMAL).read(path)
def same(a,b,message):return module(FORMAL).same(a,b,message)
def pinned(ref):return module(FORMAL).pinned(ref)
def module(path):
    need(path in FIXED and sha(path)==FIXED[path],'changed frozen source: '+str(path))
    spec=importlib.util.spec_from_file_location('_e055_'+path.parent.name+'_'+path.stem,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def source_pins():
    pins={**module(FORMAL).source_pins(),**module(SERVER).source_pins(),**{str(p):h for p,h in FIXED.items()},
          str(Path(__file__).resolve()):sha(__file__)}
    need(all(sha(p)==h for p,h in pins.items()),'complete formal source closure changed');return pins


def identity(base,arm,server,pins,*,qualification=False):
    value=copy.deepcopy(base['identity']);value.update(step_observer=qualification,qualification_trace_enabled=qualification)
    value.update(server.identity_fields(SimpleNamespace(phase='qualification' if qualification else 'formal',
        route_decode_graph='on',gdn_prework=arm),pins,installed=True,route_observer=qualification))
    return value


def bind_history(binding,base,server,pins):
    need(set(binding)=={'audit','replay_plan','child_directory'},'explicit actual E053 audit/plan/child refs')
    proof=read(pinned(binding['audit']));path=pinned(binding['replay_plan']);child=Path(binding['child_directory'])
    need(child.is_absolute() and child.resolve()==child and not child.is_symlink() and path==child/'plan.json','canonical actual replay child')
    need(proof['schema']=='E053-independent-http-replay-audit-v1' and proof['status']=='PASS'
         and proof['audit_source_sha256']==FIXED[AUDIT] and proof['child']==str(child)
         and proof['plan_sha256']==binding['replay_plan']['sha256']
         and all(proof[k] is True for k in ('qualified','recorded_history_verified','http_history_qualified','candidate_http_capture_qualified'))
         and all(proof[k] is False for k in ('performance_qualified','owner_resources_restore_qualified','offline_tensor_bytes_rechecked')),
         'actual E053 independent numerical PASS with bounded scope required')
    names=('plan.json','complete.json','base/plan.json','base/complete.json','recording.json')
    same(proof['terminal_sha256'],{str(child/n):sha(child/n) for n in names},'closed E053 terminal bytes')
    plan=read(path);top=read(child/'complete.json');capture=read(pinned(plan['capture_spec']))
    need(top['status']=='PASS' and top['plan_sha256']==binding['replay_plan']['sha256']
         and top['source_input_pins_rechecked'] is top['replay_numerical_checks_passed'] is True,'closed replay producer')
    for p in (PRODUCER,SERVER):
        need(plan['sources'][str(p)]==proof['actual_candidate_sources'][str(p)]==FIXED[p],'actual candidate source chain')
    need(plan['schema']=='c8-batched-http-replay-plan-v1' and capture['schema']=='c8-batched-http-spec-v2'
         and capture['concurrency']==8 and capture['wave_names']==['warm','screen'],'actual new C8 history')
    same(plan['qualification'],base['base_plan'],'same original source/model/config plan')
    same(capture['fixtures'],base['fixtures'],'same full prompt fixtures/caps')
    same(capture['environment'],base['environment'],'same qualified environment')
    same(plan['environment'],MLX,'same four MLX values')
    same({k:capture['transport'][k] for k in ('tcp_nodelay','io_timeout_seconds')},
         {k:base['transport'][k] for k in ('tcp_nodelay','io_timeout_seconds')},'same socket contract')
    same(capture['identity'],identity(base,'on',server,pins,qualification=True),'same captured model/entry/config identity')


def prepare(base_spec_ref,arm,label,numerical_binding):
    formal=module(FORMAL);base=formal.read(formal.pinned(base_spec_ref));formal.validate(base)
    need(base['concurrency']==8 and base['arm']=='off' and type(base['repeat']) is int and 1<=base['repeat']<=3,'original C8 OFF repeat')
    need(arm in ('off','on') and label==f'r{base["repeat"]}-c8-{arm}' and label in TRIALS,'exact unique registered arm/repeat label')
    same({k:v for k,v in base['environment'].items() if k.startswith('MLX_')},MLX,'fixed50/50/20/0 environment')
    need(len(base['jobs']['warm'])==8 and len(base['jobs']['formal'])==64,'original8/64 queue')
    command=copy.deepcopy(base['command'])
    need(command[3]==str(formal.SERVER) and '--run' not in command and all(flag not in command for flag in
         ('--qualification-trace','--route-events','--dispatch-events')),'original observer-OFF combined entry')
    for flag,value in (('--phase','formal'),('--route-decode-graph','on'),('--gdn-prework','off')):
        need(command.count(flag)==1 and command[command.index(flag)+1]==value,'original formal arm/config')
    server=module(SERVER);pins=server.source_pins();bind_history(numerical_binding,base,server,pins)
    command[3]=str(SERVER);command.insert(4,'--run');command[command.index('--gdn-prework')+1]=arm
    result=copy.deepcopy(base);result.update(schema=SCHEMA,base_spec=copy.deepcopy(base_spec_ref),arm=arm,label=label,phase=label,
        command=command,identity=identity(base,arm,server,pins),sources=source_pins(),numerical_binding=copy.deepcopy(numerical_binding),
        numerical_qualified=False,full_state_qualified=False,performance_qualified=False,observer_enabled=False,
        qualification_scope='Inherited qualification is the original wire/count contract; explicit E053 binding is history-only.',scope=SCOPE)
    return result


def validate(spec):
    need(spec.get('schema')==SCHEMA,'new paired formal specification')
    same(spec,prepare(spec['base_spec'],spec['arm'],spec['label'],spec['numerical_binding']),'changed paired formal specification')


def retained_ids(native,group):
    join=native.load('http_join');counts={};rows=[]
    for row in group['requests']:
        fixture=row['fixture_id'];index=counts.get(fixture,0);counts[fixture]=index+1
        rid,terminal,_=join.raw_completion(row)
        rows.append(dict(fixture_id=fixture,occurrence=index,request_id=row['request_id'],response_id=rid,
            raw_ids=terminal['benchmark_output_ids'],committed_ids=terminal['benchmark_committed_output_ids'],
            terminal_eos_id=terminal['benchmark_terminal_eos_id']))
    return rows


async def run(spec,out,*,checkpoint,backend=None):
    native=module(NATIVE);fn=native.run
    call=FunctionType(fn.__code__,dict(fn.__globals__,validate=validate),fn.__name__,fn.__defaults__,fn.__closure__)
    call.__kwdefaults__=dict(fn.__kwdefaults__ or {})
    result=await call(spec,out,checkpoint=checkpoint,backend=backend)
    result.update(schema='E055-c8-batched-formal-result-v1',label=spec['label'],arm=spec['arm'],phase=spec['phase'],
        observer_enabled=False,numerical_qualified=False,full_state_qualified=False,performance_qualified=False,
        cross_schedule_token_comparison='PENDING_PAIRED_AUDIT',diagnostic_throughput_only=True,
        numerical_binding=spec['numerical_binding'],scope=SCOPE)
    if result['status']=='PASS':
        rows={phase:retained_ids(native,native.read(Path(out)/(phase+'-group.json'))) for phase in ('warm','formal')}
        native.write(Path(out)/'output-ids.json',rows)
        result['output_ids']=dict(path=str(Path(out)/'output-ids.json'),sha256=sha(Path(out)/'output-ids.json'))
        result['artifacts']['output-ids.json']=dict(sha256=result['output_ids']['sha256'],bytes=(Path(out)/'output-ids.json').stat().st_size)
    native.write(Path(out)/'complete.json',result);return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args(argv)
    print(json.dumps(dict(schema=SCHEMA,default_action='describe only',trials=TRIALS,warm=8,formal=64,
        observers=False,device_run=False,performance_qualified=False,scope=SCOPE)))
    return 0


if __name__=='__main__':raise SystemExit(main())
