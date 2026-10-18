"""Offline E024 four-phase route HTTP cycle audit; never a device/service run.

Requires a closed owner and RESTORED journal before reading phase evidence.
Consumes explicit SHA-bound independent C1/C8 numerical audits, without opening
their large steps ledgers. Runtime/source files are rehashed; model payloads are
not: their full-hash pre/post receipts and current file metadata are checked.
Restoration source pins remain UNSEALED until the real predecessor is archived.
No pending source or producer PASS can grant qualification. No performance grant.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NATIVE = HERE.parent / 'pareto_realistic'
ROUTE = HERE.parent / 'route_decode_graph'
PHASES = ('c1-capture', 'c1-replay', 'c8-capture', 'c8-replay')
OWNER_SHA = 'c279e9de954080894ed2b8f35f169cb30f485f93b08568991cd61a61e13bd4f9'
RESTORE_SHA = 'a51c4a838230fb5d1e3dbbc788dea12df5f78eb360006e657e47f19ea56d83a7'
CONTROLLER_SHA = 'e6bc72007043ed0376e98ea7f28ee152587294dec200953e93a5c7c736a90694'
LEGACY_SHA = '8d8aee0372497fcf591d53e4bc02b3d1e7eac7eb1d2bf639168a72802e2395bd'
NUMERICAL_SHA = '88f32e3b3f44206b48a1f581cbfe0bde0788060cbd2a0c5030a3ec472a0139fe'
JOB = 'gui/501/local.omlx.server'
BASE_PLAN_SHA = '21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb'
REPLAY_SHA = 'd3e2a4f76b7c384f6c6c986d0838b68227618502b0552e5cdf74f149029e39e7'


class Incomplete(Exception): pass

def need(ok, message):
    if not ok: raise ValueError(message)

def same(a, b, message):
    need(json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False), message)

def read(path):
    need(Path(path).suffix != '.jsonl', 'cycle audit does not open numerical/route ledgers')
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'duplicate JSON key'); result[key] = value
        return result
    def bad(value): raise ValueError('nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs, parse_constant=bad)

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while part := stream.read(1 << 20): digest.update(part)
    return digest.hexdigest()

def ref(path): return dict(path=str(path), sha256=sha(path))
def finite(x): return type(x) in (int, float) and math.isfinite(x)

def pinned(item):
    need(type(item) is dict and set(item) == {'path', 'sha256'}, 'exact absolute path/SHA reference')
    path = Path(item['path'])
    need(path.is_absolute() and path.resolve() == path and not path.is_symlink()
         and path.is_file() and sha(path) == item['sha256'], 'pinned file changed: ' + str(path))
    return path

def load(path, digest):
    need(sha(path) == digest, 'frozen audit support changed: ' + str(path))
    spec = importlib.util.spec_from_file_location('_route_cycle_' + path.stem, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

def sealed():
    if any(len(s) != 64 or any(c not in '0123456789abcdef' for c in s)
           for s in (CONTROLLER_SHA, RESTORE_SHA)):
        raise Incomplete('controller/restoration source pins UNSEALED; no cycle qualification')

def adapter_copies(folder, expected):
    """Exact real repository-relative copies; equal basenames cannot alias."""
    manifest = read(folder / 'verified-adapter-pins.json')
    need(set(manifest) == set(expected), 'exact adapter snapshot inventory')
    copies = set()
    for source, digest in expected.items():
        path = Path(source)
        need(path.is_absolute() and path.resolve() == path and not path.is_symlink()
             and path.is_relative_to(ROOT), 'canonical repository source path')
        relative = Path('frozen-adapters') / path.relative_to(ROOT)
        row = manifest[source]; target = folder / relative
        need(set(row) == {'sha256', 'bytes', 'copy'} and row['copy'] == str(relative)
             and row['sha256'] == digest and type(row['bytes']) is int and row['bytes'] >= 0,
             'adapter copy metadata/path')
        need(target.resolve().is_relative_to((folder / 'frozen-adapters').resolve())
             and target.is_file() and not target.is_symlink() and target.stat().st_size == row['bytes']
             and sha(target) == digest, 'frozen adapter bytes/SHA')
        copies.add(str(relative))
    need(copies == {str(p.relative_to(folder)) for p in (folder/'frozen-adapters').rglob('*') if p.is_file()},
         'unexpected adapter copy or basename collision')
    return len(copies)

def preparation_pins(owner, wrapper, numeric):
    """Original157 provenance + actual new snapshots; never fake a legacy launch."""
    expected = {**numeric['execution']['pins'], wrapper['qualification']['path']: BASE_PLAN_SHA}
    need(len(numeric['execution']['pins']) == 122, 'original122 execution pins')
    manifests = {}
    for name in ('fixtures', 'source_manifest', 'model_manifest'):
        item = numeric[name]; expected[item['path']] = item['sha256']
        path = owner/'preflight/frozen-source'/item['path'].lstrip('/')
        need(sha(path) == item['sha256'], 'frozen original manifest'); manifests[name] = read(path)
    core = Path(numeric['package_root'])/'tutorial_code/miniengine'
    for name, digest in manifests['source_manifest']['files'].items():
        need(not Path(name).is_absolute() and '..' not in Path(name).parts, 'core manifest traversal')
        expected[str(core/name)] = digest
    need(len(expected) == 157, 'exact original157 provenance')
    adapters = dict(wrapper['execution']['pins'])
    for item in (wrapper['cold_restart']['controller'], *wrapper['serving'].values()):
        need(item['path'] not in adapters or adapters[item['path']] == item['sha256'], 'conflicting adapter pin')
        adapters[item['path']] = item['sha256']
    hashed = {}; models = []; copy_counts = []
    for phase in ('preflight', 'postflight'):
        folder = owner/phase
        need(sha(folder/'plan.json') == BASE_PLAN_SHA, 'original preparation plan')
        verified = read(folder/'verified-pins.json'); need(set(verified) == set(expected), 'original157 recorded keys')
        for source, digest in expected.items():
            copied = Path(source).suffix in ('.py', '.json', '.md'); row = verified[source]
            need(set(row) == {'sha256', 'bytes', 'source_copied'} and row['sha256'] == digest
                 and row['source_copied'] is copied and type(row['bytes']) is int and row['bytes'] >= 0,
                 'original recorded source/runtime pin')
            path = folder/'frozen-source'/source.lstrip('/') if copied else Path(source)
            need(path.stat().st_size == row['bytes'], 'runtime/source byte size')
            if path not in hashed: hashed[path] = sha(path)
            need(hashed[path] == digest, 'independent runtime/source SHA')
        command = [numeric['execution']['python'], '-I', '-B', numeric['execution']['qualifier']['path'],
            '--plan', wrapper['qualification']['path'], '--plan-sha256', BASE_PLAN_SHA,
            '--out', str(folder/'qualification')]
        same(read(folder/'inherited-prepare.json'), dict(command=command, executed=False,
             environment=numeric['execution']['environment']), 'actual original prepare command was not executed')
        copy_counts.append(adapter_copies(folder, adapters))
        model = manifests['model_manifest']; pins = read(folder/'verified-model-pins.json')
        wanted = {str(Path(numeric['model'])/n):h for n,h in model['files'].items()}
        need(set(pins) == set(wanted) and model['schema'] == 'E023-model-v1'
             and model['model'] == numeric['model'], 'complete model manifest receipts')
        for path, digest in wanted.items():
            row = pins[path]
            need(set(row) == {'sha256','bytes','inode','mtime_ns'} and row['sha256'] == digest
                 and all(type(row[k]) is int and row[k] > 0 for k in ('bytes','inode','mtime_ns')), 'full model hash/stat receipt')
        models.append(pins)
    same(models[0], models[1], 'model receipts changed across cycle')
    for path, row in models[1].items():
        st = Path(path).stat()
        need((st.st_size,st.st_ino,st.st_mtime_ns) == (row['bytes'],row['inode'],row['mtime_ns']), 'current model metadata')
    return dict(preparation_directories=2, verified_entries_each=157, execution_pins=122,
        adapter_copies_each=copy_counts, independent_files_hashed=len(hashed), model_files=len(models[0]),
        model_shards_rehashed=False, model_scope='Source-pinned full payload hash receipts before/after, matching manifest and current stat; no independent model payload rehash.')

def sample_names(names, prefix, count):
    """Membership only; resource contents are still read in numeric range order."""
    expected = {f'{prefix}-{i:03d}.json' for i in range(count)}
    need(len(names) == count and set(names) == expected, 'resource/absence file sequence')


def absence(row,pid):
    commands=[(['/bin/ps','-p',str(pid),'-o','pid='],1),(['/bin/launchctl','print',JOB],113),
        (['/usr/sbin/lsof','-nP','-a','-iTCP:8001','-sTCP:LISTEN','-Fpn'],1)]
    need(row['status']=='PASS' and len(row['checks'])==3,'original service absence coverage')
    for actual,(command,code) in zip(row['checks'],commands):
        same(actual,dict(command=command,returncode=code,stdout_sha256=hashlib.sha256(b'').hexdigest(),
            stdout_bytes=0,absent=True),'exact current-original absence code/output hash')



# Exact frozen 8d8aee0372497fcf591d53e4bc02b3d1e7eac7eb1d2bf639168a72802e2395bd
# resources() body, with only the filename-list check replaced by sample_names.
# This fixes lexicographic ordering at 1001 samples; numeric reads and every
# raw resource/phase/lifetime gate are unchanged. No imported globals patched.
def resources(owner,complete,children,original_pid,created):
    count=complete['samples'];need(type(count) is int and count>=4 and complete['absence_probes']==count,'resource sample count')
    for prefix in ('resource','absence'):
        sample_names([p.name for p in owner.glob(prefix+'-*.json')], prefix, count)
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



def numerical_binding(reference, target, plan, plan_sha, C, pid):
    audit = read(pinned(reference))
    need(audit['schema'] == 'E024-recorded-route-http-independent-audit-v2' and audit['status'] == 'PASS'
         and audit['qualified'] is audit['http_history_qualified'] is True
         and audit['audit_source_sha256'] == NUMERICAL_SHA and audit['concurrency'] == C
         and audit['plan_sha256'] == plan_sha and audit['base_plan_sha256'] == BASE_PLAN_SHA
         and audit['child_directory'] == str(target), 'actual route numerical audit binding')
    terminal = {name:sha(target/name) for name in ('plan.json','complete.json','base/plan.json','base/complete.json','recording.json')}
    same(audit['terminal_sha256'], terminal, 'independent numerical terminal SHA set')
    for left,right in (('sources','sources'),('raw_inputs','inputs'),('capture','capture'),('serving_prerequisite','serving')):
        same(audit[left], plan[right], 'independent numerical binding: ' + left)
    need(audit['ledger']['requests'] == (8 if C == 1 else 16) and audit['routes']['replay_pid'] == pid
         and audit['http']['max_http_outstanding'] == C and audit['http']['fully_drained'] is True
         and audit['performance_qualified'] is audit['owner_resources_restore_qualified'] is False
         and audit['offline_tensor_bytes_rechecked'] is audit['weight_shards_rehashed'] is False
         and audit['original_37_request_gate_replaced'] is False, 'recorded HTTP coverage/scope')
    return {**reference, 'requests':audit['ledger']['requests'], 'ledger_sha256':audit['ledger']['sha256'],
        'route_calls':audit['routes']['calls'], 'route_events':audit['routes']['captured']['sha256'],
        'ledger_binding_scope':'Explicit independently pinned numerical audit; this cycle audit does not reread the steps ledger.'}

def phases(owner, wrapper, numeric, specs, complete, numerical, producer, raw_audit):
    index = read(owner/'children.json')['children']; expected = list(PHASES)
    need([r['phase'] for r in index] == [r['phase'] for r in complete['phases']] == expected, 'all four observed/completed phases')
    same(read(owner/'capture-specs.json'), specs, 'actual C1/C8 source-bound capture specs')
    children = []; audits = {}
    for i,row in enumerate(index):
        stage = row['phase']; folder = owner/'phases'/stage; C = 1 if i < 2 else 8
        child, release = read(folder/'child.json'), read(folder/'release.json')
        need(type(child['pid']) is int and child['pid'] > 1 and child['identity'] is not None
             and child['identity']['pid'] == child['pid'] and child['owner_pid'] == child['identity']['ppid'] == complete['owner_pid'],
             'actual direct child identity required for qualification')
        need(release['pid'] == child['pid'] and release['waited_reaped'] is release['strict_absence'] is True
             and finite(child['spawned_at']) and finite(release['time']) and child['spawned_at'] <= release['time']
             and (not children or children[-1]['released_at'] <= child['spawned_at']), 'sequential strict phase reap')
        fact = dict(phase=stage,pid=child['pid'],spawned_at=child['spawned_at'],released_at=release['time'],returncode=release['returncode'])
        item = complete['phases'][i]; need(item['status'] == 'PASS', 'failed phase cannot qualify')
        target = folder/('capture' if stage.endswith('capture') else 'qualification')
        same(item['result'], dict(path=str((target/'complete.json').relative_to(owner)),sha256=sha(target/'complete.json')), 'phase terminal reference')
        result = read(target/'complete.json'); need(result['status'] == 'PASS', 'non-PASS child')
        replay_folder = owner/'phases'/f'c{C}-replay'; plan_path = replay_folder/'plan.json'; plan_sha = sha(plan_path)
        plan, replay_numeric, _, _ = producer.validate(plan_path, plan_sha)
        same(replay_numeric, numeric, 'same original numerical plan')
        same(plan['qualification'], wrapper['qualification'], 'original plan reference')
        same(plan['serving'], wrapper['serving'], 'same real37/42 serving prerequisite')
        captured = owner/'phases'/f'c{C}-capture/capture'
        same(plan['capture'], {k:ref(captured/n) for k,n in producer.CAPTURE_NAMES.items()}, 'actual closed capture references')
        same(plan['inputs'], {k:ref(captured/n) for k,n in producer.INPUT_NAMES.items()}, 'actual group/trace/route-event references')
        need(plan['concurrency'] == C and plan['qualifier_sha256'] == REPLAY_SHA, 'actual route replay identity')
        if stage.endswith('capture'):
            spec, header = raw_audit.capture_binding(producer, plan)
            same(spec, specs[i//2], 'canonical route capture specification')
            same(read(folder/'launch-context.json'), dict(cwd=str(ROOT),spec_sources=spec['sources'],environment=spec['environment']), 'actual direct capture cwd/source/environment')
            need(header['pid'] == child['pid'] and result['schema'] == 'E024-route-http-capture-result-v1'
                 and result['errors'] == [] and all(result[k] is True for k in ('capture_complete','cleanup_complete',
                 'server_reaped','cleanup_task_resolved','client_task_resolved','route_events_retained','small_pins_rechecked')),
                 'complete route capture/cleanup required')
            inner = read(target/'release.json')
            need(inner['pid'] == child['pid'] and inner['returncode'] == release['returncode']
                 and child['spawned_at'] <= inner['time'] <= release['time'], 'same inner/outer capture PID/exit')
            identity = read(target/'child-identity.json')
            need(identity['stdout'].split()[:2] == [str(child['pid']),str(complete['owner_pid'])], 'raw capture PPID')
            fact['inner_released_at'] = inner['time']
        else:
            command = [numeric['execution']['python'],'-I','-B',str(ROUTE/'http_qualification_v2.py'),'--run',
                '--plan',str(plan_path),'--plan-sha256',plan_sha,'--out',str(target)]
            same(read(folder/'launch.json'),dict(command=command,environment=wrapper['http']['environment'],cwd=str(ROOT)), 'real replay --run CLI/cwd/environment')
            need(release['returncode'] == 0 and result['schema'] == 'E024-recorded-route-http-result-v1'
                 and result['replay_numerical_checks_passed'] is result['input_source_pins_rechecked'] is True
                 and result['numerical_qualified'] is result['http_qualified'] is result['performance_qualified'] is False
                 and result['independent_audit_required'] is True and result['concurrency'] == C
                 and result['total_requests'] == (8 if C == 1 else 16) and result['plan_sha256'] == plan_sha,
                 'real replay terminal contract; producer is not independent qualification')
            for key in ('capture','inputs','sources','serving'): same(result[key], plan[key], 'replay terminal binding: '+key)
            audits[str(C)] = numerical_binding(numerical[str(C)],target,plan,plan_sha,C,child['pid'])
        children.append(fact)
    need(len({c['pid'] for c in children}) == 4, 'four distinct actual direct children')
    return children, audits

def audit(owner, journal, plan_sha256, numerical):
    owner,journal = Path(owner).resolve(),Path(journal).resolve()
    result = dict(schema='E024-route-http-cycle-independent-audit-v1',status='INCOMPLETE',qualified=False,
        http_history_qualified=False,independent_resource_verified=False,restoration_verified=False,
        performance_qualified=False,ledger_opened=False,offline_tensor_bytes_rechecked=False,
        audit_source_sha256=sha(__file__),owner_directory=str(owner))
    try:
        data = read(journal/'journal.json')
        if data.get('state') != 'RESTORED': raise Incomplete('journal not terminal RESTORED; owner/ledger unopened')
        complete = read(owner/'complete.json')
        if complete.get('status') not in ('PASS','FAIL'): raise Incomplete('owner not terminal; ledger unopened')
        sealed()
        need(sha(HERE/'controller.py') == CONTROLLER_SHA, 'actual restoration controller source')
        restore = load(HERE/'audit_restore.py',RESTORE_SHA)
        restoration = restore.audit(journal); result['restoration'] = restoration
        if restoration.get('status') == 'INCOMPLETE': raise Incomplete('restoration terminal evidence incomplete')
        need(restoration['status'] == 'PASS' and restoration['qualified'] is True, 'independent restoration/reap')
        result['restoration_verified'] = True
        if complete['status'] != 'PASS':
            result.update(status='FAIL',producer_errors=complete.get('errors')); return result
        need(data['schema'] == 'E024-route-http-cold-journal-v1' and data['owner_directory'] == str(owner)
             and data['owner_plan']['sha256'] == plan_sha256 == sha(owner/'owner-plan.json')
             and data['owner_source'] == dict(path=str(HERE/'owner.py'),sha256=OWNER_SHA), 'actual owner/journal binding')
        owned = load(HERE/'owner.py',OWNER_SHA)
        _,wrapper,_,numeric,specs = owned.validate_plan(SimpleNamespace(plan=owner/'owner-plan.json',plan_sha256=plan_sha256,out=owner))
        need(complete['schema'] == 'E024-route-http-owner-result-v1' and complete['owner_plan_sha256'] == plan_sha256
             and complete['errors'] == [] and complete['first_failure'] is None and complete['numerical_replay_complete'] is True
             and complete['all_children_reaped'] is True and complete['independently_qualified'] is complete['performance_qualified'] is False,
             'successful full owner terminal contract')
        need(set(numerical) == {'1','8'}, 'two explicit independent numerical audit references')
        stable = {p:sha(p) for p in (journal/'journal.json',owner/'complete.json',owner/'owner-plan.json')}
        result['pins'] = preparation_pins(owner,wrapper,numeric)
        raw_audit = load(ROUTE/'http_audit_v2.py',NUMERICAL_SHA)
        producer = load(ROUTE/'http_qualification_v2.py',REPLAY_SHA)
        children,audits = phases(owner,wrapper,numeric,specs,complete,numerical,producer,raw_audit)
        legacy = load(NATIVE/'audit_http_owner_v2.py',LEGACY_SHA)
        need(tuple(legacy.PHASES) == PHASES, 'same frozen four-phase resource contract')
        result['resources'] = resources(owner,complete,children,data['original']['pid'],read(owner/'attempt.json')['created_at'])
        need(all(sha(p) == h for p,h in stable.items()) and restoration['journal_sha256'] == stable[journal/'journal.json']
             and restoration['owner_plan_sha256'] == plan_sha256, 'terminal evidence changed')
        sources = {**wrapper['execution']['pins'], str(HERE/'controller.py'):CONTROLLER_SHA,
            str(HERE/'audit_restore.py'):RESTORE_SHA, str(ROUTE/'http_audit_v2.py'):NUMERICAL_SHA,
            str(ROUTE/'http_qualification_v2.py'):REPLAY_SHA, str(NATIVE/'audit_http_owner_v2.py'):LEGACY_SHA,
            str(Path(__file__).resolve()):result['audit_source_sha256']}
        for source,digest in sources.items(): need(sha(source) == digest, 'source changed during cycle audit')
        for item in numerical.values(): pinned(item)
        result.update(status='PASS',qualified=True,http_history_qualified=True,independent_resource_verified=True,
            owner_plan_sha256=plan_sha256,owner_complete_sha256=stable[owner/'complete.json'],journal_sha256=restoration['journal_sha256'],
            numerical_audits=audits,children=children,resource_algorithm_sha256=LEGACY_SHA,resource_adapter_delta='filename membership set+count only; original numeric order/raw gates unchanged',
            source_pins=sources,
            scope='Only actual C1x8/C8x16 captured route-ON histories, source/runtime provenance, observed resources and restoration. No future arrivals, full arithmetic reexecution, formal timing or Pareto grant.')
    except (Incomplete,FileNotFoundError) as exc: result.update(status='INCOMPLETE',reason=str(exc))
    except Exception as exc: result.update(status='FAIL',first_failure=dict(type=type(exc).__name__,message=str(exc)))
    return result

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('owner','journal','plan-sha256','c1-audit','c1-audit-sha256','c8-audit','c8-audit-sha256','out'):
        parser.add_argument('--'+name,required=True)
    args = parser.parse_args(argv); out = Path(args.out)
    need(out.parent.is_dir() and not out.exists(), 'new audit output required')
    numerical = {str(C):dict(path=str(Path(getattr(args,f'c{C}_audit')).resolve()),sha256=getattr(args,f'c{C}_audit_sha256')) for C in (1,8)}
    result = audit(args.owner,args.journal,args.plan_sha256,numerical)
    with out.open('x') as stream: json.dump(result,stream,indent=2,allow_nan=False); stream.write('\n')
    return {'PASS':0,'FAIL':1,'INCOMPLETE':2}[result['status']]

if __name__ == '__main__': raise SystemExit(main())
