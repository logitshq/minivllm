"""Offline GDN-only four-phase HTTP cycle binding; no model/service execution.

Run the frozen gdn_prework_http/audit.py separately once for each closed replay,
then supply both explicit result path/SHA refs here. This shell rechecks their
terminal/input/source bindings and calls that auditor's real capture binder.
It directly calls the frozen route audit's preparation_pins/resources functions;
no copied resource algorithm, spoofed source identity or altered globals.
Missing closure remains INCOMPLETE. Restored producer failures remain FAIL.
Successful original tensor bytes are not present: no offline arithmetic replay,
combined candidate, future-arrival, performance or global quality qualification.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
GDN = HERE.parent / 'gdn_prework_http'
PHASES = ('c1-capture', 'c1-replay', 'c8-capture', 'c8-replay')
OWNER_SHA = 'a4776f1125b9b8db82d83b0f2cebd78c57cf15894354d884d0e2cd07224a7ddd'
RESTORE_SHA = '40d07aabef5d3588b4eac57a2acadf4278eee2328a3304da2a041a234f66ff5e'
CONTROLLER_SHA = '04200ccde66bb6525f274282bd65338e222855201f38296bb46144abfc5372c5'
SUPPORT = HERE.parent / 'route_http_cycle/audit_cycle.py'
SUPPORT_SHA = 'bdcd32a06133bae37786eb148318a0d38847df2f2a20c94c984104394570cca4'
NUMERICAL_SHA = '5ccb357e1f6a1296ab8673a604c48f2e183d01d83671e8d7857edc85cded7cd0'
BASE_PLAN_SHA = '21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb'
REPLAY_SHA = '3dddc8d00b42e0ebe34883805f13a018325f83b232cf007f170cd01a27c18209'


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
    spec = importlib.util.spec_from_file_location('_gdn_cycle_' + path.stem, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

def sealed():
    if any(len(s) != 64 or any(c not in '0123456789abcdef' for c in s)
           for s in (OWNER_SHA, CONTROLLER_SHA, RESTORE_SHA)):
        raise Incomplete('controller/restoration source pins UNSEALED; no cycle qualification')

def numerical_binding(reference, target, plan, plan_sha, C, pid):
    if type(reference) is dict and 'path' in reference and not Path(reference['path']).exists():
        raise Incomplete('missing independent numerical audit: ' + str(reference['path']))
    audit = read(pinned(reference))
    need(audit['schema'] == 'GDN-recorded-http-independent-v1' and audit['status'] == 'PASS'
         and audit['qualified'] is audit['http_history_qualified'] is True
         and audit['audit_source_sha256'] == NUMERICAL_SHA and audit['concurrency'] == C
         and audit['plan_sha256'] == plan_sha and audit['base_plan_sha256'] == BASE_PLAN_SHA
         and audit['child_directory'] == str(target), 'actual GDN numerical audit binding')
    terminal = {name:sha(target/name) for name in ('plan.json','complete.json','base/plan.json','base/complete.json','recording.json')}
    same(audit['terminal_sha256'], terminal, 'independent numerical terminal SHA set')
    for left,right in (('sources','sources'),('raw_inputs','inputs'),('capture','capture'),('serving_prerequisite','serving')):
        same(audit[left], plan[right], 'independent numerical binding: ' + left)
    need(audit['ledger']['requests'] == (8 if C == 1 else 16) and audit['dispatch']['replay_pid'] == pid
         and audit['http']['max_http_outstanding'] == C and audit['http']['fully_drained'] is True
         and audit['performance_qualified'] is audit['owner_resources_restore_qualified'] is False
         and audit['offline_tensor_bytes_rechecked'] is audit['weight_shards_rehashed'] is False
         and audit['original_37_request_gate_replaced'] is False
         and audit['combined_candidates_qualified'] is audit['parameter_replacement_qualified'] is False,
         'recorded HTTP coverage/scope')
    same(audit['dispatch']['captured']['sha256'], plan['inputs']['dispatch_events']['sha256'],
         'independent actual captured dispatch bytes')
    return {**reference, 'concurrency':C, 'child_directory':str(target), 'plan_sha256':plan_sha,
        'requests':audit['ledger']['requests'], 'ledger_sha256':audit['ledger']['sha256'],
        'dispatch_calls':audit['dispatch']['calls'], 'dispatch_events':audit['dispatch']['captured']['sha256'],
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
        same(plan['serving'], wrapper['serving'], 'same real37 serving prerequisite')
        captured = owner/'phases'/f'c{C}-capture/capture'
        same(plan['capture'], {k:ref(captured/n) for k,n in producer.CAPTURE_NAMES.items()}, 'actual closed capture references')
        same(plan['inputs'], {k:ref(captured/n) for k,n in producer.INPUT_NAMES.items()}, 'actual group/trace/dispatch-event references')
        need(plan['concurrency'] == C and plan['qualifier_sha256'] == REPLAY_SHA, 'actual GDN replay identity')
        if stage.endswith('capture'):
            spec, header = raw_audit.capture_binding(producer, plan)
            same(spec, specs[i//2], 'canonical GDN capture specification')
            same(read(folder/'launch-context.json'), dict(cwd=str(ROOT),spec_sources=spec['sources'],environment=spec['environment']), 'actual direct capture cwd/source/environment')
            need(header['pid'] == child['pid'] and result['schema'] == 'GDN-prework-http-capture-result-v1'
                 and result['errors'] == [] and all(result[k] is True for k in ('capture_complete','cleanup_complete',
                 'server_reaped','cleanup_task_resolved','client_task_resolved','dispatch_events_retained','small_pins_rechecked')),
                 'complete GDN capture/cleanup required')
            inner = read(target/'release.json')
            need(inner['pid'] == child['pid'] and inner['returncode'] == release['returncode']
                 and child['spawned_at'] <= inner['time'] <= release['time'], 'same inner/outer capture PID/exit')
            identity = read(target/'child-identity.json')
            need(identity['stdout'].split()[:2] == [str(child['pid']),str(complete['owner_pid'])], 'raw capture PPID')
            fact['inner_released_at'] = inner['time']
        else:
            command = [numeric['execution']['python'],'-I','-B',str(GDN/'http_qualification.py'),'--run',
                '--plan',str(plan_path),'--plan-sha256',plan_sha,'--out',str(target)]
            same(read(folder/'launch.json'),dict(command=command,environment=wrapper['http']['environment'],cwd=str(ROOT)), 'real replay --run CLI/cwd/environment')
            need(release['returncode'] == 0 and result['schema'] == 'GDN-recorded-http-result-v1'
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
    result = dict(schema='GDN-prework-http-cycle-independent-v1',status='INCOMPLETE',qualified=False,
        http_history_qualified=False,independent_resource_verified=False,restoration_verified=False,
        performance_qualified=False,ledger_opened=False,offline_tensor_bytes_rechecked=False,combined_candidates_qualified=False,
        parameter_replacement_qualified=False,
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
        need(data['schema'] == 'GDN-prework-http-cold-journal-v1' and data['owner_directory'] == str(owner)
             and data['owner_plan']['sha256'] == plan_sha256 == sha(owner/'owner-plan.json')
             and data['owner_source'] == dict(path=str(HERE/'owner.py'),sha256=OWNER_SHA), 'actual owner/journal binding')
        owned = load(HERE/'owner.py',OWNER_SHA)
        _,wrapper,_,numeric,specs = owned.validate_plan(SimpleNamespace(plan=owner/'owner-plan.json',plan_sha256=plan_sha256,out=owner))
        need(complete['schema'] == 'GDN-prework-http-owner-result-v1' and complete['owner_plan_sha256'] == plan_sha256
             and complete['errors'] == [] and complete['first_failure'] is None and complete['numerical_replay_complete'] is True
             and complete['all_children_reaped'] is True and complete['independently_qualified'] is complete['performance_qualified'] is False,
             'successful full owner terminal contract')
        need(set(numerical) == {'1','8'}, 'two explicit independent numerical audit references')
        stable = {p:sha(p) for p in (journal/'journal.json',owner/'complete.json',owner/'owner-plan.json')}
        reused = load(SUPPORT, SUPPORT_SHA)
        need(tuple(reused.PHASES) == PHASES, 'same frozen four-phase resource contract')
        result['pins'] = reused.preparation_pins(owner,wrapper,numeric)
        raw_audit = load(GDN/'audit.py',NUMERICAL_SHA)
        producer = load(GDN/'http_qualification.py',REPLAY_SHA)
        children,audits = phases(owner,wrapper,numeric,specs,complete,numerical,producer,raw_audit)
        result['resources'] = reused.resources(owner,complete,children,data['original']['pid'],read(owner/'attempt.json')['created_at'])
        need(all(sha(p) == h for p,h in stable.items()) and restoration['journal_sha256'] == stable[journal/'journal.json']
             and restoration['owner_plan_sha256'] == plan_sha256, 'terminal evidence changed')
        sources = {**wrapper['execution']['pins'], str(HERE/'controller.py'):CONTROLLER_SHA,
            str(HERE/'audit_restore.py'):RESTORE_SHA, str(GDN/'audit.py'):NUMERICAL_SHA,
            str(GDN/'http_qualification.py'):REPLAY_SHA, str(SUPPORT):SUPPORT_SHA,
            str(Path(__file__).resolve()):result['audit_source_sha256']}
        for source,digest in sources.items(): need(sha(source) == digest, 'source changed during cycle audit')
        for item in numerical.values(): pinned(item)
        result.update(status='PASS',qualified=True,http_history_qualified=True,independent_resource_verified=True,
            owner_plan_sha256=plan_sha256,owner_complete_sha256=stable[owner/'complete.json'],journal_sha256=restoration['journal_sha256'],
            numerical_audits=audits,children=children,resource_algorithm_sha256=SUPPORT_SHA,resource_adapter_delta='None: direct call to frozen route audit resources/preparation_pins; no helper globals changed',
            source_pins=sources,
            scope='Only actual C1x8/C8x16 captured GDN-only histories, source/runtime provenance, observed resources and restoration. No future arrivals, full arithmetic reexecution, formal timing or Pareto grant.')
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
