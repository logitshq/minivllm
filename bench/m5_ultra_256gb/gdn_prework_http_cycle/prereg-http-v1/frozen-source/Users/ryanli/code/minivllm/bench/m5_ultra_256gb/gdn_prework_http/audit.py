"""Independent CLOSED GDN-only HTTP replay audit; never model/service work.

Rejoin actual SSE/UUID/prompt/trace using the frozen tokenizer contract, then
reuse the unchanged HTTP full-logit/state/ahead checker. Independent dispatch
expectations come from reconstructed semantic calls, excluding consumed ahead.
No successful tensor payloads are available for offline arithmetic recomputation.
PASS is this history only; resources/reap/restoration need the future outer audit.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
NATIVE=HERE.parent/'pareto_realistic'
PRODUCER=HERE/'http_qualification.py'
PRODUCER_SHA='3dddc8d00b42e0ebe34883805f13a018325f83b232cf007f170cd01a27c18209'
HTTP_AUDIT=NATIVE/'audit_http_replay_v2.py'
HTTP_AUDIT_SHA='7cdcf5a2f02768ef75686b0ff0b0da9d21f5f9c82ee3866e24c1f869c709f031'
BASE=NATIVE/'audit_qualification.py'
BASE_SHA='b6d90413911f2b17e0951a2fcfdb5fe00666aa06d4382b603e29e5dc4659d4f1'
DONOR=HERE.parent/'route_decode_graph/http_audit_v2.py'
DONOR_SHA='88f32e3b3f44206b48a1f581cbfe0bde0788060cbd2a0c5030a3ec472a0139fe'
BASE_PLAN_SHA='21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb'
SUPPORT={PRODUCER:PRODUCER_SHA,HTTP_AUDIT:HTTP_AUDIT_SHA,BASE:BASE_SHA,DONOR:DONOR_SHA}


class Incomplete(Exception): pass

def need(ok, message):
    if not ok: raise ValueError(message)

def same(a, b, message):
    need(json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False), message)

def parse(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'duplicate JSON key: ' + key); result[key] = value
        return result
    def bad(value): raise ValueError('nonfinite JSON constant: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)

def read(path):
    need(Path(path).suffix != '.jsonl', 'stream JSONL only')
    return parse(Path(path).read_bytes())

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while part := stream.read(1 << 20): digest.update(part)
    return digest.hexdigest()

def pinned(ref):
    need(type(ref) is dict and set(ref) == {'path', 'sha256'}, 'exact path/SHA reference')
    path = Path(ref['path'])
    need(path.is_absolute() and path.resolve() == path and not path.is_symlink() and path.is_file()
         and sha(path) == ref['sha256'], 'canonical pinned file: ' + str(path))
    return path

def load(path, expected):
    need(sha(path) == expected, 'frozen helper changed: ' + str(path))
    spec = importlib.util.spec_from_file_location('_gdn_http_audit_' + path.stem, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

def finite(x): return type(x) in (int, float) and math.isfinite(x) and x >= 0

def file_identity(path):
    s = Path(path).stat(); return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)


class DispatchCursor:
    def __init__(self,stream,header):
        self.stream=stream;self.digest=hashlib.sha256();self.bytes=self.lines=0
        self.counts=dict(candidate=0,fallback=0,b1_calls=0,multirow_calls=0,prefill_calls=0,short_bypass_calls=0)
        same(self.take(),header,'actual dispatch startup identity')
    def take(self):
        raw=self.stream.readline();need(raw and raw.endswith(b'\n'),'missing/truncated dispatch event')
        self.digest.update(raw);self.bytes+=len(raw);self.lines+=1;return parse(raw)
    def call(self,phase,step,key):
        mode=key[0]
        if mode=='decode':
            B=len(key[1]);used=max(key[2])+1
            need(B>0 and len(key[2])==len(key[3])==B,'actual decode semantic geometry')
            path='short-packed-bypass' if used<=256 else 'B1-candidate' if B==1 else 'B>1-fallback'
        else:
            need(mode=='prefill' and type(key[3]) is int and key[3]>0,'original ordinary prefill only')
            B=key[3];used=0;path='prefill-untouched'
        candidate=48 if path=='B1-candidate' else 0;fallback=48 if path=='B>1-fallback' else 0
        expected=dict(event='dispatch-call',status='SUBMITTED',phase=phase,step=step,mode=mode,B=B,used=used,
                      path=path,candidate=candidate,fallback=fallback)
        same(self.take(),expected,'independently derived actual GDN call/order/48-layer delta')
        self.counts['candidate']+=candidate;self.counts['fallback']+=fallback
        self.counts[{'B1-candidate':'b1_calls','B>1-fallback':'multirow_calls',
                     'prefill-untouched':'prefill_calls','short-packed-bypass':'short_bypass_calls'}[path]]+=1
    def finish(self):
        need(self.stream.read(1)==b'','unconsumed/extra dispatch event')
        return dict(sha256=self.digest.hexdigest(),bytes=self.bytes,lines=self.lines,counts=self.counts)


class DualDispatch:
    def __init__(self,captured,replay,header,ref):
        # Read/reseek only the small replay header, retain one streaming digest.
        pos=replay.tell();row=parse(replay.readline());replay.seek(pos)
        pid=row.get('pid');need(type(pid) is int and pid>1 and pid!=header['pid'],'separate replay PID')
        replay_header=dict(event='replay-dispatch-startup',schema='GDN-prework-http-replay-events-v1',pid=pid,
                           phase=header['phase'],captured_dispatch_events=ref)
        self.cursors=[DispatchCursor(captured,header),DispatchCursor(replay,replay_header)]
        self.pid=pid;self.calls=0
    def call(self,phase,step,key):
        for cursor in self.cursors:cursor.call(phase,step,key)
        self.calls+=1
    def finish(self,C):
        rows=[c.finish() for c in self.cursors];same(rows[0]['counts'],rows[1]['counts'],'two actual dispatch streams')
        c=rows[0]['counts'];need(self.calls>0,'actual model invocations required')
        if C==1:need(c['b1_calls']>0 and c['multirow_calls']==0,'actual B1 candidate domain')
        else:need(c['multirow_calls']>0,'actual C8 multirow fallback domain; candidate hit not required')
        return dict(captured=rows[0],replay=rows[1],calls=self.calls,replay_pid=self.pid,
                    scope='Independent semantic call order/context/counters; no token values in sidecar, no Metal completion count.')


def dispatch_case_type(base,dispatch,coverage):
    class GDNCase(base.CaseAudit):
        def check_arrays(self,row,calls,pre_staging,has_future,B):
            super().check_arrays(row,calls,pre_staging,has_future,B)
            consumed=row['ahead']['candidate']['consumed_prior_future']
            for key in calls:
                if key!=consumed:dispatch.call(self.name,self.step,key)
            c=coverage.setdefault(self.name,dict(consumed=0,discarded_prior=0,partial_retirement=0))
            c['consumed']+=int(consumed is not None)
            c['discarded_prior']+=int(row['ahead']['candidate']['discarded_prior_future'] is not None)
            after=row['after']['engine']['batch_rids'];decode=row['schedule']['decode']
            c['partial_retirement']+=int(bool(after) and bool(set(decode)-set(after)))
    return GDNCase


def expected_coverage(C,counts,calls):
    return dict(enabled=True,layer_installs=48,layer_restores=48,observer_restores=2,calls=calls,
        per_case={f'D-C{C}':counts},final_counters=[[dict(candidate=counts['b1_calls'],fallback=counts['multirow_calls']) for _ in range(48)]],
        identity_refill_enabled=False,route_graph_enabled=False)

def capture_binding(producer, plan):
    spec, header = producer.capture_gate(plan)  # Real factory + fixed original receipt schema.
    directory = Path(plan['capture']['spec']['path']).parent
    complete = read(directory/'complete.json'); release = read(directory/'release.json'); pid = header['pid']
    # The actual frozen capture inventories top-level files, not a recursive
    # archive. Its group.json holds every original raw SSE chunk used below.
    expected_files = {p.name for p in directory.iterdir() if p.is_file() and p.name != 'complete.json'}
    need(set(complete['artifacts']) == expected_files, 'complete closed top-level capture inventory')
    for rel, row in complete['artifacts'].items():
        path = directory/rel
        need(not Path(rel).is_absolute() and '..' not in Path(rel).parts and not path.is_symlink()
             and path.resolve().is_relative_to(directory.resolve()), 'capture artifact stays inside directory')
        same(row, dict(sha256=sha(path), bytes=path.stat().st_size), 'capture artifact bytes/SHA: ' + rel)
    same(read(directory/'launch.json'), dict(command=spec['command']+['--qualification-trace',str(directory/'trace.jsonl'),
         '--dispatch-events',str(directory/'dispatch-events.jsonl')], environment=spec['environment']), 'actual capture CLI/environment')
    port = spec['transport']['port']
    listener_command = ['/usr/sbin/lsof','-nP','-a']
    port_arguments = [f'-iTCP:{port}','-sTCP:LISTEN','-Fpn']
    commands = {
        'child-identity.json': ['/bin/ps','-p',str(pid),'-o','pid=,ppid=,lstart=,comm='],
        'child-absence.json': ['/bin/ps','-p',str(pid),'-o','pid='],
        'listener.json': listener_command+['-p',str(pid)]+port_arguments,
        'port-before.json': listener_command+port_arguments,
        'port-after.json': listener_command+port_arguments,
    }
    for name, command in commands.items():
        same(read(directory/name)['command'], command, 'actual PID/port probe command: '+name)
    identity = read(directory/'child-identity.json'); fields = identity['stdout'].split()
    need(type(identity['returncode']) is int and identity['returncode'] == 0 and len(fields) >= 2 and fields[0] == str(pid)
         and fields[1].isdigit() and int(fields[1]) > 1, 'actual capture process identity receipt')
    capture = producer.load(HERE/'http_capture.py', '_gdn_independent_capture_receipts')
    need(capture.no_process(read(directory/'port-before.json')), 'registered capture port initially absent')
    for name in ('ready.json', 'ready-after-client.json'):
        need(capture.ready_identity(read(directory/name), spec['identity']), 'actual GDN server readiness identity')
    need(capture.owned_listener(read(directory/'listener.json'), pid, spec['transport']['port']), 'actual owned listener')
    events = []
    with (directory/'events.jsonl').open('rb') as stream:
        for raw in stream:
            need(raw.endswith(b'\n'), 'truncated capture event'); events.append(parse(raw))
    ownterm = any(r.get('label') == 'server-term' and r.get('pid') == pid and finite(r.get('time'))
                  and r['time'] <= release['time'] for r in events)
    need(release['returncode'] == 0 or (release['returncode'] == -15 and ownterm), 'normal or own explained TERM only')
    need(not any(r.get('label') == 'server-kill' for r in events), 'KILL cannot qualify captured HTTP')
    return spec, header

def binding(child, expected_sha, top, claimed, producer, http, donor):
    need(sha(child/'plan.json') == expected_sha == top['plan_sha256'], 'explicit actual replay plan SHA')
    plan, numeric, _, _ = producer.validate(child/'plan.json', expected_sha)
    need(plan['qualifier_sha256'] == PRODUCER_SHA and plan['qualification']['sha256'] == BASE_PLAN_SHA
         and sha(child/'base/plan.json') == BASE_PLAN_SHA, 'actual qualifier/original numeric plan')
    producer.serving_gate(plan['serving'])
    C = plan['concurrency']; count = 8 if C == 1 else 16
    need(top['schema'] == 'GDN-recorded-http-result-v1' and top['prepare_only'] is False
         and top['replay_numerical_checks_passed'] is top['input_source_pins_rechecked'] is top['independent_audit_required'] is True
         and top['numerical_qualified'] is top['http_qualified'] is top['performance_qualified'] is top['owner_resources_restore_qualified'] is False
         and top['concurrency'] == C and type(top['total_requests']) is int and top['total_requests'] == count, 'new producer terminal scope')
    for name in ('sources', 'inputs', 'capture', 'serving'): same(top[name], plan[name], 'terminal binding: '+name)
    same(top['base_result'], dict(path='base/complete.json',sha256=sha(child/'base/complete.json'),returncode=0,status='PASS'), 'original numerical terminal bytes')
    need(claimed['plan_sha256'] == BASE_PLAN_SHA and claimed['qualifier_sha256'] == producer.FIXED[producer.BASE]
         and claimed['numeric_only_not_http'] is claimed['source_model_pins_rechecked'] is True
         and type(claimed['total_requests']) is int and claimed['total_requests'] == count
         and len(claimed['cases']) == 1 and claimed['cases'][0]['name'] == f'HTTP-C{C}', 'one-C base completed domain')
    same(claimed['candidate'], numeric['candidate'], 'actual candidate settings')
    spec, header = capture_binding(producer, plan)
    fixtures, config, tokenizer, token_pins = donor.tokenizer_and_manifests(numeric, http)
    jobs = producer.load(NATIVE/'http_qualification.py', '_gdn_audit_jobs').jobs_for(fixtures, C)
    same(spec['jobs'], jobs, 'exact HTTP8/16 original body jobs')
    group = read(pinned(plan['inputs']['group'])); timing = donor.http_intervals(group, spec)
    join = producer.load(NATIVE/'http_join.py', '_gdn_audit_join')
    with pinned(plan['inputs']['trace']).open('rb') as trace:
        recording = join.join(group, trace, jobs, {f['fixture_id']:f for f in fixtures}, spec['identity'],
                              lambda ids:tokenizer.decode(ids, skip_special_tokens=False))
    need(sha(child/'recording.json') == top['recording_sha256'], 'recording bytes SHA')
    same(read(child/'recording.json'), recording, 'independently joined raw SSE/trace versus recording')
    same(top['replay_dispatch_events'], dict(path='replay-dispatch-events.jsonl', sha256=sha(child/'replay-dispatch-events.jsonl')), 'actual emitted GDN dispatch sidecar')
    return plan, recording, fixtures, config, token_pins, timing, header


def audit(child_dir,expected_plan_sha256):
    child=Path(child_dir).resolve()
    result=dict(schema='GDN-recorded-http-independent-v1',status='INCOMPLETE',qualified=False,
        child_directory=str(child),ledger_opened=False,http_history_qualified=False,performance_qualified=False,
        owner_resources_restore_qualified=False,offline_tensor_bytes_rechecked=False,weight_shards_rehashed=False,
        combined_candidates_qualified=False,parameter_replacement_qualified=False,
        audit_source_sha256=sha(__file__),support_sources={str(p):h for p,h in SUPPORT.items()},
        scope='Actual captured GDN-only HTTP history, source-pinned full64 device bit/finite receipts; no future workload/performance/composition grant.')
    try:
        top=read(child/'complete.json')
        if top.get('status') in ('INCOMPLETE','PREPARED'):raise Incomplete('replay not terminal; ledger unopened')
        claimed=read(child/'base/complete.json') if (child/'base/complete.json').is_file() else {}
        if top.get('status')=='PASS' and (not claimed or claimed.get('status')=='INCOMPLETE'):
            raise Incomplete('base not terminal; ledger unopened')
        if top.get('status')!='PASS' or claimed.get('status')!='PASS':
            result.update(status='FAIL',producer_failure=dict(top_status=top.get('status'),base_status=claimed.get('status'),
                top_error=top.get('error'),base_error=claimed.get('error')));return result
        for name in ('plan.json','base/plan.json','recording.json','base/steps.jsonl','replay-dispatch-events.jsonl'):
            if not (child/name).is_file():raise Incomplete('missing '+name)
        producer,http,base,donor=[load(p,h) for p,h in SUPPORT.items()]
        stable={n:sha(child/n) for n in ('plan.json','complete.json','base/plan.json','base/complete.json','recording.json')}
        plan,recording,fixtures,config,tokens,timing,header=binding(child,expected_plan_sha256,top,claimed,producer,http,donor)
        cap=Path(plan['inputs']['dispatch_events']['path']);rep=child/'replay-dispatch-events.jsonl'
        identities={p:file_identity(p) for p in (cap,rep,child/'base/steps.jsonl')};coverage={}
        with cap.open('rb') as captured,rep.open('rb') as replay:
            dispatch=DualDispatch(captured,replay,header,plan['inputs']['dispatch_events'])
            # Explicit facade, no mutation of imported helper globals or paths.
            facade=SimpleNamespace(**{**vars(base),'CaseAudit':dispatch_case_type(base,dispatch,coverage)})
            result['ledger_opened']=True
            ledger=http.stream_ledger(child/'base/steps.jsonl',recording,fixtures,config,claimed['cases'][0],facade)
            rows=dispatch.finish(plan['concurrency'])
        C=plan['concurrency'];stats=rows['captured'];counts=stats['counts']
        same(top['coverage'],expected_coverage(C,counts,rows['calls']),'independent one-pair hook/call accounting')
        same(top['dispatch_comparison'],dict(counts=counts,captured_sha256=stats['sha256'],bytes=stats['bytes'],lines=stats['lines'],
            scope='Ordered actual submissions, including ahead; full numerical gate separate, no performance claim.'),'producer dispatch claims')
        need(stats['sha256']==plan['inputs']['dispatch_events']['sha256']
             and rows['replay']['sha256']==top['replay_dispatch_events']['sha256'],'actual dispatch hashes')
        for p,v in identities.items():need(file_identity(p)==v,'closed stream changed during audit')
        for n,v in stable.items():need(sha(child/n)==v,'terminal changed during audit')
        for ref in (*plan['inputs'].values(),*plan['capture'].values(),*plan['serving'].values()):pinned(ref)
        for p,h in {**plan['sources'],**tokens,**{str(p):h for p,h in SUPPORT.items()}}.items():need(sha(p)==h,'source changed during audit')
        need(ledger['requests']==(8 if C==1 else 16),'complete fixed8/16 history')
        result.update(status='PASS',qualified=True,http_history_qualified=True,concurrency=C,
            plan_sha256=expected_plan_sha256,base_plan_sha256=BASE_PLAN_SHA,terminal_sha256=stable,
            raw_inputs=plan['inputs'],capture=plan['capture'],serving_prerequisite=plan['serving'],sources=plan['sources'],
            tokenizer_runtime_pins=tokens,ledger=ledger,dispatch=rows,http=timing,ahead_retirement=coverage,
            peak_bytes_diagnostic=claimed['peak_bytes'],original_37_request_gate_replaced=False,
            unobserved='Only observed cancellation/EOS/ahead invalidation counts; no unobserved coverage or later formal speed grant.')
    except (Incomplete,FileNotFoundError) as exc:result.update(status='INCOMPLETE',reason=str(exc))
    except Exception as exc:result.update(status='FAIL',first_failure=dict(type=type(exc).__name__,message=str(exc)))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child',required=True,type=Path);parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--out',required=True,type=Path);args=parser.parse_args(argv)
    need(args.out.parent.is_dir() and not args.out.exists(),'new audit output required')
    result=audit(args.child,args.plan_sha256)
    with args.out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    return {'PASS':0,'FAIL':1,'INCOMPLETE':2}[result['status']]

if __name__=='__main__':raise SystemExit(main())
