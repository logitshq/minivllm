"""Offline one-C captured-HTTP numerical audit; no model/device/service work.

--child DIR --plan-sha256 SHA --out NEW_JSON. Terminal files are checked before
opening any ledger. Rejoin original SSE/trace with the pinned CPU tokenizer,
then reuse the frozen independent FCFS/full-logits/state/ahead auditor with
explicit recorded arrivals. The original 37-request gate is not modified.
PASS covers this captured schedule only, not future formal arrivals, resource
ownership, restoration, speed, or re-execution of the recorded GPU comparisons.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
AUDITOR_SHA='b6d90413911f2b17e0951a2fcfdb5fe00666aa06d4382b603e29e5dc4659d4f1'
FROZEN_V1_SHA='1e26e35bdde5cf7b81da37101326a4ef6f87ef7db2775c6324b56d053645c995'
BASE_PLAN_SHA='21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb'
SOURCES={
    'qualify_native.py':'26964fcbed9a2e5dcefaee4888244987897f3138d4a2b7add59bf63a7461b7bf',
    'client.py':'ef64a114f4f7d90ec5044d6bdcf058ca05f98dea445d6639db706a4fc75217dc',
    'native_server.py':'9bcb37eff8bdecd79800afdedc82deacf9f7a0a9a7a1e0d62dc8ba49e70fb68f',
    'http_qualification.py':'471bebd8a114d3a7f5a190c20909f6896f17fd39f282656546c1366c3fa2c6e2',
    'http_join.py':'a362c321c2329d7ab53ab6a1a5a96981bb95b44733739fdbb56dd3d38bef7b4e',
    'http_replay.py':'1304c58c66e56b3aeaa582ef9f7d61317280cf3107f8f600485426119970fb65',
}


class Incomplete(Exception):pass
def need(ok,message):
    if not ok:raise ValueError(message)
def same(a,b,message):
    need(json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False),message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while part:=f.read(1<<20):h.update(part)
    return h.hexdigest()
def read(path):
    need(Path(path).suffix!='.jsonl','ledger/trace must be streamed')
    with Path(path).open() as f:return json.load(f)
def load(name,digest):
    path=HERE/name;need(sha(path)==digest,'fixed support SHA: '+name)
    spec=importlib.util.spec_from_file_location('_http_audit_'+path.stem,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def pinned(item):
    need(set(item)=={'path','sha256'} and Path(item['path']).is_absolute(),'absolute file/SHA required')
    need(sha(item['path'])==item['sha256'],'input SHA changed: '+item['path'])
    return Path(item['path'])


def binding(child,expected_sha,top,claimed):
    need(sha(child/'plan.json')==expected_sha==top['plan_sha256'],'explicit replay plan binding')
    plan=read(child/'plan.json')
    need(set(plan)=={'schema','qualification','concurrency','transport','inputs','sources'}
         and plan['schema']=='E023-recorded-http-plan-v1','replay plan schema')
    C=plan['concurrency'];need(type(C) is int and C in (1,8),'one C1/C8 domain')
    expected={str(HERE/n):h for n,h in SOURCES.items()}
    same(plan['sources'],expected,'six fixed sources');same(top['sources'],expected,'terminal source pins')
    for p,h in expected.items():need(sha(p)==h,'measurement source changed')
    need(plan['qualification']['sha256']==BASE_PLAN_SHA and sha(pinned(plan['qualification']))==BASE_PLAN_SHA
         and sha(child/'base/plan.json')==BASE_PLAN_SHA,'unchanged original numeric plan')
    numeric=read(child/'base/plan.json');count=8 if C==1 else 16
    need(top['schema']=='E023-recorded-http-result-v1' and top['prepare_only'] is False
         and top['numerical_qualified'] is True and top['performance_qualified'] is False
         and top['input_source_pins_rechecked'] is True and top['concurrency']==C
         and type(top['total_requests']) is int and top['total_requests']==count,'terminal captured replay domain')
    same(top['base_result'],dict(path='base/complete.json',sha256=sha(child/'base/complete.json'),returncode=0,status='PASS'),'base result digest')
    need(claimed['plan_sha256']==BASE_PLAN_SHA and claimed['qualifier_sha256']==SOURCES['qualify_native.py']
         and claimed['numeric_only_not_http'] is True and claimed['source_model_pins_rechecked'] is True
         and type(claimed['total_requests']) is int and claimed['total_requests']==count,'base completed one-C contract')
    same(claimed['candidate'],numeric['candidate'],'candidate gates')
    need(len(claimed['cases'])==1 and claimed['cases'][0]['name']==f'HTTP-C{C}','one recorded case only')
    manifests={n:read(pinned(numeric[n])) for n in ('fixtures','model_manifest','source_manifest')}
    model=manifests['model_manifest'];root=Path(numeric['package_root'])/'tutorial_code/miniengine'
    need(model['schema']=='E023-model-v1' and model['model']==numeric['model']
         and model['revision']==Path(numeric['model']).name,'model manifest identity')
    core=manifests['source_manifest']['files']
    need(set(core)=={str(p.relative_to(root)) for p in root.rglob('*.py')},'exact frozen core source set')
    for rel,h in core.items():
        need(not Path(rel).is_absolute() and '..' not in Path(rel).parts and sha(root/rel)==h,'core SHA')
    config_path=Path(numeric['model'])/'config.json';need(sha(config_path)==model['files']['config.json'],'config SHA')
    config=read(config_path);config=config.get('text_config',config)
    need(len(config['layer_types'])==64 and config['layer_types'].count('full_attention')==16
         and config['vocab_size']==248320,'actual64-layer full vocabulary')
    fixtures=manifests['fixtures'];need(numeric['fixtures']['sha256']=='6a4f7ce1578a0c6fead596e8b3d646033d90a2e2b843263747b89a3c6b3c88db'
         and fixtures['model']==numeric['model'],'fixed D fixtures')
    q=load('http_qualification.py',SOURCES['http_qualification.py']);q.validate(child/'plan.json',expected_sha)
    fixtures=fixtures['fixtures'];jobs=q.jobs_for(fixtures,C)
    identity=q.expected_identity(numeric,fixtures,manifests['source_manifest'],model,C,plan['transport'])
    # The existing CPU tokenizer is pinned, not reimplemented. This performs
    # no weight/device load. Source/runtime ownership is separately audited.
    token_pins={p:h for p,h in numeric['execution']['pins'].items() if 'tokenizers' in Path(p).parts
                or Path(p).parent.name=='tokenizers-0.23.3.dist-info'}
    need(token_pins and any(p.endswith('.so') for p in token_pins),'tokenizer runtime pins')
    for p,h in token_pins.items():need(sha(p)==h,'tokenizer runtime SHA')
    import tokenizers
    need(str(Path(tokenizers.__file__).resolve()) in token_pins,'loaded tokenizer package path')
    tokenizer_path=Path(numeric['model'])/'tokenizer.json'
    need(sha(tokenizer_path)==model['files']['tokenizer.json'],'tokenizer model SHA')
    tokenizer=tokenizers.Tokenizer.from_file(str(tokenizer_path))
    for f in fixtures:
        same(tokenizer.encode(f['prompt_text'],add_special_tokens=False).ids,f['prompt_ids'],'full fixture prompt encoding')
    join=load('http_join.py',SOURCES['http_join.py'])
    same(top['inputs'],plan['inputs'],'raw HTTP input pins')
    with pinned(plan['inputs']['trace']).open('rb') as stream:
        recording=join.join(read(pinned(plan['inputs']['group'])),stream,jobs,{f['fixture_id']:f for f in fixtures},
                            identity,lambda ids:tokenizer.decode(ids,skip_special_tokens=False))
    need(sha(child/'recording.json')==top['recording_sha256'],'recording bytes SHA')
    same(read(child/'recording.json'),recording,'reconstructed raw SSE/trace differs from replay input')
    return plan,recording,fixtures,config,token_pins


def case_type(base):
    # Reuse frozen primitives, with a byte-for-byte copied accept method below.
    # The only arithmetic-free expectation change is HTTP temperature float0.0.
    need,same,array_refs=base.need,base.same,base.array_refs
    class HTTPCase(base.CaseAudit):
        def __init__(self,recording,fixtures,config):
            C=recording['concurrency'];super().__init__(f'D-C{C}',fixtures,config,recording['eos_token_ids'])
            self.name=f'HTTP-C{C}';self.recording=recording;self.C=C
            by_id={f['fixture_id']:f for f in fixtures}
            self.jobs=[dict(rid=rid,fixture=by_id[j['fixture_id']],cap=j['cap']) for rid,j in recording['jobs'].items()]
            self.by_rid={j['rid']:j for j in self.jobs};self.seen=set()

        def _accept_http(self, row):
            self.step += 1
            need(row['case'] == self.name and row['step'] == self.step and row['status'] == 'PASS', 'case/continuous step/PASS mismatch')
            self.admit()
            schedule = self.schedule()
            same(row['schedule'], schedule, 'independent FCFS/metadata schedule mismatch')
            decode, prefill = schedule['decode'], schedule['prefill']
            B = len(decode); self.hist[B] += 1
            calls = []
            if decode:
                current = ['decode', decode, [self.requests[r]['computed'] for r in decode], [self.requests[r]['ids'][-1] for r in decode]]
                calls.append(current)
            has_future = bool(decode and not prefill and all(len(self.requests[r]['ids']) + 1 < self.requests[r]['job']['cap']
                and len(self.requests[r]['job']['fixture']['prompt_ids']) + len(self.requests[r]['ids']) + 1 < 2048 for r in decode))
            same(row['reference_delivered'], row['candidate_delivered'], 'two arm output tuples differ')
            delivered = row['reference_delivered']
            expected_delivered = decode + [rid for rid, n in prefill if self.requests[rid]['computed'] + n == len(self.requests[rid]['job']['fixture']['prompt_ids'])]
            same([x[0] for x in delivered], expected_delivered, 'missing/duplicate/reordered delivered request')
            future = ['decode', decode, [self.requests[r]['computed'] + 1 for r in decode], [x[1] for x in delivered]] if has_future else None
            if has_future:
                calls.append(future)
            calls += [['prefill', rid, self.requests[rid]['computed'], n] for rid, n in prefill]
            same(row['call_keys'], calls, 'semantic call coverage mismatch')
            consumed = current if decode and current == self.previous_ahead else None
            same(row['reference_discarded_ahead'], self.previous_ahead, 'reference ahead discard mismatch')
            for rid, n in prefill:
                self.requests[rid]['computed'] += n
            for rid in decode:
                self.requests[rid]['computed'] += 1
            for rid, token, done in delivered:
                need(type(token) is int and 0 <= token < self.config['vocab_size'] and type(done) is bool, 'token/done types/range')
                req = self.requests[rid]
                need(not req['ids'] or req['ids'][-1] not in self.eos, 'output after EOS')
                req['ids'].append(token)
                cap = req['job']['cap']
                need(len(req['ids']) <= cap, 'over-cap raw output')
                expected_done = token in self.eos or len(req['ids']) == cap
                need(done == expected_done, 'normal EOS/cap retirement mismatch')
                if done:
                    self.completed.append({**req['entry'], 'raw_ids': list(req['ids']),
                        'committed_ids': req['ids'][:-1] if token in self.eos else list(req['ids'])})
            finished = [rid for rid in self.running if any(x[0] == rid and x[2] for x in delivered)]
            for rid in finished:
                self.free.append(self.requests[rid]['slot'])
                del self.requests[rid]
            self.running = [rid for rid in self.running if rid not in finished]
            self.finished = finished
            pre_staging = (self.staging - set(decode)) | {r for r, _ in prefill}
            self.staging = pre_staging - set(finished)
            carried = future if not finished else None
            same(row['ahead']['reference'], {'carried_future': carried, 'consumed_prior_future': None,
                 'discarded_prior_future': self.previous_ahead, 'discarded_current_calls': [future] if future and not carried else []}, 'reference ahead accounting')
            same(row['ahead']['candidate'], {'carried_future': carried, 'consumed_prior_future': consumed,
                 'discarded_prior_future': self.previous_ahead if consumed is None else None,
                 'discarded_current_calls': [future] if future and not carried else []}, 'candidate ahead accounting')
            batch = [rid for rid in decode if rid not in finished]
            after = row['after']['engine']
            expected_after = {'requests': [self.metadata(rid) for rid in self.requests], 'waiting': self.waiting,
                'running': self.running, 'free_slots': self.free, 'finished_pending': finished,
                'batch_rids': batch, 'batch_lens': [self.requests[r]['computed'] for r in batch],
                'has_work': bool(self.waiting or self.running), 'ahead_key': None}
            if carried:
                expected_after['ahead_key'] = [carried[1], carried[2],
                    [[0.0, 1.0, 0, self.requests[r]['job']['cap'], False, []] for r in decode], carried[3]]
            same(after, expected_after, 'after-retirement request/slot/batch/ahead metadata')
            expected_after_arrays = self.cache_names('root/committed') if batch else set()
            for rid in self.staging:
                expected_after_arrays |= self.cache_names('root/staging/' + rid, single=True)
            if carried:
                expected_after_arrays |= self.cache_names('root/ahead/layers') | {'root/ahead/tokens'}
            need(array_refs(row['after']['state']) == expected_after_arrays, 'after-state full tree coverage')
            self.check_arrays(row, calls, pre_staging, has_future, B)
            self.previous_arrays = expected_after_arrays
            self.previous_batch = len(batch)
            self.previous_ahead = carried
            self.ahead_counts.update(produced=int(has_future), carried=int(carried is not None), consumed=int(consumed is not None), invalidated=int(has_future and not carried))

        def metadata(self,rid):
            row=super().metadata(rid)
            # Frozen HTTP replay explicitly constructs SamplingParams(temperature=0.0).
            row['sampling']['temperature']=0.0
            return row

        def admit(self):
            expected=self.recording['steps'][self.step-1]
            for rid in expected['arrivals']:
                need(rid in self.by_rid and rid not in self.seen,'unknown/repeated HTTP arrival')
                job=self.by_rid[rid];wire=self.recording['jobs'][rid]
                entry=dict(rid=rid,fixture_id=job['fixture']['fixture_id'],client_request_id=wire['client_request_id'],
                           before_step=self.step,cap=job['cap'])
                self.admissions.append(entry);self.seen.add(rid);self.next+=1
                self.requests[rid]=dict(job=job,entry=entry,ids=[],computed=0,slot=None,status='WAITING')
                self.waiting.append(rid)
            need(len(self.requests)<=self.C,'HTTP exceeds live client concurrency')
            same(dict(running=self.running,waiting=self.waiting),expected['before'],'independent before/arrival order')

        def accept(self,row):
            need(self.step<len(self.recording['steps']),'extra ledger step')
            expected=self.recording['steps'][self.step]
            same(row['recorded'],expected,'ledger embedded recording changed')
            same(row['candidate_schedule'],row['schedule'],'actual arm schedules differ')
            same({k:row['schedule'][k] for k in ('decode','prefill','finished_ids')},expected['schedule'],'HTTP versus replay schedule')
            same(row['reference_delivered'],expected['results'],'HTTP versus replay delivered raw IDs')
            same(row['after']['engine']['batch_rids'],expected['surviving_decode_rows'],'HTTP surviving decode rows')
            same([rid for rid,token,done in row['reference_delivered'] if done],expected['retired'],'HTTP retirement sequence')
            self._accept_http(row)  # Original gates; explicit HTTP float temperature only.

        def finish(self,claimed):
            need(self.step==len(self.recording['steps']) and self.next==len(self.jobs) and not self.requests
                 and not self.waiting and not self.running and not self.staging and self.previous_ahead is None,'fully drained recorded workload')
            completed=[{k:r[k] for k in ('rid','raw_ids','committed_ids')} for r in self.completed]
            need(len({r['rid'] for r in completed})==len(self.jobs),'unique complete HTTP UUIDs')
            for row in completed:
                same(row,{k:self.recording['jobs'][row['rid']][k] for k in row},'terminal raw/committed IDs versus received SSE')
            hist={str(k):v for k,v in self.hist.items()};same(hist,self.recording['histogram'],'actual decode histogram')
            if self.C==8:need(max(self.hist,default=0)>=2,'C8 actual multi-row decode required')
            expected=dict(name=self.name,status='PASS',steps=self.step,
                step_limit=sum(len(j['fixture']['prompt_ids'])+j['cap'] for j in self.jobs)+1,
                admissions=self.admissions,completed=completed,decode_batch_histogram=hist,
                raw_output_count=sum(len(r['raw_ids']) for r in completed),
                committed_output_count=sum(len(r['committed_ids']) for r in completed),
                scope='Exact recorded-arrival numerical replay only; not future formal schedules.',performance_qualified=False)
            same(claimed,expected,'case receipt differs from independent streamed reconstruction')
            return dict(requests=len(completed),steps=self.step,raw_output_count=expected['raw_output_count'],
                committed_output_count=expected['committed_output_count'],observed_terminal_eos=sum(bool(r['raw_ids'] and r['raw_ids'][-1] in self.eos) for r in completed),
                decode_batch_histogram=hist,array_counts=dict(self.array_counts),ahead_counts=dict(self.ahead_counts))
    return HTTPCase


def stream_ledger(path,recording,fixtures,config,claimed,base):
    case=case_type(base)(recording,fixtures,config);events=[];digest=hashlib.sha256();size=lines=0
    before=Path(path).stat()
    expected_events=['load-pair-start','load-pair-complete','release-pair-start','release-pair-complete']
    with Path(path).open('rb') as stream:
        for raw in stream:
            size+=len(raw);lines+=1;digest.update(raw);need(raw.endswith(b'\n'),'truncated ledger line')
            row=json.loads(raw)
            if 'event' in row:
                need(len(events)<4 and row['event']==expected_events[len(events)],'one-pair lifecycle event order')
                elapsed=row['elapsed_wall_s'];need(type(elapsed) in (float,int) and math.isfinite(elapsed) and elapsed>=0,'finite lifecycle time')
                need(not events or elapsed>=events[-1]['elapsed_wall_s'],'lifecycle clock order')
                if row['event'].startswith('load-pair'):
                    need(case.step==0 and row['max_seqs']==recording['concurrency'],'load pair before steps/domain')
                    if row['event']=='load-pair-start':need(row['model_instances']==2,'independent engine pair')
                    else:need(row['independent_weight_arrays']==1349,'all independent weight handles')
                else:
                    need(case.step==len(recording['steps']),'release before all recorded steps')
                    need(row.get('model_instances')==2 if row['event']=='release-pair-start' else row.get('qualification_only') is True,'release lifecycle receipt')
                events.append(row)
            else:
                need(len(events)==2,'step outside loaded/not-yet-released pair')
                try:case.accept(row)
                except Exception as error:raise ValueError(f'ledger line {lines}, step {row.get("step")}: {error}') from error
    after=Path(path).stat()
    need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==
         (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns) and after.st_size==size,'ledger changed during streaming')
    need(len(events)==4,'complete pair release missing')
    result=case.finish(claimed)
    return dict(sha256=digest.hexdigest(),bytes=size,lines=lines,events=events,**result,
                bit_equal_checks=case.array_counts['all_arrays'],finite_checks=2*case.array_counts['all_arrays'])


def audit(child,expected_plan_sha256):
    child=Path(child).resolve();result=dict(schema='E023-recorded-http-independent-audit-v2',status='INCOMPLETE',qualified=False,child_directory=str(child),
        ledger_opened=False,performance_qualified=False,scope=__doc__.split('PASS covers',1)[1].strip(),audit_source_sha256=sha(__file__),
        frozen_v1_source_sha256=FROZEN_V1_SHA)
    try:
        top=read(child/'complete.json')
        if top.get('status') in ('INCOMPLETE','PREPARED'):raise Incomplete('replay not terminal; ledger not opened')
        claimed=read(child/'base/complete.json') if (child/'base/complete.json').is_file() else {}
        if top.get('status')=='PASS' and (not claimed or claimed.get('status')=='INCOMPLETE'):raise Incomplete('base replay not terminal; ledger not opened')
        if top.get('status')!='PASS' or claimed.get('status')!='PASS':
            result.update(status='FAIL',producer_failure=dict(top_status=top.get('status'),base_status=claimed.get('status'),
                top_error=top.get('error'),base_error=claimed.get('error'),cleanup_error=claimed.get('cleanup_error')))
            return result
        terminal_hashes={n:sha(child/n) for n in ('complete.json','base/complete.json')}
        base=load('audit_qualification.py',AUDITOR_SHA)
        plan,recording,fixtures,config,token_pins=binding(child,expected_plan_sha256,top,claimed)
        result['ledger_opened']=True
        result['ledger']=stream_ledger(child/'base/steps.jsonl',recording,fixtures,config,claimed['cases'][0],base)
        count=8 if plan['concurrency']==1 else 16
        need(result['ledger']['requests']==count,'all fixed8/16 jobs required')
        for p,h in {**plan['sources'],**token_pins}.items():need(sha(p)==h,'source/runtime changed during audit')
        for item in plan['inputs'].values():pinned(item)
        need(sha(child/'plan.json')==expected_plan_sha256 and all(sha(child/n)==h for n,h in terminal_hashes.items())
             and sha(child/'recording.json')==top['recording_sha256'],'terminal artifacts changed during audit')
        result.update(status='PASS',qualified=True,concurrency=plan['concurrency'],plan_sha256=expected_plan_sha256,
            base_plan_sha256=BASE_PLAN_SHA,child_complete_sha256=sha(child/'complete.json'),base_complete_sha256=sha(child/'base/complete.json'),
            raw_inputs=plan['inputs'],recording_sha256=top['recording_sha256'],sources=plan['sources'],tokenizer_runtime_pins=token_pins,
            base_auditor_sha256=AUDITOR_SHA,peak_bytes_diagnostic=claimed['peak_bytes'],weight_shards_rehashed=False,
            owner_resources_restore_qualified=False,original_37_request_gate_replaced=False,
            evidence_limit='Recorded original-dtype bit/finite comparisons are fully accounted; GPU arithmetic is not rerun. Only this captured HTTP arrival schedule is qualified.')
    except (Incomplete,FileNotFoundError) as error:result.update(status='INCOMPLETE',reason=str(error))
    except Exception as error:result.update(status='FAIL',first_failure=dict(type=type(error).__name__,message=str(error)))
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--child',required=True,type=Path)
    p.add_argument('--plan-sha256',required=True);p.add_argument('--out',required=True,type=Path);args=p.parse_args(argv)
    need(not args.out.exists(),'new output file required');result=audit(args.child,args.plan_sha256)
    with args.out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    return {'PASS':0,'FAIL':1,'INCOMPLETE':2}[result['status']]


if __name__=='__main__':raise SystemExit(main())
