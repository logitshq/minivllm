"""Synthetic descriptor/HTTP corruption tests only; no model or device proof."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bench.m5_ultra_256gb.pareto_realistic import audit_http_replay as a
from bench.m5_ultra_256gb.pareto_realistic import test_http_join_cpu as wire_fixture


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')


def synthetic(C):
    """Build only metadata/array-check descriptors, never actual model values."""
    base=a.load('audit_qualification.py',a.AUDITOR_SHA)
    fixtures=[dict(fixture_id=f'f{i}',prompt_ids=[1,2]) for i in range(8)]
    config=dict(vocab_size=248320,layer_types=['full_attention','linear_attention','linear_attention','linear_attention']*16)
    count=8 if C==1 else 16
    jobs={f'{i+1:032x}':dict(rid=f'{i+1:032x}',fixture_id=f'f{i%8}',client_request_id=f'client-{i}',
        prompt_ids=[1,2],cap=3,raw_ids=[99] if C==1 else [7,8,99],committed_ids=[] if C==1 else [7,8]) for i in range(count)}
    recording=dict(schema='E023-recorded-http-v1',concurrency=C,eos_token_ids=[99],jobs=jobs,steps=[],histogram={})
    state=a.case_type(base)(recording,fixtures,config);rows=[]
    remaining=list(jobs)
    while remaining or state.requests:
        arrivals=[remaining.pop(0) for _ in range(min(C,len(remaining)))] if not state.requests else []
        record=dict(step=len(rows)+1,arrivals=arrivals,
                    before=dict(running=list(state.running),waiting=list(state.waiting)+arrivals))
        recording['steps'].append(record);state.step+=1;state.admit()
        schedule=copy.deepcopy(state.schedule());decode,prefill=schedule['decode'],schedule['prefill'];B=len(decode)
        delivered=[]
        for rid in decode+[r for r,n in prefill]:
            prior=len(state.requests[rid]['ids']);token=jobs[rid]['raw_ids'][prior]
            delivered.append([rid,token,token==99])
        current=['decode',decode,[state.requests[r]['computed'] for r in decode],
                 [state.requests[r]['ids'][-1] for r in decode]] if decode else None
        future=['decode',decode,[state.requests[r]['computed']+1 for r in decode],[x[1] for x in delivered]] \
            if decode and not prefill and all(len(state.requests[r]['ids'])+1<jobs[r]['cap'] for r in decode) else None
        calls=([current] if current else [])+([future] if future else [])+[['prefill',r,state.requests[r]['computed'],n] for r,n in prefill]
        consumed=current if current and current==state.previous_ahead else None
        pre_staging=(state.staging-set(decode))|{r for r,n in prefill}
        for r,n in prefill:state.requests[r]['computed']+=n
        for r in decode:state.requests[r]['computed']+=1
        for r,t,done in delivered:
            state.requests[r]['ids'].append(t)
            if done:
                req=state.requests[r];state.completed.append({**req['entry'],'raw_ids':list(req['ids']),'committed_ids':req['ids'][:-1]})
        finished=[r for r in state.running if any(x[0]==r and x[2] for x in delivered)]
        for r in finished:state.free.append(state.requests[r]['slot']);del state.requests[r]
        state.running=[r for r in state.running if r not in finished];state.finished=finished
        state.staging=pre_staging-set(finished);batch=[r for r in decode if r not in finished]
        carried=future if not finished else None
        after=dict(requests=[state.metadata(r) for r in state.requests],waiting=state.waiting,running=state.running,
            free_slots=list(state.free),finished_pending=finished,batch_rids=batch,batch_lens=[state.requests[r]['computed'] for r in batch],
            has_work=bool(state.waiting or state.running),ahead_key=None)
        if carried:after['ahead_key']=[carried[1],carried[2],[[0,1.0,0,jobs[r]['cap'],False,[]] for r in decode],carried[3]]
        post=state.cache_names('root/committed') if B else set()
        for r in pre_staging:post|=state.cache_names('root/staging/'+r,single=True)
        if future:post|=state.cache_names('root/ahead/layers')|{'root/ahead/tokens'}
        final=state.cache_names('root/committed') if batch else set()
        for r in state.staging:final|=state.cache_names('root/staging/'+r,single=True)
        if carried:final|=state.cache_names('root/ahead/layers')|{'root/ahead/tokens'}
        groups={'before-step-state':state.previous_arrays,'before-retirement-state':post,'after-retirement-state':final}
        for call in calls:
            single=call[0]=='prefill';literal=tuple(call) if single else ('decode',tuple(call[1]),tuple(call[2]),tuple(call[3]))
            groups['logits '+repr(literal)]={'root'};groups['call-state '+repr(literal)]=state.cache_names(single=single)
        if B:
            for arm in ('reference','candidate'):groups[arm+'/committed-not-rewritten-by-ahead']=state.cache_names()
        arrays=[]
        for label,names in groups.items():
            for name in sorted(names):
                field=name.rsplit('/',1)[-1];single='/staging/' in name or label.startswith("call-state ('prefill',")
                width=1 if single else state.previous_batch if label=='before-step-state' else len(batch) if label=='after-retirement-state' else B
                dtype='mlx.core.bfloat16'
                if label.startswith('logits '):shape=[1 if "('prefill'," in label else B,248320]
                elif field in ('K','V','keys','values'):shape=[width,4,256,256]
                elif field in ('conv','conv_state'):shape=[3,10240] if single else [width,3,10240]
                elif field=='S':shape=[48,128,128] if single else [width,48,128,128];dtype='mlx.core.float32'
                elif field=='tokens':shape=[width];dtype='mlx.core.int32'
                arrays.append(dict(check=label,name=name,shape=shape,dtype=dtype,bit_equal=True,reference_finite=True,candidate_finite=True))
        record.update(schedule={k:schedule[k] for k in ('decode','prefill','finished_ids')},postrun_decode_rows=decode,
            results=delivered,retired=[r for r,t,d in delivered if d],surviving_decode_rows=batch)
        ahead=dict(reference=dict(carried_future=carried,consumed_prior_future=None,discarded_prior_future=state.previous_ahead,
                   discarded_current_calls=[future] if future and not carried else []),
            candidate=dict(carried_future=carried,consumed_prior_future=consumed,discarded_prior_future=state.previous_ahead if consumed is None else None,
                   discarded_current_calls=[future] if future and not carried else []))
        rows.append(copy.deepcopy(dict(case=f'HTTP-C{C}',step=state.step,status='PASS',recorded=record,schedule=schedule,candidate_schedule=schedule,
            reference_discarded_ahead=state.previous_ahead,reference_delivered=delivered,candidate_delivered=delivered,
            call_keys=calls,arrays=arrays,ahead=ahead,after=dict(engine=after,state=[{'array':x} for x in sorted(final)]))))
        state.previous_arrays=final;state.previous_batch=len(batch);state.previous_ahead=carried
        state.hist[B]+=1
    recording['histogram']={str(k):v for k,v in state.hist.items()}
    completed=[{k:r[k] for k in ('rid','raw_ids','committed_ids')} for r in state.completed]
    claimed=dict(name=f'HTTP-C{C}',status='PASS',steps=len(rows),step_limit=count*5+1,admissions=state.admissions,
        completed=completed,decode_batch_histogram=recording['histogram'],raw_output_count=sum(len(r['raw_ids']) for r in completed),
        committed_output_count=sum(len(r['committed_ids']) for r in completed),
        scope='Exact recorded-arrival numerical replay only; not future formal schedules.',performance_qualified=False)
    events=[dict(event='load-pair-start',elapsed_wall_s=0,max_seqs=C,model_instances=2),
        dict(event='load-pair-complete',elapsed_wall_s=1,max_seqs=C,independent_weight_arrays=1349),
        dict(event='release-pair-start',elapsed_wall_s=2,model_instances=2),dict(event='release-pair-complete',elapsed_wall_s=3,qualification_only=True)]
    return base,recording,fixtures,config,claimed,events[:2]+rows+events[2:]


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)

    def streamed(self,data):
        base,recording,fixtures,config,claimed,rows=data;path=self.root/'steps.jsonl'
        path.write_bytes(b''.join((json.dumps(r)+'\n').encode() for r in rows))
        return a.stream_ledger(path,recording,fixtures,config,claimed,base)

    def test_full64_array_coverage_c1_c8_and_terminal_eos(self):
        for C,count in ((1,8),(8,16)):
            result=self.streamed(synthetic(C));self.assertEqual(result['requests'],count)
            self.assertEqual(result['observed_terminal_eos'],count)
            self.assertEqual(result['raw_output_count']-result['committed_output_count'],count)
            self.assertGreater(result['bit_equal_checks'],count*128)
            if C==8:self.assertGreater(result['ahead_counts']['consumed'],0)

    def test_missing_bit_finite_layer_vocabulary_and_slot_rejected(self):
        for mutation in ('bit','finite','layer','vocab','slot'):
            data=synthetic(1);row=data[-1][2]
            if mutation=='bit':row['arrays'][0]['bit_equal']=False
            if mutation=='finite':row['arrays'][0]['candidate_finite']=False
            if mutation=='layer':row['arrays'].pop()
            if mutation=='vocab':next(x for x in row['arrays'] if x['check'].startswith('logits '))['shape'][-1]-=1
            if mutation=='slot':row['after']['engine']['free_slots']=[]
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.streamed(data)

    def test_recorded_uuid_schedule_raw_and_actual_candidate_schedule_rejected(self):
        for mutation in ('uuid','schedule','raw','candidate','ahead'):
            data=synthetic(8);row=data[-1][3]
            if mutation=='uuid':row['recorded']['arrivals']=['0'*32]
            if mutation=='schedule':row['recorded']['schedule']['decode'].reverse()
            if mutation=='raw':row['reference_delivered'][0][1]+=1
            if mutation=='candidate':row['candidate_schedule']['finished_ids']=['missing']
            if mutation=='ahead':row['ahead']['candidate']['carried_future']=None
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.streamed(data)

    def test_release_sequence_and_complete_claim_are_required(self):
        for mutation in ('missing-release','early-release','weight-count','completed','count'):
            data=synthetic(1)
            if mutation=='missing-release':data[-1].pop()
            if mutation=='early-release':data[-1].insert(2,data[-1].pop())
            if mutation=='weight-count':data[-1][1]['independent_weight_arrays']=1
            if mutation=='completed':data[4]['completed'].pop()
            if mutation=='count':data[4]['raw_output_count']+=1
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.streamed(data)

    def test_raw_sse_and_trace_rejoin_reject_not_derived_receipts(self):
        join=a.load('http_join.py',a.SOURCES['http_join.py'])
        data=wire_fixture.fixture();group,events,jobs,fixtures,identity,terminals=data
        result=join.join(group,wire_fixture.lines(events),jobs,fixtures,identity,wire_fixture.decode)
        self.assertEqual(result['jobs'][wire_fixture.A]['raw_ids'],[10,99])
        terminals[0]['benchmark_committed_output_ids']=[10,99];wire_fixture.replace_wire(group['requests'][0],terminals[0])
        with self.assertRaises(ValueError):join.join(group,wire_fixture.lines(events),jobs,fixtures,identity,wire_fixture.decode)

    def test_incomplete_or_failed_producer_never_opens_ledger_or_tokenizer(self):
        for status,expected in (('INCOMPLETE','INCOMPLETE'),('PREPARED','INCOMPLETE'),('FAIL','FAIL')):
            write(self.root/'complete.json',dict(status=status,error={'message':'diagnostic'}))
            with patch.object(a,'binding',side_effect=AssertionError('no heavy binding')),patch.object(a,'load',side_effect=AssertionError('no imports')):
                result=a.audit(self.root,'0'*64)
            self.assertEqual(result['status'],expected);self.assertFalse(result['ledger_opened'] or result['qualified'])
            if status=='FAIL':self.assertEqual(result['producer_failure']['top_error']['message'],'diagnostic')

    def test_terminal_pass_still_reconstructs_all_rows(self):
        data=synthetic(1);base,recording,fixtures,config,claimed,rows=data
        write(self.root/'recording.json',recording)
        write(self.root/'complete.json',dict(status='PASS',recording_sha256=a.sha(self.root/'recording.json')))
        write(self.root/'base/complete.json',dict(status='PASS',cases=[claimed],peak_bytes=1))
        write(self.root/'plan.json',{})
        (self.root/'base/steps.jsonl').write_bytes(b''.join((json.dumps(r)+'\n').encode() for r in rows))
        plan=dict(concurrency=1,sources={},inputs={})
        with patch.object(a,'binding',return_value=(plan,recording,fixtures,config,{})):
            result=a.audit(self.root,a.sha(self.root/'plan.json'))
            self.assertEqual(result['status'],'PASS',result)
            self.assertFalse(result['performance_qualified'] or result['original_37_request_gate_replaced'])
            rows[2]['arrays'].pop();(self.root/'base/steps.jsonl').write_bytes(b''.join((json.dumps(r)+'\n').encode() for r in rows))
            result=a.audit(self.root,a.sha(self.root/'plan.json'));self.assertEqual(result['status'],'FAIL')


if __name__=='__main__':unittest.main()
