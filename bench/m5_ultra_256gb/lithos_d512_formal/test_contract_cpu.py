"""CPU-only frozen-client wire fixtures, no socket/server/model/tokenizer execution."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from bench.m5_ultra_256gb.lithos_d512_formal import contract as c

ROOT = Path(__file__).resolve().parents[3]
OWNER = ROOT/'.artifacts/m5-ultra-256gb/lithos-d512-qualification-v2-owner'
ARCHIVE = ROOT/'docs/experiments/m5-ultra-256gb/E023-lithos-d512-qualification-v2'
def ref(path): return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
def binding():
    return dict(owner_directory=str(OWNER), owner_plan=ref(OWNER/'owner-plan.json'),
        **{key: ref(ARCHIVE/name) for key, name in (('audit','independent-qualification.json'),
            ('resource_audit','resource-summary.json'),('restoration_audit','restoration-audit.json'))})


def fixture(spec, phase='warm', variants=None):
    """Exercise frozen parser/finalizer with explicit synthetic wire/time, no network."""
    client = c.load('../pareto_realistic/lithos_client.py'); q = c.count_contract(spec,phase)
    rows=[]; C=spec['concurrency']; variants=variants or {}
    for i,job in enumerate(spec['jobs'][phase]):
        f=spec['fixtures'][i%8]; start=100+(i//C)*3+(i%C)*.01; end=start+2
        n,finish=variants.get(i,(512,'length')); text='' if n==1 and finish=='stop' else 'a visible answer'
        row=client._new_row(copy.deepcopy(job)); row.update(status='done',http_status=200,
            start_perf_counter_s=start,absolute_deadline_perf_counter_s=start+spec['deadlines']['request'])
        body=json.dumps(job['body']).encode(); row.update(request_body_bytes_base64=base64.b64encode(body).decode(),request_body_bytes_sha256=hashlib.sha256(body).hexdigest())
        parser=client._NDJSON(row)
        for obj,stamp in [(dict(model=spec['served_model'],created_at='2026-10-17T00:00:00Z',
            message=dict(role='assistant',content=text),done=False),start+.2),
            (dict(model=spec['served_model'],created_at='2026-10-17T00:00:02Z',message=dict(role='assistant',content=''),
                done=True,done_reason=finish,total_duration=200,load_duration=0,prompt_eval_count=len(f['prompt_ids']),
                prompt_eval_duration=100,eval_count=n,eval_duration=100),end)]:
            raw=(json.dumps(obj)+'\n').encode();client._retain(row,raw,stamp,8<<20);parser.feed(raw,stamp)
        parser.eof(end+.1); client._finalize_row(row,q,[]);rows.append(row)
    last=max(r['done_perf_counter_s'] for r in rows)
    cleanup=dict(start_perf_counter_s=last+.02,end_perf_counter_s=last+.2,budget_s=spec['deadlines']['close'],complete=True,
        deadline_exceeded=False,pending_tasks=[],errors=[],cancelled_again=False,client_close_started=True)
    group=dict(schema=client.SCHEMA,protocol=client.PROTOCOL,cache_contract=client.CACHE_CONTRACT,phase=phase,concurrency=C,
        qualification=q,qualification_errors=[],group_error=None,cleanup=cleanup,requests=rows,
        frozen_scheduler_sha256=client.BASE_SHA256,base_url='http://127.0.0.1:'+str(spec['port']),
        request_timeout_s=spec['deadlines']['request'],group_timeout_s=spec['deadlines'][phase],
        barrier_release_perf_counter_s=99,group_deadline_perf_counter_s=99+spec['deadlines'][phase],
        connection_policy=dict(max_connections=2*C,max_keepalive_connections=0,retries=0,follow_redirects=False))
    group['summary']=client._summarize(rows,[],None,cleanup,last+.01)
    group['summary']['formal_point_eligible']=phase=='formal'
    return group


def raw_change(row, change):
    ch=row['raw_ndjson_chunks'][-1];obj=json.loads(base64.b64decode(ch['body_bytes_base64']));change(obj)
    raw=(json.dumps(obj)+'\n').encode();old=base64.b64decode(ch['body_bytes_base64']);ch['body_bytes_base64']=base64.b64encode(raw).decode()
    row['raw_body_bytes']+=len(raw)-len(old)


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binding=binding();cls.base=json.loads((OWNER/'owner-plan.json').read_bytes())['qualification']
        model=Path(json.loads(Path(cls.base['path']).read_bytes())['model']).name
        cls.specs={C:c.prepare(cls.base,cls.binding,C,1,model) for C in (1,8)}
        for spec in cls.specs.values():spec.update(port=18881,deadlines=dict(request=600,warm=720,formal=2400,close=10))

    def test_actual_closed_receipts_and_original_messages(self):
        for C,spec in self.specs.items():
            self.assertEqual(spec['qualification']['audit']['sha256'],c.RECEIPTS['audit'])
            self.assertEqual(spec['qualification_identity']['owner_plan_sha256'],c.OWNER_PLAN_SHA)
            for phase,N in [('warm',8),('formal',64)]:
                self.assertEqual(len(spec['jobs'][phase]),N)
                for i,job in enumerate(spec['jobs'][phase]):
                    self.assertEqual(job['body']['messages'],spec['fixtures'][i%8]['messages'])
                    self.assertNotEqual(job['body']['messages'][0]['content'],spec['fixtures'][i%8]['prompt_text'])
                self.assertEqual(c.load('../pareto_realistic/lithos_client.py')._qualification_errors(c.count_contract(spec,phase),spec['jobs'][phase],C),[])
        self.assertFalse(set(j['request_id'] for j in self.specs[1]['jobs']['warm']) & set(j['request_id'] for j in self.specs[1]['jobs']['formal']))

    def test_eos_short_at_cap_immediate_and_length_once(self):
        spec=self.specs[1];group=fixture(spec,variants={0:(1,'stop'),1:(512,'stop'),2:(21,'stop')})
        out=c.check_group(group,spec,'warm');self.assertEqual([r['committed_output_tokens'] for r in out['requests'][:4]],[0,511,20,512])
        self.assertEqual(out['point']['committed_output_tokens'],0+511+20+5*512)
        self.assertEqual(out['point']['eos_requests'],3);self.assertFalse(out['performance_qualified']);self.assertFalse(out['full_state_qualified'])

    def test_real_raw_formula_c1_c8_formal64(self):
        for C,spec in self.specs.items():
            out=c.check_group(fixture(spec,'formal'),spec,'formal');point=out['point']
            self.assertEqual(point['requests'],64);self.assertEqual(point['max_http_outstanding'],C)
            expected_window=(64//C-1)*3+(C-1)*.01+2
            self.assertAlmostEqual(point['aggregate_output_tok_s'],32768/expected_window)
            self.assertEqual(point['e2e_p50_s'],2);self.assertEqual(point['e2e_p95_s'],2)

    def test_actual_binder_rejects_nonmatching_receipt_and_identity(self):
        for key in ('audit','resource_audit','restoration_audit','owner_plan'):
            b=copy.deepcopy(self.binding);b[key]['sha256']='0'*64
            with self.assertRaises(ValueError):c.prepare(self.base,b,1,1,self.specs[1]['served_model'])
        s=copy.deepcopy(self.specs[1]);s['qualification_identity']['pack_dir']='/wrong'
        with self.assertRaisesRegex(ValueError,'identity'):c.count_contract(s,'warm')
        s=copy.deepcopy(self.specs[1]);s['fixtures'][0]['prompt_ids'][0]+=1
        with self.assertRaisesRegex(ValueError,'fixtures'):c.count_contract(s,'warm')

    def test_changed_receipt_file_with_updated_sha_cannot_manufacture_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'audit.json';obj=json.loads(Path(self.binding['audit']['path']).read_bytes());obj['status']='FAIL';p.write_text(json.dumps(obj))
            q=copy.deepcopy(self.binding);q['audit']=ref(p)
            with self.assertRaisesRegex(ValueError,'unrecognized'):c.prepare(self.base,q,1,1,self.specs[1]['served_model'])

    def test_self_consistent_sent_double_template_rejected(self):
        spec=self.specs[1];g=fixture(spec);row=g['requests'][0];body=copy.deepcopy(row['body'])
        body['messages']=[dict(role='user',content=spec['fixtures'][0]['prompt_text'])]
        raw=json.dumps(body).encode();row['request_body_bytes_base64']=base64.b64encode(raw).decode();row['request_body_bytes_sha256']=hashlib.sha256(raw).hexdigest()
        with self.assertRaisesRegex(ValueError,'raw sent original'):c.check_group(g,spec,'warm')

    def test_raw_bad_counts_not_hidden_by_derived_fields(self):
        for value,finish in [(511,'length'),(513,'stop'),(0,'stop'),(True,'stop'),(512,'invalid')]:
            g=fixture(self.specs[1]);raw_change(g['requests'][0],lambda x:x.update(eval_count=value,done_reason=finish))
            with self.subTest(value=value,finish=finish),self.assertRaisesRegex(ValueError,'raw normal'):c.check_group(g,self.specs[1],'warm')
        g=fixture(self.specs[1]);raw_change(g['requests'][0],lambda x:x.update(prompt_eval_count=999))
        with self.assertRaisesRegex(ValueError,'prompt count'):c.check_group(g,self.specs[1],'warm')

    def test_terminal_tail_truncated_and_malformed(self):
        for suffix in (b'{}\n',b'junk',b'\n'):
            g=fixture(self.specs[1]);row=g['requests'][0];ch=row['raw_ndjson_chunks'][-1]
            old=base64.b64decode(ch['body_bytes_base64']);ch['body_bytes_base64']=base64.b64encode(old+suffix).decode();row['raw_body_bytes']+=len(suffix)
            with self.subTest(suffix=suffix),self.assertRaises(ValueError):c.check_group(g,self.specs[1],'warm')
        g=fixture(self.specs[1]);raw_change(g['requests'][0],lambda x:x.update(done=False))
        with self.assertRaises(ValueError):c.check_group(g,self.specs[1],'warm')

    def test_cleanup_cancel_and_unknown_cache_rejected(self):
        for field,value in [('complete',False),('cancelled_again',True),('pending_tasks',['response']),('errors',[{'type':'CancelledError'}]),('client_close_started',False)]:
            g=fixture(self.specs[1]);g['cleanup'][field]=value
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'drain/close'):c.check_group(g,self.specs[1],'warm')
        g=fixture(self.specs[1]);g['requests'][0]['reported_cached_prompt_tokens']=0
        with self.assertRaisesRegex(ValueError,'cache unknown'):c.check_group(g,self.specs[1],'warm')

    def test_bad_time_and_serial_mislabeled_c8(self):
        for field,value in [('start_perf_counter_s',float('nan')),('eof_perf_counter_s',1),('absolute_deadline_perf_counter_s',101),('done_perf_counter_s',float('inf'))]:
            g=fixture(self.specs[1]);g['requests'][0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):c.check_group(g,self.specs[1],'warm')
        # A complete valid C1 wire history is relabeled C8 with all supplied
        # contracts corrected. Only the actual intervals reveal serialization.
        serial=fixture(self.specs[1]);spec=self.specs[8]
        for row,job in zip(serial['requests'],spec['jobs']['warm']):row.update(copy.deepcopy(job))
        q=c.count_contract(spec,'warm');serial.update(concurrency=8,qualification=q,connection_policy=dict(max_connections=16,max_keepalive_connections=0,retries=0,follow_redirects=False))
        with self.assertRaisesRegex(ValueError,'actual HTTP concurrency'):c.check_group(serial,spec,'warm')

    def test_summary_cannot_replace_raw_metrics_or_endpoint(self):
        for mutate in [lambda g:g['summary'].__setitem__('aggregate_output_tok_s',99999),lambda g:g.__setitem__('base_url','http://foreign:18881'),
                       lambda g:g.__setitem__('request_timeout_s',601),lambda g:g['requests'][0]['count_normalization'].__setitem__('terminal_eos_subtracted',1)]:
            g=fixture(self.specs[1]);mutate(g)
            with self.assertRaises(ValueError):c.check_group(g,self.specs[1],'warm')

    def test_missing_fixture_cannot_be_hidden_by_rebuilt_contract(self):
        spec=copy.deepcopy(self.specs[1]);jobs=spec['jobs']['formal']
        jobs[1]={**copy.deepcopy(jobs[0]),'request_id':jobs[1]['request_id']}
        spec['contracts']['formal']=c._contract(jobs,spec['fixtures'],spec['qualification'],1)
        with self.assertRaisesRegex(ValueError,'fixture cycle'):c.count_contract(spec,'formal')

    def test_strict_c_and_repeat_types(self):
        for C,R in [(True,1),(1,False),(4,1),(1,4)]:
            with self.assertRaises(ValueError):c.prepare(self.base,self.binding,C,R,self.specs[1]['served_model'])

if __name__=='__main__':unittest.main()
