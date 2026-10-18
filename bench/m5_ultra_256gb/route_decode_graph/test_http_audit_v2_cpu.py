"""CPU receipt/protocol mechanics only; synthetic bits are not device evidence.

Written during a formal device experiment; execution is deferred to root.
Uses frozen 64-layer comparison-descriptor fixtures and actual capture cleanup
with a fake backend. No sockets, process spawn, model or tokenizer imports.
"""
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bench.m5_ultra_256gb.route_decode_graph import http_audit_v2 as a
from bench.m5_ultra_256gb.route_decode_graph import http_qualification_v2 as producer
from bench.m5_ultra_256gb.route_decode_graph import http_capture as capture
from bench.m5_ultra_256gb.route_decode_graph.test_http_capture_cpu import Backend
from bench.m5_ultra_256gb.pareto_realistic import test_audit_http_replay_v2_cpu as numeric


def stream(rows): return io.BytesIO(b''.join((json.dumps(r)+'\n').encode() for r in rows))
def ref(path): return dict(path=str(Path(path).resolve()),sha256=a.sha(path))


class ReceiptBackend(Backend):
    """Only the fake receipts use real argv syntax; never executes commands."""
    async def identity(self, pid):
        value=await super().identity(pid)
        return {**value,'command':['/bin/ps','-p',str(pid),'-o','pid=,ppid=,lstart=,comm=']}
    async def absent(self, pid):
        value=await super().absent(pid)
        return {**value,'command':['/bin/ps','-p',str(pid),'-o','pid=']}
    async def port(self, port, pid=None):
        value=await super().port(port,pid)
        return {**value,'command':['/usr/sbin/lsof','-nP','-a']+(['-p',str(pid)] if pid is not None else [])
                +[f'-iTCP:{port}','-sTCP:LISTEN','-Fpn']}

def headers(C=8):
    captured = dict(event='route-startup',schema='E024-route-http-events-v1',pid=12345,phase=f'HTTP-C{C}',identity={'CPU':'only'})
    source = dict(path='/CPU/captured-route-events.jsonl',sha256='a'*64)
    replay = dict(event='replay-route-startup',schema='E024-route-http-replay-events-v1',pid=12346,
                  phase=captured['phase'],captured_route_events=source)
    return captured,replay,source

def row(step=3,B=8,used=1023,L=1536,functions=1,C=8):
    return dict(event='route-call',phase=f'HTTP-C{C}',step=step,direct_label=None,status='SUBMITTED',
        B=B,used=used,capacities=[[L,L] for _ in range(16)],expected_graph=L==1536,
        actual_invocations=int(L==1536),functions=functions)

def dual(left,right=None,C=8):
    cap,rep,source=headers(C);serving=a.load(a.SERVING,a.SERVING_SHA)
    return a.DualRoutes(serving,stream([cap,*left]),stream([rep,*(left if right is None else right)]),cap,source)

def clock_group(C=8,serial=False):
    spec=dict(concurrency=C,transport={'port':18181},deadlines=dict(request=100,group=200,close=10))
    count=8 if C==1 else 16;rows=[]
    for i in range(count):
        start=float(i if serial else i//C)+1;end=start+1
        rows.append(dict(start_perf_counter_s=start,done_perf_counter_s=end,terminal_perf_counter_s=end,
            absolute_deadline_perf_counter_s=start+100,e2e_s=1.0))
    group=dict(base_url='http://127.0.0.1:18181',request_timeout_s=100,group_timeout_s=200,
        connection_policy=dict(max_connections=2*C,max_keepalive_connections=0,retries=0,follow_redirects=False),
        barrier_release_perf_counter_s=1.0,group_deadline_perf_counter_s=201.0,requests=rows,
        cleanup=dict(adapter='native-close-cancellation-v1',
            frozen_client_sha256='ef64a114f4f7d90ec5044d6bdcf058ca05f98dea445d6639db706a4fc75217dc',
            budget_s=10,client_close_started=True,complete=True,errors=[],pending_tasks=[],
            deadline_exceeded=False,cancelled_again=False,start_perf_counter_s=20.0,end_perf_counter_s=21.0))
    return group,spec


class Tests(unittest.TestCase):
    def test_half_open_actual_http_concurrency_and_serial_mislabeled_c8(self):
        for C in (1,8):
            group,spec=clock_group(C);self.assertEqual(a.http_intervals(group,spec)['max_http_outstanding'],C)
        group,spec=clock_group(8,serial=True)
        with self.assertRaisesRegex(ValueError,'actual HTTP concurrency'):a.http_intervals(group,spec)
        group,spec=clock_group(8);group['requests'][8]['start_perf_counter_s']=1.5
        group['requests'][8]['absolute_deadline_perf_counter_s']=101.5;group['requests'][8]['e2e_s']=1.5
        with self.assertRaisesRegex(ValueError,'actual HTTP concurrency'):a.http_intervals(group,spec)

    def test_bad_request_group_and_close_clock_reject(self):
        for corruption in ('zero','nan','request-budget','close','cleanup','origin'):
            group,spec=clock_group();r=group['requests'][0]
            if corruption=='zero':r['done_perf_counter_s']=r['start_perf_counter_s']
            elif corruption=='nan':r['start_perf_counter_s']=float('nan')
            elif corruption=='request-budget':r['absolute_deadline_perf_counter_s']=2
            elif corruption=='close':group['cleanup']['start_perf_counter_s']=1
            elif corruption=='cleanup':group['cleanup']['cancelled_again']=True
            else:group['base_url']='http://wrong.invalid'
            with self.subTest(corruption=corruption),self.assertRaises(ValueError):a.http_intervals(group,spec)

    def test_equal_but_identically_wrong_sidecars_are_not_an_oracle(self):
        good=row()
        for key,value in (('B',1),('used',1024),('actual_invocations',0),('expected_graph',False),
                          ('functions',2),('step',True),('direct_label','bad'),('status','RUNNING')):
            bad={**good,key:value};routes=dual([bad])
            with self.subTest(key=key),self.assertRaises(ValueError):routes.route('HTTP-C8',3,None,8,1023,1536)
        routes=dual([good]);routes.route('HTTP-C8',3,None,8,1023,1536)
        self.assertEqual(routes.finish()['calls'],1)

    def test_missing_extra_reordered_truncated_and_wrong_pid_sidecars_reject(self):
        first=row();second=row(used=1024,functions=2)
        for left,right in (([first],[]),([second,first],[second,first]),([first],[first,second])):
            routes=dual(left,right)
            with self.assertRaises(ValueError):
                routes.route('HTTP-C8',3,None,8,1023,1536);routes.finish()
        cap,rep,source=headers();serving=a.load(a.SERVING,a.SERVING_SHA)
        with self.assertRaisesRegex(ValueError,'distinct actual replay PID'):
            a.DualRoutes(serving,stream([cap]),stream([{**rep,'pid':cap['pid']}]),cap,source)
        routes=dual([first]);routes.cursors[1].stream=io.BytesIO(json.dumps(first).encode())
        with self.assertRaisesRegex(ValueError,'truncated'):routes.route('HTTP-C8',3,None,8,1023,1536)

    def case(self,routes):
        base=a.load(a.BASE,a.BASE_SHA);counts={}
        cls=a.route_case_type(base,routes,counts)
        # Numerical gate is mocked only in the isolated routing tests below.
        config={'vocab_size':248320,'layer_types':['linear_attention','linear_attention','linear_attention','full_attention']*16}
        case=cls('D-C8',numeric.synthetic(8)[2],config,[99]);case.name='HTTP-C8';case.step=3
        return base,case,counts

    def test_current_and_future_calls_then_consumed_current_skipped(self):
        current=['decode',['x'],[1022],[7]];future=['decode',['x'],[1023],[8]]
        for consumed in (None,current):
            events=[] if consumed else [row(B=1)]
            events.append(row(B=1,used=1024,functions=2));routes=dual(events)
            if consumed:
                for cursor in routes.cursors:cursor.families={1023}
            base,case,counts=self.case(routes);case.old_ids=['x'];case.capacity=1536
            r=dict(schedule=dict(decode=['x'],requests=[dict(rid='x',prompt_ids=[1]*978,sampling={'max_tokens':512})]),
                arrays=[],ahead={'candidate':dict(consumed_prior_future=consumed,discarded_prior_future=None)},
                after={'engine':{'batch_rids':['x']}})
            with patch.object(base.CaseAudit,'check_arrays') as inherited:
                case.check_arrays(r,[current,future],set(),True,1);inherited.assert_called_once()
            self.assertEqual(routes.finish()['calls'],1 if consumed else 2)
            self.assertEqual(counts['HTTP-C8']['consumed'],int(consumed is not None))

    def test_derived_rebuild_capacity_fallback_and_array_capacity_gate(self):
        routes=dual([row(B=1,used=257,L=768,functions=0)])
        base,case,_=self.case(routes)
        r=dict(schedule=dict(decode=['new'],requests=[dict(rid='new',prompt_ids=[1]*256,sampling={'max_tokens':512})]),
            arrays=[dict(check="call-state ('decode', anything)",name='root/3/K',shape=[1,4,768,256])],
            ahead={'candidate':dict(consumed_prior_future=None,discarded_prior_future=None)},after={'engine':{'batch_rids':['new']}})
        with patch.object(base.CaseAudit,'check_arrays'):
            case.check_arrays(r,[['decode',['new'],[256],[7]]],set(),False,1)
        self.assertEqual(case.capacity,768);self.assertEqual(routes.cursors[0].fallbacks,1)
        with self.assertRaisesRegex(ValueError,'no actual graph'):routes.finish()
        base,case,_=self.case(dual([]));r['arrays'][0]['shape'][2]=1536
        with patch.object(base.CaseAudit,'check_arrays'),self.assertRaisesRegex(ValueError,'reconstructed padded'):case.check_arrays(r,[],set(),False,1)

    def test_inherited_full64_bits_finite_and_lifecycle_not_weakened(self):
        http=a.load(a.HTTP_AUDIT,a.HTTP_AUDIT_SHA)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'steps.jsonl'
            for corruption in (None,'bit','finite','missing-layer','unfinished-release'):
                base,recording,fixtures,config,claimed,rows=numeric.synthetic(8)
                routes=SimpleNamespace(route=lambda *args: self.fail('short toy should not take long route'))
                base.CaseAudit=a.route_case_type(base,routes,{})
                if corruption=='bit':rows[2]['arrays'][0]['bit_equal']=False
                elif corruption=='finite':rows[2]['arrays'][0]['candidate_finite']=False
                elif corruption=='missing-layer':rows[2]['arrays'].pop()
                elif corruption=='unfinished-release':rows.pop()
                path.write_bytes(b''.join((json.dumps(r)+'\n').encode() for r in rows))
                if corruption:
                    with self.subTest(corruption=corruption),self.assertRaises(ValueError):http.stream_ledger(path,recording,fixtures,config,claimed,base)
                else:
                    result=http.stream_ledger(path,recording,fixtures,config,claimed,base)
                    self.assertEqual(result['requests'],16);self.assertGreater(result['bit_equal_checks'],16*128)

    def test_active_failed_missing_terminal_never_opens_ledger(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'base').mkdir();(root/'base/steps.jsonl').write_text('must not parse\n')
            for status,expected in (('INCOMPLETE','INCOMPLETE'),('PREPARED','INCOMPLETE'),('FAIL','FAIL'),('PASS','INCOMPLETE')):
                (root/'complete.json').write_text(json.dumps({'status':status}))
                with patch.object(a,'load',side_effect=AssertionError('no helper/ledger before terminal')):
                    result=a.audit(root,'not-a-plan')
                self.assertEqual(result['status'],expected);self.assertFalse(result['ledger_opened'] or result['qualified'])

    def test_prerequisite_is_exact_archived_receipt_not_a_same_schema_pass(self):
        binding = dict(plan=ref(a.HERE/'prereg-serving-v1/serving-plan.json'), audit=ref(a.SERVING_RECEIPT))
        result = a.serving_prerequisite(producer, binding)
        self.assertEqual(result['audit_sha256'], a.SERVING_RECEIPT_SHA)
        with tempfile.TemporaryDirectory() as temp:
            other=Path(temp)/'copied-pass.json';other.write_bytes(a.SERVING_RECEIPT.read_bytes())
            # Even unchanged receipt bytes at an unregistered path do not
            # silently substitute an arbitrary claimed serving prerequisite.
            with self.assertRaisesRegex(ValueError,'immutable real37/42'):
                a.serving_prerequisite(producer,{**binding,'audit':ref(other)})
        with self.assertRaisesRegex(ValueError,'immutable real37/42'):
            a.serving_prerequisite(producer,{**binding,'audit':{**binding['audit'],'sha256':'0'*64}})


class CaptureBindingTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_capture_cleanup_receipts_and_top_level_inventory(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp).resolve()/'capture';base=capture.donor.HERE/'prereg-qualification-v1/plan.json'
            numeric_plan=producer.read(base)
            spec=capture.prepare(base,producer.BASE_PLAN_SHA,8,python=numeric_plan['execution']['python'],
                environment=numeric_plan['execution']['environment'],
                transport=dict(port=18181,tcp_nodelay=True,io_timeout_seconds=300),
                deadlines=dict(ready=1,request=1,group=1,close=.05,term=.05,kill=.05,overall=3,resource_interval=.01))
            async def checkpoint(*args):pass
            result=await capture.capture(spec,out,checkpoint=checkpoint,backend=ReceiptBackend(spec))
            self.assertEqual(result['status'],'PASS')
            plan=dict(concurrency=8,qualification=spec['base_plan'],
                capture={k:ref(out/name) for k,name in producer.CAPTURE_NAMES.items()},
                inputs={k:ref(out/name) for k,name in producer.INPUT_NAMES.items()})
            got,header=a.capture_binding(producer,plan);self.assertEqual(got,spec);self.assertEqual(header['pid'],23456)
            # Rebound hashes cannot turn a successful probe of another PID or
            # port into evidence about this owned child/listener.
            for name in ('child-identity.json','child-absence.json','listener.json','port-after.json'):
                original=(out/name).read_bytes();changed=a.read(out/name)
                command=changed['command']
                if '-p' in command:command[command.index('-p')+1]='23457'
                else:command[3]='-iTCP:18182'
                (out/name).write_text(json.dumps(changed)+'\n')
                for key,filename in producer.CAPTURE_NAMES.items():
                    if filename==name:plan['capture'][key]=ref(out/name)
                complete=a.read(out/'complete.json')
                complete['artifacts'][name]={'sha256':a.sha(out/name),'bytes':(out/name).stat().st_size}
                (out/'complete.json').write_text(json.dumps(complete)+'\n');plan['capture']['complete']=ref(out/'complete.json')
                with self.subTest(probe=name),self.assertRaisesRegex(ValueError,'actual PID/port probe command'):
                    a.capture_binding(producer,plan)
                (out/name).write_bytes(original)
                for key,filename in producer.CAPTURE_NAMES.items():
                    if filename==name:plan['capture'][key]=ref(out/name)
                complete['artifacts'][name]={'sha256':a.sha(out/name),'bytes':(out/name).stat().st_size}
                (out/'complete.json').write_text(json.dumps(complete)+'\n');plan['capture']['complete']=ref(out/'complete.json')
            (out/'server.stderr').write_bytes(b'changed after closure')
            with self.assertRaisesRegex(ValueError,'artifact'):a.capture_binding(producer,plan)


if __name__=='__main__':unittest.main()
