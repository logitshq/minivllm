"""Independent evidence negatives; fake model/OS only, no numerical device claim."""
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bench.m5_ultra_256gb.gdn_graph_combined import http_audit as a, http_capture as capture, http_qualification as producer
from bench.m5_ultra_256gb.gdn_graph_combined import test_qualify_cpu as f
from bench.m5_ultra_256gb.gdn_graph_combined.test_http_cpu import Backend, arguments
from bench.m5_ultra_256gb.route_decode_graph.test_http_server_cpu import context
from bench.m5_ultra_256gb.route_decode_graph.test_http_audit_v2_cpu import clock_group
from bench.m5_ultra_256gb.pareto_realistic import test_audit_http_replay_v2_cpu as numeric


def stream(rows):return io.BytesIO(b''.join((json.dumps(r)+'\n').encode() for r in rows))
def ref(path):return dict(path=str(path),sha256=a.sha(path))
def rows(C=1):
    graph,proxy,witness,stats,events=f.proxy_fixture();witness.phase=f'HTTP-C{C}';witness.step=1
    for _ in range(2):witness(SimpleNamespace(shape=(C,)),context(B=C))
    return events

def dual(left,right=None,C=1):
    captured=dict(event='route-startup',schema='GDN-graph-combined-http-events-v1',pid=23456,phase=f'HTTP-C{C}',identity={})
    source={'path':'/CPU/route-events.jsonl','sha256':'a'*64}
    replay=dict(event='replay-route-startup',schema='GDN-graph-combined-http-replay-events-v1',pid=23457,
                phase=f'HTTP-C{C}',captured_route_events=source)
    return a.DualRoutes(a.load(a.SERVING,a.SERVING_SHA),stream([captured,*left]),
        stream([replay,*(left if right is None else right)]),captured,source)


class Tests(unittest.TestCase):
    def test_real_producer_trace_and_reuse_independently_checked_each_stream(self):
        for C in (1,8):
            events=rows(C);r=dual(events,C=C)
            for _ in range(2):r.route(f'HTTP-C{C}',1,None,C,1023,1536)
            result=r.finish();self.assertEqual(result['calls'],2)
            self.assertEqual(result['combined']['compiled-tracing'],1);self.assertEqual(result['combined']['compiled-reuse'],1)
            self.assertEqual(result['combined']['traced_batches'][0]['B'],C)

    def test_both_identically_wrong_sidecars_do_not_supply_oracle(self):
        for field,value in (('B',8),('used',1024),('step',True),('installation',1),
                            ('actual_invocations',0),('kind','compiled-reuse'),('per_layer_candidate_delta',True)):
            bad=rows();bad[0][field]=value;r=dual(bad)
            with self.subTest(field=field),self.assertRaises(ValueError):r.route('HTTP-C1',1,None,1,1023,1536)
        bad=rows();bad[0]['after'][-1]['candidate']=0;r=dual(bad)
        with self.assertRaises(ValueError):r.route('HTTP-C1',1,None,1,1023,1536)

    def test_reuse_requires_own_prior_trace_and_continuous_counters(self):
        good=rows();r=dual(good[2:])
        with self.assertRaises(ValueError):r.route('HTTP-C1',1,None,1,1023,1536)
        bad=copy.deepcopy(good);bad[2]['before'][0]['candidate']+=1;r=dual(bad)
        r.route('HTTP-C1',1,None,1,1023,1536)
        with self.assertRaises(ValueError):r.route('HTTP-C1',1,None,1,1023,1536)

    def test_missing_extra_and_reordered_rows_reject(self):
        good=rows()
        for bad in (good[:-1],good+[good[-1]],good[1:]+good[:1]):
            r=dual(good,bad)
            with self.assertRaises(ValueError):
                for _ in range(2):r.route('HTTP-C1',1,None,1,1023,1536)
                r.finish()

    def test_frozen_full64_bit_finite_shape_and_terminal_gates_retained(self):
        http=a.load(a.HTTP_AUDIT,a.HTTP_AUDIT_SHA)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'steps.jsonl'
            for corruption in (None,'bit','finite','missing-layer','unfinished-release'):
                base,recording,fixtures,config,claimed,ledger=numeric.synthetic(8)
                original=base.CaseAudit
                routes=SimpleNamespace(route=lambda *args:self.fail('short fixture cannot long-route'),
                    external_decode=lambda *args:self.fail('short fixture cannot external-decode'))
                facade=SimpleNamespace(**{**vars(base),'CaseAudit':a.route_case_type(base,routes,{})})
                if corruption=='bit':ledger[2]['arrays'][0]['bit_equal']=False
                elif corruption=='finite':ledger[2]['arrays'][0]['candidate_finite']=False
                elif corruption=='missing-layer':ledger[2]['arrays'].pop()
                elif corruption=='unfinished-release':ledger.pop()
                path.write_bytes(stream(ledger).getvalue())
                if corruption:
                    with self.subTest(corruption=corruption),self.assertRaises(ValueError):http.stream_ledger(path,recording,fixtures,config,claimed,facade)
                else:self.assertEqual(http.stream_ledger(path,recording,fixtures,config,claimed,facade)['requests'],16)
                self.assertIs(base.CaseAudit,original)

    def test_consumed_ahead_is_skipped_but_beyond_bound_fallback_advances(self):
        base=a.load(a.BASE,a.BASE_SHA);events=[]
        route=SimpleNamespace(route=lambda *args:events.append(('route',args)),external_decode=lambda *args:events.append(('external',args)))
        cls=a.route_case_type(base,route,{})
        config=dict(vocab_size=248320,layer_types=['linear_attention']*3+['full_attention'])
        case=cls('D-C1',numeric.synthetic(1)[2],config,[99]);case.name='HTTP-C1';case.step=2;case.old_ids=['x'];case.capacity=1792
        current=['decode',['x'],[1535],[7]];future=['decode',['x'],[1536],[8]]
        row=dict(schedule=dict(decode=['x'],requests=[dict(rid='x',prompt_ids=[1]*1000,sampling={'max_tokens':512})]),arrays=[],
            ahead={'candidate':dict(consumed_prior_future=current,discarded_prior_future=None)},after={'engine':{'batch_rids':['x']}})
        with patch.object(base.CaseAudit,'check_arrays') as inherited:case.check_arrays(row,[current,future],set(),True,1)
        inherited.assert_called_once();self.assertEqual(events,[('external',('HTTP-C1',1))])

    def test_actual_post_done_concurrency_not_claimed_C(self):
        group,spec=clock_group(8);self.assertEqual(a.http_intervals(group,spec)['max_http_outstanding'],8)
        group,spec=clock_group(8,serial=True)
        with self.assertRaises(ValueError):a.http_intervals(group,spec)

    def test_active_failed_or_missing_terminal_never_opens_ledger(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'base').mkdir();(root/'base/steps.jsonl').write_text('not read')
            for status,expected in (('INCOMPLETE','INCOMPLETE'),('PREPARED','INCOMPLETE'),('FAIL','FAIL'),('PASS','INCOMPLETE')):
                (root/'complete.json').write_text(json.dumps({'status':status}))
                with patch.object(a,'load',side_effect=AssertionError('no helper before terminal')):result=a.audit(root,'0'*64)
                self.assertEqual(result['status'],expected);self.assertFalse(result['ledger_opened'] or result['qualified'])


class ReceiptBackend(Backend):
    async def identity(self,pid):
        return {**await super().identity(pid),'command':['/bin/ps','-p',str(pid),'-o','pid=,ppid=,lstart=,comm=']}
    async def absent(self,pid):return {**await super().absent(pid),'command':['/bin/ps','-p',str(pid),'-o','pid=']}
    async def port(self,port,pid=None):
        return {**await super().port(port,pid),'command':['/usr/sbin/lsof','-nP','-a']+
            (['-p',str(pid)] if pid is not None else [])+[f'-iTCP:{port}','-sTCP:LISTEN','-Fpn']}


class ReceiptTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_cleanup_binding_and_wrong_pid_even_rehashed_reject(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp).resolve()/'capture';path,kw=arguments();spec=capture.prepare(path,capture.sha(path),1,**kw)
            async def checkpoint(*args):pass
            result=await capture.capture(spec,out,checkpoint=checkpoint,backend=ReceiptBackend(spec))
            self.assertEqual(result['status'],'PASS')
            plan=dict(concurrency=1,qualification=spec['base_plan'],capture={k:ref(out/n) for k,n in producer.CAPTURE_NAMES.items()},
                inputs={k:ref(out/n) for k,n in producer.INPUT_NAMES.items()})
            actual,header=a.capture_binding(producer,plan);self.assertEqual(actual,spec)
            for name in ('child-identity.json','child-absence.json','listener.json','port-after.json'):
                saved=(out/name).read_bytes();bad=a.read(out/name)
                if '-p' in bad['command']:bad['command'][bad['command'].index('-p')+1]='23457'
                else:bad['command'][3]='-iTCP:18182'
                producer.write(out/name,bad)
                for key,n in producer.CAPTURE_NAMES.items():
                    if n==name:plan['capture'][key]=ref(out/name)
                result['artifacts'][name]=dict(sha256=a.sha(out/name),bytes=(out/name).stat().st_size)
                producer.write(out/'complete.json',result);plan['capture']['complete']=ref(out/'complete.json')
                with self.subTest(name=name),self.assertRaises(ValueError):a.capture_binding(producer,plan)
                (out/name).write_bytes(saved)
                for key,n in producer.CAPTURE_NAMES.items():
                    if n==name:plan['capture'][key]=ref(out/name)
                result['artifacts'][name]=dict(sha256=a.sha(out/name),bytes=(out/name).stat().st_size)
                producer.write(out/'complete.json',result);plan['capture']['complete']=ref(out/'complete.json')


if __name__=='__main__':unittest.main()
