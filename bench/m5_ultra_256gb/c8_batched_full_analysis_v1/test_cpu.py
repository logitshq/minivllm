"""Small synthetic descriptor checks; no active E051 output is opened."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parents[2]))
s=importlib.util.spec_from_file_location('_e051_audit_tests',HERE/'audit.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
from bench.m5_ultra_256gb.pareto_realistic import test_audit_http_replay_v2_cpu as fixture


def terminal():
    case=dict(status='PASS',name='HTTP-C8',steps=1030,completed=[dict(rid=str(i)) for i in range(16)],
        raw_output_count=8192,committed_output_count=8192)
    inner=dict(status='PASS',plan_sha256=m.BASE_PLAN_SHA,
        qualifier_sha256='26964fcbed9a2e5dcefaee4888244987897f3138d4a2b7add59bf63a7461b7bf',
        numeric_only_not_http=True,source_model_pins_rechecked=True,total_requests=16,cases=[case])
    top=dict(schema='c8-batched-gdn-graph-full-result-v1',status='PASS',source_input_pins_rechecked=True,
        full_history_qualified=True,http_qualified=False,performance_qualified=False,
        qualification_scope='Only the exact registered E047 recording; no new candidate HTTP capture.',
        pid=123,ppid=122,steps=1030,total_requests=16,case=case)
    return top,inner


class FullAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous,cls.a=m.support();cls.http=cls.a.load(cls.a.HTTP_AUDIT,cls.a.HTTP_AUDIT_SHA)

    def test_full_terminal_accepts_only_complete_recorded_scope(self):
        top,inner=terminal();self.assertEqual(m.full_terminal(self.a,top,inner)['steps'],1030)
        for mutate in (lambda t,i:t.update(steps=49),lambda t,i:t.update(total_requests=0),
                       lambda t,i:t.update(http_qualified=True),lambda t,i:t.update(performance_qualified=True),
                       lambda t,i:i['cases'][0]['completed'].pop(),lambda t,i:i['cases'][0].update(raw_output_count=8191)):
            t,i=terminal();mutate(t,i)
            with self.assertRaises(ValueError):m.full_terminal(self.a,t,i)

    def streamed(self,mutation=None):
        base,recording,fixtures,config,claimed,rows=fixture.synthetic(8)
        base.CaseAudit=self.a.route_case_type(base,SimpleNamespace(route=lambda *args:self.fail('short toy cannot call long route')), {})
        if mutation:mutation(rows,claimed)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'steps.jsonl'
            path.write_bytes(b''.join((json.dumps(r)+'\n').encode() for r in rows))
            return self.http.stream_ledger(path,recording,fixtures,config,claimed,base)

    def test_actual_full_checker_drains_synthetic16_with_complete_array_coverage(self):
        result=self.streamed();self.assertEqual(result['requests'],16)
        self.assertGreater(result['bit_equal_checks'],16*128)
        self.assertGreater(result['ahead_counts']['consumed'],0)

    def test_full_checker_rejects_missing_state_or_bad_finite_receipt(self):
        for mutate in (lambda r,c:r[2]['arrays'].pop(),
                       lambda r,c:r[2]['arrays'][0].update(candidate_finite=False),
                       lambda r,c:r[2]['arrays'][0].update(bit_equal=False)):
            with self.assertRaises(ValueError):self.streamed(mutate)

    def test_full_checker_rejects_missing_release_or_false_completed_count(self):
        for mutate in (lambda r,c:r.pop(),lambda r,c:c['completed'].pop()):
            with self.assertRaises(ValueError):self.streamed(mutate)

    def dual(self,extra=False):
        header=dict(event='route-startup',schema='E024-route-http-events-v1',pid=123,phase='HTTP-C8',identity={})
        ref=dict(path='/cpu-only/routes.jsonl',sha256='0'*64)
        replay=dict(event='replay-route-startup',schema='E024-route-http-replay-events-v1',pid=124,phase='HTTP-C8',captured_route_events=ref)
        call=dict(event='route-call',phase='HTTP-C8',step=3,direct_label=None,status='SUBMITTED',B=8,used=1023,
                  capacities=[[1536,1536] for _ in range(16)],expected_graph=True,actual_invocations=1,functions=1)
        def stream(items):return io.BytesIO(b''.join((json.dumps(r)+'\n').encode() for r in items))
        return self.a.DualRoutes(self.a.load(self.a.SERVING,self.a.SERVING_SHA),stream([header,call]),
            stream([replay,call]+([call] if extra else [])),header,ref)

    def test_both_route_streams_require_exact_eof(self):
        routes=self.dual();routes.route('HTTP-C8',3,None,8,1023,1536);self.assertEqual(routes.finish()['calls'],1)
        routes=self.dual(extra=True);routes.route('HTTP-C8',3,None,8,1023,1536)
        with self.assertRaises(ValueError):routes.finish()
        with self.assertRaises(ValueError):self.dual().finish()

    def test_frozen_actual_qualifier_sha_without_importing_producer(self):
        self.assertEqual(self.a.sha(m.QUALIFIER),m.QUALIFIER_SHA)
        self.assertEqual(self.a.sha(m.PREVIOUS),m.PREVIOUS_SHA)


if __name__=='__main__':unittest.main(verbosity=2)
