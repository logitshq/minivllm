"""Dynamic-history and unchanged full-checker tests; no actual E053 audit."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parents[2]))
s=importlib.util.spec_from_file_location('_e053_audit_tests',HERE/'audit.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
from bench.m5_ultra_256gb.pareto_realistic import test_audit_http_replay_v2_cpu as fixture


def terminal(steps):
    case=dict(name='HTTP-C8',status='PASS',steps=steps,completed=[dict(rid=str(i)) for i in range(16)])
    top=dict(schema='c8-batched-http-replay-result-v1',status='PASS',replay_numerical_checks_passed=True,
        source_input_pins_rechecked=True,independent_audit_required=True,numerical_qualified=False,http_qualified=False,
        performance_qualified=False,owner_resources_restore_qualified=False,pid=123,ppid=122,steps=steps,total_requests=16,case=case)
    inner=dict(status='PASS',plan_sha256='21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb',
        qualifier_sha256='26964fcbed9a2e5dcefaee4888244987897f3138d4a2b7add59bf63a7461b7bf',
        numeric_only_not_http=True,source_model_pins_rechecked=True,total_requests=16,cases=[case])
    return top,inner


class DynamicAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.routes,cls.a=m.support()

    def test_new_steps_are_not_assumed_equal_to_old1030(self):
        for steps in (1019,1030,1041):
            top,inner=terminal(steps);self.assertEqual(m.terminal(self.a,top,inner)['steps'],steps)
        for mutate in (lambda t,i:t.update(steps=0),lambda t,i:t.update(steps=True),
                       lambda t,i:t.update(performance_qualified=True),lambda t,i:i['cases'][0]['completed'].pop()):
            top,inner=terminal(1019);mutate(top,inner)
            with self.assertRaises(ValueError):m.terminal(self.a,top,inner)

    def test_dynamic_output_counts_preserve_terminal_eos_difference(self):
        recording=dict(steps=[{}]*23,jobs={str(i):dict(raw_ids=[7,99],committed_ids=[7]) for i in range(16)})
        self.assertEqual(m.dynamic_counts(recording),dict(expected_steps=23,expected_requests=16,
            expected_raw_outputs=32,expected_committed_outputs=16))

    def test_full_math_still_rejects_wrong_array_and_unfinished_release(self):
        http=self.a.load(self.a.HTTP_AUDIT,self.a.HTTP_AUDIT_SHA)
        for bad in (None,'array','release'):
            base,recording,fixtures,config,claimed,rows=fixture.synthetic(8)
            dummy=SimpleNamespace(route=lambda *args:self.fail('toy long route'),external_decode=lambda *args:self.fail('toy external decode'))
            facade=SimpleNamespace(**{**vars(base),'CaseAudit':self.routes.route_case_type(base,dummy,{})})
            if bad=='array':rows[2]['arrays'][0]['bit_equal']=False
            elif bad=='release':rows.pop()
            with tempfile.TemporaryDirectory() as temp:
                path=Path(temp)/'steps.jsonl';path.write_bytes(b''.join((json.dumps(r)+'\n').encode() for r in rows))
                if bad:
                    with self.assertRaises(ValueError):http.stream_ledger(path,recording,fixtures,config,claimed,facade)
                else:
                    result=http.stream_ledger(path,recording,fixtures,config,claimed,facade)
                    self.assertEqual(result['requests'],16);self.assertGreater(result['bit_equal_checks'],16*128)

    def test_actual_producer_private_trace_sources_are_pinned_without_import(self):
        for path,digest in ((m.QUALIFIER,m.QUALIFIER_SHA),(m.SERVER,m.SERVER_SHA),(m.TRIAL,m.TRIAL_SHA),
                            (m.INTEGRATION,m.INTEGRATION_SHA),(m.GDN,m.GDN_SHA),(m.ROUTES,m.ROUTES_SHA)):
            self.assertEqual(self.a.sha(path),digest)


if __name__=='__main__':unittest.main(verbosity=2)
