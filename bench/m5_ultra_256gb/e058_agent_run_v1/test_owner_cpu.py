"""Prepared-only CPU mechanisms: every qualifier, process and signal is mocked."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


h = load('_e058_test_hold', HERE/'terminal_hold.py')
r = load('_e058_test_runner', HERE/'runner.py')
e = load('_e058_test_execute', HERE/'execute.py')


class TerminalHold(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.output = self.root/'numerical'; self.output.mkdir()
        self.command = [str(h.R/'.venv/bin/python'),'-I','-B',str(h.QUALIFIER),
                        '--run','--plan','/cpu-only/plan.json','--plan-sha256','a'*64,'--out',str(self.output)]
        self.path = self.root/'command.json'; self.path.write_text(json.dumps(self.command))
        self.waits = []; self.qualifier_calls = []; self.exit_code = 0

    def qualifier(self, path, run_name):
        self.qualifier_calls.append((path,run_name,list(sys.argv)))
        self.assertEqual(path, str(h.QUALIFIER)); self.assertEqual(run_name, '__main__')
        self.assertEqual(sys.argv, self.command[3:])
        (self.output/'complete.json').write_text(json.dumps(dict(
            status='PASS' if self.exit_code == 0 else 'FAIL', pid=123, ppid=456)))
        raise SystemExit(self.exit_code)

    def ack(self, delay, change=None):
        self.waits.append(delay)
        done = self.root/'numerical-done.json'
        self.assertTrue(done.is_file(), 'must wait after original qualifier completes')
        self.assertFalse((self.root/'numerical-exit-ack.json').exists())
        row = dict(pid=123,owner_pid=456,done_sha256=h.sha(done),command=self.command,
                   final_physical_and_metadata_complete=True)
        if change: change(row)
        h.atomic(self.root/'numerical-exit-ack.json',row)

    def call(self, *, change=None, clock=None, parents=None, ack=True):
        previous = list(sys.argv)
        with ExitStack() as stack:
            stack.enter_context(patch.object(h.runpy,'run_path',side_effect=self.qualifier))
            stack.enter_context(patch.object(h.os,'getpid',return_value=123))
            stack.enter_context(patch.object(h.os,'getppid',side_effect=parents) if parents is not None
                                else patch.object(h.os,'getppid',return_value=456))
            stack.enter_context(patch.object(h.time,'monotonic',side_effect=clock) if clock is not None
                                else patch.object(h.time,'monotonic',return_value=100))
            sleep = stack.enter_context(patch.object(h.time,'sleep',side_effect=(lambda delay:self.ack(delay,change)) if ack else None))
            value = h.run(self.path,h.sha(self.path),self.root)
        self.assertEqual(sys.argv,previous,'original argv restored even after SystemExit')
        self.assertEqual(len(self.qualifier_calls),1)
        return value,sleep.call_count

    def test_original_systemexit_zero_waits_for_valid_final_sample_ack(self):
        code, waits = self.call()
        self.assertEqual((code,waits),(0,1))
        self.assertEqual(json.loads((self.root/'numerical-done.json').read_text())['qualifier_returncode'],0)
        self.assertFalse((self.root/'terminal-hold-error.json').exists())

    def test_original_systemexit_one_stays_failure_after_valid_ack(self):
        self.exit_code=1; code,waits=self.call()
        self.assertEqual((code,waits),(1,1))
        self.assertEqual(json.loads((self.root/'numerical-done.json').read_text())['qualifier_returncode'],1)
        self.assertFalse((self.root/'terminal-hold-error.json').exists(),'original failure is not an ack failure')

    def rejected_ack(self, change):
        code,waits=self.call(change=change)
        self.assertEqual((code,waits),(1,1))
        self.assertTrue((self.root/'terminal-hold-error.json').is_file())
        self.assertEqual(json.loads((self.root/'numerical-done.json').read_text())['qualifier_returncode'],0)

    def test_ack_without_final_resource_sample_is_rejected(self):
        self.rejected_ack(lambda row:row.update(final_physical_and_metadata_complete=False))

    def test_wrong_owner_binding_is_rejected(self):
        self.rejected_ack(lambda row:row.update(owner_pid=999))

    def test_wrong_child_binding_is_rejected(self):
        self.rejected_ack(lambda row:row.update(pid=999))

    def test_wrong_original_command_is_rejected(self):
        self.rejected_ack(lambda row:row.update(command=self.command+['changed']))

    def test_wrong_done_hash_is_rejected(self):
        self.rejected_ack(lambda row:row.update(done_sha256='0'*64))

    def test_parent_change_during_hold_is_rejected(self):
        code,waits=self.call(parents=[456,999],ack=False)
        self.assertEqual((code,waits),(1,0))
        self.assertIn('owner changed',json.loads((self.root/'terminal-hold-error.json').read_text())['error'])

    def test_parent_change_when_ack_arrives_is_rejected(self):
        code,waits=self.call(parents=[456,456,999])
        self.assertEqual((code,waits),(1,1))
        self.assertIn('owner changed at ack',json.loads((self.root/'terminal-hold-error.json').read_text())['error'])

    def test_late_ack_does_not_extend_sixty_second_bound(self):
        code,waits=self.call(clock=[100,100,161])
        self.assertEqual((code,waits),(1,1))
        self.assertIn('60-second',json.loads((self.root/'terminal-hold-error.json').read_text())['error'])

    def test_missing_ack_hits_sixty_second_bound_without_real_sleep(self):
        code,waits=self.call(clock=[100,161],ack=False)
        self.assertEqual((code,waits),(1,0))
        done=json.loads((self.root/'numerical-done.json').read_text())
        self.assertEqual(done['ack_deadline_monotonic'],160)
        self.assertIn('60-second',json.loads((self.root/'terminal-hold-error.json').read_text())['error'])


class OwnerMechanisms(unittest.TestCase):
    def test_capture_predecessor_is_sealed_v2_not_failed_v1(self):
        with patch.object(e,'sha',return_value=e.CAPTURE_PLAN_SHA) as digest:
            self.assertEqual(e.capture_predecessor(),e.B/'rowwise_mlp_shared_cost_device_v1/prereg-v1/plan.json')
        self.assertEqual(digest.call_count,1)
        self.assertEqual(e.sha(e.SCREEN),e.SCREEN_SHA)

    def test_unsealed_capture_is_rejected_before_reading_predecessor(self):
        with patch.object(e,'CAPTURE_PLAN_SHA','UNSEALED'),patch.object(e,'sha',side_effect=AssertionError('must not read')) as digest:
            with self.assertRaisesRegex(AssertionError,'unsealed'):e.capture_predecessor()
        digest.assert_not_called()

    def test_child_environment_replaces_all_mlx_keys_without_parent_mutation(self):
        with patch.dict(os.environ,{'MLX_STALE':'bad','MLX_MAX_OPS_PER_BUFFER':'999'},clear=False):
            before=dict(os.environ); expected={'MLX_MAX_OPS_PER_BUFFER':'50'}
            actual=r.child_environment(expected)
            self.assertEqual({k:v for k,v in actual.items() if k.startswith('MLX_')},expected)
            self.assertEqual(dict(os.environ),before)

    def test_observation_failure_reaps_original_popen_and_preserves_fail(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            out=Path(tmp)/'owner'; parent=os.getpid()
            service=dict(pid=111,ppid=1,start='service',executable='omlx-server')
            identity=dict(pid=222,ppid=parent,start='child',executable='python')
            class Child:
                pid=222; returncode=None
                def __init__(self): self.actions=[]; self.waits=[]
                def poll(self): return self.returncode
                def terminate(self): self.actions.append('TERM'); self.returncode=-15
                def kill(self): self.actions.append('KILL'); self.returncode=-9
                def wait(self,timeout=None): self.waits.append(timeout); return self.returncode
            child=Child(); probes=[]
            def command(argv,**kw):
                probes.append(argv)
                if argv[:2]==['/bin/ps','-p']:
                    if argv[-1]=='pid=': return subprocess.CompletedProcess(argv,1,'','')
                    return subprocess.CompletedProcess(argv,0,'T\n','')
                return subprocess.CompletedProcess(argv,0,json.dumps(dict(power_metadata={'source':'AC Power'},processinfo_thermal_metadata={'code':0})),'')
            stack.enter_context(patch.object(r,'verify_pins'))
            stack.enter_context(patch.object(r,'process',side_effect=lambda pid:service if pid==111 else identity))
            stack.enter_context(patch.object(r.subprocess,'run',side_effect=command))
            stack.enter_context(patch.object(r.subprocess,'Popen',return_value=child))
            stack.enter_context(patch.object(r.signal,'getsignal',return_value=signal.SIG_DFL))
            stack.enter_context(patch.object(r.signal,'signal'))
            stack.enter_context(patch.object(r.time,'monotonic',return_value=100))
            stack.enter_context(patch.object(r.time,'sleep'))
            stack.enter_context(patch.object(r.physical,'measure',side_effect=[dict(started_monotonic=100,ended_monotonic=100),RuntimeError('injected post-spawn observation failure')]))
            stack.enter_context(patch.object(r.physical,'checked',return_value=dict(top_bytes=1,aux_current_bytes=1,aux_peak_bytes=1,swap_bytes=0,swapouts=0)))
            stack.enter_context(patch.object(r.resources,'guard'))
            plan=dict(pins={},service=dict(pid=111,process=service),physical_baseline=dict(swap_bytes=0,swapouts=0),
                      screen_deadline_seconds=1650,max_observation_gap_seconds=12,
                      numerical_command=['/cpu-only/qualifier'],child_environment={},outputs={'numerical':str(Path(tmp)/'numerical')})
            self.assertEqual(r.run(plan,out),1)
            result=json.loads((out/'result.json').read_text()); reap=json.loads((out/'child-reap-raw.json').read_text())
            self.assertEqual(result['status'],'FAIL'); self.assertTrue(result['child_reaped'])
            self.assertIn('injected post-spawn',result['errors'][0]['message'])
            self.assertEqual(child.actions,['TERM']); self.assertTrue(child.waits)
            self.assertEqual(reap['status'],'REAPED'); self.assertEqual(reap['returncode'],-15)
            self.assertTrue(reap['waited_reaped'] and reap['strict_absence'])
            self.assertIn(['/bin/ps','-p','222','-o','pid='],probes)

    def test_final_physical_sample_ack_then_normal_exit_is_reaped(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            out=Path(tmp)/'owner'; numerical=Path(tmp)/'numerical';numerical.mkdir();parent=os.getpid()
            service=dict(pid=111,ppid=1,start='service',executable='omlx-server')
            identity=dict(pid=222,ppid=parent,start='child',executable='python')
            terminal=complete();terminal.update(pid=222,ppid=parent)
            r.write(numerical/'complete.json',terminal)
            command=['/cpu-only/qualifier'];actions=[]
            class Child:
                pid=222;returncode=None
                def poll(self):
                    if not (out/'numerical-done.json').exists():
                        r.hold.atomic(out/'numerical-done.json',dict(schema='E058-terminal-hold-v1',pid=222,ppid=parent,
                            command=command,qualifier_returncode=0,wrapper_errors=[],ack_deadline_monotonic=160,
                            qualifier_complete=dict(path=str(numerical/'complete.json'),sha256=r.sha(numerical/'complete.json'))))
                    return self.returncode
                def wait(self,timeout=None):
                    ack=json.loads((out/'numerical-exit-ack.json').read_text())
                    assert ack['final_physical_and_metadata_complete'] is True
                    assert ack['done_sha256']==r.sha(out/'numerical-done.json')
                    actions.append('wait-after-final-sample');self.returncode=0;return 0
                def terminate(self):raise AssertionError('normal held exit must not be terminated')
                def kill(self):raise AssertionError('normal held exit must not be killed')
            child=Child()
            def process_command(argv,**kwargs):
                if argv[:2]==['/bin/ps','-p']:
                    return subprocess.CompletedProcess(argv,1 if argv[-1]=='pid=' else 0,'' if argv[-1]=='pid=' else 'T\n','')
                return subprocess.CompletedProcess(argv,0,json.dumps(dict(power_metadata={'source':'AC Power'},processinfo_thermal_metadata={'code':0})),'')
            def measure(directory,pid):
                directory.mkdir();row=dict(started_monotonic=100,ended_monotonic=100)
                r.write(directory/'sample.json',row);actions.append('physical-'+str(pid));return row
            stack.enter_context(patch.object(r,'verify_pins'))
            stack.enter_context(patch.object(r,'process',side_effect=lambda pid:service if pid==111 else identity))
            stack.enter_context(patch.object(r.subprocess,'run',side_effect=process_command))
            stack.enter_context(patch.object(r.subprocess,'Popen',return_value=child))
            stack.enter_context(patch.object(r.signal,'getsignal',return_value=signal.SIG_DFL))
            stack.enter_context(patch.object(r.signal,'signal'))
            stack.enter_context(patch.object(r.time,'monotonic',return_value=100))
            stack.enter_context(patch.object(r.time,'sleep'))
            stack.enter_context(patch.object(r.physical,'measure',side_effect=measure))
            stack.enter_context(patch.object(r.physical,'checked',return_value=dict(top_bytes=1,aux_current_bytes=1,aux_peak_bytes=1,swap_bytes=0,swapouts=0)))
            stack.enter_context(patch.object(r.resources,'guard'))
            plan=dict(pins={},service=dict(pid=111,process=service),physical_baseline=dict(swap_bytes=0,swapouts=0),
                screen_deadline_seconds=1650,max_observation_gap_seconds=12,numerical_plan_sha256='abc',
                numerical_command=command,child_environment={},outputs={'numerical':str(numerical)})
            self.assertEqual(r.run(plan,out),0)
            result=json.loads((out/'result.json').read_text())
            self.assertEqual(result['status'],'PASS_CAPTURE_ONLY');self.assertEqual(result['physical_samples'],2)
            self.assertTrue(result['child_reaped']);self.assertEqual(result['child_returncode'],0)
            self.assertLess(actions.index('physical-222'),actions.index('wait-after-final-sample'))

    def test_functiontype_reuse_rebinds_all_local_globals_without_shared_mutation(self):
        # These deliberately refer to names supplied by the cloned globals dict.
        def inherited_bind(*args): return runner,HERE,verify_payloads
        def inherited_cleanup(*args): return bind_runner()
        def inherited_run(expected):
            runner.seen=(expected,HERE,verify_payloads,bind_runner(),bounded_cleanup())
            raise RuntimeError('captured cloned namespace before any runtime action')
        fake=SimpleNamespace(seen=None); replacement=lambda *args:None
        original=(e.outer.runner,e.outer.HERE,e.outer.verify_payloads)
        with patch.object(e,'runner',fake),patch.object(e,'preserve_sources',replacement),\
             patch.object(e.outer,'bind_runner',inherited_bind),patch.object(e.outer,'bounded_cleanup',inherited_cleanup),\
             patch.object(e.outer,'run',inherited_run):
            with self.assertRaisesRegex(RuntimeError,'captured cloned namespace'): e.run('prepared-sha')
        expected=(fake,e.HERE,replacement)
        self.assertEqual(fake.seen,('prepared-sha',e.HERE,replacement,expected,expected))
        self.assertEqual((e.outer.runner,e.outer.HERE,e.outer.verify_payloads),original)


def complete():
    t=r.load(HERE/'timing.py','_e058_test_timing')
    rows=[dict(width=w,group=g,repeat=j,ordinal=o,arm=arm,elapsed_ns=100)
          for w,g,j,o,arm in t.schedule()]
    return dict(schema='E058-feature-result-v1',status='PASS_FEATURES_LOCAL_COST',pid=123,ppid=456,
        plan_sha256='abc',numerical_screen_qualified=True,pre_timing_parameters_unchanged=True,
        device_cleanup_completed=True,source_model_pins_rechecked=True,canonical_unchanged=True,
        independent_raw_audit_pending=True,http_qualified=False,performance_qualified=False,engine_pareto_point=False,
        completed_pairs=24,timings=rows,timing_summary=json.loads(json.dumps(t.reduce(rows))),raw_manifest=dict(path='pairs.json',sha256='a'*64))

class CostCompletion(unittest.TestCase):
    def test_complete_qualified_cost_without_engine_performance_claim(self):
        valid=complete();plan=dict(numerical_plan_sha256='abc');r.validate_complete(valid,123,456,plan)
        for key,value in [('status','PASS'),('numerical_screen_qualified',False),('device_cleanup_completed',False),
                          ('http_qualified',True),('performance_qualified',True),('ppid',999),('completed_pairs',23),
                          ('canonical_unchanged',False),('pre_timing_parameters_unchanged',False),('timings',valid['timings'][:-1])]:
            with self.subTest(field=key),self.assertRaises(AssertionError):r.validate_complete({**valid,key:value},123,456,plan)
    def test_repeated_pair_wrong_order_nonpositive_or_changed_summary_rejected(self):
        for change in (lambda d:d['timings'].__setitem__(2,dict(d['timings'][0])),
                       lambda d:d['timings'][0].update(arm='shared_with_features'),
                       lambda d:d['timings'][0].update(elapsed_ns=0),
                       lambda d:d.update(timing_summary={}),
                       lambda d:d['raw_manifest'].update(sha256='bad')):
            bad=complete();change(bad)
            with self.assertRaises(AssertionError):r.validate_complete(bad,123,456,dict(numerical_plan_sha256='abc'))

if __name__ == '__main__': unittest.main()
