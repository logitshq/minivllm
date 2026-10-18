"""Fake owner integration only; no real model/process/service/physical sampling."""
import ast
import contextlib
import importlib.util
import inspect
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PATH = Path(__file__).with_name('owner.py').resolve()
spec = importlib.util.spec_from_file_location('glm_dual_owner_cpu', PATH)
owner = importlib.util.module_from_spec(spec); spec.loader.exec_module(owner)


class FakeBase:
    journal_lock = staticmethod(lambda path: contextlib.nullcontext())
    @staticmethod
    def durable(path, value, exclusive=True, raw=False):
        with Path(path).open('xb' if exclusive else 'wb') as stream:
            stream.write(value if raw else json.dumps(value).encode())


class Tests(unittest.TestCase):
    def test_description_has_no_backend_and_namespace_is_actual_new_source(self):
        with patch.object(owner, 'run', side_effect=AssertionError('no execution')), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(owner.main([]), 0)
            with self.assertRaises(ValueError): owner.main(['--plan', '/missing'])
        self.assertEqual(owner.HERE, PATH.parent)
        self.assertEqual(owner.run.__globals__['__file__'], str(PATH))
        self.assertNotEqual(owner._original.HERE, owner.HERE)
        self.assertEqual(owner.SCHEMA, 'GLM-cold-load-owner-plan-v3')

    def test_exact_source_delta_keeps_prepare_reap_postflight_original(self):
        restored = owner._run_source
        for before, after in reversed(owner.RUN_DELTAS):
            self.assertEqual(restored.count(after), 1); restored = restored.replace(after, before, 1)
        self.assertEqual(ast.dump(ast.parse(restored)), ast.dump(ast.parse(inspect.getsource(owner._original.run))))
        for name in ('verify_inventory', 'child_command', 'probe_terminal', 'checkpoint'):
            self.assertIs(getattr(owner, name).__code__, getattr(owner._original, name).__code__)

    def test_missing_real_controller_fails_before_any_backend_or_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve(); probe = root/'probe-plan.json'; probe.write_text('{"source_pins": {}}')
            plan = dict(schema=owner.SCHEMA, probe_plan=owner.ref(probe), verification={}, physical_policy={},
                cold_restart=dict(controller=dict(path=str(root/'missing-controller.py'), sha256='0'*64)),
                outputs={}, budgets={}, monitor_policy=owner.MONITOR_POLICY)
            plan['execution'] = dict(owner=owner.ref(PATH), pins=owner.source_pins(plan))
            path = root/'owner-plan.json'; path.write_text(json.dumps(plan))
            args = SimpleNamespace(plan=path, plan_sha256=owner.sha(path), out=root/'out')
            with self.assertRaises((FileNotFoundError, ValueError)):
                owner.validate_plan(args)
            self.assertFalse(args.out.exists())

    def test_policy_cannot_weaken_physical_limits_or_reuse_serial_contract(self):
        base = dict(monitor_policy=dict(owner.MONITOR_POLICY), physical_policy=dict(
                    interval_seconds=1., probe_seconds=5., child_limit_bytes=220*(1 << 30)))
        with patch.object(owner, '_validated_original_contract', return_value=(base, None, None, None)):
            owner.validate_plan(None)
            for section, key, bad in [('physical_policy', 'interval_seconds', 2),
                                      ('physical_policy', 'child_limit_bytes', 240*(1 << 30)),
                                      ('physical_policy', 'probe_seconds', 6),
                                      ('monitor_policy', 'metadata_per_physical_sample', True),
                                      ('monitor_policy', 'metadata_max_gap_seconds', 6)]:
                original = base[section][key]; base[section][key] = bad
                with self.subTest(key=key), self.assertRaises(ValueError): owner.validate_plan(None)
                base[section][key] = original

    def test_live_adapter_routes_original_validators_and_keeps_separate_raw_receipts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve(); args = SimpleNamespace(out=root)
            child = SimpleNamespace(pid=543210); parent=os.getpid()
            identity=dict(pid=child.pid,ppid=parent,start='Sun Oct 18 10:00:00 2026',comm='fixture')
            policy=dict(probe_seconds=5.,interval_seconds=1.,work_seconds=1800.,child_limit_bytes=220*(1<<30))
            plan=dict(physical_policy=policy); state={}; live=[]; checkpoints=[]; calls=[]
            def read(argv, timeout=5):
                if argv[-1]=='pid=,ppid=,lstart=,comm=':
                    text=f'{child.pid} {parent} Sun Oct 18 10:00:00 2026 fixture\n';code=0;error=''
                elif argv[:2]==['/bin/ps','-p'] or argv[0]=='/usr/sbin/lsof':text=error='';code=1
                elif argv[0]=='/bin/launchctl':text='';error='not found';code=113
                elif argv[0]=='/usr/bin/pmset':text="Now drawing from 'AC Power'";error='';code=0
                elif argv[0]=='/usr/bin/osascript':text='0';error='';code=0
                else:text=f'{child.pid} {parent} fixture\n';error='';code=0
                return dict(command=argv,returncode=code,stdout=text,stderr=error)
            commands=SimpleNamespace(read=read,run=object())
            def measure(path,pid,*,timeout_seconds,run):
                self.assertIs(run,commands.run);self.assertLessEqual(timeout_seconds,5)
                Path(path).mkdir();row={'pid':pid,'ordinal':len(live)}
                FakeBase.durable(Path(path)/'sample.json',row);return row
            def observe(row,limits,cold,**kwargs):
                self.assertIs(limits,policy)
                self.assertEqual(kwargs['previous'],live[-1] if live else None);calls.append(row)
            def monitor(handle,*,out,physical,metadata,**kwargs):
                self.assertIs(handle,child);self.assertEqual(kwargs['physical_interval'],1)
                self.assertEqual(kwargs['metadata_interval'],5);out.mkdir()
                physical(0,commands,owner.time.monotonic()+5)
                metadata(0,commands,owner.time.monotonic()+5)
                physical(1,commands,owner.time.monotonic()+5)
                report=dict(status='COMPLETE',read_commands_settled=True)
                FakeBase.durable(out/'monitor.json',report);return report
            with patch.object(owner._monitor,'monitor',monitor):
                now=owner.time.monotonic()
                owner._live(args,plan,SimpleNamespace(measure=measure,observe=observe),child,identity,
                            {'replacement':{'pid':543200}},parent,{},now,now,live,checkpoints,state,
                            lambda:None,FakeBase.durable)
            self.assertEqual(len(live),2);self.assertEqual(calls,live);self.assertEqual(len(checkpoints),1)
            self.assertEqual(checkpoints[0]['label'],'running-metadata')
            self.assertTrue(state['read_commands_settled']);self.assertEqual(state['status'],'COMPLETE')
            self.assertEqual(state['receipt'],owner.ref(root/'monitor/monitor.json'))

    def fake_run(self, root, mode):
        plan_path = root/'plan.json'; plan_path.write_text('{}')
        args = SimpleNamespace(plan=plan_path, plan_sha256='a'*64, out=root/'owner')
        parent = os.getpid(); child_pid = parent+100000; events = []
        child = SimpleNamespace(pid=child_pid, returncode=0, wait=lambda timeout: 0)
        plan = dict(cold_restart={'journal_path': str(root/'journal')},
            execution={'owner': {'path': str(PATH), 'sha256': 'b'*64}, 'pins': {}},
            physical_policy={'probe_seconds':5, 'work_seconds':1800, 'interval_seconds':1},
            monitor_policy=owner.MONITOR_POLICY, budgets={'checkpoint_seconds':5},
            probe_plan={'path':str(plan_path),'sha256':'a'*64},
            outputs={'probe':str(args.out/'phases'/owner.PHASE/'probe')})
        probe = SimpleNamespace(SNAPSHOT=root/'model', VENV=root/'venv', HERE=root/'probe', ENVIRONMENT={})
        binding = dict(id='journal', removed_at=1, controller_sha256='c'*64)
        def identity(pid):
            if pid == child_pid and mode == 'identity-failed': raise ValueError('identity after returned Popen failed')
            return dict(pid=pid, ppid=1 if pid==parent else parent, start='fixture', comm='python')
        def spawn(*args):
            events.append('spawn')
            if mode == 'unknown-spawn': raise KeyboardInterrupt('unknown Popen outcome')
            return child
        system = SimpleNamespace(identity=identity, spawn=spawn,
            read=lambda argv, timeout: dict(command=argv, returncode=1, stdout='', stderr=''))
        def measure(path, pid, **kwargs):
            Path(path).mkdir(); FakeBase.durable(Path(path)/'sample.json', {}); return dict(pid=pid)
        def reap(handle, *, expected_pid, owner_pid, absence, out):
            self.assertIs(handle, child); events.append('reap')
            row = dict(status='REAPED', pid=child_pid, absence=absence(child_pid))
            FakeBase.durable(out, row); return row
        guard = SimpleNamespace(measure=measure, cold_admission=lambda *a, **k: {}, reap_owned=reap,
            diagnostic_report=lambda *a, **k: {'status':'PASS' if mode=='complete' else 'FAIL'})
        def checkpoint(*a, record=None, **kwargs):
            row = {'status':'PASS'}
            if record: record(row)
            return row
        def live(*args):
            state = args[12]
            state.update(started=True, read_commands_settled=mode not in ('unknown-probe', 'unjoined-probe'))
            if mode != 'complete': raise RuntimeError(mode)
            path = Path(plan['outputs']['probe']); path.mkdir(); (path/'complete.json').write_text('{}')
        def inventory(*args):
            if mode == 'preflight-failed': raise ValueError('changed runtime')
            return {}
        with contextlib.ExitStack() as stack:
            replacements = dict(validate_plan=lambda a:(plan, probe, guard, {'replacement':{'pid':8000}}),
                helper=lambda p:FakeBase, stopped=lambda *a:(b'{}', binding), verify_inventory=inventory,
                source_pins=lambda p:{}, checkpoint=checkpoint, _live=live, probe_terminal=lambda *a:None)
            for name, value in replacements.items(): stack.enter_context(patch.object(owner, name, value))
            stack.enter_context(patch.dict(os.environ, {}, clear=True))
            stack.enter_context(patch.object(owner.time, 'sleep', lambda seconds:None))
            for name, value in [('getitimer',(0.,0.)), ('getsignal','old-handler'), ('setitimer',None), ('signal',None)]:
                stack.enter_context(patch.object(owner.signal, name, return_value=value))
            code = owner.run(args, system=system)
        return args.out, code, events

    def test_actual_run_closes_both_lanes_before_canonical_release(self):
        with tempfile.TemporaryDirectory() as temp:
            out, code, events = self.fake_run(Path(temp).resolve(), 'complete')
            self.assertEqual(code, 0); self.assertEqual(events, ['spawn', 'reap'])
            value = owner.read(out/'complete.json')
            self.assertEqual(value['schema'], 'GLM-cold-load-owner-result-v3')
            self.assertTrue(value['all_children_reaped']); self.assertTrue(value['monitor']['read_commands_settled'])
            self.assertFalse(value['legacy_checkpoint_per_sample_contract'])
            self.assertFalse(value['performance_qualified'])

    def test_failed_settled_monitor_allows_recovery_but_not_experiment_success(self):
        with tempfile.TemporaryDirectory() as temp:
            out, code, events = self.fake_run(Path(temp).resolve(), 'physical-limit-failed')
            self.assertEqual(code, 1); self.assertIn('reap', events)
            self.assertTrue(owner.read(out/'complete.json')['all_children_reaped'])
            self.assertTrue((out/'release.json').is_file())

    def test_unsettled_monitor_never_issues_all_children_reaped_release(self):
        for mode in ('unknown-probe', 'unjoined-probe'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                out, code, events = self.fake_run(Path(temp).resolve(), mode)
                self.assertEqual(code, 1); self.assertIn('reap', events)
                self.assertTrue((out/'phases'/owner.PHASE/'reap.json').is_file())
                self.assertFalse((out/'release.json').exists())
                self.assertFalse(owner.read(out/'complete.json')['all_children_reaped'])

    def test_original_preintent_unknown_launch_and_known_pid_failure_still_distinct(self):
        for mode in ('preflight-failed', 'unknown-spawn', 'identity-failed'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                out, code, events = self.fake_run(Path(temp).resolve(), mode)
                self.assertEqual(code, 1)
                if mode == 'preflight-failed':
                    self.assertEqual(owner.read(out/'terminal.json')['status'], 'NO_CHILD_STARTED')
                    self.assertNotIn('spawn', events)
                elif mode == 'unknown-spawn':
                    self.assertTrue((out/'spawn-intent.json').exists())
                    self.assertFalse((out/'terminal.json').exists()); self.assertFalse((out/'release.json').exists())
                else:
                    self.assertIn('reap', events); self.assertTrue((out/'release.json').is_file())


if __name__ == '__main__': unittest.main()
