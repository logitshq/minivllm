"""Prepared pure-fake tests. No device or live service coverage is implied.

Not run during the active GPU measurement. Runtime helpers are replaced in the
run-path tests; the module imported here contains only Python standard library.
"""
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


PATH = Path(__file__).resolve().with_name('owner.py')
spec = importlib.util.spec_from_file_location('glm_owner_under_test', PATH)
owner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(owner)


class ReadOnlyFake:
    def __init__(self, *, child=9002, owner_pid=9001, foreign=False, omit=False, fail_at=None):
        self.child, self.owner = child, owner_pid
        self.foreign, self.omit, self.fail_at = foreign, omit, fail_at
        self.calls=[]

    def read(self, argv, timeout):
        self.calls.append(argv)
        if len(self.calls)==self.fail_at:
            raise subprocess.TimeoutExpired(argv,timeout,output=b'partial')
        rc, out, err=0,'',''
        if argv[:2]==['/bin/ps','-p']: rc=1
        elif argv[0]=='/bin/launchctl': rc,err=113,'Could not find service\n'
        elif argv[0]=='/usr/sbin/lsof': rc=1
        elif argv[0]=='/usr/bin/pmset': out="Now drawing from 'AC Power'\n"
        elif argv[0]=='/usr/bin/osascript': out='0\n'
        elif argv[0]=='/bin/ps':
            out=f'{self.owner} 1 /usr/bin/python3\n'
            if self.child is not None and not self.omit: out+=f'{self.child} {self.owner} /usr/bin/python3\n'
            if self.foreign:
                out += '9003 1 /bin/omlx\n' if argv[-1].endswith('comm=') else '9003 1 /bin/omlx serve --api-key secret\n'
        else: raise AssertionError(argv)
        return dict(command=argv,returncode=rc,stdout=out,stderr=err)


class OwnerMechanismTests(unittest.TestCase):
    def test_describe_and_partial_arguments_never_run(self):
        with patch.object(owner,'run',side_effect=AssertionError('execution')), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(owner.main([]),0)
            with self.assertRaises(ValueError): owner.main(['--plan','/missing'])
            with self.assertRaises(ValueError): owner.main(['--run','--plan','/missing'])

    def test_actual_probe_argv_requires_run_and_admission_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp).resolve(); admission=base/'admission.json';admission.write_text('{}')
            probe=SimpleNamespace(VENV=base/'venv',HERE=base/'probe')
            plan={'probe_plan':{'path':str(base/'plan.json'),'sha256':'a'*64}}
            argv=owner.child_command(plan,probe,admission,base/'output')
            self.assertEqual(argv,[str(base/'venv/bin/python'),'-I','-B',str(base/'probe/load_probe.py'),'--run',
                '--plan',str(base/'plan.json'),'--plan-sha256','a'*64,'--admission',str(admission),
                '--admission-sha256',owner.sha(admission),'--out',str(base/'output')])

    def test_checkpoint_actual_child_and_raw_absence(self):
        rows=[]; system=ReadOnlyFake()
        result=owner.checkpoint(system,old_pid=8000,child_pid=9002,owner_pid=9001,deadline=1e20,record=rows.append)
        self.assertEqual(result['status'],'PASS'); self.assertEqual(len(rows),1)
        self.assertEqual(result['recognized'],[dict(kind='native',pid=9002,ppid=9001),dict(kind='legacy',pid=9002,ppid=9001)])
        self.assertEqual([r['returncode'] for r in result['commands'][:3]],[1,113,1])
        self.assertNotIn('stdout',result['commands'][-1])
        self.assertEqual(result['authorized_suspended_pids'],[])

    def test_missing_actual_child_inventory_fails_and_is_retained(self):
        rows=[]
        with self.assertRaisesRegex(ValueError,'missing'):
            owner.checkpoint(ReadOnlyFake(omit=True),old_pid=8000,child_pid=9002,owner_pid=9001,deadline=1e20,record=rows.append)
        self.assertEqual(rows[0]['status'],'FAIL'); self.assertEqual(len(rows[0]['commands']),7)

    def test_foreign_process_rejects_and_redacts_even_on_failure(self):
        rows=[]
        with self.assertRaisesRegex(ValueError,'foreign'):
            owner.checkpoint(ReadOnlyFake(foreign=True),old_pid=8000,child_pid=9002,owner_pid=9001,deadline=1e20,record=rows.append)
        self.assertNotIn('secret',json.dumps(rows));self.assertNotIn('stdout',rows[0]['commands'][-1])
        self.assertEqual(rows[0]['status'],'FAIL')

    def test_checkpoint_timeout_keeps_partial_raw_command(self):
        rows=[]
        with self.assertRaises(subprocess.TimeoutExpired):
            owner.checkpoint(ReadOnlyFake(fail_at=3),old_pid=8000,child_pid=9002,owner_pid=9001,deadline=1e20,record=rows.append)
        self.assertEqual(rows[0]['commands'][-1]['stdout'],'partial')
        self.assertIsNone(rows[0]['commands'][-1]['returncode'])
        self.assertEqual(rows[0]['status'],'FAIL')

    def test_same_pid_wrong_parent_rejects(self):
        with self.assertRaisesRegex(ValueError,'foreign'):
            owner.checkpoint(ReadOnlyFake(owner_pid=7),old_pid=8000,child_pid=9002,owner_pid=9001,deadline=1e20)

    def test_inventory_record_types_cannot_alias_bool_and_int(self):
        record=dict(sha256='a'*64,stat=dict(bytes=0,mtime_ns=2,inode=3,device=4))
        self.assertTrue(owner.file_record(record))
        for key in record['stat']:
            bad=copy.deepcopy(record);bad['stat'][key]=False
            self.assertFalse(owner.file_record(bad))
        bad=copy.deepcopy(record);bad['sha256']='unknown'
        self.assertFalse(owner.file_record(bad))

    def test_describe_exit_or_wrong_probe_scope_cannot_pass(self):
        probe=SimpleNamespace(SNAPSHOT=Path('/original/snapshot'),EXPECTED={'mlx':'0.32.3'})
        plan={'probe_plan':{'sha256':'a'*64}}
        good=dict(schema='GLM-official-load-result-v1',status='PASS',pid=9002,owner_pid=9001,
            plan_sha256='a'*64,admission_sha256='b'*64,probe_sha256=owner.FIXED['../glm_flash/load_probe.py'],
            snapshot=str(probe.SNAPSHOT),versions=probe.EXPECTED,short_lifecycle_diagnostic_complete=True,
            shutdown={'complete':True},completed_requests=[{'cap':n} for n in (8,32,32,32)],
            performance_qualified=False,full_model_correctness_qualified=False,chat_quality_qualified=False,
            physical_resource_safety_qualified=False)
        owner.probe_terminal(good,plan,'b'*64,9002,9001,probe)
        for bad in ({},dict(good,performance_qualified=True),dict(good,pid=9003),dict(good,shutdown={'complete':False})):
            with self.assertRaises(ValueError): owner.probe_terminal(bad,plan,'b'*64,9002,9001,probe)


class FakeBase:
    journal_lock=staticmethod(lambda path:contextlib.nullcontext())
    signal_events=None
    @staticmethod
    def durable(path,value,exclusive=True,raw=False):
        path=Path(path)
        with path.open(('xb' if exclusive else 'wb')) as stream:
            stream.write(value if raw else json.dumps(value).encode())
        if FakeBase.signal_events is not None and path.name in ('terminal.json','release.json','complete.json'):
            FakeBase.signal_events.append(('durable',path.name))


class RunFailureTests(unittest.TestCase):
    def fake_run(self, temp, mode):
        root=Path(temp).resolve(); plan_path=root/'plan.json';plan_path.write_text('{}')
        args=SimpleNamespace(plan=plan_path,plan_sha256='a'*64,out=root/'owner')
        parent=os.getpid(); child_pid=parent+100000; events=[]; child=SimpleNamespace(pid=child_pid,returncode=-15)
        plan=dict(cold_restart={'journal_path':str(root/'journal')},execution={'owner':{'path':str(PATH),'sha256':'b'*64},'pins':{}},
            physical_policy={'probe_seconds':5,'work_seconds':1800,'interval_seconds':5},
            budgets={'checkpoint_seconds':5},probe_plan={'path':str(plan_path),'sha256':'a'*64},
            outputs={'probe':str(args.out/'phases'/owner.PHASE/'probe')})
        probe=SimpleNamespace(SNAPSHOT=root/'model',VENV=root/'venv',HERE=root/'probe',ENVIRONMENT={})
        prior={'replacement':{'pid':8000}}
        binding=dict(id='journal',removed_at=1,controller_sha256='c'*64)
        def identity(pid):
            if pid != parent: raise ValueError('identity unavailable after returned Popen')
            return dict(pid=parent,ppid=1,start='fixture',comm='python')
        def spawn(*unused):
            events.append(('spawn',))
            if mode=='unknown-spawn': raise KeyboardInterrupt('unknown Popen outcome')
            return child
        system=SimpleNamespace(identity=identity,spawn=spawn,
            read=lambda argv,timeout:dict(command=argv,returncode=1,stdout='',stderr=''))
        def measure(path,pid,**kwargs):
            Path(path).mkdir();FakeBase.durable(Path(path)/'sample.json',{})
            return dict(pid=pid)
        def reap(handle,*,expected_pid,owner_pid,absence,out):
            self.assertIs(handle,child);self.assertEqual(expected_pid,child_pid);events.append(('reap',child_pid))
            row=dict(status='REAPED',pid=child_pid,absence=absence(child_pid))
            FakeBase.durable(out,row);return row
        guard=SimpleNamespace(measure=measure,cold_admission=lambda *a,**k:{},reap_owned=reap,
            diagnostic_report=lambda *a,**k:{'status':'FAIL'})
        def checkpoint(*a,record=None,**k):
            row={'status':'PASS'}
            if record:record(row)
            return row
        def inventory(*a):
            if mode=='prepare-fail':raise ValueError('changed registered runtime')
            return {}
        FakeBase.signal_events=events
        with contextlib.ExitStack() as stack:
            replacements={'validate_plan':lambda a:(plan,probe,guard,prior),'helper':lambda p:FakeBase,
                'stopped':lambda *a:(b'{}',binding),'verify_inventory':inventory,
                'source_pins':lambda p:{},'checkpoint':checkpoint}
            for name,value in replacements.items():stack.enter_context(patch.object(owner,name,value))
            stack.enter_context(patch.dict(os.environ,{},clear=True))
            stack.enter_context(patch.object(owner.time,'sleep',lambda seconds:None))
            stack.enter_context(patch.object(owner.signal,'getitimer',return_value=(0.0,0.0)))
            stack.enter_context(patch.object(owner.signal,'getsignal',return_value='old-handler'))
            stack.enter_context(patch.object(owner.signal,'setitimer',return_value=None))
            stack.enter_context(patch.object(owner.signal,'signal',side_effect=lambda s,h:events.append(('signal',s,h))))
            code=owner.run(args,system=system)
        FakeBase.signal_events=None
        self.assertEqual(code,1)
        complete=owner.read(args.out/'complete.json')
        self.assertEqual(complete['status'],'FAIL')
        self.assertFalse(complete['performance_qualified'])
        # Durable recovery and result records precede restored signal handlers.
        last_result=max(i for i,event in enumerate(events) if event[:2]==('durable','complete.json'))
        restored=[i for i,event in enumerate(events) if event[0]=='signal' and event[2]=='old-handler']
        self.assertTrue(restored and min(restored)>last_result)
        return args.out,events

    def test_prepare_failure_has_no_child_terminal_not_fake_release(self):
        with tempfile.TemporaryDirectory() as temp:
            out,events=self.fake_run(temp,'prepare-fail')
            self.assertEqual(owner.read(out/'terminal.json')['status'],'NO_CHILD_STARTED')
            self.assertFalse((out/'spawn-intent.json').exists());self.assertFalse((out/'release.json').exists())
            self.assertNotIn(('spawn',),events)

    def test_unknown_spawn_keeps_intent_and_blocks_no_child_claim(self):
        with tempfile.TemporaryDirectory() as temp:
            out,events=self.fake_run(temp,'unknown-spawn')
            self.assertTrue((out/'spawn-intent.json').exists())
            self.assertFalse((out/'terminal.json').exists());self.assertFalse((out/'release.json').exists())
            self.assertFalse(any(event[0]=='reap' for event in events))
            self.assertIsNone(owner.read(out/'complete.json')['child_started'])

    def test_returned_pid_reaped_even_when_identity_read_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            out,events=self.fake_run(temp,'known-child')
            self.assertTrue(owner.read(out/'release.json')['all_children_reaped'])
            self.assertFalse((out/'terminal.json').exists())
            entry=owner.read(out/'children.json')['children'][0]
            self.assertIsNone(entry['identity']);self.assertIsNotNone(entry['release'])
            self.assertEqual(sum(event[0]=='reap' for event in events),1)


if __name__=='__main__':unittest.main()
