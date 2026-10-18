"""CPU-only direct-child protocol fixtures adapted from frozen retry tests.
No service/runtime backend is called. Owner-plan validation is explicitly mocked
inside receipt tests; predecessor fixtures bind actual reviewed source identities
but are synthetic JSON, not evidence of a device experiment or restoration.
The --help safety test spawns only an isolated Python process with OS/model calls
blocked. These tests must not be executed during the parent's GPU measurement.
"""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from unittest.mock import patch
PATH=Path(__file__).with_name('controller.py')
sp=importlib.util.spec_from_file_location('omlx_formal_retry_controller_cpu',PATH)
c=importlib.util.module_from_spec(sp);sp.loader.exec_module(c)
ORIGINAL=dict(pid=25221,ppid=1,start='CPU restored service',comm='omlx-server')
class Fake:
    def __init__(self):self.live={};self.calls=[]
    def process(self,pid):self.calls.append(pid);return self.live.get(pid)
class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.j=c.Journal(self.root/'journal',create=True);self.addCleanup(self.j.close)
        self.owner=self.root/'owner';self.s=Fake();self.source=self.root/'owner.py';self.source.write_text('# fixed mock owner\n')
        self.j.data=dict(schema='E023-omlx-D512-formal-cold-journal-v1',id='cpu-journal',state='STOPPED',
            removed_at=time.time()-10,controller_sha256=c.b.sha(PATH),original=ORIGINAL,
            predecessor={'cpu':'fixture'},predecessor_copies={},owner_directory=str(self.owner),
            owner_plan={'path':str(self.root/'wrapper.json'),'sha256':'a'*64},trials=list(c.PHASES),
            owner_source={'path':str(self.source),'sha256':c.b.sha(self.source)},resource_samples=[],events=[])
        self.wrapper=dict(cold_restart={'journal_path':str(self.j.directory)},schema='E023-omlx-D512-formal-owner-plan-v1',trials=list(c.PHASES))
        self.write_plan()
        self.j.save('cpu-fixture')
        self.plan_mock=patch.object(c,'owner_plan',side_effect=lambda *a:self.wrapper)
        self.plan_mock.start();self.addCleanup(self.plan_mock.stop)

    def write_plan(self):
        path=Path(self.j.data['owner_plan']['path']);c.b.atomic_write(path,self.wrapper)
        self.j.data['owner_plan']['sha256']=c.b.sha(path)
        self.j.data['trials']=self.wrapper['trials']

    def receipts(self,count=6,nochild=False):
        self.owner.mkdir();raw=self.j.path.read_bytes();(self.owner/'stopped-journal.json').write_bytes(raw)
        a=dict(schema='cold-service-owner-attempt-v1',owner_pid=81000,
            owner_identity=dict(pid=81000,ppid=1,start='cpu',comm='python'),journal_id=self.j.data['id'],
            journal_path=str(self.j.directory),controller_sha256=self.j.data['controller_sha256'],
            owner_plan_sha256=self.j.data['owner_plan']['sha256'],owner_source=self.j.data['owner_source'],removed_at=self.j.data['removed_at'],
            created_at=time.time(),phase='PRESPAWN',stopped_journal_sha256=hashlib.sha256(raw).hexdigest())
        c.b.atomic_write(self.owner/'attempt.json',a)
        binding={k:a[k] for k in ('owner_pid','owner_identity','owner_plan_sha256','journal_id','owner_source','removed_at')}
        binding['attempt_sha256']=c.b.sha(self.owner/'attempt.json')
        if nochild:
            c.b.atomic_write(self.owner/'terminal.json',dict(binding,schema='cold-service-owner-terminal-v1',
                status='NO_CHILD_STARTED',terminal_at=time.time(),spawn_intent_written=False,child_started=False))
            return
        c.b.atomic_write(self.owner/'spawn-intent.json',binding)
        children=[]
        for n,phase in enumerate(c.PHASES[:count]):
            folder=self.owner/'phases'/phase;folder.mkdir(parents=True)
            identity=dict(pid=81001+n,ppid=81000,start='cpu',comm='python')
            c.b.atomic_write(folder/'intent.json',dict(binding,phase=phase,time=time.time()))
            c.b.atomic_write(folder/'child.json',dict(phase=phase,pid=identity['pid'],owner_pid=81000,identity=identity))
            c.b.atomic_write(folder/'release.json',dict(pid=identity['pid'],returncode=0 if n<count-1 else 1,
                waited_reaped=True,strict_absence=True,time=time.time()))
            row=dict(phase=phase,pid=identity['pid'],identity=identity)
            for kind in ('intent','child','release'):
                path=folder/(kind+'.json');row[kind]=dict(path=str(path.relative_to(self.owner)),sha256=c.b.sha(path))
            children.append(row)
        c.b.atomic_write(self.owner/'children.json',dict(schema='cold-service-owner-children-v1',children=children))
        c.b.atomic_write(self.owner/'release.json',dict(binding,schema='cold-service-owner-release-v2',
            children=children,children_sha256=c.b.sha(self.owner/'children.json'),all_children_reaped=True,time=time.time()))

    def rewrite_index(self,index):
        c.b.atomic_write(self.owner/'children.json',index)
        final=c.b.read_json(self.owner/'release.json');final.update(children=index['children'],children_sha256=c.b.sha(self.owner/'children.json'))
        c.b.atomic_write(self.owner/'release.json',final)

    def test_six_children_reaped_and_every_pid_absent(self):
        self.receipts();result=c.reaped(self.j,self.s);self.assertFalse(result['no_child_started'])
        self.assertEqual(set(self.s.calls),set(range(81000,81007)))
        for pid in range(81000,81007):
            self.s.live[pid]={'pid':pid}
            with self.subTest(pid=pid),self.assertRaises(c.NotReady):c.reaped(self.j,self.s)
            self.s.live.clear()

    def test_failed_or_killed_experiment_is_recoverable_only_after_actual_reap(self):
        self.receipts(1);index=c.b.read_json(self.owner/'children.json');row=index['children'][0]
        path=self.owner/row['release']['path'];receipt=c.b.read_json(path)
        # Recovery is distinct from qualification: owner rejects -9 as a passing
        # trial, but restoring the original service is allowed after strict reap.
        receipt['returncode']=-9;c.b.atomic_write(path,receipt)
        row['release']['sha256']=c.b.sha(path);self.rewrite_index(index)
        self.assertFalse(c.reaped(self.j,self.s)['no_child_started'])
        receipt['waited_reaped']=False;c.b.atomic_write(path,receipt)
        row['release']['sha256']=c.b.sha(path);self.rewrite_index(index)
        with self.assertRaises(c.Refuse):c.reaped(self.j,self.s)

    def test_failed_short_prefix_allowed_but_omitted_unknown_intent_rejected(self):
        self.receipts(1);self.assertFalse(c.reaped(self.j,self.s)['no_child_started'])
        path=self.owner/'phases/r1-c8/intent.json';path.parent.mkdir();path.write_text('{}')
        with self.assertRaisesRegex(c.Refuse,'omitted'):c.reaped(self.j,self.s)

    def test_actual_capture_inner_receipts_are_same_child_and_preserved(self):
        self.receipts(1);folder=self.owner/'phases/r1-c1/trial';folder.mkdir()
        intent=c.b.read_json(self.owner/'phases/r1-c1/intent.json')
        c.b.atomic_write(folder/'spawn-intent.json',dict(command=['CPU mock'],time=intent['time']))
        c.b.atomic_write(folder/'child.json',dict(pid=81001))
        c.b.atomic_write(folder/'release.json',dict(pid=81001,waited_reaped=False))
        result=c.reaped(self.j,self.s)
        self.assertIn(str(folder/'release.json'),[r['path'] for r in result['receipts']])
        c.b.atomic_write(folder/'child.json',dict(pid=91000))
        with self.assertRaisesRegex(c.Refuse,'nested PID'):c.reaped(self.j,self.s)

    def test_known_popen_with_failed_identity_probe_requires_explicit_ownership(self):
        self.receipts(1);index=c.b.read_json(self.owner/'children.json');row=index['children'][0]
        path=self.owner/row['child']['path'];child=c.b.read_json(path)
        child.update(identity=None,ownership='direct-Popen-returned');c.b.atomic_write(path,child)
        row.update(identity=None);row['child']['sha256']=c.b.sha(path);self.rewrite_index(index)
        self.assertFalse(c.reaped(self.j,self.s)['no_child_started'])
        child.pop('ownership');c.b.atomic_write(path,child);row['child']['sha256']=c.b.sha(path);self.rewrite_index(index)
        with self.assertRaises((c.Refuse,c.formal.Refuse)):c.reaped(self.j,self.s)

    def test_index_cannot_omit_completed_phase_or_accept_false_reap_or_parent(self):
        self.receipts();index=c.b.read_json(self.owner/'children.json');saved=copy.deepcopy(index)
        index['children'].pop();self.rewrite_index(index)
        with self.assertRaises((c.Refuse,c.formal.Refuse)):c.reaped(self.j,self.s)
        self.rewrite_index(saved)
        for kind,field,value in [('release','waited_reaped',False),('child','owner_pid',99999)]:
            index=copy.deepcopy(saved);row=index['children'][0];path=self.owner/row[kind]['path'];raw=path.read_bytes()
            payload=json.loads(raw);payload[field]=value;c.b.atomic_write(path,payload)
            row[kind]['sha256']=c.b.sha(path);self.rewrite_index(index)
            with self.subTest(kind=kind),self.assertRaises((c.Refuse,c.formal.Refuse)):c.reaped(self.j,self.s)
            path.write_bytes(raw);self.rewrite_index(saved)

    def test_legacy_release_and_nonprefix_and_receipt_hash_are_rejected(self):
        self.receipts();path=self.owner/'release.json';raw=path.read_bytes()
        c.b.atomic_write(path,dict(pid=81001,returncode=0,waited_reaped=True,strict_absence=True))
        with self.assertRaises((c.Refuse,c.formal.Refuse)):c.reaped(self.j,self.s)
        path.write_bytes(raw);index=c.b.read_json(self.owner/'children.json');index['children'].reverse();self.rewrite_index(index)
        with self.assertRaises((c.Refuse,c.formal.Refuse)):c.reaped(self.j,self.s)
        index['children'].reverse();index['children'][0]['child']['sha256']='0'*64;self.rewrite_index(index)
        with self.assertRaises((c.Refuse,c.formal.Refuse)):c.reaped(self.j,self.s)

    def test_no_child_terminal_remains_strict_and_source_pinned(self):
        self.receipts(nochild=True);self.assertTrue(c.reaped(self.j,self.s)['no_child_started'])
        path=self.owner/'spawn-intent.json';path.write_text('{}')
        with self.assertRaises((c.Refuse,c.formal.Refuse)):c.reaped(self.j,self.s)
        path.unlink();self.source.write_text('changed')
        with self.assertRaises((c.Refuse,c.formal.Refuse)):c.reaped(self.j,self.s)

    def test_first_capture_pre_popen_rejection_requires_complete_proof(self):
        self.receipts(nochild=True);folder=self.owner/'phases/r1-c1/trial';folder.mkdir(parents=True)
        inner=folder/'spawn-intent.json';a=c.b.read_json(self.owner/'attempt.json')
        c.b.atomic_write(inner,dict(command=['python','-I','-B','CPU-fixture'],time=time.time()))
        proof_path=folder.parent/'no-popen.json';proof=dict(schema='cold-service-no-popen-v1',phase='r1-c1',
            owner_pid=a['owner_pid'],journal_id=a['journal_id'],owner_plan_sha256=a['owner_plan_sha256'],
            attempt_sha256=c.b.sha(self.owner/'attempt.json'),ownership='tracked-backend-rejected-before-Popen',
            popen_called=False,time=time.time(),command=c.b.read_json(inner)['command'],
            inner_spawn_intent=dict(path=str(inner.relative_to(self.owner)),sha256=c.b.sha(inner)))
        terminal=c.b.read_json(self.owner/'terminal.json');terminal['terminal_at']=time.time()
        c.b.atomic_write(self.owner/'terminal.json',terminal)
        with self.assertRaises((c.Refuse,c.formal.Refuse)):c.reaped(self.j,self.s)
        c.b.atomic_write(proof_path,proof)
        terminal['no_popen_receipt']=dict(path=str(proof_path.relative_to(self.owner)),sha256=c.b.sha(proof_path))
        c.b.atomic_write(self.owner/'terminal.json',terminal)
        result=c.reaped(self.j,self.s);self.assertTrue(result['no_child_started'])
        self.assertIn(str(proof_path),[row['path'] for row in result['receipts']])
        for key,value in [('popen_called',True),('attempt_sha256','wrong'),('command',['changed']),('time',True)]:
            c.b.atomic_write(proof_path,dict(proof,**{key:value}))
            terminal['no_popen_receipt']['sha256']=c.b.sha(proof_path);c.b.atomic_write(self.owner/'terminal.json',terminal)
            with self.subTest(key=key),self.assertRaises((c.Refuse,c.formal.Refuse)):c.reaped(self.j,self.s)
        c.b.atomic_write(proof_path,proof)
        terminal['no_popen_receipt']['sha256']=c.b.sha(proof_path);c.b.atomic_write(self.owner/'terminal.json',terminal)
        child=folder/'child.json';child.write_text('{"pid":81001}')
        with self.assertRaises((c.Refuse,c.formal.Refuse)):c.reaped(self.j,self.s)

    def test_unresolved_prespawn_crash_is_not_a_forged_terminal(self):
        self.receipts(nochild=True);(self.owner/'terminal.json').unlink()
        with self.assertRaises(c.NotReady):c.reaped(self.j,self.s)
        with self.assertRaises((c.Refuse,c.formal.Refuse)):c.reaped(self.j,self.s,no_experiment=True)

    def test_later_pre_popen_rejection_requires_next_trial_and_full_proof(self):
        self.receipts(1)
        phase='r1-c8';folder=self.owner/'phases'/phase/'trial';folder.mkdir(parents=True)
        inner=folder/'spawn-intent.json';a=c.b.read_json(self.owner/'attempt.json')
        c.b.atomic_write(inner,dict(command=['python','-I','-B','CPU-fixture'],time=time.time()))
        proof_path=folder.parent/'no-popen.json'
        proof=dict(schema='cold-service-no-popen-v1',phase=phase,owner_pid=a['owner_pid'],
            journal_id=a['journal_id'],owner_plan_sha256=a['owner_plan_sha256'],
            attempt_sha256=c.b.sha(self.owner/'attempt.json'),ownership='tracked-backend-rejected-before-Popen',
            popen_called=False,time=time.time(),command=c.b.read_json(inner)['command'],
            inner_spawn_intent=dict(path=str(inner.relative_to(self.owner)),sha256=c.b.sha(inner)))
        c.b.atomic_write(proof_path,proof)
        final=c.b.read_json(self.owner/'release.json');final['time']=time.time()
        c.b.atomic_write(self.owner/'release.json',final)
        with self.assertRaisesRegex(c.Refuse,'omitted'):c.reaped(self.j,self.s)
        final['no_popen_receipt']=dict(path=str(proof_path.relative_to(self.owner)),sha256=c.b.sha(proof_path))
        c.b.atomic_write(self.owner/'release.json',final)
        result=c.reaped(self.j,self.s)
        self.assertEqual(set(self.s.calls),{81000,81001})
        self.assertIn(str(proof_path),[r['path'] for r in result['receipts']])
        for field,value in [('phase','r2-c8'),('popen_called',True),('owner_pid',99999),
                            ('time',a['created_at']-1),('attempt_sha256','wrong')]:
            c.b.atomic_write(proof_path,dict(proof,**{field:value}))
            final['no_popen_receipt']['sha256']=c.b.sha(proof_path);c.b.atomic_write(self.owner/'release.json',final)
            with self.subTest(field=field),self.assertRaises((c.Refuse,c.formal.Refuse)):c.reaped(self.j,self.s)
        c.b.atomic_write(proof_path,proof);final['no_popen_receipt']['sha256']=c.b.sha(proof_path)
        c.b.atomic_write(self.owner/'release.json',final)
        for relative in ['phases/r1-c8/intent.json','phases/r1-c8/trial/child.json',
                         'phases/r2-c8/no-popen.json']:
            path=self.owner/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
            with self.subTest(relative=relative),self.assertRaises((c.Refuse,c.formal.Refuse)):c.reaped(self.j,self.s)
            path.unlink()

    def test_help_safe_without_runtime_network_or_subprocess(self):
        import contextlib,io
        with patch.object(c,'System',side_effect=AssertionError('OS backend')), \
             patch.object(c.b,'System',side_effect=AssertionError('base OS backend')), \
             patch.object(c.subprocess,'Popen',side_effect=AssertionError('subprocess')), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as result:c.main(['--help'])
        self.assertEqual(result.exception.code,0)

    def test_real_worker_argv_and_cancel_escalate_reap_only_owned_http_client(self):
        class Child:
            pid=81009
            returncode=None
            def poll(self):return self.returncode
            def terminate(self):calls.append('terminate')
            def kill(self):calls.append('kill')
            def wait(self,timeout):
                calls.append(('wait',timeout))
                if timeout==5 and 'kill' not in calls:raise subprocess.TimeoutExpired('CPU',timeout)
                self.returncode=-9;return self.returncode
        class Backend:
            def process(self,pid):
                return None if pid==81009 else dict(pid=pid,ppid=1,start='CPU',comm='python')
        calls=[];child=Child()
        def popen(argv,**kwargs):calls.append(argv);return child
        with patch.object(c,'sample',side_effect=KeyboardInterrupt('CPU cancellation')):
            with self.assertRaises(KeyboardInterrupt):
                c.monitored_load(self.j,Backend(),ORIGINAL,dict(swap_bytes=0),popen=popen)
        argv=calls[0]
        self.assertEqual(argv[:5],[sys.executable,'-I','-B',str(PATH),'load-worker'])
        self.assertEqual(argv[5:7],['--journal',str(self.j.directory)])
        self.assertIn('terminate',calls);self.assertIn('kill',calls)
        release=c.b.read_json(self.j.directory/'load-worker/release.json')
        self.assertEqual(release,dict(pid=81009,returncode=-9,waited_reaped=True,strict_absence=True,scope='HTTP client worker only'))
        self.assertTrue(self.j.data['post_monitor_failed'])
        self.assertEqual(self.j.data['cancellation_scope'],'HTTP wait only; service loader may remain active')

    def test_stop_recover_worker_and_phase_logic_preserve_frozen_ast(self):
        def functions(path):return {n.name:n for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        here,stable,formal=map(functions,(PATH,c.STABLE_PATH,c.FORMAL_PATH))
        for name in ('stop','recover','monitored_load','main'):
            self.assertEqual(ast.dump(here[name]),ast.dump(stable[name]),name)
        for name in ('phase_reaps','reaped'):
            self.assertEqual(ast.dump(here[name]),ast.dump(formal[name]),name)
        self.assertIs(c.System,c.stable.System)
        self.assertIs(c.no_popen_proof,c.formal.no_popen_proof)
        text=ast.get_source_segment(PATH.read_text(),here['monitored_load'])
        self.assertIn("str(Path(__file__).resolve()),'load-worker'",text)
        # No imported module global assignments or source-path impersonation.
        for n in ast.walk(ast.parse(PATH.read_text())):
            if isinstance(n,ast.Assign):
                self.assertFalse(any(isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name)
                    and t.value.id in ('b','stable','formal') for t in n.targets))


class SourceAndPredecessorTests(unittest.TestCase):
    def binding(self):
        return dict(path=str(c.PREDECESSOR_ARCHIVE/'restored-journal.json'),sha256=c.PREDECESSOR_JOURNAL_SHA,
            audit=dict(path=str(c.PREDECESSOR_ARCHIVE/'restoration-audit.json'),sha256=c.PREDECESSOR_AUDIT_SHA))

    def test_actual_archived_predecessor_without_backend_and_exact_hash_reject(self):
        binding=self.binding()
        with patch.object(c,'System',side_effect=AssertionError('OS backend')), \
             patch.object(c.b,'System',side_effect=AssertionError('base OS backend')):
            prior,raw=c.predecessor(binding)
            self.assertEqual(prior['replacement']['pid'],98900)
            self.assertEqual(prior['replacement']['start'],'Sat Oct 17 17:21:25 2026')
            self.assertEqual(json.loads(raw[1])['status'],'PASS')
            self.assertEqual(json.loads(raw[1])['owner_reap']['owner_status'],'FAIL')
            for name in ('sha256','path'):
                bad=copy.deepcopy(binding);bad[name]='0'*64 if name=='sha256' else '/tmp/not-archived'
                with self.subTest(name=name),self.assertRaises((c.Refuse,FileNotFoundError)):c.predecessor(bad)

    def test_actual_local_sources_and_fixed_formal_cannot_be_omitted(self):
        sp=importlib.util.spec_from_file_location('omlx_formal_retry_owner_binding_cpu',PATH.with_name('owner.py'))
        owner=importlib.util.module_from_spec(sp);sp.loader.exec_module(owner)
        plan=dict(execution=dict(owner=dict(path=str(PATH.with_name('owner.py')),sha256=c.OWNER_SHA),pins=owner.source_pins()))
        pins=c.source_pins(plan)
        for name in (*owner.LOCAL,*owner.REQUIRED_AUDIT_SOURCES):
            path=PATH.with_name(name);self.assertEqual(pins[str(path)],c.b.sha(path))
            bad=copy.deepcopy(plan);bad['execution']['pins'].pop(str(path))
            with self.subTest(name=name),self.assertRaises(c.Refuse):c.source_pins(bad)

    def test_six_trial_plan_template_and_original_qualified_inventory(self):
        sp=importlib.util.spec_from_file_location('omlx_formal_retry_template_cpu',PATH.with_name('owner.py'))
        owner=importlib.util.module_from_spec(sp);sp.loader.exec_module(owner)
        original=PATH.parent.parent/'omlx_d512/prereg-qualification-v1/owner-plan.json'
        old=json.loads(original.read_bytes())
        # Restoration follows failed formal-v1; count qualification remains the
        # independently successful earlier C1/C8 qualification, not that failure.
        archive=c.ARCHIVE_ROOT/'E023-omlx-d512-qualification-v1'
        refs={key:dict(path=str(archive/name),sha256=c.b.sha(archive/name)) for key,name in
              (('audit','independent-qualification.json'),('resource_audit','resource-summary.json'),('restoration_audit','restoration-audit.json'))}
        q=dict(owner_directory=str(PATH.parents[3]/'.artifacts/m5-ultra-256gb/omlx-d512-qualification-v1-owner'),
               owner_plan=dict(path=str(original),sha256=c.b.sha(original)),**refs)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();cold=dict(journal_path=str(root/'journal'),controller=dict(path=str(PATH),sha256=c.b.sha(PATH)),
                approved_contract=c.CONTRACT,proposal_plan_sha256=c.PLAN_SHA,predecessor=self.binding())
            http=dict(port=18023,api_key='CPU-fixture',environment={},deadlines=dict(overall=3600,close=10,term=15,kill=15,resource_interval=5))
            plan=owner.plan_template(q,cold,http,old['verification']);path=root/'plan.json';c.b.atomic_write(path,plan)
            self.assertEqual(c.owner_plan(path,c.b.sha(path),cold['journal_path']),plan)
            self.assertEqual(owner.qualified_plan(q,old['verification'])['schema'],'E023-native-qualification-plan-v1')
            for field in ('phases','deadline','verification'):
                bad=copy.deepcopy(plan)
                if field=='phases':bad['trials']=list(reversed(bad['trials']))
                elif field=='deadline':bad['deadline_seconds']+=1
                else:bad['verification']['groups']['omlx_source']={}
                c.b.atomic_write(path,bad)
                with self.subTest(field=field),self.assertRaises(c.Refuse):c.owner_plan(path,c.b.sha(path),cold['journal_path'])

if __name__=='__main__':unittest.main()
