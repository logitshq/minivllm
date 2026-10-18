"""CPU-only direct-child protocol fixtures adapted from frozen retry tests.
No service/runtime backend is called. Owner-plan validation is explicitly mocked
inside receipt tests; predecessor fixtures bind actual reviewed source identities
but are synthetic JSON, not evidence of a device experiment or restoration.
The --help safety test calls argparse in-process with both backend constructors
and subprocess creation blocked. These tests must not be executed during the parent's GPU measurement.
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
sp=importlib.util.spec_from_file_location('route_http_controller_cpu',PATH)
c=importlib.util.module_from_spec(sp);sp.loader.exec_module(c)
ORIGINAL=dict(pid=25221,ppid=1,start='CPU restored service',comm='omlx-server')
class Fake:
    def __init__(self):self.live={};self.calls=[]
    def process(self,pid):self.calls.append(pid);return self.live.get(pid)
class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.j=c.Journal(self.root/'journal',create=True);self.addCleanup(self.j.close)
        self.owner=self.root/'owner';self.s=Fake();self.source=self.root/'owner.py';self.source.write_text('# fixed mock owner\n')
        self.j.data=dict(schema='E024-route-http-cold-journal-v1',id='cpu-journal',state='STOPPED',
            removed_at=time.time()-10,controller_sha256=c.b.sha(PATH),original=ORIGINAL,
            predecessor={'cpu':'fixture'},predecessor_copies={},owner_directory=str(self.owner),
            owner_plan={'path':str(self.root/'wrapper.json'),'sha256':'a'*64},phases=list(c.PHASES),
            owner_source={'path':str(self.source),'sha256':c.b.sha(self.source)},resource_samples=[],events=[])
        self.wrapper=dict(cold_restart={'journal_path':str(self.j.directory)},schema='E024-route-http-owner-plan-v1',phases=list(c.PHASES))
        self.write_plan()
        self.j.save('cpu-fixture')
        self.plan_mock=patch.object(c,'owner_plan',side_effect=lambda *a:self.wrapper)
        self.plan_mock.start();self.addCleanup(self.plan_mock.stop)

    def write_plan(self):
        path=Path(self.j.data['owner_plan']['path']);c.b.atomic_write(path,self.wrapper)
        self.j.data['owner_plan']['sha256']=c.b.sha(path)
        self.j.data['phases']=self.wrapper['phases']

    def receipts(self,count=4,nochild=False):
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

    def test_four_children_reaped_and_every_pid_absent(self):
        self.receipts();result=c.reaped(self.j,self.s);self.assertFalse(result['no_child_started'])
        self.assertEqual(set(self.s.calls),set(range(81000,81005)))
        for pid in range(81000,81005):
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
        path=self.owner/'phases/c1-replay/intent.json';path.parent.mkdir();path.write_text('{}')
        with self.assertRaisesRegex(c.Refuse,'omitted'):c.reaped(self.j,self.s)

    def test_actual_capture_inner_receipts_are_same_child_and_preserved(self):
        self.receipts(1);folder=self.owner/'phases/c1-capture/capture';folder.mkdir()
        intent=c.b.read_json(self.owner/'phases/c1-capture/intent.json')
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
        self.receipts(nochild=True);folder=self.owner/'phases/c1-capture/capture';folder.mkdir(parents=True)
        inner=folder/'spawn-intent.json';a=c.b.read_json(self.owner/'attempt.json')
        c.b.atomic_write(inner,dict(command=['python','-I','-B','CPU-fixture'],time=time.time()))
        proof_path=folder.parent/'no-popen.json';proof=dict(schema='cold-service-no-popen-v1',phase='c1-capture',
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
        self.receipts(2)
        phase='c8-capture';folder=self.owner/'phases'/phase/'capture';folder.mkdir(parents=True)
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
        self.assertEqual(set(self.s.calls),{81000,81001,81002})
        self.assertIn(str(proof_path),[r['path'] for r in result['receipts']])
        for field,value in [('phase','c8-replay'),('popen_called',True),('owner_pid',99999),
                            ('time',a['created_at']-1),('attempt_sha256','wrong')]:
            c.b.atomic_write(proof_path,dict(proof,**{field:value}))
            final['no_popen_receipt']['sha256']=c.b.sha(proof_path);c.b.atomic_write(self.owner/'release.json',final)
            with self.subTest(field=field),self.assertRaises((c.Refuse,c.formal.Refuse)):c.reaped(self.j,self.s)
        c.b.atomic_write(proof_path,proof);final['no_popen_receipt']['sha256']=c.b.sha(proof_path)
        c.b.atomic_write(self.owner/'release.json',final)
        for relative in ['phases/c8-capture/intent.json','phases/c8-capture/capture/child.json',
                         'phases/c8-replay/no-popen.json']:
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

    def test_stop_recover_worker_keep_frozen_ast_and_real_identity(self):
        def functions(path):return {n.name:n for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        here,stable=map(functions,(PATH,c.STABLE_PATH))
        for name in ('stop','recover','monitored_load','main'):
            self.assertEqual(ast.dump(here[name]),ast.dump(stable[name]),name)
        self.assertIs(c.System,c.stable.System)
        text=ast.get_source_segment(PATH.read_text(),here['monitored_load'])
        self.assertIn("str(Path(__file__).resolve()),'load-worker'",text)
        for n in ast.walk(ast.parse(PATH.read_text())):
            if isinstance(n,ast.Assign):
                self.assertFalse(any(isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name)
                    and t.value.id in ('b','stable','formal') for t in n.targets))

    def test_replay_rejects_any_nested_spawn_and_fake_no_popen(self):
        self.receipts(2)
        folder=self.owner/'phases/c1-replay/capture';folder.mkdir()
        c.b.atomic_write(folder/'child.json',dict(pid=81002))
        with self.assertRaisesRegex(c.Refuse,'omitted'):c.reaped(self.j,self.s)
        a=c.b.read_json(self.owner/'attempt.json')
        with self.assertRaisesRegex(c.Refuse,'only capture'):
            c.no_popen_proof(self.j,self.owner,a,'c1-replay',0,1,{})


class SourceAndPredecessorTests(unittest.TestCase):
    def test_unsealed_predecessor_refuses_before_backend_or_evidence_read(self):
        with patch.object(c,'PREDECESSOR_JOURNAL_SHA','UNSEALED'), \
             patch.object(c,'System',side_effect=AssertionError('OS backend')), \
             patch.object(c.b,'System',side_effect=AssertionError('base OS backend')), \
             patch.object(c.Path,'read_bytes',side_effect=AssertionError('no receipt read before seal')):
            with self.assertRaisesRegex(c.Refuse,'not sealed'):c.predecessor({})

    def test_exact_synthetic_restoration_predecessor_rejects_partial_or_false_claim(self):
        # Fixture-only archive: exact hashes are derived from these small fake
        # receipts. The real formal-v2 remains unsealed and is never consulted.
        with tempfile.TemporaryDirectory() as tmp:
            archive=Path(tmp).resolve();jp=archive/'restored-journal.json';ap=archive/'restoration-audit.json'
            status=dict(version='0.7.0',default_model=c.b.TARGET,loaded=[c.b.TARGET],loading=0,
                target_loading=False,path=c.b.MODEL,pinned=False,engine_type='vlm',config_model_type='qwen4_exp')
            prior=dict(schema=c.PREDECESSOR_SCHEMA,controller_sha256=c.PREDECESSOR_CONTROLLER_SHA,
                state='RESTORED',approved_contract=c.CONTRACT,plan_sha256=c.PLAN_SHA,support_sha256=c.BASE_SHA,
                replacement=ORIGINAL,final_status=status,owner_plan={'sha256':'a'*64},
                log_window={'sha256':'b'*64,'bytes':1},exact_runtime_signature_restored=False,warm_kv_restored=False)
            audit=dict(schema='cold-service-restoration-audit-v1',status='PASS',qualified=True,
                controller_sha256=c.PREDECESSOR_CONTROLLER_SHA,audit_source_sha256=c.PREDECESSOR_AUDITOR_SHA,
                owner_plan_sha256='a'*64,service_restore=dict(status='PERSISTED_VISIBLE_RECEIPTS_CONSISTENT',
                    replacement=ORIGINAL,raw_log_sha256='b'*64,raw_log_bytes=1,http_worker_returncode=0))
            def receipt_pair(current,checked):
                c.b.atomic_write(jp,current);checked=copy.deepcopy(checked);checked['journal_sha256']=c.b.sha(jp)
                c.b.atomic_write(ap,checked)
                return dict(path=str(jp),sha256=c.b.sha(jp),audit=dict(path=str(ap),sha256=c.b.sha(ap)))
            def call(current,checked):
                ref=receipt_pair(current,checked)
                with patch.object(c,'ARCHIVE_ROOT',archive),patch.object(c,'PREDECESSOR_ARCHIVE',archive), \
                     patch.object(c,'PREDECESSOR_JOURNAL_SHA',ref['sha256']), \
                     patch.object(c,'PREDECESSOR_AUDIT_SHA',ref['audit']['sha256']), \
                     patch.object(c,'System',side_effect=AssertionError('backend')), \
                     patch.object(c.b,'System',side_effect=AssertionError('base backend')):
                    return c.predecessor(ref)
            self.assertEqual(call(prior,audit)[0],prior)
            for field,value in [('state','LOAD_INTENT'),('post_monitor_failed',True),('warm_kv_restored',True)]:
                with self.subTest(field=field),self.assertRaises(c.Refuse):call(dict(prior,**{field:value}),audit)
            for field,value in [('qualified',False),('audit_source_sha256','0'*64)]:
                with self.subTest(field=field),self.assertRaises(c.Refuse):call(prior,dict(audit,**{field:value}))

    def test_actual_owner_manifest_missing_dependency_rejected(self):
        owner=c.load(PATH.with_name('owner.py'),c.OWNER_SHA,'route_http_owner_manifest_test')
        manifest=owner.source_pins()
        plan=dict(execution=dict(owner=dict(path=str(PATH.with_name('owner.py')),sha256=c.OWNER_SHA),pins=manifest))
        pins=c.source_pins(plan)
        self.assertTrue(all(pins[p]==h for p,h in manifest.items()))
        for path in [str(PATH.with_name('owner.py')),str(owner.MODULES['http_capture']),str(owner.MODULES['http_qualification'])]:
            bad=copy.deepcopy(plan);bad['execution']['pins'].pop(path)
            with self.subTest(path=path),self.assertRaises(c.Refuse):c.source_pins(bad)

    def test_plan_shape_budget_and_journal_binding_without_backend(self):
        # Actual source manifest and original numeric/serving receipt gates run;
        # only the unfinished future predecessor is a synthetic fixture here.
        owner=c.load(PATH.with_name('owner.py'),c.OWNER_SHA,'route_http_owner_template_test')
        base=PATH.parent.parent/'pareto_realistic/prereg-http-baseline-v2/owner-plan.json'
        old=json.loads(base.read_bytes())
        serving_plan=PATH.parent.parent/'route_decode_graph/prereg-serving-v1/serving-plan.json'
        serving=dict(plan=dict(path=str(serving_plan),sha256=c.b.sha(serving_plan)),
                     audit=dict(path=str(owner.SERVING_AUDIT),sha256=owner.SERVING_AUDIT_SHA))
        with tempfile.TemporaryDirectory() as temp, patch.object(c,'predecessor',return_value=({},[])), \
             patch.object(c,'System',side_effect=AssertionError('OS backend')), \
             patch.object(c.b,'System',side_effect=AssertionError('base OS backend')):
            root=Path(temp).resolve();cold=dict(journal_path=str(root/'journal'),controller=dict(path=str(PATH),sha256=c.b.sha(PATH)),
                approved_contract=c.CONTRACT,proposal_plan_sha256=c.PLAN_SHA,predecessor={})
            plan=owner.plan_template(old['qualification'],cold,old['http'],serving)
            path=root/'plan.json';c.b.atomic_write(path,plan)
            self.assertEqual(c.owner_plan(path,c.b.sha(path),cold['journal_path']),plan)
            for field in ('schema','deadline_seconds','http','serving'):
                bad=copy.deepcopy(plan)
                if field=='schema':bad[field]='old schema'
                elif field=='deadline_seconds':bad[field]+=1
                elif field=='http':bad[field]['deadlines']['overall']=901
                else:bad[field]['audit']['sha256']='0'*64
                c.b.atomic_write(path,bad)
                with self.subTest(field=field),self.assertRaises(c.Refuse):c.owner_plan(path,c.b.sha(path),cold['journal_path'])

if __name__=='__main__':unittest.main()
