"""Focused CPU protocol tests; all process/HTTP/service operations are fakes."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

PATH=Path(__file__).with_name('controller_v3.py')
sp=importlib.util.spec_from_file_location('cold_v3_cpu',PATH)
c=importlib.util.module_from_spec(sp);sp.loader.exec_module(c)
ORIGINAL=dict(pid=74456,ppid=1,start='Sat Oct 17 07:37:30 2026',comm='omlx-server')


class Fake:
    def __init__(self):
        self.live={};self.job_value=None;self.listener_value=None;self.calls=[]
    def process(self,pid):self.calls.append(pid);return self.live.get(pid)
    def job(self):return self.job_value
    def listener(self):return self.listener_value


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.j=c.Journal(self.root/'journal',create=True);self.addCleanup(self.j.close)
        self.owner=self.root/'owner';self.s=Fake();self.source=self.root/'owner.py';self.source.write_text('# fixed mock owner\n')
        self.j.data=dict(schema='omlx-cold-restart-journal-v3',id='cpu-journal',state='STOPPED',
            removed_at=time.time()-10,controller_sha256=c.b.sha(PATH),original=ORIGINAL,
            predecessor={'cpu':'fixture'},predecessor_copies={},owner_directory=str(self.owner),
            owner_plan={'path':str(self.root/'wrapper.json'),'sha256':'a'*64},
            owner_source={'path':str(self.source),'sha256':c.b.sha(self.source)},resource_samples=[],events=[])
        self.j.save('cpu-fixture')

    def predecessor_fixture(self):
        status=dict(version='0.7.0',default_model=c.b.TARGET,loaded=[c.b.TARGET],loading=0,
            target_loading=False,path=c.b.MODEL,pinned=False,engine_type='vlm',config_model_type='qwen4_exp')
        prior=dict(schema='omlx-cold-restart-journal-v3',controller_sha256=c.b.sha(PATH),
            state='RESTORED',approved_contract=c.CONTRACT,plan_sha256=c.PLAN_SHA,support_sha256=c.BASE_SHA,
            replacement=ORIGINAL,final_status=status,owner_plan={'sha256':'a'*64},
            log_window={'sha256':'b'*64,'bytes':123},events=[],exact_runtime_signature_restored=False,
            warm_kv_restored=False,global_settings_stat={},model_metadata={})
        p=self.root/'restored.json';c.b.atomic_write(p,prior)
        audit=dict(schema='cold-service-restoration-audit-v1',status='FAIL',qualified=False,
            journal_sha256=c.b.sha(p),owner_plan_sha256='a'*64,
            service_restore=dict(status='PERSISTED_VISIBLE_RECEIPTS_CONSISTENT',replacement=ORIGINAL,
                raw_log_sha256='b'*64,raw_log_bytes=123,http_worker_returncode=0))
        a=self.root/'audit.json';c.b.atomic_write(a,audit)
        binding=dict(path=str(p),sha256=c.b.sha(p),audit=dict(path=str(a),sha256=c.b.sha(a)))
        return binding,prior,audit

    def receipts(self,count=4,nochild=False):
        self.owner.mkdir();raw=self.j.path.read_bytes();(self.owner/'stopped-journal.json').write_bytes(raw)
        a=dict(schema='cold-service-owner-attempt-v1',owner_pid=81000,
            owner_identity=dict(pid=81000,ppid=1,start='cpu',comm='python'),journal_id=self.j.data['id'],
            journal_path=str(self.j.directory),controller_sha256=self.j.data['controller_sha256'],
            owner_plan_sha256='a'*64,owner_source=self.j.data['owner_source'],removed_at=self.j.data['removed_at'],
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

    def test_actual_initial_archived_pair_is_exact_and_cpu_only(self):
        folder=c.ARCHIVE_ROOT/'E028-identity-refill-model-v1'
        binding=dict(path=str(folder/'restored-journal.json'),sha256=c.INITIAL_PREDECESSOR_SHA,
                     audit=dict(path=str(folder/'independent-audit-v2.json'),sha256=c.INITIAL_AUDIT_SHA))
        with patch.object(c.b,'System',side_effect=AssertionError('no live backend')):
            prior,_=c.predecessor(binding)
        self.assertEqual(prior['replacement'],ORIGINAL)
        changed=copy.deepcopy(binding);changed['audit']['sha256']='0'*64
        with self.assertRaises(c.Refuse):c.predecessor(changed)

    def test_future_restored_failed_experiment_allowed_but_journal_binding_required(self):
        binding,_,audit=self.predecessor_fixture()
        with patch.object(c,'ARCHIVE_ROOT',self.root):
            self.assertEqual(c.predecessor(binding)[0]['replacement'],ORIGINAL)
            for key,value in [('journal_sha256','wrong'),('schema','E028-model-owner-audit-v2')]:
                bad=dict(audit,**{key:value});c.b.atomic_write(binding['audit']['path'],bad)
                changed=copy.deepcopy(binding);changed['audit']['sha256']=c.b.sha(binding['audit']['path'])
                with self.subTest(key=key),self.assertRaises(c.Refuse):c.predecessor(changed)

    def test_predecessor_mutation_live_location_and_incomplete_restore_rejected(self):
        binding,prior,audit=self.predecessor_fixture()
        with self.assertRaises(c.Refuse):c.predecessor(binding)
        with patch.object(c,'ARCHIVE_ROOT',self.root):
            for changes in [dict(state='LOAD_INTENT'),dict(post_monitor_failed=True),dict(warm_kv_restored=True),
                            dict(events=[dict(event='post_worker_cleanup_failed')]),dict(replacement=dict(ORIGINAL,pid=True))]:
                c.b.atomic_write(binding['path'],dict(prior,**changes));changed=copy.deepcopy(binding)
                with self.assertRaises(c.Refuse):c.predecessor(changed)  # unchanged immutable SHA
                changed['sha256']=c.b.sha(binding['path']);aa=dict(audit,journal_sha256=changed['sha256'])
                if 'replacement' in changes:aa['service_restore']=dict(aa['service_restore'],replacement=changes['replacement'])
                c.b.atomic_write(binding['audit']['path'],aa);changed['audit']['sha256']=c.b.sha(binding['audit']['path'])
                with self.subTest(changes=changes),self.assertRaises(c.Refuse):c.predecessor(changed)

    def test_original_and_absence_use_new_full_identity_not_old_pid_default(self):
        self.s.live[74456]=ORIGINAL;self.s.job_value=dict(pid=74456,state='running',path=c.b.PLIST);self.s.listener_value=74456
        self.assertEqual(c.original_identity(self.s,ORIGINAL),ORIGINAL)
        self.s.live[74456]=dict(ORIGINAL,start='reused')
        with self.assertRaises(c.Refuse):c.original_identity(self.s,ORIGINAL)
        self.s.live={};self.s.job_value=None;self.s.listener_value=None;c.absent_service(self.s,ORIGINAL)
        self.assertNotIn(74384,self.s.calls)
        with self.assertRaises(TypeError):c.absent_service(self.s)

    def test_replacement_requires_different_pid_same_fixed_job_and_listener(self):
        self.s.live[74456]=ORIGINAL;self.s.job_value=dict(pid=74456,state='running',path=c.b.PLIST);self.s.listener_value=74456
        with self.assertRaises(c.Refuse):c.replacement(self.s,ORIGINAL)
        other=dict(ORIGINAL,pid=90000);self.s.live[90000]=other;self.s.job_value['pid']=90000;self.s.listener_value=90000
        self.assertEqual(c.replacement(self.s,ORIGINAL),other)
        self.s.job_value['path']='other'
        with self.assertRaises(c.Refuse):c.replacement(self.s,ORIGINAL)

    def test_prepare_freezes_predecessor_and_authorize_rejects_changed_copy(self):
        self.j.data={};model=self.root/'model';model.mkdir();(model/'config.json').write_text('{}')
        settings=self.root/'settings.json';settings.write_text('{}')
        with (patch.object(c,'ARCHIVE_ROOT',self.root),patch.object(c.b,'MODEL',str(model)),
              patch.object(c.b,'GLOBAL_SETTINGS',settings),patch.object(c.b,'pins',return_value={}),
              patch.object(c.os,'getuid',return_value=501)):
            binding,prior,audit=self.predecessor_fixture()
            prior.update(global_settings_stat=c.b.file_stat(settings),model_metadata={str(model/'config.json'):c.b.sha(model/'config.json')})
            c.b.atomic_write(binding['path'],prior);binding['sha256']=c.b.sha(binding['path'])
            audit['journal_sha256']=binding['sha256'];c.b.atomic_write(binding['audit']['path'],audit)
            binding['audit']['sha256']=c.b.sha(binding['audit']['path'])
            wrapper=self.root/'wrapper.json'
            plan=dict(schema='E023-http-service-absent-owner-plan-v1',execution={'owner':{'path':str(self.source),'sha256':c.b.sha(self.source)}},
                cold_restart=dict(journal_path=str(self.j.directory),controller={'path':str(PATH),'sha256':c.b.sha(PATH)},
                    approved_contract=c.CONTRACT,proposal_plan_sha256=c.PLAN_SHA,predecessor=binding))
            c.b.atomic_write(wrapper,plan)
            self.s.live[74456]=ORIGINAL;self.s.job_value=dict(pid=74456,state='running',path=c.b.PLIST);self.s.listener_value=74456
            self.s.log_identity=lambda:{'cpu':'log-stat'}
            with patch.object(c.b,'service_status',return_value=prior['final_status']):
                c.prepare(self.j,self.s,self.owner,wrapper,c.b.sha(wrapper))
            self.assertEqual(self.j.data['state'],'PREPARED');self.assertEqual(self.j.data['original'],ORIGINAL)
            self.assertEqual((self.j.directory/'predecessor.json').read_bytes(),Path(binding['path']).read_bytes())
            c.authorize(self.j,c.CONTRACT,c.PLAN_SHA)
            (self.j.directory/'predecessor.json').write_text('{}')
            with self.assertRaisesRegex(c.Refuse,'copy changed'):c.authorize(self.j,c.CONTRACT,c.PLAN_SHA)

    def test_all_four_children_reaped_and_every_pid_absent(self):
        self.receipts();result=c.reaped(self.j,self.s);self.assertFalse(result['no_child_started'])
        self.assertEqual(set(self.s.calls),{81000,81001,81002,81003,81004})
        for pid in range(81000,81005):
            self.s.live[pid]={'pid':pid}
            with self.subTest(pid=pid),self.assertRaises(c.NotReady):c.reaped(self.j,self.s)
            self.s.live.clear()

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
        with self.assertRaises(c.Refuse):c.reaped(self.j,self.s)

    def test_index_cannot_omit_completed_phase_or_accept_false_reap_or_parent(self):
        self.receipts();index=c.b.read_json(self.owner/'children.json');saved=copy.deepcopy(index)
        index['children'].pop();self.rewrite_index(index)
        with self.assertRaises(c.Refuse):c.reaped(self.j,self.s)
        self.rewrite_index(saved)
        for kind,field,value in [('release','waited_reaped',False),('child','owner_pid',99999)]:
            index=copy.deepcopy(saved);row=index['children'][0];path=self.owner/row[kind]['path'];raw=path.read_bytes()
            payload=json.loads(raw);payload[field]=value;c.b.atomic_write(path,payload)
            row[kind]['sha256']=c.b.sha(path);self.rewrite_index(index)
            with self.subTest(kind=kind),self.assertRaises(c.Refuse):c.reaped(self.j,self.s)
            path.write_bytes(raw);self.rewrite_index(saved)

    def test_legacy_release_and_nonprefix_and_receipt_hash_are_rejected(self):
        self.receipts();path=self.owner/'release.json';raw=path.read_bytes()
        c.b.atomic_write(path,dict(pid=81001,returncode=0,waited_reaped=True,strict_absence=True))
        with self.assertRaises(c.Refuse):c.reaped(self.j,self.s)
        path.write_bytes(raw);index=c.b.read_json(self.owner/'children.json');index['children'].reverse();self.rewrite_index(index)
        with self.assertRaises(c.Refuse):c.reaped(self.j,self.s)
        index['children'].reverse();index['children'][0]['child']['sha256']='0'*64;self.rewrite_index(index)
        with self.assertRaises(c.Refuse):c.reaped(self.j,self.s)

    def test_no_child_terminal_remains_strict_and_source_pinned(self):
        self.receipts(nochild=True);self.assertTrue(c.reaped(self.j,self.s)['no_child_started'])
        path=self.owner/'spawn-intent.json';path.write_text('{}')
        with self.assertRaises(c.Refuse):c.reaped(self.j,self.s)
        path.unlink();self.source.write_text('changed')
        with self.assertRaises(c.Refuse):c.reaped(self.j,self.s)

    def test_prespawn_capture_diagnostics_with_empty_index_can_restore(self):
        self.receipts(nochild=True)
        folder=self.owner/'phases/c1-capture/capture';folder.mkdir(parents=True)
        (folder/'diagnostic.json').write_text('{"error":"CPU port preflight failure"}')
        index=self.owner/'children.json';c.b.atomic_write(index,dict(schema='cold-service-owner-children-v1',children=[]))
        result=c.reaped(self.j,self.s);self.assertTrue(result['no_child_started'])
        self.assertIn(str(index),[row['path'] for row in result['receipts']])
        for name in ('intent.json','child.json','release.json','spawn-intent.json'):
            path=folder/name;path.write_text('{}')
            with self.subTest(name=name),self.assertRaises(c.Refuse):c.reaped(self.j,self.s)
            path.unlink()
        c.b.atomic_write(index,dict(schema='cold-service-owner-children-v1',children=[{}]))
        with self.assertRaises(c.Refuse):c.reaped(self.j,self.s)

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
        with self.assertRaises(c.Refuse):c.reaped(self.j,self.s)
        c.b.atomic_write(proof_path,proof)
        result=c.reaped(self.j,self.s);self.assertTrue(result['no_child_started'])
        self.assertIn(str(proof_path),[row['path'] for row in result['receipts']])
        for key,value in [('popen_called',True),('attempt_sha256','wrong'),('command',['changed']),('time',True)]:
            c.b.atomic_write(proof_path,dict(proof,**{key:value}))
            with self.subTest(key=key),self.assertRaises(c.Refuse):c.reaped(self.j,self.s)
        c.b.atomic_write(proof_path,proof)
        child=folder/'child.json';child.write_text('{"pid":81001}')
        with self.assertRaises(c.Refuse):c.reaped(self.j,self.s)

    def test_unresolved_prespawn_crash_is_not_a_forged_terminal(self):
        self.receipts(nochild=True);(self.owner/'terminal.json').unlink()
        with self.assertRaises(c.NotReady):c.reaped(self.j,self.s)
        with self.assertRaises(c.Refuse):c.reaped(self.j,self.s,no_experiment=True)

    def test_resource_and_raw_capture_functions_unchanged_from_v2(self):
        old=ast.parse(PATH.with_name('controller_v2.py').read_text());new=ast.parse(PATH.read_text())
        for name in ('System','parse_resources','sample','check_resource','resource_gate','atomic_bytes','exclusive_json'):
            a=next(x for x in old.body if getattr(x,'name',None)==name);b=next(x for x in new.body if getattr(x,'name',None)==name)
            self.assertEqual(ast.dump(a,include_attributes=False),ast.dump(b,include_attributes=False),name)
        code=ast.get_source_segment(PATH.read_text(),next(x for x in new.body if getattr(x,'name',None)=='monitored_load'))
        self.assertIn("str(Path(__file__).resolve()),'load-worker'",code)
        self.assertNotIn('controller_v2.py',code)

    def test_help_safe_without_runtime_network_or_subprocess(self):
        script='''
import http.client,importlib.abc,importlib.util,socket,subprocess,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if name.split('.')[0] in {'mlx','numpy','torch','tutorial_code'}:raise AssertionError(name)
sys.meta_path.insert(0,Block())
socket.socket=lambda *a,**k:(_ for _ in ()).throw(AssertionError('socket'))
subprocess.Popen=lambda *a,**k:(_ for _ in ()).throw(AssertionError('subprocess'))
sp=importlib.util.spec_from_file_location('cold_v3',sys.argv[1]);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
raise SystemExit(m.main(['--help']))
'''
        r=subprocess.run([sys.executable,'-I','-B','-c',script,str(PATH)],capture_output=True,text=True,timeout=5)
        self.assertEqual(r.returncode,0,r.stderr)


if __name__=='__main__':unittest.main()
