"""Pure CPU fakes: no controller mutation, HTTP, process signals or live probe."""
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
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

PATH=Path(__file__).with_name('controller_v2.py')
sp=importlib.util.spec_from_file_location('cold_v2_cpu',PATH)
c=importlib.util.module_from_spec(sp);sp.loader.exec_module(c)


class Fake:
    def __init__(self):
        self.clock=time.time();self.pressure=1;self.swap=0;self.footprint=30*c.GIB
        self.live={};self.samples=0;self.mutations=[]
    def process(self,pid):
        if pid==os.getpid():return dict(pid=pid,ppid=123,start='cpu-test',comm='python')
        return self.live.get(pid)
    def resources(self):
        self.samples+=1
        return dict(time=self.clock,system_footprint=self.footprint,physical_ram=256*c.GIB,
                    pressure=self.pressure,swap_bytes=self.swap,raw={'fixture':True})
    def sleep(self,n):self.clock+=n


class Child:
    pid=81234
    def __init__(self,returncode=None):self.returncode=returncode;self.calls=[];self.wait_raises=False
    def poll(self):return self.returncode
    def terminate(self):self.calls.append('terminate');self.returncode=-15
    def kill(self):self.calls.append('kill');self.returncode=-9
    def wait(self,timeout):
        self.calls.append(('wait',timeout))
        if self.wait_raises:
            self.wait_raises=False;raise subprocess.TimeoutExpired('fake',timeout)
        return self.returncode


class V2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.j=c.Journal(self.root/'journal',create=True)
        self.addCleanup(self.j.close);self.owner=self.root/'owner';self.s=Fake()
        (self.root/'owner.py').write_text('# fixed CPU owner fixture\n')
        self.j.data=dict(schema='omlx-cold-restart-journal-v2',id='cpu-journal',state='STOPPED',
            removed_at=time.time()-10,controller_sha256=c.b.sha(PATH),owner_directory=str(self.owner),
            owner_plan={'path':str(self.root/'wrapper.json'),'sha256':'a'*64},
            owner_source={'path':str(self.root/'owner.py'),'sha256':c.b.sha(self.root/'owner.py')},resource_samples=[],events=[])
        self.j.save('cpu-fixture')

    def receipts(self,nochild=True):
        self.owner.mkdir()
        raw=self.j.path.read_bytes();(self.owner/'stopped-journal.json').write_bytes(raw)
        a=dict(schema='cold-service-owner-attempt-v1',owner_pid=81000,
               owner_identity=dict(pid=81000,ppid=1,start='cpu',comm='python'),
               journal_id=self.j.data['id'],journal_path=str(self.j.directory),
               controller_sha256=self.j.data['controller_sha256'],owner_plan_sha256='a'*64,
               owner_source=self.j.data['owner_source'],removed_at=self.j.data['removed_at'],
               created_at=time.time(),phase='PRESPAWN',stopped_journal_sha256=hashlib.sha256(raw).hexdigest())
        c.b.atomic_write(self.owner/'attempt.json',a)
        if nochild:
            terminal=dict(schema='cold-service-owner-terminal-v1',status='NO_CHILD_STARTED',
                owner_pid=81000,owner_identity=a['owner_identity'],owner_source=a['owner_source'],
                owner_plan_sha256='a'*64,journal_id=a['journal_id'],removed_at=a['removed_at'],
                terminal_at=time.time(),phase='validation',error='CPU failure',
                attempt_sha256=c.b.sha(self.owner/'attempt.json'),spawn_intent_written=False,child_started=False)
            c.b.atomic_write(self.owner/'terminal.json',terminal)
        else:
            c.b.atomic_write(self.owner/'spawn-intent.json',dict(owner_pid=81000,journal_id=a['journal_id'],
                owner_plan_sha256='a'*64,attempt_sha256=c.b.sha(self.owner/'attempt.json')))
            c.b.atomic_write(self.owner/'child.json',dict(pid=81001))
            c.b.atomic_write(self.owner/'release.json',dict(pid=81001,returncode=-15,waited_reaped=True,strict_absence=True))

    def test_durable_pre_spawn_failure_can_recover_without_fake_child(self):
        self.receipts()
        row=c.reaped(self.j,self.s)
        self.assertTrue(row['no_child_started']);self.assertEqual(row['owner_pid'],81000)
        self.assertFalse((self.owner/'child.json').exists())
        self.assertEqual(len(row['receipts']),3)

    def test_no_child_proof_rejects_intent_and_binding_corruption(self):
        self.receipts()
        for name in ('spawn-intent.json','child.json','release.json'):
            path=self.owner/name;path.write_text('{}')
            with self.assertRaises(c.Refuse):c.reaped(self.j,self.s)
            path.unlink()
        path=self.owner/'terminal.json';original=c.b.read_json(path)
        for key,value in (('owner_plan_sha256','wrong'),('attempt_sha256','wrong'),('child_started',True),
                          ('spawn_intent_written',True),('terminal_at',True),('owner_pid',81001)):
            c.b.atomic_write(path,dict(original,**{key:value}))
            with self.subTest(key=key):
                with self.assertRaises(c.Refuse):c.reaped(self.j,self.s)
        c.b.atomic_write(path,original)
        self.s.live[81000]={'pid':81000}
        with self.assertRaises(c.NotReady):c.reaped(self.j,self.s)

    def test_child_branch_requires_durable_intent_reap_and_both_absent(self):
        self.receipts(False);self.assertFalse(c.reaped(self.j,self.s)['no_child_started'])
        for pid in (81000,81001):
            self.s.live[pid]={'pid':pid}
            with self.assertRaises(c.NotReady):c.reaped(self.j,self.s)
            del self.s.live[pid]
        (self.owner/'spawn-intent.json').unlink()
        with self.assertRaises(c.NotReady):c.reaped(self.j,self.s)
        with self.assertRaises(c.Refuse):c.reaped(self.j,self.s,True)

    def test_no_experiment_requires_owner_directory_never_appeared(self):
        self.assertTrue(c.reaped(self.j,self.s,True)['no_experiment'])
        self.owner.mkdir()
        with self.assertRaises(c.Refuse):c.reaped(self.j,self.s,True)

    def test_no_child_proof_also_rechecks_exact_owner_source(self):
        self.receipts();(self.root/'owner.py').write_text('# replaced owner\n')
        with self.assertRaisesRegex(c.Refuse,'pinned owner source changed'):c.reaped(self.j,self.s)

    def test_physical_parse_uses_system_field_and_strict_byte_units(self):
        raw={'unit':'byte','bytes per unit':1,'errors':[],'warnings':[],
             'total footprint':1,'auxiliary':{'sys_footprint':30*c.GIB}}
        row=c.parse_resources(raw,'1\n','vm.swapusage: used = 12.00M','274877906944\n',1)
        self.assertEqual(row['system_footprint'],30*c.GIB)
        for key,value in (('unit','KiB'),('bytes per unit',True),('errors',['bad']),('warnings',['bad'])):
            with self.assertRaises(c.Refuse):c.parse_resources(dict(raw,**{key:value}),'1','used = 0M',str(256*c.GIB),1)
        broken=copy.deepcopy(raw);broken['auxiliary']['sys_footprint']=True
        with self.assertRaises(c.Refuse):c.parse_resources(broken,'1','used = 0M',str(256*c.GIB),1)

    def test_every_gate_sample_is_retained_not_overwritten(self):
        c.resource_gate(self.j,self.s);c.resource_gate(self.j,self.s,loaded=True)
        self.assertEqual(len(self.j.data['resource_samples']),4)
        self.assertEqual(len(c.b.read_json(self.j.path)['resource_samples']),4)
        self.s.pressure=2
        with self.assertRaises(c.NotReady):c.resource_gate(self.j,self.s)
        self.assertEqual(len(self.j.data['resource_samples']),6)

    def test_system_resources_preserves_complete_raw_and_commands_offline(self):
        backend=c.System(self.j.directory)
        raw={'unit':'byte','bytes per unit':1,'errors':[],'warnings':[],
             'auxiliary':{'sys_footprint':30*c.GIB},'unparsed_evidence':{'kept':123}}
        def run(argv,**kw):
            if argv[0]=='/usr/bin/footprint':
                Path(argv[-1]).write_text(json.dumps(raw));out='fixture footprint stdout'
            elif argv[-1]=='kern.memorystatus_vm_pressure_level':out='1'
            elif argv[-1]=='vm.swapusage':out='used = 10M'
            else:out=str(256*c.GIB)
            return NS(returncode=0,stdout=out,stderr='')
        with patch.object(c.subprocess,'run',side_effect=run):row=backend.resources()
        saved=Path(row['raw']['path'])
        self.assertEqual(json.loads(saved.read_text()),raw)
        self.assertEqual(c.b.sha(saved),row['raw']['sha256'])
        self.assertEqual(len(c.b.read_json(row['raw']['commands_path'])),4)

    def test_log_window_bytes_sha_offset_retained_even_when_markers_fail(self):
        path=self.root/'service.log';path.write_bytes(b'old\n');before=c.b.file_stat(path)
        new=b'2026 - INFO - incomplete load\n';path.write_bytes(b'old\n'+new)
        backend=c.System(self.j.directory)
        with patch.object(c.b,'LOG',path):lines,receipt=backend.log_window(before)
        self.assertEqual(Path(receipt['path']).read_bytes(),new)
        self.assertEqual(receipt['sha256'],hashlib.sha256(new).hexdigest())
        self.assertEqual((receipt['start_offset'],receipt['end_offset']),(4,4+len(new)))
        with self.assertRaises(c.Refuse):c.b.verify_logs(lines)
        self.assertTrue(Path(receipt['path']).exists())
        with patch.object(c.b,'LOG',path):
            with self.assertRaises(c.Refuse):backend.log_window(dict(before,ino=before['ino']+1))

    def test_monitored_post_wait_retains_samples_and_direct_reap(self):
        child=Child(0);identity={'pid':90001}
        with patch.object(c.b,'replacement',return_value=identity):
            c.monitored_load(self.j,self.s,identity,{'swap_bytes':0},popen=lambda *a,**k:child)
        self.assertEqual(self.j.data['state'],'LOAD_INTENT')
        self.assertEqual(len(self.j.data['resource_samples']),1)
        release=c.b.read_json(self.j.directory/'load-worker/release.json')
        self.assertTrue(release['waited_reaped']);self.assertTrue(release['strict_absence'])
        self.assertFalse(self.s.mutations)

    def test_monitor_breach_cancels_only_fake_http_wait_and_preserves_intent(self):
        child=Child();self.s.pressure=2
        with patch.object(c.b,'replacement',return_value={'pid':90001}):
            with self.assertRaises(c.NotReady):
                c.monitored_load(self.j,self.s,{'pid':90001},{'swap_bytes':0},popen=lambda *a,**k:child)
        self.assertEqual(child.calls[0],'terminate');self.assertEqual(self.j.data['state'],'LOAD_INTENT')
        self.assertTrue(self.j.data['post_monitor_failed']);self.assertEqual(self.s.mutations,[])
        self.assertIn('service loader may remain active',self.j.data['cancellation_scope'])

    def test_probe_failure_and_cleanup_error_do_not_replace_primary(self):
        child=Child();child.wait_raises=True
        def fail():raise ValueError('original probe error')
        self.s.resources=fail
        with self.assertRaisesRegex(ValueError,'original probe error'):
            c.monitored_load(self.j,self.s,{}, {'swap_bytes':0},popen=lambda *a,**k:child)
        self.assertIn('kill',child.calls)
        self.assertTrue(c.b.read_json(self.j.directory/'load-worker/release.json')['strict_absence'])

    def test_monitored_post_has_absolute_bound_without_retry(self):
        child=Child();calls=[];clock=[0]
        def spawn(*a,**kw):calls.append(a);clock[0]=601;return child
        with patch.object(c.time,'monotonic',side_effect=lambda:clock[0]):
            with self.assertRaisesRegex(c.Refuse,'absolute monitored POST'):
                c.monitored_load(self.j,self.s,{}, {'swap_bytes':0},popen=spawn)
        self.assertEqual(len(calls),1);self.assertIn('terminate',child.calls)

    def test_internal_worker_rejects_wrong_nonce_without_backend_or_signal(self):
        self.j.save('fixture-load-intent',state='LOAD_INTENT',approved_contract=c.CONTRACT,
                    load_worker={'nonce':'fixed','directory':str(self.j.directory/'load-worker')})
        with patch.object(c.b,'System',side_effect=AssertionError('no backend')):
            with self.assertRaises(c.Refuse):c.load_worker(self.j.directory,'wrong')

    def test_missing_approval_does_not_open_journal_or_instantiate_system(self):
        with (patch.object(c,'Journal',side_effect=AssertionError('no journal')),
              patch.object(c,'System',side_effect=AssertionError('no system'))):
            with self.assertRaises(c.Refuse):
                c.main(['stop','--journal',str(self.root/'unused'),'--plan-sha256',c.PLAN_SHA])

    def test_default_help_safe_without_runtime_socket_or_commands(self):
        code='''
import http.client,importlib.abc,importlib.util,socket,subprocess,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if name.split('.')[0] in {'mlx','numpy','torch','tutorial_code'}:raise AssertionError(name)
sys.meta_path.insert(0,Block())
socket.socket=lambda *a,**k:(_ for _ in ()).throw(AssertionError('socket'))
subprocess.Popen=lambda *a,**k:(_ for _ in ()).throw(AssertionError('process'))
sp=importlib.util.spec_from_file_location('v2',sys.argv[1]);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
raise SystemExit(m.main(['--help']))
'''
        r=subprocess.run([sys.executable,'-I','-B','-c',code,str(PATH)],capture_output=True,text=True,timeout=5)
        self.assertEqual(r.returncode,0,r.stderr)


if __name__=='__main__':unittest.main()
