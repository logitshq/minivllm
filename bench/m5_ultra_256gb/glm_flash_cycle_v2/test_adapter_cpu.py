"""Source-prepared fake checks; no model, GPU, sockets or service operations."""
import copy
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent

def module(name):
    sp=importlib.util.spec_from_file_location('glm_v2_test_'+name,HERE/(name+'.py'))
    m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m

c=module('controller');o=module('owner');a=module('audit_restore')
p=module('preregister');x=module('execute');closed=module('audit_closed')

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')
def ref(path):return dict(path=str(path),sha256=c.b.sha(path))


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()

    def prior(self):
        folder=self.root/'archive'/'future-gdn';folder.mkdir(parents=True)
        replacement=dict(pid=20004,ppid=1,start='synthetic',comm='omlx-server')
        prior=dict(schema='GDN-prework-http-formal-cold-journal-v1',controller_sha256=c.PRIOR_SHA,
            trials=list(c.PRIOR_PHASES),state='RESTORED',approved_contract=c.CONTRACT,
            plan_sha256=c.PLAN_SHA,support_sha256=c.BASE_SHA,events=[],replacement=replacement,
            owner_plan={'sha256':'b'*64},log_window=dict(sha256='c'*64,bytes=5),
            exact_runtime_signature_restored=False,warm_kv_restored=False,
            final_status=dict(version='0.7.0',default_model=c.b.TARGET,loaded=[c.b.TARGET],
                loading=0,target_loading=False,path=c.b.MODEL,pinned=False,engine_type='vlm',config_model_type='qwen4_exp'))
        jp=folder/'restored-journal.json';ap=folder/'restoration-audit.json';write(jp,prior)
        proof=dict(schema='cold-service-restoration-audit-v1',status='PASS',qualified=True,
            audit_source_sha256=c.PRIOR_AUDIT_SHA,journal_sha256=c.b.sha(jp),controller_sha256=c.PRIOR_SHA,
            owner_plan_sha256='b'*64,owner_reap={'owner_status':'FAIL'},
            service_restore=dict(status='PERSISTED_VISIBLE_RECEIPTS_CONSISTENT',replacement=replacement,
                raw_log_sha256='c'*64,raw_log_bytes=5,http_worker_returncode=0))
        write(ap,proof)
        return jp,ap,prior,proof

    def test_future_actual_pair_binds_restoration_not_experiment_success(self):
        jp,ap,prior,proof=self.prior()
        with patch.object(c,'ARCHIVE_ROOT',self.root/'archive'):
            got,raw=c.predecessor(dict(ref(jp),audit=ref(ap)))
        self.assertEqual(got,prior);self.assertEqual(raw,[jp.read_bytes(),ap.read_bytes()])
        self.assertEqual(proof['owner_reap']['owner_status'],'FAIL')

    def test_missing_future_pair_is_not_a_prepared_receipt(self):
        with self.assertRaises(Exception):c.predecessor({})
        binding=dict(path=str(self.root/'missing/restored-journal.json'),sha256='a'*64,
            audit=dict(path=str(self.root/'missing/restoration-audit.json'),sha256='b'*64))
        with patch.object(c,'ARCHIVE_ROOT',self.root):
            with self.assertRaises(FileNotFoundError):c.predecessor(binding)
        self.assertFalse((self.root/'missing').exists())

    def test_wrong_history_incomplete_recovery_and_stale_sha_rejected(self):
        jp,ap,prior,proof=self.prior()
        with patch.object(c,'ARCHIVE_ROOT',self.root/'archive'):
            for key,value in [('schema','E024-route-http-formal-cold-journal-v1'),('trials',['glm-official-load']),
                              ('controller_sha256','0'*64),('state','LOAD_INTENT'),('warm_kv_restored',True)]:
                bad=dict(prior);bad[key]=value;write(jp,bad);write(ap,dict(proof,journal_sha256=c.b.sha(jp)))
                with self.subTest(key=key),self.assertRaises(Exception):c.predecessor(dict(ref(jp),audit=ref(ap)))
            write(jp,prior);write(ap,proof);binding=dict(ref(jp),audit=ref(ap));write(jp,dict(prior,extra='tamper'))
            with self.assertRaises(Exception):c.predecessor(binding)

    def test_wrong_auditor_status_or_worker_receipt_rejected(self):
        jp,ap,prior,proof=self.prior()
        with patch.object(c,'ARCHIVE_ROOT',self.root/'archive'):
            variants=[dict(proof,status='FAIL'),dict(proof,qualified=False),dict(proof,audit_source_sha256='0'*64),
                dict(proof,service_restore=dict(proof['service_restore'],http_worker_returncode=1)),
                dict(proof,service_restore=dict(proof['service_restore'],replacement=dict(prior['replacement'],pid=99)))]
            for bad in variants:
                write(ap,bad)
                with self.assertRaises(Exception):c.predecessor(dict(ref(jp),audit=ref(ap)))

    def test_archive_root_and_same_parent_are_required(self):
        jp,ap,_,proof=self.prior();other=self.root/'archive'/'different'/'restoration-audit.json';write(other,proof)
        with patch.object(c,'ARCHIVE_ROOT',self.root/'archive'):
            with self.assertRaises(Exception):c.predecessor(dict(ref(jp),audit=ref(other)))
        with patch.object(c,'ARCHIVE_ROOT',self.root/'unrelated'):
            with self.assertRaises(Exception):c.predecessor(dict(ref(jp),audit=ref(ap)))

    def test_original_algorithms_and_actual_module_identity(self):
        for new,old,names in [(o,o._original,('run','validate_plan','checkpoint','verify_inventory','child_command','probe_terminal')),
            (a,a._original,('audit','owner_reap','child_absence')),
            (closed,closed._original,('audit','request_counts','raw_sample','checkpoints')),
            (x,x._original,('execute','preflight'))]:
            for name in names:
                self.assertIs(getattr(new,name).__code__,getattr(old,name).__code__)
                self.assertIs(getattr(new,name).__globals__,vars(new))
                self.assertNotEqual(getattr(new,name).__globals__['__file__'],getattr(old,name).__globals__['__file__'])
        self.assertIs(o.System,o._original.System)
        self.assertEqual(o.FIXED,o._original.FIXED)

    def test_recovery_injection_and_donor_globals_remain_separate(self):
        self.assertIs(c.stop.__code__,c._original.stop.__code__)
        self.assertIs(c.recover.__code__,c._original.recover.__code__)
        self.assertIs(c.stop.__globals__['authorize'],c.authorize)
        self.assertIs(c.recover.__globals__['reaped'],c.reaped)
        self.assertIs(c.recover.__globals__['monitored_load'],c.monitored_load)
        self.assertEqual(Path(c.monitored_load.__globals__['__file__']).resolve(),HERE/'controller.py')
        self.assertEqual(c._original.PRIOR_PATH.parent.name,'route_http_formal')
        self.assertEqual(c.PRIOR_PATH.parent.name,'gdn_prework_formal')
        self.assertIsNot(c._original.recover.__globals__['monitored_load'],c.monitored_load)

    def test_source_pins_new_dict_and_all_original_sources_retained(self):
        pins={str(self.root/'synthetic.py'):'a'*64}
        plan={'execution':dict(owner=dict(path=str(HERE/'owner.py'),sha256=c.OWNER_SHA),pins=dict(pins))}
        with patch.object(c,'owner_module',return_value=SimpleNamespace(source_pins=lambda plan:dict(plan['execution']['pins']))):
            result=c.source_pins(plan)
        self.assertEqual(plan['execution']['pins'],pins)
        self.assertIn(str(HERE/'_reuse.py'),result)
        for name,digest in c._adapter.FIXED.items():self.assertEqual(result[str(HERE.parent/'glm_flash_cycle'/name)],digest)

    def test_new_owner_plan_receives_actual_registered_output(self):
        plan=dict(schema=c.OWNER_SCHEMA,outputs={'owner':str(self.root/'owner')},
            cold_restart=dict(journal_path=str(self.root/'journal'),controller=ref(HERE/'controller.py')))
        path=self.root/'plan.json';write(path,plan);seen=[]
        def validate(args):seen.append(args);return plan,None,None,None
        with patch.object(c,'owner_module',return_value=SimpleNamespace(validate_plan=validate)),patch.object(c,'source_pins',return_value={}):
            self.assertEqual(c.owner_plan(path,c.b.sha(path),self.root/'journal'),plan)
            self.assertEqual(seen[0].out,self.root/'owner')
            with self.assertRaises(Exception):c.owner_plan(path,c.b.sha(path),self.root/'wrong')

    def test_default_descriptions_do_not_run_or_prepare(self):
        for m,method in ((o,'run'),(p,'prepare'),(x,'execute')):
            with patch.object(m,method,side_effect=AssertionError('must remain inert')),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(m.main([]),0)
                with self.assertRaises((ValueError,SystemExit)):m.main(['--out',str(self.root/'unused')])
        self.assertFalse((self.root/'unused').exists())

    def test_saved_commands_actual_v2_sources_venv_and_plan_sha(self):
        plan=dict(execution={'owner':{'path':str(HERE/'owner.py')}},outputs={'owner':str(self.root/'owner')},
            cold_restart=dict(controller={'path':str(HERE/'controller.py')},journal_path=str(self.root/'journal'),
                proposal_plan_sha256='a'*64,approved_contract=c.CONTRACT))
        python='/synthetic/venv/bin/python';digest='b'*64
        commands=p.build_commands(self.root/'plan.json',digest,plan,self.root/'audit.json',python)
        for name in ('controller_prepare','controller_stop','controller_recover'):
            self.assertEqual(commands[name][:4],[python,'-I','-B',str(HERE/'controller.py')])
        self.assertEqual(commands['owner'][:5],[python,'-I','-B',str(HERE/'owner.py'),'--run'])
        self.assertEqual(commands['closed_audit'][3],str(HERE/'audit_closed.py'))
        for name,flag in [('controller_prepare','--owner-plan-sha256'),('owner','--plan-sha256'),('closed_audit','--plan-sha256')]:
            argv=commands[name];self.assertEqual(argv[argv.index(flag)+1],digest)
        with self.assertRaises(ValueError):p.validate_outputs([self.root/'r',self.root/'o',self.root/'j',self.root/'o'/'audit'])

    def driver_context(self):
        journal=self.root/'journal';write(journal/'journal.json',{'state':'STOPPED'})
        return dict(execution=self.root/'execution',owner=self.root/'owner',journal=journal,
            commands=dict(controller_stop=['stop'],owner=['owner','--run'],controller_recover=['recover']),
            receipt_raw=b'{}',prepared_raw=b'{}',owner_plan_sha256='a'*64)

    def test_unknown_owner_invocation_still_recovers_without_no_experiment(self):
        ctx=self.driver_context();seen=[]
        def run(folder,name,command):
            seen.append((name,command))
            if name=='glm-owner':raise RuntimeError('unknown outcome')
            if name=='recover':write(ctx['journal']/'journal.json',{'state':'RESTORED'})
            return 0
        with self.assertRaisesRegex(RuntimeError,'unknown outcome'):x.execute(ctx,run=run)
        self.assertEqual(seen[-1],('recover',['recover']))
        final=x.read(ctx['execution']/'final.json')
        self.assertTrue(final['owner_invocation_attempted']);self.assertFalse(final['owner_invocation_returned'])

    def test_waited_nonzero_owner_may_prove_no_start_but_cannot_succeed(self):
        ctx=self.driver_context();seen=[]
        def run(folder,name,command):
            seen.append((name,command))
            if name=='glm-owner':return 2
            if name=='recover':write(ctx['journal']/'journal.json',{'state':'RESTORED'})
            return 0
        self.assertEqual(x.execute(ctx,run=run),1)
        self.assertEqual(seen[-1],('recover',['recover','--no-experiment']))
        self.assertEqual(x.read(ctx['execution']/'final.json')['owner_returncode'],2)

    def nochild_fixture(self):
        owner=self.root/'owner';owner.mkdir();journal=self.root/'journal';journal.mkdir()
        now=time.time()-100;source=self.root/'source.py';source.write_text('# fake owner source\n')
        plan=dict(schema=c.OWNER_SCHEMA,outputs={'owner':str(owner)},
            cold_restart=dict(journal_path=str(journal),controller=ref(HERE/'controller.py')))
        pp=self.root/'owner-plan.json';write(pp,plan)
        data=dict(schema=c.JOURNAL_SCHEMA,state='STOPPED',id='CPU',removed_at=now,
            controller_sha256=c.b.sha(HERE/'controller.py'),owner_directory=str(owner),owner_plan=ref(pp),
            owner_source=ref(source),trials=[c.PHASE],original=dict(pid=20001,ppid=1,start='old',comm='omlx-server'),
            predecessor={},predecessor_copies={})
        write(owner/'stopped-journal.json',data)
        attempt=dict(schema='cold-service-owner-attempt-v1',phase='PRESPAWN',owner_pid=20002,
            owner_identity=dict(pid=20002,ppid=2,start='owner',comm='python'),owner_source=data['owner_source'],
            owner_plan_sha256=data['owner_plan']['sha256'],journal_id='CPU',journal_path=str(journal),
            controller_sha256=data['controller_sha256'],removed_at=now,created_at=now+1,
            stopped_journal_sha256=c.b.sha(owner/'stopped-journal.json'))
        write(owner/'attempt.json',attempt);(owner/'phases'/c.PHASE).mkdir(parents=True)
        write(owner/'children.json',dict(schema='cold-service-owner-children-v1',children=[]))
        bound={k:attempt[k] for k in ('owner_pid','owner_identity','owner_source','owner_plan_sha256','journal_id','removed_at')}
        write(owner/'terminal.json',dict(bound,schema='cold-service-owner-terminal-v1',status='NO_CHILD_STARTED',
            attempt_sha256=c.b.sha(owner/'attempt.json'),terminal_at=now+2,spawn_intent_written=False,child_started=False))
        def validate(args):
            self.assertEqual(c.b.sha(args.plan),args.plan_sha256)
            self.assertEqual(args.out,owner)
            return plan,None,None,None
        return owner,pp,SimpleNamespace(data=data,directory=journal),SimpleNamespace(validate_plan=validate)

    def test_empty_preintent_directory_uses_actual_external_plan(self):
        owner,pp,j,validator=self.nochild_fixture()
        with patch.object(c,'owner_module',return_value=validator),patch.object(c,'source_pins',return_value={}):
            got=c.reaped(j,SimpleNamespace(process=lambda pid:None))
            self.assertTrue(got['no_child_started']);self.assertFalse((owner/'owner-plan.json').exists())
            write(pp,dict(c.b.read_json(pp),changed=True))
            with self.assertRaises(AssertionError):c.reaped(j,SimpleNamespace(process=lambda pid:None))

    def test_unknown_intent_cannot_use_no_child_terminal(self):
        owner,pp,j,validator=self.nochild_fixture();write(owner/'spawn-intent.json',{})
        with patch.object(c,'owner_module',return_value=validator),patch.object(c,'source_pins',return_value={}):
            with self.assertRaises(Exception):c.reaped(j,SimpleNamespace(process=lambda pid:None))

    def test_missing_terminal_does_not_grant_diagnostic_or_restore_pass(self):
        self.assertEqual(a.audit(self.root/'missing')['status'],'INCOMPLETE')
        result=closed.audit(self.root/'missing','a'*64)
        self.assertEqual(result['status'],'INCOMPLETE')
        self.assertFalse(result['observed_diagnostic_complete']);self.assertFalse(result['performance_qualified'])


if __name__=='__main__':unittest.main()
