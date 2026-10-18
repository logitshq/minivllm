"""Narrow retry contracts: real CLI/factory identity and failed-v1 predecessor.

No backend, process, HTTP, service, model or inference operation is used.
"""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bench.m5_ultra_256gb.gdn_graph_combined_formal_retry_v2 import owner as o
from bench.m5_ultra_256gb.gdn_graph_combined_formal_retry_v2 import controller as c
from bench.m5_ultra_256gb.gdn_graph_combined_formal_retry_v2 import audit_restore as a
from bench.m5_ultra_256gb.gdn_graph_combined_formal_retry_v2 import execute as e

JOURNAL_SHA = 'd2eb8f5fe8f579462960be97cbaa4cc70b24b2d071fd2683c60869fa53933934'
AUDIT_SHA = 'f0d067cf06319d2a1203129b2ace9c6252cceaf18f6687659975bf748e7990fc'


def binding(root):
    return dict(path=str(root/'restored-journal.json'),sha256=JOURNAL_SHA,
                audit=dict(path=str(root/'restoration-audit.json'),sha256=AUDIT_SHA))


def save_pair(root, prior, receipt):
    root.mkdir(exist_ok=True)
    jp=root/'restored-journal.json';ap=root/'restoration-audit.json'
    jp.write_text(json.dumps(prior)+'\n')
    jd=hashlib.sha256(jp.read_bytes()).hexdigest()
    receipt=copy.deepcopy(receipt);receipt['journal_sha256']=jd
    ap.write_text(json.dumps(receipt)+'\n')
    return dict(path=str(jp),sha256=jd,audit=dict(path=str(ap),sha256=hashlib.sha256(ap.read_bytes()).hexdigest()))


class RetryTests(unittest.TestCase):
    def test_owner_algorithm_preserved_in_real_retry_namespace(self):
        for name in ('run','full_prepare','validate_plan','main','source_pins'):
            self.assertIs(getattr(o,name).__code__,getattr(o._original,name).__code__)
            self.assertIs(getattr(o,name).__globals__,vars(o))
        self.assertEqual(o.HERE,Path(o.__file__).resolve().parent)
        self.assertEqual(o._original.HERE,o.ORIGINAL)
        self.assertEqual(o._original.PLAN_SCHEMA,'GDN-graph-combined-http-formal-owner-plan-v1')
        self.assertEqual(o.PLAN_SCHEMA,'GDN-graph-combined-http-formal-owner-plan-v2')
        self.assertIs(o.System,o._original.System)

    def test_factory_is_original_file_not_copied_or_spoofed(self):
        with patch.object(o,'module',return_value='factory') as load:
            self.assertEqual(o.load('formal'),'factory')
        self.assertEqual(load.call_args.args[:2],(o.ORIGINAL/'formal.py',o.FORMAL_SHA))
        self.assertEqual(o.sha(o.ORIGINAL/'formal.py'),o.FORMAL_SHA)
        self.assertFalse((o.HERE/'formal.py').exists())

    def test_actual_three_argument_cli_calls_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();out=root/'new-owner';plan=root/'plan.json'
            plan=root/'owner-plan.json'
            with patch.object(e,'P',root),patch.object(o,'run',return_value=7) as run:
                command=e.owner_command('a'*64,out)
                self.assertEqual(command[:4],[e.PY,'-I','-B',str(o.HERE/'owner.py')])
                self.assertNotIn('--run',command)
                self.assertEqual(o.main(command[4:]),7)
            args=run.call_args.args[0]
            self.assertEqual((args.plan,args.out,args.plan_sha256),(plan,out,'a'*64))
            self.assertFalse(args.prepare_only);self.assertFalse(out.exists())

    def test_unsupported_run_flag_rejects_before_owner_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();out=root/'new-owner'
            with patch.object(o,'run') as run,contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as error:
                o.main(['--run','--plan',str(root/'p'),'--plan-sha256','a'*64,'--out',str(out)])
            self.assertEqual(error.exception.code,2);run.assert_not_called();self.assertFalse(out.exists())

    def test_prepare_only_emits_true_original_prepare_schema_and_plan_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();out=root/'prepared';digest='b'*64
            def durable(path,value,exclusive):
                self.assertTrue(exclusive)
                with path.open('x') as stream:json.dump(value,stream)
            base=SimpleNamespace(durable=durable)
            plan={'execution':{'pins':{'actual':'source'}}};specs=[{'CPU':'same original factory'}]
            with patch.object(e,'P',root),patch.object(o,'validate_plan',return_value=(base,plan,b'',{},specs)),patch.object(o,'run') as run:
                command=e.owner_command(digest,out,prepare_only=True)
                self.assertEqual(command[-1],'--prepare-only');self.assertNotIn('--run',command)
                self.assertEqual(o.main(command[4:]),0)
            run.assert_not_called();row=json.loads((out/'prepared.json').read_text())
            self.assertEqual(row['schema'],'GDN-graph-combined-http-formal-owner-prepare-v1')
            self.assertEqual(row['owner_plan_sha256'],digest);self.assertEqual(row['formal_specs'],specs)
            self.assertIs(row['runtime_weights_verified'],False)

    def test_template_exact_twelve_specs_budget_and_new_source_identity(self):
        with patch.object(o,'source_pins',return_value={'test':'pin'}):
            plan=o.plan_template({'q':1},{'cold':2},{'deadlines':o.DEADLINES},{'base':3},{'route':4},{'combined':5},{'specs':6})
        self.assertEqual(len(plan),11);self.assertEqual(plan['schema'],o.PLAN_SCHEMA)
        self.assertEqual(plan['trials'],list(o.PHASES));self.assertEqual(plan['deadline_seconds'],40524)
        self.assertEqual(plan['execution']['owner'],dict(path=str(o.HERE/'owner.py'),sha256=o.sha(o.__file__)))
        self.assertEqual(plan['prepared_specs'],{'specs':6})

    def test_controller_bindings_preserve_algorithms_with_actual_entry(self):
        for name in ('prepare','authorize','load_worker','monitored_load','main'):
            self.assertIs(getattr(c,name).__code__,getattr(c._original,name).__code__)
            self.assertIs(getattr(c,name).__globals__,vars(c))
        for name in ('stop','phase_reaps','reaped','recover'):
            self.assertIs(getattr(c,name).__code__,getattr(c._original,name).__code__)
            self.assertIsNot(getattr(c,name).__globals__,getattr(c._original,name).__globals__)
        self.assertIs(c.recover.__globals__['monitored_load'],c.monitored_load)
        self.assertIs(c.recover.__globals__['authorize'],c.authorize)
        self.assertIs(c.reaped.__globals__['phase_reaps'],c.phase_reaps)
        self.assertEqual(c.monitored_load.__globals__['__file__'],c.__file__)
        self.assertEqual(c._original.OWNER_SCHEMA,'GDN-graph-combined-http-formal-owner-plan-v1')

    def test_actual_closed_no_experiment_archive_accepted_by_exact_bytes(self):
        prior,raw=c.predecessor(binding(c.PREDECESSOR_ARCHIVE))
        self.assertEqual([hashlib.sha256(x).hexdigest() for x in raw],[JOURNAL_SHA,AUDIT_SHA])
        self.assertEqual(prior['state'],'RESTORED');self.assertEqual(prior['reap'],{'no_experiment':True})
        self.assertEqual(prior['replacement']['pid'],91373)
        self.assertFalse(Path(prior['owner_directory']).exists())

    def test_predecessor_rejects_incomplete_wrong_source_or_experiment_claim(self):
        original=json.loads((c.PREDECESSOR_ARCHIVE/'restored-journal.json').read_text())
        receipt=json.loads((c.PREDECESSOR_ARCHIVE/'restoration-audit.json').read_text())
        changes=[('state','LOAD_INTENT'),('schema','GDN-graph-combined-http-cold-journal-v1'),
                 ('controller_sha256','0'*64),('reap',{'no_experiment':False}),
                 ('post_monitor_failed',True),('exact_runtime_signature_restored',True)]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve()
            for key,value in changes:
                with self.subTest(key=key):
                    prior=copy.deepcopy(original);prior[key]=value
                    ref=save_pair(root,prior,receipt)
                    with patch.object(c,'PREDECESSOR_ARCHIVE',root),self.assertRaises(c.Refuse):c.predecessor(ref)
            bad=copy.deepcopy(receipt);bad['owner_reap']['children']=[123]
            ref=save_pair(root,original,bad)
            with patch.object(c,'PREDECESSOR_ARCHIVE',root),self.assertRaises(c.Refuse):c.predecessor(ref)

    def test_predecessor_rejects_new_owner_directory_and_bad_ref_digest(self):
        prior=json.loads((c.PREDECESSOR_ARCHIVE/'restored-journal.json').read_text())
        receipt=json.loads((c.PREDECESSOR_ARCHIVE/'restoration-audit.json').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();prior['owner_directory']=str(root/'appeared-owner');Path(prior['owner_directory']).mkdir()
            ref=save_pair(root,prior,receipt)
            with patch.object(c,'PREDECESSOR_ARCHIVE',root),self.assertRaises(c.Refuse):c.predecessor(ref)
            Path(prior['owner_directory']).rmdir();ref['sha256']='0'*64
            with patch.object(c,'PREDECESSOR_ARCHIVE',root),self.assertRaises(c.Refuse):c.predecessor(ref)

    def test_restore_no_experiment_is_absence_not_fabricated_child(self):
        self.assertIs(a.owner_reap.__code__,a._original.owner_reap.__code__)
        with tempfile.TemporaryDirectory() as tmp:
            owner=Path(tmp).resolve()/'owner'
            data=dict(owner_directory=str(owner),reap={'no_experiment':True},removed_at=20)
            self.assertEqual(a.owner_reap(data,None,None,None,None),(20,dict(status='NO_EXPERIMENT',children=[])))
            owner.mkdir()
            with self.assertRaises(ValueError):a.owner_reap(data,None,None,None,None)

    def test_driver_failed_cpu_preflight_never_stops_or_recovers_service(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();P=root/'registered';J=root/'journal';D=root/'execution';O=root/'owner'
            P.mkdir();J.mkdir();(P/'owner-plan.json').write_text('{}\n')
            (J/'journal.json').write_text('{"state":"PREPARED"}\n')
            C=o.HERE/'controller.py';OWNER=o.HERE/'owner.py'
            pin_paths=[Path(e.__file__).resolve(),C,OWNER,P/'owner-plan.json']
            receipt=dict(source_pins={str(p):e.sha(p) for p in pin_paths},
                         owner_plan_sha256=e.sha(P/'owner-plan.json'),prepared_journal_sha256=e.sha(J/'journal.json'))
            (P/'receipt.json').write_text(json.dumps(receipt)+'\n')
            calls=[]
            def rejected(name,command):
                calls.append((name,command));return 2
            with patch.multiple(e,P=P,J=J,D=D,O=O,C=C,OWNER=OWNER),patch.object(e,'run',side_effect=rejected),patch.object(sys,'argv',[e.__file__,'--run']):
                self.assertEqual(e.main(),1)
            self.assertEqual([x[0] for x in calls],['owner-cli-prepare'])
            self.assertIn('--prepare-only',calls[0][1]);self.assertNotIn('--run',calls[0][1])
            final=json.loads((D/'final.json').read_text())
            self.assertEqual(final['journal_state'],'PREPARED');self.assertEqual(final['prepare_returncode'],2)
            self.assertIsNone(final['stop_returncode']);self.assertIsNone(final['recovery_returncode'])
            self.assertFalse(final['owner_invocation_attempted']);self.assertFalse(O.exists())

    def test_nonterminal_restore_incomplete_without_source_or_owner_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();(root/'journal.json').write_text(json.dumps({'state':'LOAD_INTENT'}))
            with patch.object(a,'sources',side_effect=AssertionError('must not inspect incomplete source')):
                result=a.audit(root)
        self.assertEqual(result['status'],'INCOMPLETE')
        self.assertFalse(result['qualified'] or result['performance_qualified'] or result['numerical_qualified'])


if __name__=='__main__':unittest.main()
