"""CPU-only retry metadata regressions; no transport, model or service calls."""
import ast
import builtins
import hashlib
import json
from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bench.m5_ultra_256gb.gdn_graph_combined_formal_retry_v2 import audit_formal as a

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False)+'\n')

def ref(root,path):return dict(path=str(path.relative_to(root)),sha256=a.sha(path))

def cmd_port(port):return ['/usr/sbin/lsof','-nP','-a',f'-iTCP:{port}','-sTCP:LISTEN','-Fpn']

def cmd_pid(pid):return ['/bin/ps','-p',str(pid),'-o','pid=']

def raw(command,inner=False):return dict(command=command,returncode=1,stdout='',**({'stderr':''} if inner else {}))

def lifecycle(root,*,old_source=False):
    root.mkdir();source=dict(path=str((a.DONOR.parent if old_source else a.HERE)/'owner.py'),sha256=(a.donor.OWNER_SHA if old_source else a.OWNER_SHA))
    cold=dict(journal_path=str(root/'journal'),controller=dict(sha256=a.CONTROLLER_SHA),predecessor={'mock':'closed'})
    plan=dict(schema='GDN-graph-combined-http-formal-owner-plan-v2',trials=list(a.PHASES),execution=dict(owner=source),cold_restart=cold,
              http=dict(transport=dict(port=18181)))
    write(root/'owner-plan.json',plan);digest=a.sha(root/'owner-plan.json')
    stopped=dict(state='STOPPED',owner_directory=str(root),owner_plan=dict(sha256=digest),owner_source=source,
        id='fake-cycle',removed_at=10,trials=list(a.PHASES),predecessor=cold['predecessor'],controller_sha256=a.CONTROLLER_SHA)
    write(root/'stopped-journal.json',stopped)
    attempt=dict(schema='cold-service-owner-attempt-v1',owner_pid=900,owner_identity=dict(pid=900),owner_source=source,
        owner_plan_sha256=digest,journal_id='fake-cycle',removed_at=10,created_at=11,journal_path=cold['journal_path'],
        controller_sha256=a.CONTROLLER_SHA,stopped_journal_sha256=a.sha(root/'stopped-journal.json'))
    write(root/'attempt.json',attempt);bound={k:attempt[k] for k in
        ('owner_pid','owner_identity','owner_source','owner_plan_sha256','journal_id','removed_at')}
    bound['attempt_sha256']=a.sha(root/'attempt.json');write(root/'spawn-intent.json',bound)
    children=[]
    for i,phase in enumerate(a.PHASES):
        folder=root/'phases'/phase;trial=folder/'trial';pid=1000+i;start=20+i*10
        write(folder/'intent.json',dict(phase=phase,time=start,**bound))
        identity=dict(pid=pid,ppid=900)
        write(folder/'child.json',dict(phase=phase,pid=pid,owner_pid=900,identity=identity,
            ownership='direct-Popen-returned',spawned_at=start+1))
        write(trial/'spawn-intent.json',dict(time=start,command=['CPU-FAKE']))
        write(trial/'child.json',dict(pid=pid,time=start+1))
        write(trial/'release.json',dict(pid=pid,returncode=-15,waited_reaped=True,strict_absence=True,port_absent=True,time=start+5))
        write(folder/'child-absence-00.json',raw(cmd_pid(pid)));write(folder/'port-absence-00.json',raw(cmd_port(18181)))
        write(folder/'release.json',dict(phase=phase,pid=pid,returncode=-15,waited_reaped=True,strict_absence=True,
              time=start+6,absence=ref(root,folder/'child-absence-00.json')))
        for name,command in (('port-before.json',cmd_port(18181)),('child-absence.json',cmd_pid(pid)),('port-after.json',cmd_port(18181))):
            write(trial/name,raw(command,True))
        write(trial/'complete.json',dict(artifacts={p.name:dict(sha256=a.sha(p),bytes=p.stat().st_size)
                                                   for p in trial.iterdir() if p.is_file()}))
        children.append(dict(phase=phase,pid=pid,identity=identity,**{name:ref(root,folder/(name+'.json')) for name in ('intent','child','release')}))
    write(root/'children.json',dict(schema='cold-service-owner-children-v1',children=children))
    write(root/'release.json',dict(schema='cold-service-owner-release-v2',children=children,children_sha256=a.sha(root/'children.json'),
        all_children_reaped=True,time=150,**bound))
    return plan,dict(owner_pid=900,owner_plan_sha256=digest)

class RetryBindingTests(unittest.TestCase):
    def test_frozen_raw_functions_keep_original_factory_and_globals(self):
        self.assertEqual(a.sha(a.DONOR),a.DONOR_SHA)
        self.assertEqual(a.sha(a.FORMAL),a.FORMAL_SHA)
        self.assertEqual(a.FORMAL,a.DONOR.parent/'formal.py')
        self.assertNotEqual(a.FORMAL.parent,a.HERE)
        for name in ('observer_identity','audit_trials','pinned','outer_absence','inner_absence',
                     'adapter_copies','preparation_pins','tokenizer_reference','raw_metrics',
                     'resources','paired_summary','input_inventory'):
            self.assertIs(getattr(a,name),getattr(a.donor,name),name)
        self.assertEqual(a.audit_trials.__globals__['HERE'],a.DONOR.parent)
        self.assertEqual(a.audit_trials.__globals__['__file__'],str(a.DONOR))
        self.assertEqual(a.donor.HERE,a.DONOR.parent)

    def test_only_declared_owner_schema_and_entry_metadata_delta(self):
        original=a.DONOR.read_text();new=Path(a.__file__).read_text()
        old={n.name:n for n in ast.parse(original).body if isinstance(n,ast.FunctionDef)}
        now={n.name:n for n in ast.parse(new).body if isinstance(n,ast.FunctionDef)}
        source=ast.get_source_segment(original,old['owner_binding']).replace(
            "'GDN-graph-combined-http-formal-owner-plan-v1'","'GDN-graph-combined-http-formal-owner-plan-v2'")
        self.assertEqual(ast.dump(ast.parse(source).body[0]),ast.dump(now['owner_binding']))
        source=ast.get_source_segment(original,old['audit']).replace(
            "'GDN-graph-combined-http-formal-independent-audit-v1'","'GDN-graph-combined-http-formal-independent-audit-v2'")
        source=source.replace("str(DONOR):DONOR_SHA,str(SUPPORT):SUPPORT_SHA", "str(DONOR):DONOR_SHA,str(donor.DONOR):donor.DONOR_SHA,str(SUPPORT):SUPPORT_SHA")
        source=source.replace("str(HERE/'formal.py'):FORMAL_SHA", "str(FORMAL):FORMAL_SHA")
        source=source.replace("reused_audit_source_sha256=DONOR_SHA,raw_algorithm_source_sha256=SUPPORT_SHA,", "reused_audit_source_sha256=DONOR_SHA,unchanged_factory_source=dict(path=str(FORMAL),sha256=FORMAL_SHA),raw_algorithm_source_sha256=SUPPORT_SHA,")
        self.assertEqual(ast.dump(ast.parse(source).body[0]),ast.dump(now['audit']))
        self.assertNotIn('audit_trials',now)

    def test_actual_retry_lifecycle_accepts_and_old_plan_schema_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            owner=Path(tmp).resolve()/'owner';plan,complete=lifecycle(owner)
            _,_,children,proofs=a.owner_binding(owner,plan,complete)
            self.assertEqual([row['phase'] for row in children],list(a.PHASES))
            self.assertEqual(len(proofs),12)
            plan['schema']='GDN-graph-combined-http-formal-owner-plan-v1'
            with self.assertRaisesRegex(ValueError,'real plan/source'):
                a.owner_binding(owner,plan,complete)

    def test_consistent_old_owner_receipts_still_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            owner=Path(tmp).resolve()/'owner';plan,complete=lifecycle(owner,old_source=True)
            with self.assertRaisesRegex(ValueError,'real plan/source'):
                a.owner_binding(owner,plan,complete)

    def test_failed_experiment_preserves_restore_without_raw_or_performance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();owner=root/'owner';journal=root/'journal'
            write(owner/'owner-plan.json',{'CPU':'failed-new-plan'})
            digest=a.sha(owner/'owner-plan.json')
            write(owner/'complete.json',dict(status='FAIL',errors=['CPU injected stage failure']))
            write(journal/'journal.json',dict(state='RESTORED',owner_directory=str(owner),
                owner_plan=dict(sha256=digest),owner_source=dict(path=str(a.HERE/'owner.py'),sha256=a.OWNER_SHA)))
            restoration=dict(status='PASS',qualified=True,performance_qualified=False,
                owner_plan_sha256=digest,journal_sha256=a.sha(journal/'journal.json'))
            def loader(path,sha,name):
                self.assertEqual(Path(path),a.HERE/'audit_restore.py');self.assertEqual(sha,a.RESTORE_SHA)
                return SimpleNamespace(audit=lambda actual:restoration)
            with patch.object(a,'load',side_effect=loader):result=a.audit(owner,journal,digest)
            self.assertEqual(result['status'],'FAIL');self.assertEqual(result['restoration'],restoration)
            self.assertTrue(result['partial_points_unqualified'])
            self.assertFalse(result['qualified'] or result['performance_qualified'] or result['http_raw_opened'])
            self.assertEqual(result['unchanged_factory_source'],dict(path=str(a.FORMAL),sha256=a.FORMAL_SHA))

    def test_import_and_missing_evidence_cannot_enter_backend(self):
        original=builtins.__import__
        def guard(name,*args,**kwargs):
            if name.split('.')[0] in ('mlx','mlx_lm','mlx_vlm','omlx','tokenizers'):
                raise AssertionError('backend import: '+name)
            return original(name,*args,**kwargs)
        with patch('builtins.__import__',side_effect=guard):
            runpy.run_path(a.__file__)
            with tempfile.TemporaryDirectory() as tmp:
                result=a.audit(Path(tmp)/'absent-owner',Path(tmp)/'absent-journal','missing')
        self.assertEqual(result['status'],'INCOMPLETE')
        self.assertFalse(result['qualified'] or result['performance_qualified'] or result['http_raw_opened'])


if __name__=='__main__':unittest.main()
