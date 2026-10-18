"""Offline fake receipts only; no process queries, signals or service actions."""
import ast
import copy
import importlib.util
import inspect
import json
from pathlib import Path
import tempfile
import unittest


def load(name):
    path=Path(__file__).with_name(name+'.py')
    spec=importlib.util.spec_from_file_location('glm_v3_cpu_'+name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


c=load('controller');audit=load('audit_restore')


class Tests(unittest.TestCase):
    def write(self,path,value): path.write_text(json.dumps(value))

    def evidence(self,root):
        folder=root/'monitor';folder.mkdir()
        for lane in ('physical','metadata'): (folder/lane).mkdir()
        row=dict(command=['/usr/bin/vm_stat'],started_monotonic=20.,started_at=200.,deadline_monotonic=25.,
                 pid=8103,waited_reaped=True,unknown_launch=False,returncode=-15,status='INCOMPLETE')
        stem=folder/'physical/command-0000'
        self.write(stem.with_suffix('.intent.json'),{k:row[k] for k in
                   ('command','started_monotonic','started_at','deadline_monotonic')})
        self.write(stem.with_suffix('.child.json'),dict(pid=8103,owner_pid=8101,
                   ownership='direct-Popen-returned',command=row['command']))
        self.write(stem.with_suffix('.json'),row)
        report=dict(schema='GLM-dual-cadence-monitor-v1',status='INCOMPLETE',owner_pid=8101,pid=8102,
            read_commands_settled=True,unjoined_workers=[],unresolved_read_commands=[],
            commands=dict(physical=[row],metadata=[]))
        self.write(folder/'monitor.json',report)
        complete=dict(schema='GLM-cold-load-owner-result-v3',owner_pid=8101,status='FAIL',
            monitor=dict(started=True,read_commands_settled=True,status='INCOMPLETE',
                         receipt=dict(path=str(folder/'monitor.json'),sha256=c.b.sha(folder/'monitor.json'))))
        self.write(root/'complete.json',complete)
        return report,complete

    def rebound(self,root,report,complete):
        self.write(root/'monitor/monitor.json',report)
        complete['monitor']['receipt']['sha256']=c.b.sha(root/'monitor/monitor.json')
        self.write(root/'complete.json',complete)

    def test_unsealed_actual_predecessor_has_no_success_path(self):
        with self.assertRaisesRegex(Exception,'not sealed'): c.predecessor_manifest()
        with self.assertRaisesRegex(Exception,'not sealed'): c.predecessor({})

    def test_failed_but_waited_probe_allows_restoration_file_proof(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();self.evidence(root)
            files=c.monitor_receipts(root,8101,8102)
            self.assertEqual(len(files),4)
            self.assertEqual(files[0],root/'monitor/monitor.json')

    def test_unjoined_unknown_or_unreaped_monitor_rejected_even_with_rebound_sha(self):
        changes=[('unjoined_workers',['physical']),('unresolved_read_commands',[{'pid':8103}]),
                 ('read_commands_settled',False)]
        for key,bad in changes:
            with self.subTest(key=key),tempfile.TemporaryDirectory() as temp:
                root=Path(temp).resolve();report,complete=self.evidence(root)
                report[key]=bad;self.rebound(root,report,complete)
                with self.assertRaises(Exception):c.monitor_receipts(root,8101,8102)
        for key,bad in [('unknown_launch',True),('waited_reaped',False),('pid',8102),('returncode',False)]:
            with self.subTest(key=key),tempfile.TemporaryDirectory() as temp:
                root=Path(temp).resolve();report,complete=self.evidence(root)
                report['commands']['physical'][0][key]=bad
                self.write(root/'monitor/physical/command-0000.json',report['commands']['physical'][0])
                self.rebound(root,report,complete)
                with self.assertRaises(Exception):c.monitor_receipts(root,8101,8102)

    def test_orphan_probe_intent_and_changed_returned_pid_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();report,complete=self.evidence(root)
            extra=root/'monitor/metadata/command-0000.intent.json';self.write(extra,{})
            with self.assertRaises(Exception):c.monitor_receipts(root,8101,8102)
            extra.unlink()
            child=root/'monitor/physical/command-0000.child.json';bad=json.loads(child.read_text());bad['pid']=9999
            self.write(child,bad)
            with self.assertRaises(Exception):c.monitor_receipts(root,8101,8102)

    def test_before_monitor_failure_needs_no_orphan_monitor_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve()
            self.write(root/'complete.json',dict(schema='GLM-cold-load-owner-result-v3',owner_pid=8101,monitor={}))
            self.assertEqual(c.monitor_receipts(root,8101,8102),[])
            (root/'monitor').mkdir()
            with self.assertRaises(Exception):c.monitor_receipts(root,8101,8102)

    def test_restore_audit_only_changes_owner_schema_and_keeps_original_raw_checks(self):
        restored=audit._owner_reap_source.replace(audit._after,audit._before,1)
        self.assertEqual(ast.dump(ast.parse(restored)),ast.dump(ast.parse(inspect.getsource(audit._original.owner_reap))))
        self.assertIs(audit.binding.__code__,audit._original.binding.__code__)
        self.assertIs(audit.child_absence.__code__,audit._original.child_absence.__code__)
        self.assertIs(audit.audit.__code__,audit._original.audit.__code__)
        self.assertIs(c.stop.__code__,c._original.stop.__code__)
        self.assertIs(c.recover.__code__,c._original.recover.__code__)


if __name__=='__main__':unittest.main()
