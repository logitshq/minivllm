"""Small fake raw receipts only. Original OS/token parsers remain SHA-pinned."""
import ast
import copy
import importlib.util
import inspect
import json
from pathlib import Path
import tempfile
import unittest

path=Path(__file__).with_name('audit_closed.py')
spec=importlib.util.spec_from_file_location('glm_v3_cpu_closed_audit',path)
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


class Tests(unittest.TestCase):
    def evidence(self,root):
        identity=dict(pid=8102,ppid=8101,start='Sun Oct 18 12:00:00 2026',comm='/fake/glm-python')
        child=dict(pid=8102,identity=identity,spawned_at=999.,spawned_monotonic=99.)
        release=dict(time=1010.)
        plan=dict(monitor_policy=copy.deepcopy(a.MONITOR_POLICY),physical_policy=dict(work_seconds=1800))
        complete=dict(schema='GLM-cold-load-owner-result-v3',owner_pid=8101,monitor_policy=copy.deepcopy(a.MONITOR_POLICY),
            legacy_checkpoint_per_sample_contract=False,live_physical_samples=2,metadata_live_samples=1,checkpoints=7)
        report=dict(schema='GLM-dual-cadence-monitor-v1',status='COMPLETE',pid=8102,owner_pid=8101,child_returncode=0,
            policy=copy.deepcopy(a.READ_POLICY),errors=[],unjoined_workers=[],unresolved_read_commands=[],
            read_commands_settled=True,model_child_release_proven=False,physical_resource_safety_qualified=False,
            performance_qualified=False,started_monotonic=100.,ended_monotonic=102.,work_deadline=1800.,lanes={},commands={})
        paths=[]
        for lane,windows in [('physical',[(100.01,100.21),(100.71,100.91)]),('metadata',[(100.02,101.62)])]:
            rows=[];commands=[];previous=100.
            for i,(start,end) in enumerate(windows):
                batch=[]
                for n in range(7):
                    ca=start+.01+n*(end-start-.04)/7;cb=ca+(end-start-.04)/8
                    command=dict(command=['/fake/read',str(n)],returncode=0,stdout='raw-'+str(n),stderr='',
                        started_monotonic=ca,ended_monotonic=cb,deadline_monotonic=start+5.,elapsed_seconds=cb-ca,
                        started_at=ca+900.,ended_at=cb+900.,status='COMPLETE',actions=[],waited_reaped=True,
                        unknown_launch=False,pid=9000+len(commands)+n)
                    batch.append(command)
                if lane=='physical':
                    for n in (0,6):
                        batch[n].update(command=['/bin/ps','-p','8102','-o','pid=,ppid=,lstart=,comm='],
                            stdout='8102 8101 Sun Oct 18 12:00:00 2026 /fake/glm-python\n')
                    raw=dict(commands=[a.raw_command(r) for r in batch[1:-1]],
                        started_monotonic=batch[0]['ended_monotonic'],ended_monotonic=batch[-2]['ended_monotonic']+.001)
                    raw_path=root/f'physical-{i:04d}/sample.json';paths.append(raw_path);save(raw_path,raw)
                    result=dict(sample=dict(path=str(raw_path),sha256=a.sha(raw_path)),observed_within_budget=True)
                else:
                    batch[-1].pop('stdout');batch[-1].update(stdout_sha256='a'*64,stdout_bytes=123)
                    raw=dict(commands=[a.raw_command(r) for r in batch],lane_sample_index=i,label='running-metadata',
                        monotonic=start+.001,ended_monotonic=end-.001)
                    raw_path=root/f'checkpoint-{i+5:04d}.json';save(raw_path,raw)
                    result=dict(checkpoint=dict(path=str(raw_path),sha256=a.sha(raw_path)))
                commands.extend(batch)
                row=dict(index=i,status='COMPLETE',started_monotonic=start,ended_monotonic=end,deadline_monotonic=start+5.,
                    started_at=start+900.,ended_at=end+900.,prior_end_monotonic=previous,gap_seconds=start-previous,
                    elapsed_seconds=end-start,command_range=dict(start=i*7,stop=(i+1)*7),result=result)
                rows.append(row);previous=end
            report['lanes'][lane]=dict(active=None,samples=rows);report['commands'][lane]=commands
        complete['monitor']=dict(started=True,read_commands_settled=True,status='COMPLETE',receipt={})
        self.rebind(root,report,complete)
        return plan,complete,child,release,paths,report

    def rebind(self,root,report,complete):
        for lane,state in report['lanes'].items():
            for i,row in enumerate(state['samples']):save(root/f'monitor/{lane}/sample-{i:04d}.json',row)
        path=root/'monitor/monitor.json';save(path,report)
        complete['monitor']['receipt']=dict(path=str(path),sha256=a.sha(path))

    def test_independent_slow_metadata_overlaps_multiple_physical_samples(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();plan,complete,child,release,paths,report=self.evidence(root)
            result=a.monitor_resources(root,plan,complete,child,release,paths)
            self.assertEqual(result['status'],'PASS')
            self.assertEqual(result['lanes']['physical']['samples'],2)
            self.assertEqual(result['lanes']['metadata']['samples'],1)
            self.assertGreater(result['lanes']['metadata']['maximum_callback_seconds'],1.5)
            self.assertFalse(result['checkpoint_per_physical_sample'])

    def test_gap_and_callback_deadline_reject_even_with_rebound_files(self):
        for lane,key,value in [('physical','gap_seconds',1.1),('metadata','deadline_monotonic',101.)]:
            with self.subTest(lane=lane,key=key),tempfile.TemporaryDirectory() as temp:
                root=Path(temp).resolve();plan,complete,child,release,paths,report=self.evidence(root)
                report['lanes'][lane]['samples'][-1][key]=value;self.rebind(root,report,complete)
                with self.assertRaisesRegex(ValueError,'cadence/deadline'):a.monitor_resources(root,plan,complete,child,release,paths)

    def test_missing_duplicate_command_range_and_orphan_sample_reject(self):
        for kind in ('range','missing','orphan'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as temp:
                root=Path(temp).resolve();plan,complete,child,release,paths,report=self.evidence(root)
                if kind=='range':report['lanes']['physical']['samples'][1]['command_range']['start']=0
                if kind=='missing':report['commands']['metadata'].pop()
                self.rebind(root,report,complete)
                if kind=='orphan':save(root/'monitor/physical/sample-9999.json',{})
                with self.assertRaises(ValueError):a.monitor_resources(root,plan,complete,child,release,paths)

    def test_command_elapsed_error_or_unreaped_cannot_be_masked(self):
        for key,value in [('elapsed_seconds',6.),('status','INCOMPLETE'),('waited_reaped',False),('unknown_launch',True)]:
            with self.subTest(key=key),tempfile.TemporaryDirectory() as temp:
                root=Path(temp).resolve();plan,complete,child,release,paths,report=self.evidence(root)
                report['commands']['physical'][3][key]=value;self.rebind(root,report,complete)
                with self.assertRaisesRegex(ValueError,'bounded command'):a.monitor_resources(root,plan,complete,child,release,paths)

    def test_physical_identity_and_original_stdout_binding(self):
        for index in (0,2,6):
            with self.subTest(index=index),tempfile.TemporaryDirectory() as temp:
                root=Path(temp).resolve();plan,complete,child,release,paths,report=self.evidence(root)
                report['commands']['physical'][index]['stdout']='8109 8101 Sun Oct 18 12:00:00 2026 /different\n'
                self.rebind(root,report,complete)
                with self.assertRaises(ValueError):a.monitor_resources(root,plan,complete,child,release,paths)

    def test_metadata_redacted_original_hash_and_sample_sha_are_checked(self):
        for kind in ('redacted','raw'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as temp:
                root=Path(temp).resolve();plan,complete,child,release,paths,report=self.evidence(root)
                if kind=='redacted':report['commands']['metadata'][-1]['stdout_sha256']='b'*64
                else:save(root/'checkpoint-0005.json',{})
                self.rebind(root,report,complete)
                with self.assertRaises(ValueError):a.monitor_resources(root,plan,complete,child,release,paths)

    def test_incomplete_monitor_never_grants_resource_pass(self):
        for key,value in [('status','INCOMPLETE'),('errors',[{'error':'probe failed'}]),('unjoined_workers',['physical'])]:
            with self.subTest(key=key),tempfile.TemporaryDirectory() as temp:
                root=Path(temp).resolve();plan,complete,child,release,paths,report=self.evidence(root)
                report[key]=value;self.rebind(root,report,complete)
                with self.assertRaises(ValueError):a.monitor_resources(root,plan,complete,child,release,paths)

    def test_original_raw_token_inventory_restoration_algorithms_unchanged(self):
        for name,source,deltas in [('audit',a._audit_source,a.AUDIT_DELTAS),('checkpoints',a._checkpoint_source,a.CHECKPOINT_DELTAS)]:
            for before,after in reversed(deltas):source=source.replace(after,before,1)
            self.assertEqual(ast.dump(ast.parse(source)),ast.dump(ast.parse(inspect.getsource(getattr(a._original,name)))))
        self.assertIs(a.raw_sample.__code__,a._original.raw_sample.__code__)
        self.assertIs(a.request_counts.__code__,a._original.request_counts.__code__)


if __name__=='__main__':unittest.main()
