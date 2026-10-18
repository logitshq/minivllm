"""Explicit metadata/transport mocks, not device or numerical qualification.

No tests have been executed during the live qualification cycle. The outer
central CPU runner must block subprocess/socket/device imports when running these.
Only qualification validation is faked; selection and closed-receipt gates are
real. The async smoke reuses the frozen client MockTransport and serializer.
"""
import asyncio
import copy
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bench.m5_ultra_256gb.route_http_formal import formal as m


def ref(path): return dict(path=str(path.resolve()), sha256=m.sha(path))


def fixture(root, C=1):
    """Small synthetic terminal receipts; no steps ledger or arrays are fabricated."""
    child=root/'child'; child.mkdir(); (child/'base').mkdir()
    owner=root/'owner'; owner.mkdir(); journal=root/'journal'; journal.mkdir()
    base_plan=dict(path=str(root/'numeric.json'), sha256=m.BASE_SHA)
    plan=dict(qualification=base_plan, concurrency=C, sources={'synthetic-source':'fixture'},
              inputs={'synthetic-input':'fixture'}, capture={'synthetic-capture':'fixture'}, serving={'synthetic-serving':'fixture'})
    m.write(child/'plan.json',plan); (child/'base/plan.json').write_text('synthetic numeric bytes')
    m.write(child/'base/complete.json',dict(status='PASS'))
    m.write(child/'recording.json',dict(synthetic=True))
    top=dict(schema='E024-recorded-route-http-result-v1',status='PASS',prepare_only=False,
        replay_numerical_checks_passed=True,input_source_pins_rechecked=True,independent_audit_required=True,
        numerical_qualified=False,http_qualified=False,performance_qualified=False,owner_resources_restore_qualified=False,
        total_requests=8 if C==1 else 16,concurrency=C,
        **{k:plan[k] for k in ('sources','inputs','capture','serving')},
        base_result=dict(path='base/complete.json',sha256=m.sha(child/'base/complete.json'),returncode=0,status='PASS'))
    m.write(child/'complete.json',top)
    audit=dict(schema='E024-recorded-route-http-independent-audit-v2',status='PASS',qualified=True,
        http_history_qualified=True,concurrency=C,audit_source_sha256=m.FIXED[m.AUDITOR],performance_qualified=False,
        owner_resources_restore_qualified=False,child_directory=str(child),base_plan_sha256=m.BASE_SHA,
        plan_sha256=m.sha(child/'plan.json'),terminal_sha256={name:m.sha(child/name) for name in m.TERMINALS},
        sources=plan['sources'],raw_inputs=plan['inputs'],capture=plan['capture'],serving_prerequisite=plan['serving'],
        ledger=dict(requests=8 if C==1 else 16),http=dict(max_http_outstanding=C,fully_drained=True))
    m.write(root/'audit.json',audit)
    other=copy.deepcopy(audit); otherC=8 if C==1 else 1
    other.update(concurrency=otherC,ledger=dict(requests=8 if otherC==1 else 16))
    m.write(root/'other-audit.json',other)
    m.write(owner/'owner-plan.json',dict(cold_restart=dict(journal_path=str(journal))))
    m.write(owner/'complete.json',dict(status='PASS'))
    m.write(journal/'journal.json',dict(state='RESTORED'))
    cycle=dict(schema='E024-route-http-cycle-independent-audit-v1',status='PASS',qualified=True,
        http_history_qualified=True,independent_resource_verified=True,restoration_verified=True,
        performance_qualified=False,audit_source_sha256=m.FIXED[m.CYCLE_AUDITOR],
        numerical_audits={str(C):ref(root/'audit.json'),str(otherC):ref(root/'other-audit.json')},
        owner_directory=str(owner),owner_plan_sha256=m.sha(owner/'owner-plan.json'),owner_complete_sha256=m.sha(owner/'complete.json'),
        journal_sha256=m.sha(journal/'journal.json'),source_pins={str(p):h for p,h in m.FIXED.items()})
    cycle['restoration']=dict(status='PASS',qualified=True,journal_sha256=cycle['journal_sha256'],owner_plan_sha256=cycle['owner_plan_sha256'])
    m.write(root/'cycle.json',cycle)
    binding=dict(audit=ref(root/'audit.json'), replay_plan=ref(child/'plan.json'),child_directory=str(child),cycle_audit=ref(root/'cycle.json'))
    return base_plan,binding,plan


def original(C=1):
    return dict(command=['/cpu/python','-I','-B',str(m.NATIVE/'native_server.py'),'--max-seqs',str(C)],
        identity=dict(step_observer=False,qualification_trace_enabled=False,eos_ids=[99],configuration=dict(max_seqs=C)))


class ClosedBinding(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve(); self.base,self.binding,self.plan=fixture(self.root)
        self.producer=SimpleNamespace(validate=lambda *args:(self.plan,{}, {}, {}))
        self.patcher=patch.object(m,'module',return_value=self.producer); self.patcher.start(); self.addCleanup(self.patcher.stop)
    def update(self,name,change):
        path=self.root/name; value=m.read(path); change(value); m.write(path,value)
        if name=='audit.json':
            self.binding['audit']=ref(path)
            cycle=m.read(self.root/'cycle.json'); cycle['numerical_audits']['1']=ref(path)
            m.write(self.root/'cycle.json',cycle)
        self.binding['cycle_audit']=ref(self.root/'cycle.json')
    def test_closed_two_C_cycle_and_exact_child_pass(self):
        plan,_,audit=m.bind_route(self.binding,1,self.base)
        self.assertEqual(plan,self.plan); self.assertEqual(audit['ledger']['requests'],8)
    def test_cycle_incomplete_resource_or_restoration_never_qualifies(self):
        pristine=(self.root/'cycle.json').read_bytes()
        for field,value in [('status','INCOMPLETE'),('independent_resource_verified',False),('restoration_verified',False)]:
            with self.subTest(field=field):
                (self.root/'cycle.json').write_bytes(pristine)
                self.update('cycle.json',lambda d:d.update({field:value}))
                with self.assertRaisesRegex(ValueError,'closed independently'): m.bind_route(self.binding,1,self.base)
    def test_other_C_missing_rejects_before_formal(self):
        self.update('cycle.json',lambda d:d['numerical_audits'].pop('8'))
        with self.assertRaisesRegex(ValueError,'both actual concurrency'):m.bind_route(self.binding,1,self.base)
    def test_forged_auditor_source_or_mislabeled_concurrency(self):
        pristine=(self.root/'audit.json').read_bytes()
        for changes in ({'audit_source_sha256':'0'*64},{'concurrency':8}):
            with self.subTest(changes=changes):
                (self.root/'audit.json').write_bytes(pristine)
                self.update('audit.json',lambda d:d.update(changes))
                with self.assertRaisesRegex(ValueError,'actual route numerical'):m.bind_route(self.binding,1,self.base)
    def test_changed_terminal_and_changed_raw_binding_rejected(self):
        path=self.root/'child/recording.json'; before=path.read_bytes(); path.write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError,'terminal bytes'):m.bind_route(self.binding,1,self.base)
        path.write_bytes(before)
        self.update('audit.json',lambda d:d.update(raw_inputs={'wrong':'wire'}))
        with self.assertRaisesRegex(ValueError,'actual audit binding: raw_inputs'):m.bind_route(self.binding,1,self.base)
    def test_prepare_only_or_invented_qualified_producer_rejected_even_rehashed(self):
        path=self.root/'child/complete.json'; pristine=path.read_bytes()
        for changes in ({'prepare_only':True},{'numerical_qualified':True}):
            with self.subTest(changes=changes):
                path.write_bytes(pristine); top=m.read(path); top.update(changes); m.write(path,top)
                self.update('audit.json',lambda d:d['terminal_sha256'].update({'complete.json':m.sha(path)}))
                with self.assertRaisesRegex(ValueError,'actual replay terminal scope'):m.bind_route(self.binding,1,self.base)
    def test_owner_or_journal_changes_reject_closed_binding(self):
        path=self.root/'journal/journal.json'; m.write(path,dict(state='STOPPED'))
        with self.assertRaisesRegex(ValueError,'immutable restored'):m.bind_route(self.binding,1,self.base)


class Selection(unittest.TestCase):
    def test_OFF_preserves_original_without_route_object_access(self):
        class Forbidden:
            def __getattribute__(self,name): raise AssertionError('OFF accessed route '+name)
        base=original(); expected=copy.deepcopy(base)
        command,identity=m.select_entry(base,'off',Forbidden(),Forbidden())
        self.assertEqual((command,identity),(expected['command'],expected['identity'])); self.assertEqual(base,expected)
    def test_ON_real_frozen_server_identity_observers_off_and_no_mutation(self):
        server=m.module(m.SERVER); pins=server.source_pins(); base=original(8); saved=copy.deepcopy(base)
        command,identity=m.select_entry(base,'on',server,pins)
        self.assertEqual(command[3],str(m.SERVER)); self.assertEqual(command[-4:],['--phase','formal','--route-decode-graph','on'])
        self.assertFalse(any(v in command for v in ('--qualification-trace','--route-events')))
        self.assertEqual(base,saved); self.assertFalse(identity['step_observer']); self.assertFalse(identity['qualification_trace_enabled'])
        graph=identity['route_decode_graph']
        self.assertTrue(graph['installed']); self.assertFalse(graph['route_observer']); self.assertFalse(graph['actual_compiled_invocations_observed'])
        self.assertFalse(graph['identity_refill_enabled']); self.assertEqual(graph['sources'],pins)
    def test_selection_rejects_trace_or_wrong_entry_in_original(self):
        for mutate in (lambda d:d['command'].append('--route-events'),lambda d:d['command'].__setitem__(3,'/other/server.py'),
                       lambda d:d['identity'].update(step_observer=True)):
            base=original(); mutate(base)
            with self.assertRaises(ValueError):m.select_entry(base,'off',None,None)
    def test_validate_reconstructs_not_just_approved_arm_label(self):
        expected=dict(schema='E024-route-http-formal-spec-v1',sources={},base_plan={},concurrency=1,repeat=1,
            arm='off',baseline_qualification={},route_qualification={},environment={},transport={},deadlines={},**original())
        with patch.object(m,'source_pins',return_value={}), patch.object(m,'prepare',return_value=expected):
            m.validate(expected)
            bad=copy.deepcopy(expected); bad['command'][3]=str(m.SERVER)
            with self.assertRaisesRegex(ValueError,'formal spec changed'):m.validate(bad)


class Transport(unittest.IsolatedAsyncioTestCase):
    async def test_new_run_uses_original_wire_counts_and_direct_cleanup_for_both_entries(self):
        # Same original parser/serializer/client and mock server backend. No sockets.
        from bench.m5_ultra_256gb.pareto_realistic import test_native_formal_cpu as old
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for arm in ('off','on'):
                spec=old.prepare(root/('metadata-'+arm),C=1)
                server=m.module(m.SERVER); pins=server.source_pins()
                spec['command'],spec['identity']=m.select_entry(spec,arm,server,pins)
                spec.update(schema='E024-route-http-formal-spec-v1',arm=arm,phase='r1-c1-'+arm)
                backend=old.Backend(spec); checkpoints=[]
                async def checkpoint(label,pid): checkpoints.append((label,pid))
                with patch.object(m,'validate',return_value=None):
                    result=await m.run(spec,root/arm,checkpoint=checkpoint,backend=backend)
                self.assertEqual(result['status'],'PASS'); self.assertTrue(result['count_eligible'])
                self.assertFalse(result['performance_qualified']); self.assertFalse(result['full_state_qualified'])
                self.assertEqual(result['point']['requests'],64); self.assertEqual(backend.phases,['warm','formal'])
                self.assertEqual(backend.child.actions,['TERM']); self.assertTrue(result['server_reaped'])
                self.assertIn(('spawned',backend.child.pid),checkpoints); self.assertIn(('after-reap',None),checkpoints)
                self.assertEqual(m.read(root/arm/'point.json'),dict(**result['point'],count_eligible=True,performance_qualified=False))


if __name__=='__main__':unittest.main()
