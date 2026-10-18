"""Explicit metadata/transport mocks, not device or numerical qualification.

No tests have been executed during the live qualification cycle. The outer
central CPU runner must block subprocess/socket/device imports when running these.
Synthetic terminal fixtures fake only the expensive replay validator, never the
closed-receipt gates. Async cases use the frozen real parser/MockTransport.
OFF selection cannot access GDN; both-arm preparation still requires the closed
GDN prerequisite. These CPU tests grant no device, numerical or timing result.
"""
import asyncio
import copy
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bench.m5_ultra_256gb.gdn_prework_formal import formal as m


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
    top=dict(schema='GDN-recorded-http-result-v1',status='PASS',prepare_only=False,
        replay_numerical_checks_passed=True,input_source_pins_rechecked=True,independent_audit_required=True,
        numerical_qualified=False,http_qualified=False,performance_qualified=False,owner_resources_restore_qualified=False,
        total_requests=8 if C==1 else 16,concurrency=C,
        **{k:plan[k] for k in ('sources','inputs','capture','serving')},
        base_result=dict(path='base/complete.json',sha256=m.sha(child/'base/complete.json'),returncode=0,status='PASS'))
    m.write(child/'complete.json',top)
    audit=dict(schema='GDN-recorded-http-independent-v1',status='PASS',qualified=True,
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
    cycle=dict(schema='GDN-prework-http-cycle-independent-v1',status='PASS',qualified=True,
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
        plan,_,audit=m.bind_gdn(self.binding,1,self.base)
        self.assertEqual(plan,self.plan); self.assertEqual(audit['ledger']['requests'],8)
    def test_cycle_incomplete_resource_or_restoration_never_qualifies(self):
        pristine=(self.root/'cycle.json').read_bytes()
        for field,value in [('status','INCOMPLETE'),('independent_resource_verified',False),('restoration_verified',False)]:
            with self.subTest(field=field):
                (self.root/'cycle.json').write_bytes(pristine)
                self.update('cycle.json',lambda d:d.update({field:value}))
                with self.assertRaisesRegex(ValueError,'closed independently'): m.bind_gdn(self.binding,1,self.base)
    def test_other_C_missing_rejects_before_formal(self):
        self.update('cycle.json',lambda d:d['numerical_audits'].pop('8'))
        with self.assertRaisesRegex(ValueError,'both actual concurrency'):m.bind_gdn(self.binding,1,self.base)
    def test_forged_auditor_source_or_mislabeled_concurrency(self):
        pristine=(self.root/'audit.json').read_bytes()
        for changes in ({'audit_source_sha256':'0'*64},{'concurrency':8}):
            with self.subTest(changes=changes):
                (self.root/'audit.json').write_bytes(pristine)
                self.update('audit.json',lambda d:d.update(changes))
                with self.assertRaisesRegex(ValueError,'actual GDN numerical'):m.bind_gdn(self.binding,1,self.base)
    def test_changed_terminal_and_changed_raw_binding_rejected(self):
        path=self.root/'child/recording.json'; before=path.read_bytes(); path.write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError,'terminal bytes'):m.bind_gdn(self.binding,1,self.base)
        path.write_bytes(before)
        self.update('audit.json',lambda d:d.update(raw_inputs={'wrong':'wire'}))
        with self.assertRaisesRegex(ValueError,'actual audit binding: raw_inputs'):m.bind_gdn(self.binding,1,self.base)
    def test_prepare_only_or_invented_qualified_producer_rejected_even_rehashed(self):
        path=self.root/'child/complete.json'; pristine=path.read_bytes()
        for changes in ({'prepare_only':True},{'numerical_qualified':True}):
            with self.subTest(changes=changes):
                path.write_bytes(pristine); top=m.read(path); top.update(changes); m.write(path,top)
                self.update('audit.json',lambda d:d['terminal_sha256'].update({'complete.json':m.sha(path)}))
                with self.assertRaisesRegex(ValueError,'actual replay terminal scope'):m.bind_gdn(self.binding,1,self.base)
    def test_wrong_cycle_source_or_qualified_C8_overlap_rejected(self):
        self.update('cycle.json',lambda d:d['source_pins'].__setitem__(str(m.CYCLE/'owner.py'),'0'*64))
        with self.assertRaisesRegex(ValueError,'actual adapter source'):m.bind_gdn(self.binding,1,self.base)
        self.update('cycle.json',lambda d:d['source_pins'].__setitem__(str(m.CYCLE/'owner.py'),m.FIXED[m.CYCLE/'owner.py']))
        self.update('audit.json',lambda d:d['http'].update(max_http_outstanding=8))
        with self.assertRaisesRegex(ValueError,'drained concurrency'):m.bind_gdn(self.binding,1,self.base)
    def test_owner_or_journal_changes_reject_closed_binding(self):
        path=self.root/'journal/journal.json'; m.write(path,dict(state='STOPPED'))
        with self.assertRaisesRegex(ValueError,'immutable restored'):m.bind_gdn(self.binding,1,self.base)


class Selection(unittest.TestCase):
    def test_OFF_preserves_original_without_gdn_object_access(self):
        class Forbidden:
            def __getattribute__(self,name): raise AssertionError('OFF accessed GDN '+name)
        base=original(); expected=copy.deepcopy(base)
        command,identity=m.select_entry(base,'off',Forbidden(),Forbidden())
        self.assertEqual((command,identity),(expected['command'],expected['identity'])); self.assertEqual(base,expected)
    def test_ON_real_frozen_server_identity_observers_off_and_no_mutation(self):
        server=m.module(m.SERVER); pins=server.source_pins(); base=original(8); saved=copy.deepcopy(base)
        command,identity=m.select_entry(base,'on',server,pins)
        self.assertEqual(command[3],str(m.SERVER)); self.assertEqual(command[-4:],['--phase','formal','--gdn-prework','on'])
        self.assertFalse(any(v in command for v in ('--qualification-trace','--dispatch-events')))
        self.assertEqual(base,saved); self.assertFalse(identity['step_observer']); self.assertFalse(identity['qualification_trace_enabled'])
        gdn=identity['gdn_prework']
        self.assertTrue(gdn['installed']); self.assertEqual(gdn['installed_layers'],48)
        self.assertFalse(gdn['dispatch_observer']); self.assertFalse(gdn['actual_calls_observed'])
        self.assertFalse(gdn['route_graph_enabled']); self.assertFalse(gdn['identity_refill_enabled'])
        self.assertEqual(gdn['sources'],pins)
    def test_selection_rejects_trace_or_wrong_entry_in_original(self):
        for mutate in (lambda d:d['command'].append('--dispatch-events'),lambda d:d['command'].__setitem__(3,'/other/server.py'),
                       lambda d:d['identity'].update(step_observer=True)):
            base=original(); mutate(base)
            with self.assertRaises(ValueError):m.select_entry(base,'off',None,None)
    def test_validate_reconstructs_not_just_approved_arm_label(self):
        expected=dict(schema='GDN-prework-http-formal-spec-v1',sources={},base_plan={},concurrency=1,repeat=1,
            arm='off',baseline_qualification={},gdn_qualification={},environment={},transport={},deadlines={},**original())
        with patch.object(m,'source_pins',return_value={}), patch.object(m,'prepare',return_value=expected):
            m.validate(expected)
            bad=copy.deepcopy(expected); bad['command'][3]=str(m.SERVER)
            with self.assertRaisesRegex(ValueError,'formal spec changed'):m.validate(bad)


    def test_fixed_module_bad_source_rejected_before_any_import(self):
        with patch.object(m,'sha',return_value='0'*64), patch.object(m.importlib.util,'spec_from_file_location',side_effect=AssertionError('must not import')):
            with self.assertRaisesRegex(ValueError,'frozen support changed'):m.module(m.SERVER)

    def test_prepare_both_arms_require_closed_qualification_and_same_environment(self):
        server=m.module(m.SERVER); pins=server.source_pins()
        base=original(8); base.update(base_plan={'path':'/numeric','sha256':m.BASE_SHA},concurrency=8,repeat=1)
        environment={'MLX_METAL_FAST_SYNCH':'0'}; transport={'port':18023,'tcp_nodelay':True,'io_timeout_seconds':300.0}
        baseline={'audit':{'path':'/baseline','sha256':'1'*64}}; gdn={'audit':{'path':'/gdn','sha256':'2'*64}}
        qualified=copy.deepcopy(base['identity']); qualified.update(step_observer=True,qualification_trace_enabled=True)
        qualified.update(server.identity_fields(SimpleNamespace(phase='qualification',gdn_prework='on'),pins,installed=True,dispatch_observer=True))
        capture=dict(identity=qualified,environment=environment,transport=transport)
        native=SimpleNamespace(prepare=lambda *a,**kw:copy.deepcopy(base))
        with patch.object(m,'donor',return_value=native), patch.object(m,'module',return_value=server), patch.object(m,'source_pins',return_value=pins), patch.object(m,'bind_gdn',return_value=({},capture,{})) as bind:
            for arm,binding in [('off',baseline),('on',gdn)]:
                prepared=m.prepare(base['base_plan'],8,1,arm=arm,baseline_qualification=baseline,gdn_qualification=gdn,environment=environment,transport=transport,deadlines={})
                self.assertEqual(prepared['qualification'],binding)
                self.assertFalse(prepared['observer_enabled'])
                self.assertEqual(prepared['gdn_qualification'],gdn)
            self.assertEqual(bind.call_count,2)  # OFF does not waive the global closed-cycle prerequisite.
            capture['environment']={'MLX_METAL_FAST_SYNCH':'1'}
            with self.assertRaisesRegex(ValueError,'qualified MLX environment'):
                m.prepare(base['base_plan'],8,1,arm='off',baseline_qualification=baseline,gdn_qualification=gdn,environment=environment,transport=transport,deadlines={})


class Transport(unittest.IsolatedAsyncioTestCase):
    async def test_new_run_uses_original_wire_counts_and_direct_cleanup_for_both_entries(self):
        # Same original parser/serializer/client and mock server backend. No sockets.
        from bench.m5_ultra_256gb.pareto_realistic import test_native_formal_cpu as old
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for arm in ('off','on'):
                spec=old.prepare(root/('metadata-'+arm),C=1 if arm=='off' else 8)
                server=m.module(m.SERVER); pins=server.source_pins()
                spec['command'],spec['identity']=m.select_entry(spec,arm,server,pins)
                spec.update(schema='GDN-prework-http-formal-spec-v1',arm=arm,phase='r1-c1-'+arm)
                backend=old.Backend(spec); checkpoints=[]
                async def checkpoint(label,pid): checkpoints.append((label,pid))
                with patch.object(m,'validate',return_value=None):
                    result=await m.run(spec,root/arm,checkpoint=checkpoint,backend=backend)
                self.assertEqual(result['status'],'PASS'); self.assertTrue(result['count_eligible'])
                self.assertFalse(result['performance_qualified']); self.assertFalse(result['full_state_qualified'])
                self.assertEqual(result['point']['requests'],64); self.assertEqual(backend.phases,['warm','formal'])
                self.assertEqual(backend.child.actions,['TERM']); self.assertTrue(result['server_reaped'])
                self.assertTrue(result['cleanup_complete']); self.assertTrue(result['client_task_resolved'])
                self.assertFalse((root/arm/'trace.jsonl').exists()); self.assertFalse((root/arm/'dispatch-events.jsonl').exists())
                self.assertIn(('spawned',backend.child.pid),checkpoints); self.assertIn(('after-reap',None),checkpoints)
                self.assertEqual(m.read(root/arm/'point.json'),dict(**result['point'],count_eligible=True,performance_qualified=False))


    async def test_raw_wire_counts_full_prompt_EOS_and_cleanup_are_not_trusted_summaries(self):
        from bench.m5_ultra_256gb.pareto_realistic import test_native_formal_cpu as old
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve()
            for C in (1,8):
                spec=old.prepare(root/('c'+str(C)),C=C)
                group=await old.group(spec,'formal'); saved=copy.deepcopy(group)
                checked=m.check_group(group,spec,'formal',old.decode)
                self.assertEqual(group,saved); self.assertEqual(checked['raw_requests_checked'],64)
                self.assertEqual(checked['point']['committed_output_tokens'],22+21*512)
                self.assertEqual(group['summary']['max_http_outstanding'],C)
                contract=m.count_contract(spec,'formal')
                self.assertTrue(contract['normal_eos']); self.assertEqual(contract['raw_ids_count_rule'],'native_committed_ids')
                self.assertNotIn('expected_raw_ids_by_request',contract)
                changes={
                    'count':lambda row:old.rewrite_terminal(row,lambda t:t['usage'].__setitem__('completion_tokens',512)),
                    'prompt':lambda row:old.rewrite_terminal(row,lambda t:t['benchmark_prompt_ids'].__setitem__(0,777)),
                    'EOS':lambda row:old.rewrite_terminal(row,lambda t:t.__setitem__('benchmark_terminal_eos_id',None)),
                    'body':lambda row:row['body'].__setitem__('temperature',1),
                }
                for label,mutate in changes.items():
                    bad=copy.deepcopy(group); mutate(bad['requests'][0])
                    with self.subTest(C=C,label=label),self.assertRaises(ValueError):m.check_group(bad,spec,'formal',old.decode)
                bad=copy.deepcopy(group); bad['cleanup']['complete']=False
                with self.assertRaisesRegex(ValueError,'cleanup incomplete'):m.check_group(bad,spec,'formal',old.decode)

    async def test_invalid_warm_retained_no_formal_and_no_point(self):
        from bench.m5_ultra_256gb.pareto_realistic import test_native_formal_cpu as old
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve(); spec=old.prepare(root/'metadata',C=1)
            spec.update(schema='GDN-prework-http-formal-spec-v1',arm='off',phase='r1-c1-off')
            backend=old.Backend(spec); actual=backend.group
            async def wrong(spec,phase,out):
                group=await actual(spec,phase,out); group['requests'][0]['answer']='corrupted'; return group
            backend.group=wrong
            with patch.object(m,'validate',return_value=None):
                result=await m.run(spec,root/'trial',checkpoint=lambda *a:None,backend=backend)
            self.assertEqual(result['status'],'FAIL'); self.assertEqual(backend.phases,['warm'])
            self.assertTrue(result['server_reaped']); self.assertFalse(result['count_eligible'])
            self.assertTrue((root/'trial/warm-group.json').is_file()); self.assertFalse((root/'trial/point.json').exists())

    async def test_surviving_child_rejects_count_eligible_trial(self):
        from bench.m5_ultra_256gb.pareto_realistic import test_native_formal_cpu as old
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve(); spec=old.prepare(root/'metadata',C=1)
            spec.update(schema='GDN-prework-http-formal-spec-v1',arm='off',phase='r1-c1-off')
            backend=old.Backend(spec,bad_reap=True); checkpoints=[]
            with patch.object(m,'validate',return_value=None):
                result=await m.run(spec,root/'trial',checkpoint=lambda *args:checkpoints.append(args),backend=backend)
            self.assertEqual(result['status'],'FAIL'); self.assertTrue(result['count_eligible'])
            self.assertFalse(result['server_reaped']); self.assertFalse(result['performance_qualified'])
            self.assertFalse((root/'trial/release.json').exists()); self.assertEqual(checkpoints[-1],('cleanup-incomplete',backend.child.pid))

    async def test_repeated_cancel_preserves_partial_and_term_kill_wait(self):
        from bench.m5_ultra_256gb.pareto_realistic import test_native_formal_cpu as old
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve(); spec=old.prepare(root/'metadata',C=1)
            spec.update(schema='GDN-prework-http-formal-spec-v1',arm='off',phase='r1-c1-off')
            backend=old.Backend(spec,ignore_term=True); closing,allow=asyncio.Event(),asyncio.Event()
            async def slow(spec,phase,out):
                backend.group_started.set()
                try:await asyncio.Future()
                except asyncio.CancelledError:
                    closing.set(); await allow.wait()
                    raise m.load('native_cleanup').GroupCancelled(dict(group_error={'type':'CancelledError'},cleanup={'complete':False}))
            backend.group=slow
            with patch.object(m,'validate',return_value=None):
                task=asyncio.create_task(m.run(spec,root/'trial',checkpoint=lambda *a:None,backend=backend))
                await backend.group_started.wait(); task.cancel(); await closing.wait()
                task.cancel(); await asyncio.sleep(0); task.cancel(); await asyncio.sleep(0); allow.set()
                result=await task
            self.assertEqual(result['status'],'FAIL'); self.assertTrue(result['server_reaped'])
            self.assertEqual(backend.child.actions,['TERM','KILL']); self.assertTrue(backend.waited)
            self.assertTrue((root/'trial/warm-group.json').is_file()); self.assertFalse(result['performance_qualified'])


if __name__=='__main__':unittest.main()
