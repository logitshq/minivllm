"""Small terminal/transport mocks; no device or future HTTP qualification.

Only the expensive producer validator is mocked in binding tests. The actual
native queue, SSE parser, count checker and bounded cleanup run in MockTransport.
Both formal arms use the same new entry and keep route ON.
"""
import asyncio
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bench.m5_ultra_256gb.gdn_graph_combined_formal import formal as m


def ref(path): return dict(path=str(path.resolve()),sha256=m.sha(path))


def fixture(root,C=1):
    child=root/'child';child.mkdir();(child/'base').mkdir()
    owner=root/'owner';owner.mkdir();journal=root/'journal';journal.mkdir()
    base=dict(path=str(root/'numeric.json'),sha256=m.BASE_SHA)
    plan=dict(qualification=base,concurrency=C,sources={'fixture':'source'},inputs={'fixture':'wire'},
              capture={'fixture':'capture'},serving={'fixture':'serving'})
    m.write(child/'plan.json',plan);m.write(child/'base/plan.json',{'synthetic':True})
    m.write(child/'base/complete.json',{'status':'PASS'});m.write(child/'recording.json',{'synthetic':True})
    top=dict(schema='GDN-graph-combined-recorded-http-result-v1',status='PASS',prepare_only=False,
        replay_numerical_checks_passed=True,input_source_pins_rechecked=True,independent_audit_required=True,
        numerical_qualified=False,http_qualified=False,performance_qualified=False,owner_resources_restore_qualified=False,
        total_requests=8 if C==1 else 16,concurrency=C,**{k:plan[k] for k in ('sources','inputs','capture','serving')},
        base_result=dict(path='base/complete.json',sha256=m.sha(child/'base/complete.json'),returncode=0,status='PASS'))
    m.write(child/'complete.json',top)
    audit=dict(schema='GDN-graph-combined-recorded-http-independent-v1',status='PASS',qualified=True,
        http_history_qualified=True,concurrency=C,audit_source_sha256=m.FIXED[m.AUDITOR],performance_qualified=False,
        owner_resources_restore_qualified=False,child_directory=str(child),base_plan_sha256=m.BASE_SHA,
        plan_sha256=m.sha(child/'plan.json'),terminal_sha256={n:m.sha(child/n) for n in m.TERMINALS},
        sources=plan['sources'],raw_inputs=plan['inputs'],capture=plan['capture'],serving_prerequisite=plan['serving'],
        ledger=dict(requests=8 if C==1 else 16),http=dict(max_http_outstanding=C,fully_drained=True))
    m.write(root/'audit.json',audit);other=copy.deepcopy(audit);otherC=8 if C==1 else 1
    other.update(concurrency=otherC,ledger=dict(requests=8 if otherC==1 else 16));m.write(root/'other.json',other)
    m.write(owner/'owner-plan.json',dict(cold_restart=dict(journal_path=str(journal))))
    m.write(owner/'complete.json',dict(status='PASS'));m.write(journal/'journal.json',dict(state='RESTORED'))
    cycle=dict(schema='GDN-graph-combined-http-cycle-independent-v1',status='PASS',qualified=True,
        http_history_qualified=True,independent_resource_verified=True,restoration_verified=True,
        performance_qualified=False,audit_source_sha256=m.FIXED[m.CYCLE_AUDITOR],
        numerical_audits={str(C):ref(root/'audit.json'),str(otherC):ref(root/'other.json')},
        owner_directory=str(owner),owner_plan_sha256=m.sha(owner/'owner-plan.json'),owner_complete_sha256=m.sha(owner/'complete.json'),
        journal_sha256=m.sha(journal/'journal.json'),source_pins={str(p):h for p,h in m.FIXED.items()})
    cycle['restoration']=dict(status='PASS',qualified=True,journal_sha256=cycle['journal_sha256'],owner_plan_sha256=cycle['owner_plan_sha256'])
    m.write(root/'cycle.json',cycle)
    return base,dict(audit=ref(root/'audit.json'),replay_plan=ref(child/'plan.json'),child_directory=str(child),cycle_audit=ref(root/'cycle.json')),plan


def original(C=1):
    return dict(command=['/cpu/python','-I','-B',str(m.NATIVE/'native_server.py'),'--max-seqs',str(C)],
        identity=dict(step_observer=False,qualification_trace_enabled=False,eos_ids=[99],configuration=dict(max_seqs=C)))


class Binding(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve()
        self.base,self.binding,self.plan=fixture(self.root)
        self.mock=patch.object(m,'module',return_value=SimpleNamespace(validate=lambda *a:(self.plan,{}, {}, {})))
        self.mock.start();self.addCleanup(self.mock.stop)
    def update(self,name,mutate):
        path=self.root/name;data=m.read(path);mutate(data);m.write(path,data)
        if name=='audit.json':
            self.binding['audit']=ref(path);cycle=m.read(self.root/'cycle.json')
            cycle['numerical_audits']['1']=ref(path);m.write(self.root/'cycle.json',cycle)
        self.binding['cycle_audit']=ref(self.root/'cycle.json')
    def test_closed_two_C_cycle_and_actual_child(self):
        plan,_,audit=m.bind_combined(self.binding,1,self.base)
        self.assertEqual(plan,self.plan);self.assertEqual(audit['ledger']['requests'],8)
    def test_failed_incomplete_cycle_missing_C_or_source_rejected(self):
        before=(self.root/'cycle.json').read_bytes()
        for mutate in (lambda d:d.update(status='FAIL'),lambda d:d.update(restoration_verified=False),
                       lambda d:d.update(independent_resource_verified=False),lambda d:d['numerical_audits'].pop('8'),
                       lambda d:d['source_pins'].__setitem__(str(m.CYCLE/'owner.py'),'wrong')):
            (self.root/'cycle.json').write_bytes(before);self.update('cycle.json',mutate)
            with self.assertRaises(ValueError):m.bind_combined(self.binding,1,self.base)
    def test_terminal_raw_and_unqualified_producer_never_hidden_by_rehash(self):
        path=self.root/'child/recording.json';saved=path.read_bytes();path.write_text('{}')
        with self.assertRaisesRegex(ValueError,'terminal bytes'):m.bind_combined(self.binding,1,self.base)
        path.write_bytes(saved)
        self.update('audit.json',lambda d:d.update(raw_inputs={'wrong':'wire'}))
        with self.assertRaisesRegex(ValueError,'raw_inputs'):m.bind_combined(self.binding,1,self.base)
        self.update('audit.json',lambda d:d.update(raw_inputs=self.plan['inputs']))
        path=self.root/'child/complete.json';top=m.read(path);top['numerical_qualified']=True;m.write(path,top)
        self.update('audit.json',lambda d:d['terminal_sha256'].update({'complete.json':m.sha(path)}))
        with self.assertRaisesRegex(ValueError,'terminal scope'):m.bind_combined(self.binding,1,self.base)


class Selection(unittest.TestCase):
    def test_same_real_entry_route_always_ON_only_gdn_switch_changes(self):
        server=m.module(m.SERVER);pins=server.source_pins();base=original(8);saved=copy.deepcopy(base)
        selected={arm:m.select_entry(base,arm,server,pins) for arm in ('off','on')}
        self.assertEqual(base,saved)
        self.assertEqual(selected['off'][0][:-1],selected['on'][0][:-1])
        for arm,(command,identity) in selected.items():
            self.assertEqual(command[3],str(m.SERVER));self.assertEqual(command[-6:],['--phase','formal','--route-decode-graph','on','--gdn-prework',arm])
            facts=identity['combined_candidates']
            self.assertTrue(facts['route_decode_graph_enabled']);self.assertTrue(facts['installed'])
            self.assertEqual(facts['gdn_layer_installs'],48 if arm=='on' else 0)
            self.assertFalse(facts['route_observer']);self.assertFalse(facts['actual_compiled_invocations_observed'])
            self.assertFalse(facts['identity_refill_enabled']);self.assertFalse(identity['step_observer'])
    def test_existing_observer_or_wrong_entry_rejected(self):
        for mutate in (lambda d:d['command'].append('--route-events'),lambda d:d['command'].append('--dispatch-events'),
                       lambda d:d['command'].__setitem__(3,'/wrong'),lambda d:d['identity'].update(step_observer=True)):
            base=original();mutate(base)
            with self.assertRaises(ValueError):m.select_entry(base,'off',None,None)
    def test_prepare_both_arms_require_two_same_config_histories(self):
        server=m.module(m.SERVER);pins=server.source_pins();old=m.route.module(m.route.SERVER)
        base=original(8);base.update(base_plan={'path':'/numeric','sha256':m.BASE_SHA},concurrency=8,repeat=1)
        environment={'cpu':'synthetic'};transport=dict(port=18023,tcp_nodelay=True,io_timeout_seconds=300.0)
        native_identity=copy.deepcopy(base['identity']);native_identity.update(step_observer=True,qualification_trace_enabled=True)
        identity=copy.deepcopy(native_identity);identity.update(server.identity_fields(SimpleNamespace(phase='qualification',gdn_prework='on',route_decode_graph='on'),pins,installed=True,route_observer=True))
        old_identity=copy.deepcopy(native_identity);old_identity.update(old.identity_fields(SimpleNamespace(phase='qualification',route_decode_graph='on'),old.source_pins(),installed=True,route_observer=True))
        capture=dict(identity=identity,environment=environment,transport=transport)
        old_capture=dict(identity=old_identity,environment=environment,transport=transport)
        args=dict(baseline_qualification={'audit':'baseline'},route_qualification={'audit':'route'},combined_qualification={'audit':'combined'},environment=environment,transport=transport,deadlines={})
        with patch.object(m,'donor',return_value=SimpleNamespace(prepare=lambda *a,**kw:copy.deepcopy(base))),patch.object(m,'source_pins',return_value=pins),patch.object(m,'bind_combined',return_value=({},capture,{})) as bind,patch.object(m.route,'bind_route',return_value=({},old_capture,{})):
            for arm in ('off','on'):
                spec=m.prepare(base['base_plan'],8,1,arm=arm,**args)
                self.assertEqual(spec['qualification'],args['route_qualification' if arm=='off' else 'combined_qualification'])
            self.assertEqual(bind.call_count,2)
            capture['environment']={'wrong':'MLX'}
            with self.assertRaisesRegex(ValueError,'qualified MLX environment'):m.prepare(base['base_plan'],8,1,arm='off',**args)
    def test_validate_rebuilds_actual_command_and_unsealed_sources_refused(self):
        expected=dict(schema='GDN-graph-combined-http-formal-spec-v1',sources={},base_plan={},concurrency=1,repeat=1,arm='off',
            baseline_qualification={},route_qualification={},combined_qualification={},environment={},transport={},deadlines={},**original())
        with patch.object(m,'source_pins',return_value={}),patch.object(m,'prepare',return_value=expected):
            m.validate(expected);bad=copy.deepcopy(expected);bad['command'].append('--route-events')
            with self.assertRaisesRegex(ValueError,'changed formal specification'):m.validate(bad)
        with patch.dict(m.FIXED,{m.CYCLE_AUDITOR:'UNSEALED'}):
            with self.assertRaisesRegex(ValueError,'not yet sealed'):m.source_pins()
    def test_default_description_has_exact12_and_no_action(self):
        stream=io.StringIO()
        with patch.object(m,'source_pins',side_effect=AssertionError('no prepare')),contextlib.redirect_stdout(stream):
            self.assertEqual(m.main([]),0)
        data=json.loads(stream.getvalue());self.assertEqual(len(data['trials']),12)
        self.assertEqual(len(set(data['trials'])),12);self.assertFalse(data['device_run']);self.assertTrue(data['route_always_on'])


class Transport(unittest.IsolatedAsyncioTestCase):
    async def test_frozen_run_both_arms_unchanged_namespace_and_real_wire(self):
        from bench.m5_ultra_256gb.pareto_realistic import test_native_formal_cpu as old
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();native=m.donor();before=dict(native.run.__globals__)
            for C,arm in ((1,'off'),(8,'on')):
                spec=old.prepare(root/('metadata-'+arm),C=C);server=m.module(m.SERVER)
                spec['command'],spec['identity']=m.select_entry(spec,arm,server,server.source_pins())
                spec.update(schema='GDN-graph-combined-http-formal-spec-v1',arm=arm,phase=f'r1-c{C}-{arm}')
                backend=old.Backend(spec)
                with patch.object(m,'donor',return_value=native),patch.object(m,'validate',return_value=None):
                    result=await m.run(spec,root/arm,checkpoint=lambda *a:None,backend=backend)
                self.assertEqual(result['status'],'PASS',result['errors']);self.assertEqual(result['point']['requests'],64)
                self.assertEqual(backend.phases,['warm','formal']);self.assertEqual(backend.child.actions,['TERM'])
                self.assertTrue(result['server_reaped']);self.assertFalse(result['performance_qualified'])
                self.assertEqual(m.read(root/arm/'point.json'),dict(**result['point'],count_eligible=True,performance_qualified=False))
                self.assertFalse((root/arm/'route-events.jsonl').exists())
            self.assertEqual(native.run.__globals__,before)
    async def test_invalid_warm_is_retained_and_prevents_formal(self):
        from bench.m5_ultra_256gb.pareto_realistic import test_native_formal_cpu as old
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();spec=old.prepare(root/'metadata',C=1);spec.update(arm='on',phase='r1-c1-on')
            backend=old.Backend(spec);real=backend.group
            async def wrong(spec,phase,out):
                value=await real(spec,phase,out);value['requests'][0]['answer']='wrong';return value
            backend.group=wrong
            with patch.object(m,'validate',return_value=None):result=await m.run(spec,root/'trial',checkpoint=lambda *a:None,backend=backend)
            self.assertEqual(result['status'],'FAIL');self.assertEqual(backend.phases,['warm']);self.assertTrue(result['server_reaped'])
            self.assertTrue((root/'trial/warm-group.json').exists());self.assertFalse((root/'trial/point.json').exists())
    async def test_repeated_cancel_still_retains_partial_and_TERM_KILL_wait(self):
        from bench.m5_ultra_256gb.pareto_realistic import test_native_formal_cpu as old
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();spec=old.prepare(root/'metadata',C=1);spec.update(arm='on',phase='r1-c1-on')
            backend=old.Backend(spec,ignore_term=True);closing,allow=asyncio.Event(),asyncio.Event()
            async def slow(spec,phase,out):
                backend.group_started.set()
                try:await asyncio.Future()
                except asyncio.CancelledError:
                    closing.set();await allow.wait()
                    raise m.load('native_cleanup').GroupCancelled(dict(group_error={'type':'CancelledError'},cleanup={'complete':False}))
            backend.group=slow
            with patch.object(m,'validate',return_value=None):
                task=asyncio.create_task(m.run(spec,root/'trial',checkpoint=lambda *a:None,backend=backend))
                await backend.group_started.wait();task.cancel();await closing.wait();task.cancel();await asyncio.sleep(0);allow.set()
                result=await task
            self.assertEqual(result['status'],'FAIL');self.assertTrue(result['server_reaped'])
            self.assertEqual(backend.child.actions,['TERM','KILL']);self.assertTrue(backend.waited)
            self.assertTrue((root/'trial/warm-group.json').exists())
    async def test_raw_full_prompt_EOS_counts_and_cleanup_not_summary_oracle(self):
        from bench.m5_ultra_256gb.pareto_realistic import test_native_formal_cpu as old
        with tempfile.TemporaryDirectory() as tmp:
            spec=old.prepare(Path(tmp).resolve()/'metadata',C=8);group=await old.group(spec,'formal')
            checked=m.check_group(group,spec,'formal',old.decode)
            self.assertEqual(checked['point']['committed_output_tokens'],22+21*512)
            self.assertEqual(group['summary']['max_http_outstanding'],8)
            self.assertNotIn('expected_raw_ids_by_request',m.count_contract(spec,'formal'))
            for key,value in (('benchmark_prompt_ids',[777]),('benchmark_terminal_eos_id',None),('usage',{'completion_tokens':512})):
                bad=copy.deepcopy(group);old.rewrite_terminal(bad['requests'][0],lambda t:t.__setitem__(key,value))
                with self.assertRaises((ValueError,KeyError)):m.check_group(bad,spec,'formal',old.decode)
            bad=copy.deepcopy(group);bad['cleanup']['complete']=False
            with self.assertRaises(ValueError):m.check_group(bad,spec,'formal',old.decode)


if __name__=='__main__':unittest.main()
