"""Focused synthetic file-binding tests; not actual engine/resource qualification.

Written source-only while formal GPU measurements run. Root executes later.
No real process, socket, vendor runtime, service or model is imported/launched.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from bench.m5_ultra_256gb.gdn_graph_combined_http_cycle import audit_cycle as a


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + '\n')


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve(); self.owner = self.root/'owner'; self.journal = self.root/'journal'

    def test_nonterminal_does_not_open_owner_or_load_support(self):
        write(self.journal/'journal.json', {'state':'PREPARED'})
        with patch.object(a, 'load', side_effect=AssertionError('no support import')):
            result = a.audit(self.owner,self.journal,'0'*64,{})
        self.assertEqual(result['status'],'INCOMPLETE'); self.assertFalse(result['ledger_opened'])
        self.assertFalse(result['qualified']); self.assertFalse(self.owner.exists())

    def test_terminal_with_unsealed_restoration_cannot_qualify(self):
        write(self.journal/'journal.json', {'state':'RESTORED'})
        write(self.owner/'complete.json', {'status':'PASS'})
        with patch.object(a,'RESTORE_SHA','UNSEALED'), patch.object(a,'load',side_effect=AssertionError('no import')):
            result = a.audit(self.owner,self.journal,'0'*64,{})
        self.assertEqual(result['status'],'INCOMPLETE'); self.assertIn('UNSEALED',result['reason'])

    def test_failed_experiment_can_preserve_separate_recovery_pass(self):
        write(self.journal/'journal.json', {'state':'RESTORED'})
        write(self.owner/'complete.json', {'status':'FAIL','errors':['synthetic first error']})
        restored = {'status':'PASS','qualified':True}
        with patch.object(a,'sealed'), patch.object(a,'sha',return_value='a'*64), \
             patch.object(a,'CONTROLLER_SHA','a'*64), patch.object(a,'load',return_value=NS(audit=lambda _:restored)):
            result = a.audit(self.owner,self.journal,'0'*64,{})
        self.assertEqual(result['status'],'FAIL'); self.assertTrue(result['restoration_verified'])
        self.assertFalse(result['qualified']); self.assertFalse(result['performance_qualified'])
        self.assertEqual(result['producer_errors'],['synthetic first error'])

    def test_missing_restoration_terminal_evidence_stays_incomplete(self):
        write(self.journal/'journal.json', {'state':'RESTORED'})
        write(self.owner/'complete.json', {'status':'PASS'})
        with patch.object(a,'sealed'), patch.object(a,'sha',return_value='a'*64), \
             patch.object(a,'CONTROLLER_SHA','a'*64), \
             patch.object(a,'load',return_value=NS(audit=lambda _:{'status':'INCOMPLETE','qualified':False})):
            result = a.audit(self.owner,self.journal,'0'*64,{})
        self.assertEqual(result['status'],'INCOMPLETE'); self.assertFalse(result['restoration_verified'])
        self.assertFalse(result['ledger_opened']); self.assertFalse(result['qualified'])

    def test_frozen_resource_and_provenance_functions_are_directly_reused(self):
        donor = a.load(a.SUPPORT,a.SUPPORT_SHA)
        self.assertEqual(tuple(donor.PHASES),a.PHASES)
        tree = ast.parse(Path(a.__file__).read_text())
        names = {node.name for node in tree.body if isinstance(node,ast.FunctionDef)}
        self.assertNotIn('resources',names); self.assertNotIn('preparation_pins',names)
        used = {node.func.attr for node in ast.walk(tree) if isinstance(node,ast.Call)
                and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name)
                and node.func.value.id=='reused'}
        self.assertEqual(used,{'resources','preparation_pins'})
        for count in (1000,1001):
            names=sorted(f'resource-{i:03d}.json' for i in range(count))
            donor.sample_names(names,'resource',count)
            with self.assertRaises(ValueError): donor.sample_names(names[:-1],'resource',count)

    def test_evidence_symlink_still_rejected_and_no_tokenizer_ref_is_invented(self):
        target=self.root/'actual.json';write(target,{'CPU':'only'})
        alias=self.root/'alias.json';alias.symlink_to(target)
        with self.assertRaises(ValueError):a.pinned({'path':str(alias),'sha256':a.sha(target)})
        self.assertEqual(a.pinned(a.ref(target)),target)
        # Model paths use the frozen model-manifest/stat path, not this evidence
        # binder. Keep the actual support source fixed; do not import tokenizers.
        donor=a.load(a.SUPPORT,a.SUPPORT_SHA)
        source=ast.parse(Path(donor.__file__).read_text())
        function=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='preparation_pins')
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
                             and n.func.id=='pinned' for n in ast.walk(function)))

    def cycle(self):
        """Explicit fake producer/source validation; real file/CLI/audit binders run."""
        wrapper = dict(qualification={'path':'/CPU/original.json','sha256':a.BASE_PLAN_SHA},
            serving={'plan':{'path':'/CPU/serving.json','sha256':'1'*64},'audit':{'path':'/CPU/serving-audit.json','sha256':'2'*64},
                     'complete':{'path':'/CPU/serving-complete.json','sha256':'6'*64}},
            http={'environment':{'MLX_CPU_FIXTURE':'true'}})
        numeric = {'execution':{'python':'/CPU/python'}}; complete = {'owner_pid':9000,'phases':[]}
        specs = [{'concurrency':C,'sources':{'/CPU/server.py':'3'*64},'environment':wrapper['http']['environment']} for C in (1,8)]
        plans = {}; audits = {}; index = []
        capture_names = dict(spec='spec.json',complete='complete.json',child='child.json',release='release.json',absence='child-absence.json',port='port-after.json')
        input_names = dict(group='group.json',trace='trace.jsonl',route_events='route-events.jsonl')
        for i,stage in enumerate(a.PHASES):
            folder = self.owner/'phases'/stage; C = 1 if i<2 else 8; pid = 9100+i
            child = dict(pid=pid,owner_pid=9000,identity={'pid':pid,'ppid':9000},spawned_at=10+100*i)
            release = dict(pid=pid,returncode=0,waited_reaped=True,strict_absence=True,time=90+100*i)
            write(folder/'child.json',child); write(folder/'release.json',release); index.append({'phase':stage})
            target = folder/('capture' if stage.endswith('capture') else 'qualification')
            if stage.endswith('capture'):
                flags = ('capture_complete','cleanup_complete','server_reaped','cleanup_task_resolved',
                         'client_task_resolved','route_events_retained','small_pins_rechecked')
                result = dict(schema='GDN-graph-combined-http-capture-result-v1',status='PASS',errors=[],**dict.fromkeys(flags,True))
                write(target/'complete.json',result); write(target/'spec.json',specs[i//2]); write(target/'child.json',{'pid':pid})
                write(target/'release.json',dict(pid=pid,returncode=0,time=80+100*i))
                write(target/'child-identity.json',{'stdout':f'{pid} 9000 synthetic'})
                for name in ('child-absence.json','port-after.json','group.json'): write(target/name,{})
                for name in ('trace.jsonl','route-events.jsonl'): (target/name).write_bytes(b'CPU synthetic only\n')
                write(folder/'launch-context.json',dict(cwd=str(a.ROOT),spec_sources=specs[i//2]['sources'],environment=wrapper['http']['environment']))
            else:
                captured = self.owner/'phases'/f'c{C}-capture/capture'
                plan = dict(schema='GDN-graph-combined-recorded-http-plan-v1',concurrency=C,qualification=wrapper['qualification'],
                    serving=wrapper['serving'],qualifier_sha256=a.REPLAY_SHA,sources={'/CPU/candidate.py':'4'*64},
                    capture={k:a.ref(captured/n) for k,n in capture_names.items()},inputs={k:a.ref(captured/n) for k,n in input_names.items()})
                write(folder/'plan.json',plan); plans[str(folder/'plan.json')] = plan
                digest = a.sha(folder/'plan.json')
                command = [numeric['execution']['python'],'-I','-B',str(a.GDN/'http_qualification.py'),'--run',
                    '--plan',str(folder/'plan.json'),'--plan-sha256',digest,'--out',str(target)]
                write(folder/'launch.json',dict(command=command,environment=wrapper['http']['environment'],cwd=str(a.ROOT)))
                result = dict(schema='GDN-graph-combined-recorded-http-result-v1',status='PASS',replay_numerical_checks_passed=True,
                    input_source_pins_rechecked=True,numerical_qualified=False,http_qualified=False,performance_qualified=False,
                    independent_audit_required=True,concurrency=C,total_requests=8 if C==1 else 16,plan_sha256=digest,
                    **{k:plan[k] for k in ('capture','inputs','sources','serving')})
                write(target/'complete.json',result); write(target/'plan.json',plan)
                for name in ('base/plan.json','base/complete.json','recording.json'): write(target/name,{'fixture':'CPU only'})
                report = dict(schema='GDN-graph-combined-recorded-http-independent-v1',status='PASS',qualified=True,http_history_qualified=True,
                    audit_source_sha256=a.NUMERICAL_SHA,concurrency=C,plan_sha256=digest,base_plan_sha256=a.BASE_PLAN_SHA,child_directory=str(target),
                    terminal_sha256={n:a.sha(target/n) for n in ('plan.json','complete.json','base/plan.json','base/complete.json','recording.json')},
                    sources=plan['sources'],raw_inputs=plan['inputs'],capture=plan['capture'],serving_prerequisite=plan['serving'],
                    ledger={'requests':8 if C==1 else 16,'sha256':'5'*64},routes={'replay_pid':pid,'calls':12,'combined':{'compiled-tracing':1,'compiled-reuse':11,'eager-fallback':0,'traced_batches':[1]},'captured':{'sha256':plan['inputs']['route_events']['sha256']}},
                    http={'max_http_outstanding':C,'fully_drained':True},performance_qualified=False,owner_resources_restore_qualified=False,
                    offline_tensor_bytes_rechecked=False,weight_shards_rehashed=False,original_37_request_gate_replaced=False)
                path = self.root/f'audit-c{C}.json'; write(path,report); audits[str(C)] = a.ref(path)
            complete['phases'].append(dict(phase=stage,status='PASS',result={'path':str((target/'complete.json').relative_to(self.owner)),'sha256':a.sha(target/'complete.json')}))
        write(self.owner/'children.json',{'children':index}); write(self.owner/'capture-specs.json',specs)
        producer = NS(CAPTURE_NAMES=capture_names,INPUT_NAMES=input_names,
            validate=lambda path,digest:(plans[str(path)],numeric,None,None))
        raw = NS(capture_binding=lambda producer,plan:(specs[0 if plan['concurrency']==1 else 1],
            {'pid':9100 if plan['concurrency']==1 else 9102}))
        return wrapper,numeric,specs,complete,audits,producer,raw

    def check_cycle(self, fixture): return a.phases(self.owner,*fixture)

    def test_four_phase_file_binding_with_real_gdn_cli_and_explicit_audits(self):
        facts,audits = self.check_cycle(self.cycle())
        self.assertEqual([x['phase'] for x in facts],list(a.PHASES))
        self.assertEqual([audits[k]['requests'] for k in ('1','8')],[8,16])
        self.assertLess(facts[0]['inner_released_at'],facts[1]['spawned_at'])

    def test_missing_run_flag_wrong_cwd_or_environment_rejected(self):
        fixture = self.cycle(); path = self.owner/'phases/c1-replay/launch.json'; original = a.read(path)
        for kind in ('mode','cwd','env'):
            bad = copy.deepcopy(original)
            if kind=='mode': bad['command'].remove('--run')
            if kind=='cwd': bad['cwd']='/CPU/wrong'
            if kind=='env': bad['environment']['MLX_CPU_FIXTURE']='changed'
            write(path,bad)
            with self.subTest(kind=kind),self.assertRaises(ValueError): self.check_cycle(fixture)
        write(path,original); self.check_cycle(fixture)

    def test_stale_or_wrong_c_audit_cannot_bind_after_receipt_rehash(self):
        fixture = self.cycle(); path = Path(fixture[4]['1']['path']); original = a.read(path)
        for kind in ('pid','route','terminal','scope','count','C'):
            bad = copy.deepcopy(original)
            if kind=='pid': bad['routes']['replay_pid']+=1
            if kind=='route': bad['raw_inputs']['route_events']['sha256']='0'*64
            if kind=='terminal': bad['terminal_sha256']['complete.json']='0'*64
            if kind=='scope': bad['performance_qualified']=True
            if kind=='count': bad['ledger']['requests']=16
            if kind=='C': bad['concurrency']=8
            write(path,bad); fixture[4]['1']=a.ref(path)
            with self.subTest(kind=kind),self.assertRaises(ValueError): self.check_cycle(fixture)

    def test_inner_outer_pid_exit_or_capture_reap_before_replay_required(self):
        fixture = self.cycle(); path = self.owner/'phases/c1-capture/capture/release.json'; original = a.read(path)
        for key,value in [('pid',9999),('returncode',-9),('time',150)]:
            bad = dict(original); bad[key]=value; write(path,bad)
            # Fake producer validation deliberately does not police these raw fields.
            # Rebind the captured reference so this tests the outer cross-check itself.
            plan_path = self.owner/'phases/c1-replay/plan.json'; fixture[5].validate(plan_path,'')[0]['capture']['release']=a.ref(path)
            with self.subTest(key=key),self.assertRaises(ValueError): self.check_cycle(fixture)

    def test_incomplete_four_phase_prefix_cannot_be_a_success(self):
        fixture = self.cycle(); index = a.read(self.owner/'children.json'); index['children'].pop()
        write(self.owner/'children.json',index)
        with self.assertRaises(ValueError): self.check_cycle(fixture)

    def top(self):
        wrapper,numeric,specs,complete,reports,producer,raw = self.cycle()
        wrapper['execution']={'pins':{}}
        write(self.owner/'owner-plan.json',wrapper);digest=a.sha(self.owner/'owner-plan.json')
        complete.update(schema='GDN-graph-combined-http-owner-result-v1',status='PASS',owner_plan_sha256=digest,
            errors=[],first_failure=None,numerical_replay_complete=True,all_children_reaped=True,
            independently_qualified=False,performance_qualified=False)
        write(self.owner/'complete.json',complete);write(self.owner/'attempt.json',{'created_at':1})
        write(self.journal/'journal.json',dict(schema='GDN-graph-combined-http-cold-journal-v1',state='RESTORED',
            owner_directory=str(self.owner),owner_plan={'sha256':digest},
            owner_source={'path':str(a.HERE/'owner.py'),'sha256':a.OWNER_SHA},original={'pid':8000}))
        restoration=dict(status='PASS',qualified=True,journal_sha256=a.sha(self.journal/'journal.json'),
            owner_plan_sha256=digest)
        modules={a.HERE/'audit_restore.py':NS(audit=lambda _:restoration),
            a.HERE/'owner.py':NS(validate_plan=lambda args:(None,wrapper,None,numeric,specs)),
            a.GDN/'http_qualification.py':producer,a.GDN/'http_audit.py':raw}
        return digest,reports,modules

    def test_complete_shell_delegates_resource_gate_and_never_grants_performance(self):
        digest,reports,modules=self.top();calls=[]
        def resources(owner,complete,children,pid,created):
            calls.append((owner,pid,created,[r['phase'] for r in children]));return {'samples':99}
        modules[a.SUPPORT]=NS(PHASES=a.PHASES,preparation_pins=lambda *_:{'CPU_fixture':True},resources=resources)
        with patch.object(a,'load',side_effect=lambda path,digest:modules[path]):
            result=a.audit(self.owner,self.journal,digest,reports)
        self.assertEqual(result['status'],'PASS',result)
        self.assertEqual(calls,[(self.owner,8000,1,list(a.PHASES))])
        self.assertEqual(result['resources'],{'samples':99})
        self.assertEqual(set(result['numerical_audits']),{'1','8'})
        self.assertTrue(result['http_history_qualified']);self.assertTrue(result['restoration_verified'])
        self.assertFalse(result['performance_qualified']);self.assertFalse(result['ledger_opened'])
        self.assertFalse(result['global_model_quality_qualified'])

    def test_resource_rejection_or_missing_independent_report_never_qualifies(self):
        digest,reports,modules=self.top()
        def refuse(*args):raise ValueError('synthetic raw pressure failure')
        modules[a.SUPPORT]=NS(PHASES=a.PHASES,preparation_pins=lambda *_:{},resources=refuse)
        with patch.object(a,'load',side_effect=lambda path,digest:modules[path]):
            result=a.audit(self.owner,self.journal,digest,reports)
            self.assertEqual(result['status'],'FAIL',result);self.assertFalse(result['qualified'])
            self.assertIn('raw pressure',result['first_failure']['message'])
            Path(reports['1']['path']).unlink()
            result=a.audit(self.owner,self.journal,digest,reports)
        self.assertEqual(result['status'],'INCOMPLETE',result)
        self.assertFalse(result['qualified']);self.assertFalse(result['ledger_opened'])


if __name__ == '__main__': unittest.main()
