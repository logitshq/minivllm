"""Synthetic E055 contracts only; no actual prepare, process, signal or model."""
import asyncio
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('_e055_test_trial',HERE/'trial.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def ref(path):return dict(path=str(path),sha256=m.sha(path))
def pure_read(path):return json.loads(Path(path).read_text())
def pure_same(a,b,message):
    if a!=b:raise ValueError(message)
def pure_pinned(value):
    path=Path(value['path'])
    if not path.is_absolute() or path.resolve()!=path or m.sha(path)!=value['sha256']:raise ValueError('bad ref')
    return path

def identity_fields(args,pins,*,installed,route_observer):
    return dict(server_source_path=str(m.SERVER),server_source_sha256=m.FIXED[m.SERVER],benchmark_phase=args.phase,
        combined_candidates=dict(gdn_prework_enabled=args.gdn_prework=='on',route_decode_graph_enabled=True,
            gdn_layer_installs=48 if args.gdn_prework=='on' else 0,route_observer=route_observer))
server=SimpleNamespace(identity_fields=identity_fields,source_pins=lambda:{str(m.SERVER):m.FIXED[m.SERVER]})

def base_spec():
    return dict(schema='GDN-graph-combined-http-formal-spec-v1',concurrency=8,arm='off',repeat=1,
        phase='r1-c8-off',observer_enabled=False,command=['python','-I','-B',str(m.B/'gdn_graph_combined/http_server.py'),
            '--phase','formal','--route-decode-graph','on','--gdn-prework','off'],
        identity=dict(step_observer=False,qualification_trace_enabled=False,per_token_observer=False,model='fixed'),
        environment=dict(LANG='en_US.UTF-8',**m.MLX),jobs=dict(warm=[{'i':i} for i in range(8)],formal=[{'i':i} for i in range(64)]),
        base_plan=dict(path='/original/plan.json',sha256='original'),fixtures=[dict(fixture_id='D01')],
        qualification={'audit':{'sha256':'original-wire-count'}},transport=dict(tcp_nodelay=True,io_timeout_seconds=300.0),
        deadlines={'resource_interval':5})


class FactoryTests(unittest.TestCase):
    def test_real_source_closure_is_cpu_only_and_frozen(self):
        pins=m.source_pins()
        self.assertEqual(pins[str(m.SERVER)],m.FIXED[m.SERVER]);self.assertEqual(pins[str(m.AUDIT)],m.FIXED[m.AUDIT])
        self.assertEqual(pins[str(HERE/'trial.py')],m.sha(HERE/'trial.py'))

    def test_six_trials_are_unique_three_pairs_at_c8(self):
        self.assertEqual(len(set(m.TRIALS)),6)
        self.assertEqual([x.rsplit('-',1)[1] for x in m.TRIALS],['on','off','off','on','on','off'])
        for a,b in zip(m.TRIALS[::2],m.TRIALS[1::2]):self.assertEqual(a.split('-')[:2],b.split('-')[:2])

    def test_factory_preserves_queue_environment_and_requires_exact_validation(self):
        original=base_spec();before=copy.deepcopy(original)
        formal=SimpleNamespace(read=lambda p:copy.deepcopy(original),pinned=lambda r:Path('/cpu/base.json'),
            validate=lambda v:None,SERVER=m.B/'gdn_graph_combined/http_server.py')
        modules=lambda p:formal if p==m.FORMAL else server
        with patch.object(m,'module',side_effect=modules),patch.object(m,'same',side_effect=pure_same),\
             patch.object(m,'bind_history') as history,patch.object(m,'source_pins',return_value={'fixed':'source'}):
            for arm in ('off','on'):
                spec=m.prepare({'path':'base','sha256':'old'},arm,'r1-c8-'+arm,{'actual':'E053'})
                self.assertEqual(original,before);self.assertEqual(spec['jobs'],original['jobs'])
                self.assertEqual(spec['qualification'],original['qualification']);self.assertEqual(spec['environment'],original['environment'])
                self.assertEqual(spec['command'][3:5],[str(m.SERVER),'--run'])
                self.assertEqual(spec['command'][-1],arm);self.assertFalse(spec['identity']['step_observer'])
                self.assertFalse(spec['identity']['combined_candidates']['route_observer'])
                self.assertEqual(spec['identity']['combined_candidates']['gdn_layer_installs'],48 if arm=='on' else 0)
                m.validate(spec)
                for mutate in (lambda x:x['jobs']['formal'].pop(),lambda x:x['environment'].update(MLX_MAX_MB_PER_BUFFER='200'),
                               lambda x:x.update(performance_qualified=True),lambda x:x['command'].append('--route-events')):
                    bad=copy.deepcopy(spec);mutate(bad)
                    with self.assertRaises(ValueError):m.validate(bad)
            with self.assertRaises(ValueError):m.prepare({},'on','r2-c8-on',{})
            self.assertTrue(history.called)

    def test_actual_e053_binding_rejects_pending_old_or_environment_changed_proof(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(m,'read',side_effect=pure_read),\
             patch.object(m,'same',side_effect=pure_same),patch.object(m,'pinned',side_effect=pure_pinned):
            root=Path(temp).resolve();child=root/'numerical';child.mkdir();(child/'base').mkdir();base=base_spec()
            cap=dict(schema='c8-batched-http-spec-v2',concurrency=8,wave_names=['warm','screen'],fixtures=base['fixtures'],
                environment=base['environment'],transport=base['transport'],identity=m.identity(base,'on',server,{},qualification=True))
            cap_path=root/'capture-spec.json';write(cap_path,cap)
            plan=dict(schema='c8-batched-http-replay-plan-v1',sources={str(p):m.FIXED[p] for p in (m.SERVER,m.PRODUCER)},
                capture_spec=ref(cap_path),qualification=base['base_plan'],environment=m.MLX)
            write(child/'plan.json',plan)
            write(child/'complete.json',dict(status='PASS',plan_sha256=m.sha(child/'plan.json'),source_input_pins_rechecked=True,replay_numerical_checks_passed=True))
            for name in ('base/plan.json','base/complete.json','recording.json'):write(child/name,{})
            proof=dict(schema='E053-independent-http-replay-audit-v1',status='PASS',audit_source_sha256=m.FIXED[m.AUDIT],
                child=str(child),plan_sha256=m.sha(child/'plan.json'),qualified=True,recorded_history_verified=True,
                http_history_qualified=True,candidate_http_capture_qualified=True,performance_qualified=False,
                owner_resources_restore_qualified=False,offline_tensor_bytes_rechecked=False,actual_candidate_sources=plan['sources'],
                terminal_sha256={str(child/n):m.sha(child/n) for n in ('plan.json','complete.json','base/plan.json','base/complete.json','recording.json')})
            audit=root/'audit.json';write(audit,proof)
            binding=lambda:dict(audit=ref(audit),replay_plan=ref(child/'plan.json'),child_directory=str(child))
            m.bind_history(binding(),base,server,{})
            for change in (dict(status='INCOMPLETE'),dict(http_history_qualified=False),dict(performance_qualified=True),
                           dict(schema='E051-independent-full-audit-v1'),dict(audit_source_sha256='0'*64)):
                write(audit,{**proof,**change})
                with self.assertRaises(ValueError):m.bind_history(binding(),base,server,{})
            write(audit,proof);changed=copy.deepcopy(base);changed['environment']['MLX_MAX_MB_PER_BUFFER']='200'
            with self.assertRaisesRegex(ValueError,'environment'):m.bind_history(binding(),changed,server,{})
            (child/'recording.json').write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'terminal bytes'):m.bind_history(binding(),base,server,{})

    def test_complete_ids_preserve_occurrences_and_terminal_eos(self):
        terminal=lambda row:(row['rid'],dict(benchmark_output_ids=row['raw'],benchmark_committed_output_ids=row['ids'],benchmark_terminal_eos_id=row['eos']),'')
        native=SimpleNamespace(load=lambda name:SimpleNamespace(raw_completion=terminal))
        group=dict(requests=[dict(fixture_id='D01',request_id='a',rid='u1',raw=[7,99],ids=[7],eos=99),
                             dict(fixture_id='D01',request_id='b',rid='u2',raw=[8,9],ids=[8,9],eos=None)])
        rows=m.retained_ids(native,group)
        self.assertEqual([r['occurrence'] for r in rows],[0,1]);self.assertEqual(rows[0]['raw_ids'],[7,99])
        self.assertEqual(rows[0]['committed_ids'],[7]);self.assertEqual(rows[1]['committed_ids'],[8,9])

    def test_native_run_uses_private_validator_and_retains_diagnostic_scope(self):
        # `validate` is resolved from the copied function's private globals only.
        async def fake_run(spec,out,*,checkpoint,backend=None):
            validate(spec)
            await checkpoint('cpu-only',None)
            return dict(status='PASS',server_reaped=True,cleanup_complete=True,artifacts={})
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);native=SimpleNamespace(run=fake_run,read=pure_read,write=write)
            for phase in ('warm','formal'):write(out/(phase+'-group.json'),{'requests':[]})
            async def check(label,pid):self.assertEqual((label,pid),('cpu-only',None))
            spec=dict(label='r1-c8-on',phase='r1-c8-on',arm='on',numerical_binding={'actual':'proof'})
            with patch.object(m,'module',return_value=native),patch.object(m,'validate') as gate,\
                 patch.object(m,'retained_ids',return_value=[dict(raw_ids=[1,2],committed_ids=[1,2])]):
                result=asyncio.run(m.run(spec,out,checkpoint=check))
            gate.assert_called_once_with(spec)
            self.assertNotIn('validate',fake_run.__globals__)
            self.assertTrue(result['server_reaped']);self.assertTrue(result['diagnostic_throughput_only'])
            self.assertFalse(result['performance_qualified']);self.assertFalse(result['full_state_qualified'])
            self.assertEqual(result['cross_schedule_token_comparison'],'PENDING_PAIRED_AUDIT')
            self.assertEqual(pure_read(out/'output-ids.json')['formal'][0]['raw_ids'],[1,2])


if __name__=='__main__':unittest.main(verbosity=2)
