"""Prepared C24 CPU mechanisms, not executed or device-qualified.

Real frozen transport/recording/audit control flow, fake OS/model boundaries;
the small NumPy runner is only a scheduling/token-input regression. Descriptor
fixtures exercise all128-state receipt gates without supplying model values.
"""
import ast
import copy
from contextlib import ExitStack, redirect_stderr
import hashlib
import inspect
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

from bench.m5_ultra_256gb.gdn_graph_combined_c24 import audit as a, http_capture as capture
from bench.m5_ultra_256gb.gdn_graph_combined_c24 import http_server as server, pair, qualify, replay
from bench.m5_ultra_256gb.gdn_graph_combined_c24.test_domain_cpu import numeric
from bench.m5_ultra_256gb.gdn_graph_combined.test_http_cpu import arguments
from bench.m5_ultra_256gb.gdn_graph_combined.test_http_audit_cpu import ReceiptBackend
from bench.m5_ultra_256gb.route_decode_graph.test_http_server_cpu import argv
from bench.m5_ultra_256gb.route_decode_graph.test_http_audit_v2_cpu import clock_group
from bench.m5_ultra_256gb.pareto_realistic import qualify_native as base
from bench.m5_ultra_256gb.pareto_realistic import test_audit_http_replay_v2_cpu as descriptors
from bench.m5_ultra_256gb.pareto_realistic.test_qualify_cpu import CPU, Engine, SamplingParams


def ref(path):
    return dict(path=str(path), sha256=a.sha(path))


def descriptor_fixture(C, *, tail=True, admitted=None):
    """Reuse the frozen synthetic builder, with explicit test-domain changes.

    These are fabricated comparison descriptors, never a numerical oracle.
    Unequal EOS lengths create a continuing old-row B1 after full B=C.
    """
    source = inspect.getsource(descriptors.synthetic)
    replacements = [
        ('count=8 if C==1 else 16', 'count=8'),
        ('cap=3,raw_ids=[99] if C==1 else [7,8,99],committed_ids=[] if C==1 else [7,8]',
         'cap=4,raw_ids=([7,8,9,99] if TAIL and i%C==C-1 else [7,8,99]),'
         'committed_ids=([7,8,9] if TAIL and i%C==C-1 else [7,8])'),
        ('min(C,len(remaining))', 'min(ADMIT or C,len(remaining))'),
        ('step_limit=count*5+1', 'step_limit=count*6+1'),
    ]
    for old, new in replacements:
        if source.count(old) != 1:
            raise AssertionError('frozen descriptor fixture anchor changed: '+old)
        source = source.replace(old, new)
    facade = NS(**{**vars(descriptors.a), 'case_type': a.case_type})
    namespace = dict(descriptors.synthetic.__globals__, a=facade, TAIL=tail, ADMIT=admitted)
    exec(compile(ast.parse(source), '<C24 synthetic descriptor fixture>', 'exec'), namespace)
    result = namespace['synthetic'](C)
    result[4]['domain_coverage'] = dict(full_batch=C, continuing_tail_B1=True)
    return result


def recording(C, *, tail=True):
    """Independent toy engine makes eight-job actual scheduler observations."""
    engine = Engine(C)
    jobs = {f'r{i}': dict(rid=f'r{i}', client_request_id=f'client-{i}', fixture_id=f'f{i}',
        prompt_ids=[1, 2+i], cap=5 if tail and i%C==C-1 else 3,
        raw_ids=[], committed_ids=[]) for i in range(8)}
    result = dict(schema='E023-recorded-http-v1', concurrency=C,
        eos_token_ids=sorted(engine.eos_ids), jobs=jobs, steps=[], histogram={})
    pending = list(jobs)
    original_schedule, original_run = engine.scheduler.schedule, engine.runner.run
    def schedule():
        value = original_schedule()
        observed['schedule'] = {k: base.schedule_metadata(engine, value)[k]
                               for k in ('decode', 'prefill', 'finished_ids')}
        return value
    def run(value):
        answer = original_run(value)
        observed['postrun_decode_rows'] = list(engine.runner.batch.req_ids)
        return answer
    engine.scheduler.schedule, engine.runner.run = schedule, run
    while pending or engine.has_work():
        observed = dict(step=len(result['steps'])+1, arrivals=[])
        # Separate waves ensure the final old row really continues at B1.
        if not engine.has_work():
            for _ in range(min(C, len(pending))):
                rid = pending.pop(0); job = jobs[rid]
                engine.add_request(job['prompt_ids'], SamplingParams(max_tokens=job['cap'],
                    temperature=0.0, ignore_eos=False), req_id=rid)
                observed['arrivals'].append(rid)
        observed['before'] = dict(running=[r.req_id for r in engine.scheduler.running],
                                 waiting=[r.req_id for r in engine.scheduler.waiting])
        delivered = engine.step()
        observed.update(results=base.plain(delivered), retired=[r for r,t,d in delivered if d],
                        surviving_decode_rows=list(engine.runner.batch.req_ids))
        B = str(len(observed['schedule']['decode']))
        result['histogram'][B] = result['histogram'].get(B, 0)+1
        for rid, token, done in delivered:
            jobs[rid]['raw_ids'].append(token)
            if done:
                ids = jobs[rid]['raw_ids']
                jobs[rid]['committed_ids'] = ids[:-1] if token in engine.eos_ids else list(ids)
        result['steps'].append(observed)
    return result


class EntryTests(unittest.TestCase):
    def test_actual_parser_accepts_only_c2_c4_and_both_qualification_switches(self):
        for C in (2,4):
            off = server.arguments(argv('--max-seqs', str(C)))
            self.assertEqual((off.gdn_prework, off.route_decode_graph), ('off','off'))
            on = ['--max-seqs', str(C), '--phase', 'qualification', '--gdn-prework','on',
                  '--route-decode-graph','on','--qualification-trace','/CPU/trace','--route-events','/CPU/routes']
            self.assertEqual(server.arguments(argv(*on)).max_seqs, C)
            for extra in (['--gdn-prework','off'], ['--route-events','/CPU/trace'], ['--max-seqs','8']):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    server.arguments(argv(*on,*extra))
        for C in (1,8,16):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                server.arguments(argv('--max-seqs',str(C)))

    def test_real_capture_factory_and_server_argv_preserve_eight_original_bodies(self):
        path, kw = arguments()
        original = capture.native.prepare(path,capture.sha(path),1,**kw)
        for C in (2,4):
            spec = capture.prepare(path,capture.sha(path),C,**kw); capture.validate(spec)
            self.assertEqual(len(spec['jobs']),8)
            self.assertEqual([x['body'] for x in spec['jobs']],[x['body'] for x in original['jobs']])
            self.assertEqual(spec['command'][3],str(capture.SERVER))
            args = server.arguments(spec['command'][4:]+['--qualification-trace','/CPU/trace','--route-events','/CPU/routes'])
            self.assertEqual(args.max_seqs,C)
            self.assertEqual(spec['identity']['configuration']['max_seqs'],C)
            self.assertEqual(spec['identity']['qualification_domain']['requests'],8)
            bad=copy.deepcopy(spec);bad['command'][bad['command'].index('--max-seqs')+1]='8'
            with self.assertRaises(ValueError):capture.validate(bad)


class CaptureTests(unittest.IsolatedAsyncioTestCase):
    async def test_fake_capture_to_independent_binding_uses_actual_readiness_layer(self):
        path,kw=arguments()
        async def checkpoint(*unused):pass
        for C in (2,4):
            with tempfile.TemporaryDirectory() as temporary:
                out=Path(temporary).resolve()/'capture'
                spec=capture.prepare(path,capture.sha(path),C,**kw)
                backend=ReceiptBackend(spec)
                result=await capture.capture(spec,out,checkpoint=checkpoint,backend=backend)
                self.assertEqual(result['status'],'PASS');self.assertTrue(result['server_reaped'])
                plan=dict(concurrency=C,qualification=spec['base_plan'],
                    capture={k:ref(out/n) for k,n in qualify.CAPTURE_NAMES.items()},
                    inputs={k:ref(out/n) for k,n in qualify.INPUT_NAMES.items()})
                actual,header=a.capture_binding(qualify,plan)
                self.assertEqual(actual,spec);self.assertEqual(header['phase'],f'HTTP-C{C}')
                self.assertEqual(backend.child.actions,['TERM'])


class AuditTests(unittest.TestCase):
    def streamed(self,data):
        native,record,fixtures,config,claimed,rows=data
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'steps.jsonl'
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            http=a.load(a.HTTP_AUDIT,a.HTTP_AUDIT_SHA)
            return http.stream_ledger(path,record,fixtures,config,claimed,native)

    def test_actual_eight_request_intervals_and_wrong_count_or_serial_reject(self):
        for C in (2,4):
            group,spec=clock_group(C);group['requests']=group['requests'][:8]
            self.assertEqual(a.http_intervals(group,spec)['max_http_outstanding'],C)
            extra,_=clock_group(C)
            with self.assertRaisesRegex(ValueError,'eight HTTP'):a.http_intervals(extra,spec)
            serial,_=clock_group(C,serial=True);serial['requests']=serial['requests'][:8]
            with self.assertRaisesRegex(ValueError,'actual HTTP concurrency'):a.http_intervals(serial,spec)

    def test_independent_actual_c_slots_full128_state_and_tail(self):
        for C in (2,4):
            data=descriptor_fixture(C)
            case=a.case_type(data[0])(data[1],data[2],data[3])
            self.assertEqual(case.free,list(range(C-1,-1,-1)))
            case.step=1;case.admit();scheduled=case.schedule()
            self.assertEqual([r['slot'] for r in scheduled['requests']],list(range(C)))
            result=self.streamed(data)
            self.assertEqual(result['requests'],8);self.assertEqual(result['observed_terminal_eos'],8)
            self.assertEqual(result['domain_coverage'],dict(full_batch=C,continuing_tail_B1=True))
            self.assertGreater(result['bit_equal_checks'],8*128)

    def test_actual_not_claimed_fullbatch_tail_and_original_array_gates(self):
        for data in (descriptor_fixture(4,tail=False),descriptor_fixture(4,admitted=2)):
            with self.assertRaisesRegex(ValueError,'actual full B=C'):self.streamed(data)
        for corruption in ('claim','bit','finite','slot','layer','release'):
            data=descriptor_fixture(2);row=data[-1][2]
            if corruption=='claim':data[4]['domain_coverage']['continuing_tail_B1']=False
            elif corruption=='bit':row['arrays'][0]['bit_equal']=False
            elif corruption=='finite':row['arrays'][0]['candidate_finite']=False
            elif corruption=='slot':row['schedule']['requests'][0]['slot']=7
            elif corruption=='layer':row['arrays'].pop()
            else:data[-1].pop()
            with self.subTest(corruption=corruption),self.assertRaises(ValueError):self.streamed(data)

    def test_no_terminal_or_unsealed_source_cannot_grant_qualification(self):
        self.assertEqual(a.PRODUCER_SHA,'UNSEALED')
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'base').mkdir();(root/'base/steps.jsonl').write_text('must not be read')
            (root/'complete.json').write_text(json.dumps({'status':'INCOMPLETE'}))
            with patch.object(a,'load',side_effect=AssertionError('no ledger helper')):result=a.audit(root,'0'*64)
            self.assertEqual(result['status'],'INCOMPLETE');self.assertFalse(result['ledger_opened'] or result['qualified'])


class ReplayTests(unittest.TestCase):
    def test_replay_ast_diff_is_only_new_domain_and_actual_coverage(self):
        current=inspect.getsource(replay.run_recorded_case)
        original=inspect.getsource(descriptors.a.load('http_replay.py',descriptors.a.SOURCES['http_replay.py']).run_recorded_case)
        changes=[
            ('concurrency not in (2, 4)','concurrency not in (1, 8)'),
            ('nonempty finite C2/C4 recording required','nonempty finite C1/C8 recording required'),
            ('    previous_decode, tail_continuation = [], False\n',''),
            ("            current_decode = list(reference.schedule['decode'])\n"
             '            if len(previous_decode) > 1 and len(current_decode) == 1 and current_decode[0] in previous_decode:\n'
             '                tail_continuation = True\n'
             '            previous_decode = current_decode\n'
             '            B = len(current_decode)',"            B = len(reference.schedule['decode'])"),
            ('if max(map(int, histogram), default=0) != concurrency or not tail_continuation:',
             'if concurrency == 8 and max(map(int, histogram), default=0) < 2:'),
            ('C2/C4 requires full B=C and continuing multirow-to-B1 tail coverage',
             'C8 recorded replay did not exercise multi-row decode'),
            ("'performance_qualified': False,\n            'domain_coverage': {'full_batch': concurrency, 'continuing_tail_B1': tail_continuation}",
             "'performance_qualified': False"),
        ]
        for old,new in changes:
            self.assertEqual(current.count(old),1,old);current=current.replace(old,new)
        self.assertEqual(ast.dump(ast.parse(current)),ast.dump(ast.parse(original)))

    def run_replay(self,data):
        self.engines=[Engine(data['concurrency']),Engine(data['concurrency'])]
        with tempfile.TemporaryDirectory() as temporary:
            cmp=base.Comparator(CPU(),temporary);ledger=io.StringIO()
            observers=[base.Observer(e,cmp,arm) for e,arm in zip(self.engines,('reference','candidate'))]
            try:return replay.run_recorded_case(data,self.engines,observers,SamplingParams,
                NS(prefill_batch_backend_supported=lambda:True),cmp,ledger,lambda:None,base=base)
            finally:
                for observer in observers:observer.restore()

    def test_real_cpu_scheduler_two_independent_arms_fullbatch_and_continuing_tail(self):
        for C in (2,4):
            result=self.run_replay(recording(C))
            self.assertEqual(len(result['completed']),8)
            self.assertEqual(result['domain_coverage'],dict(full_batch=C,continuing_tail_B1=True))
            self.assertFalse(result['performance_qualified'])
        with self.assertRaisesRegex(base.QualificationFailure,'continuing multirow'):
            self.run_replay(recording(4,tail=False))

    def test_recorded_token_remains_comparison_target_not_model_input(self):
        data=recording(2);expected=data['steps'][0]['results'][0][1]
        data['steps'][0]['results'][0][1]=expected+1
        with self.assertRaisesRegex(base.QualificationFailure,'recorded delivered'):self.run_replay(data)
        for engine in self.engines:self.assertEqual(engine.requests['r0'].output_token_ids,[expected])


class PairTests(unittest.TestCase):
    def failed_pair(self, *, constructor=False, cleanup_error=False):
        """Actual pair.run finally; explicit fake import/module/OS boundaries."""
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root=Path(temporary).resolve();package=root/'tutorial_code/miniengine';package.mkdir(parents=True)
            (package/'__init__.py').write_text('');model=root/'model';model.mkdir()
            (model/'weights.safetensors').write_bytes(b'')
            n=numeric();n.update(model=str(model),package_root=str(root),source_manifest='source',model_manifest='model',fixtures='fixtures')
            fixtures=[dict(fixture_id=f'f{i}',prompt_ids=[1]*513,normal_eos=True,max_output_tokens=512) for i in range(8)]
            manifests=dict(source={'files':{'__init__.py':'fake-source'}},model=dict(schema='E023-model-v1',
                model=str(model),revision='model',files={'config.json':'fake','generation_config.json':'fake','weights.safetensors':'fake'}),
                fixtures=dict(model=str(model),fixtures=fixtures))
            plan=root/'plan.json';plan.write_text(json.dumps(n));digest=hashlib.sha256(plan.read_bytes()).hexdigest()
            events=[];owned=[]
            class Observer:
                def __init__(self,arm):self.arm=arm;self.calls={'kept':object()};self.previous_future=self.postrun=object()
                def restore(self):
                    events.append(self.arm)
                    if cleanup_error and self.arm=='reference':raise RuntimeError('restore failure')
            def load(engines,observers,**kw):
                engines.extend([object(),object()])
                for arm in ('reference','candidate'):
                    if constructor and arm=='candidate':raise RuntimeError('constructor failure')
                    ob=Observer(arm);owned.append(ob);observers.append(ob)
            def fail_step(*args):raise RuntimeError('step failure')
            fake=NS(__file__=base.__file__,pinned_json=lambda key:manifests[key],verify_files=lambda *a:None,
                Comparator=lambda *a:object(),Observer=Observer,workloads=lambda *a:[dict(name='D-C2',concurrency=2)],
                run_case=fail_step,write=lambda path,value:Path(path).write_text(json.dumps(value)))
            modules={name:ModuleType(name) for name in ('mlx','mlx.core','mlx.utils','tutorial_code',
                'tutorial_code.miniengine','tutorial_code.miniengine.engine','tutorial_code.miniengine.engine.llm_engine',
                'tutorial_code.miniengine.engine.model_runner','tutorial_code.miniengine.sampling_params')}
            mx=modules['mlx.core'];mx.synchronize=lambda:events.append('sync');mx.clear_cache=lambda:events.append('clear')
            modules['mlx'].core=mx;modules['mlx.utils'].tree_flatten=lambda x:[]
            modules['tutorial_code'].miniengine=modules['tutorial_code.miniengine']
            modules['tutorial_code.miniengine'].engine=modules['tutorial_code.miniengine.engine']
            modules['tutorial_code.miniengine.engine'].model_runner=modules['tutorial_code.miniengine.engine.model_runner']
            modules['tutorial_code.miniengine'].__file__=str(package/'__init__.py')
            modules['tutorial_code.miniengine.engine.llm_engine'].LLMEngine=object
            modules['tutorial_code.miniengine.sampling_params'].SamplingParams=object
            stack.enter_context(patch.dict(sys.modules,modules));stack.enter_context(patch.object(sys,'path',list(sys.path)))
            stack.enter_context(patch.object(pair.domain,'BASE_PLAN_SHA',digest))
            stack.enter_context(patch.object(pair.domain,'load_pair_into',side_effect=load))
            stack.enter_context(patch.object(pair.signal,'getitimer',return_value=(0.0,0.0)))
            handler=stack.enter_context(patch.object(pair.signal,'signal',return_value='previous-handler'))
            alarm=stack.enter_context(patch.object(pair.signal,'setitimer'))
            out=root/'out';rc=pair.run(fake,dict(path=str(plan),sha256=digest),n,fixtures,2,out)
            result=json.loads((out/'complete.json').read_text())
            self.assertEqual(rc,1);self.assertEqual(result['status'],'FAIL')
            self.assertEqual(result['error']['message'],'constructor failure' if constructor else 'step failure')
            self.assertEqual([x for x in events if x in ('reference','candidate')],[x.arm for x in owned])
            self.assertEqual(events[-2:],['sync','clear'])
            self.assertTrue(all(not x.calls and x.previous_future is x.postrun is None for x in owned))
            alarm.assert_called_with(pair.signal.ITIMER_REAL,0)
            handler.assert_called_with(pair.signal.SIGALRM,'previous-handler')
            return result

    def test_partial_observer_construction_retains_primary_failure_and_releases(self):
        self.failed_pair(constructor=True)

    def test_restore_failure_still_attempts_second_observer_and_clears_alarm(self):
        result=self.failed_pair(cleanup_error=True)
        self.assertEqual(result['cleanup_error']['message'],'restore failure')


if __name__=='__main__':unittest.main()
