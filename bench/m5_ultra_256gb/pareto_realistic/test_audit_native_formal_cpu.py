"""Offline CPU protocol/receipt corruptions; no real service, model, GPU or socket.

The trial fixture runs the frozen formal runner with fake process methods and
real native client/serializer over MockTransport. It cannot grant model evidence.
"""
import asyncio
import base64
import copy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import httpx

from bench.m5_ultra_256gb.pareto_realistic import audit_native_formal as a
from bench.m5_ultra_256gb.pareto_realistic import native_formal as formal
from bench.m5_ultra_256gb.pareto_realistic import native_server as server
from bench.m5_ultra_256gb.pareto_realistic import test_native_formal_cpu as trial_cpu
from bench.m5_ultra_256gb.pareto_realistic import test_audit_http_owner_v2_cpu as pins_cpu


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False)+'\n')


def absence(pid):
    commands=[(['/bin/ps','-p',str(pid),'-o','pid='],1),(['/bin/launchctl','print',a.JOB],113),
              (['/usr/sbin/lsof','-nP','-a','-iTCP:8001','-sTCP:LISTEN','-Fpn'],1)]
    return dict(status='PASS',checks=[dict(command=c,returncode=rc,stdout_sha256=hashlib.sha256(b'').hexdigest(),
        stdout_bytes=0,absent=True) for c,rc in commands])


class UniqueBackend(trial_cpu.Backend):
    def spawn(self,*args):
        child=super().spawn(*args);self.spawned_at=time.time();return child
    async def group(self,spec,phase,output):
        self.phases.append(phase)
        fixtures={f['prompt_text']:f for f in spec['fixtures']};counter=0
        async def handler(request):
            nonlocal counter
            i=counter;counter+=1;body=json.loads(request.content);f=fixtures[body['prompt']]
            eos=spec['identity']['eos_ids'][0]
            raw=[10,eos] if i%3==0 else [eos] if i%3==1 else [11]*512
            rid=f'{spec["concurrency"]*100000+(1000 if phase=="formal" else 1)+i:032x}'
            terminal=server.completion(server.Job(rid,f['prompt_ids'],512),raw,set(spec['identity']['eos_ids']))
            text=trial_cpu.decode(terminal['benchmark_committed_output_ids'])
            delta=trial_cpu.wire(dict(id='cmpl-'+rid,choices=[dict(index=0,text=text,finish_reason=None)])) if text else b''
            return httpx.Response(200,stream=trial_cpu.Stream(delta+trial_cpu.wire(terminal)+trial_cpu.wire('[DONE]')),
                headers={'content-type':'text/event-stream'})
        # MockTransport intercepts this exact registered local address; no socket.
        return await formal.load('native_cleanup').run_group(f"http://127.0.0.1:{spec['transport']['port']}",spec['jobs'][phase],spec['concurrency'],
            phase=phase,qualification=formal.count_contract(spec,phase),request_timeout_s=2,group_timeout_s=2,
            cleanup_timeout_s=.1,transport=httpx.MockTransport(handler),output_dir=output)


async def trial_fixture(root):
    owner=root/'owner';owner.mkdir();specs=[];index=[];stages=[]
    for C in (1,8):
        spec=trial_cpu.prepare(root/f'mock-numerical-C{C}',C);specs.append(spec)
        stage=f'r1-c{C}';folder=owner/'phases'/stage;folder.mkdir(parents=True)
        backend=UniqueBackend(spec);backend.child.pid+=C
        result=await formal.run(spec,folder/'trial',checkpoint=lambda *args:None,backend=backend)
        if result['status']!='PASS':raise AssertionError(result['errors'])
        child=dict(phase=stage,pid=backend.child.pid,owner_pid=__import__('os').getpid(),
            identity=dict(pid=backend.child.pid,ppid=__import__('os').getpid()),
            ownership='direct-Popen-returned',spawned_at=backend.spawned_at)
        release=dict(phase=stage,pid=backend.child.pid,returncode=backend.child.returncode,waited_reaped=True,
                     strict_absence=True,time=time.time())
        write(folder/'child.json',child);write(folder/'release.json',release)
        index.append(dict(phase=stage,pid=child['pid']))
        refs=lambda name:dict(path=str((folder/'trial'/name).relative_to(owner)),sha256=a.sha(folder/'trial'/name))
        stages.append(dict(phase=stage,status='PASS',result=refs('complete.json'),point=refs('point.json'),
                           numerical_audit=spec['qualification']['audit']))
    write(owner/'children.json',dict(schema='cold-service-owner-children-v1',children=index))
    write(owner/'formal-specs.json',specs)
    write(owner/'points.json',dict(schema='E023-native-formal-points-v1',trials=stages,performance_qualified=False))
    wrapper=dict(trials=['r1-c1','r1-c8'],http={'environment':specs[0]['environment'],'transport':specs[0]['transport']})
    complete=dict(owner_pid=__import__('os').getpid(),phases=stages)
    return owner,wrapper,specs,complete


class RawTrialTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.owner,self.wrapper,self.specs,self.complete=await trial_fixture(self.root)
    def check(self):return a.audit_trials(self.owner,self.wrapper,self.specs,self.complete,trial_cpu.decode)

    def test_actual_mock_producer_two_trials_variable_eos_and_common_windows(self):
        children,points=self.check()
        self.assertEqual([c['phase'] for c in children],['r1-c1','r1-c8'])
        self.assertEqual([p['max_http_outstanding'] for p in points],[1,8])
        self.assertEqual([p['committed_output_tokens'] for p in points],[22+21*512]*2)
        self.assertTrue(all(p['requests']==64 and p['warm_requests_checked']==8 for p in points))
        for stage in self.wrapper['trials']:
            self.assertFalse(a.read(self.owner/'phases'/stage/'trial/point.json')['performance_qualified'])

    def test_original_point_claim_tamper_and_observer_change_rejected(self):
        path=self.owner/'phases/r1-c1/trial/point.json';raw=path.read_bytes()
        row=json.loads(raw);row['performance_qualified']=True;write(path,row)
        with self.assertRaises(ValueError):self.check()
        path.write_bytes(raw)
        spec=self.specs[0];spec['identity']['step_observer']=True
        with self.assertRaises(ValueError):self.check()

    def test_raw_full_prompt_same_length_text_eos_and_clock_corruption(self):
        path=self.owner/'phases/r1-c1/trial/formal-group.json';original=a.read(path)
        spec=self.specs[0]
        for kind in ('prompt','text','eos','clock','usage','cap'):
            group=copy.deepcopy(original);row=group['requests'][0]
            if kind=='prompt':trial_cpu.rewrite_terminal(row,lambda t:t['benchmark_prompt_ids'].__setitem__(0,777))
            if kind=='text':row['answer']='bad'
            if kind=='eos':trial_cpu.rewrite_terminal(row,lambda t:t.__setitem__('benchmark_terminal_eos_id',None))
            if kind=='clock':row['done_perf_counter_s']+=1
            if kind=='usage':trial_cpu.rewrite_terminal(row,lambda t:t['usage'].__setitem__('completion_tokens',True))
            if kind=='cap':row['body']['max_tokens']=511
            with self.subTest(kind=kind),self.assertRaises(ValueError):formal.check_group(group,spec,'formal',trial_cpu.decode)

    def test_metric_oracle_ignores_derived_summary_and_uses_wire(self):
        group=a.read(self.owner/'phases/r1-c8/trial/formal-group.json')
        before=a.raw_metrics(group)
        group['summary']={'aggregate_output_tok_s':999999,'e2e_s':{'p50':0}}
        for row in group['requests']:row['qualified_output_tokens']=999;row['reported_completion_tokens']=999
        self.assertEqual(a.raw_metrics(group),before)
        group['group_deadline_perf_counter_s']=before['last_done_s']-1
        with self.assertRaises(ValueError):a.raw_metrics(group)

    def test_fixed_endpoint_timeout_and_request_deadline_rejected(self):
        folder=self.owner/'phases/r1-c1/trial';complete=a.read(folder/'complete.json')
        original=a.read(folder/'formal-group.json')
        for kind in ('endpoint','group-timeout','request-deadline'):
            group=copy.deepcopy(original)
            if kind=='endpoint':group['base_url']='http://unregistered.invalid'
            if kind=='group-timeout':group['group_timeout_s']+=1
            if kind=='request-deadline':group['requests'][0]['absolute_deadline_perf_counter_s']+=1
            if kind=='request-deadline':
                with self.subTest(kind=kind),self.assertRaises(ValueError):a.raw_metrics(group)
                continue
            for file in (folder/'formal-group.json',folder/'formal/group.json'):write(file,group)
            complete['artifacts']['formal-group.json'].update(sha256=a.sha(folder/'formal-group.json'),
                bytes=(folder/'formal-group.json').stat().st_size)
            write(folder/'complete.json',complete)
            self.complete['phases'][0]['result']['sha256']=a.sha(folder/'complete.json')
            write(self.owner/'points.json',dict(schema='E023-native-formal-points-v1',
                trials=self.complete['phases'],performance_qualified=False))
            with self.subTest(kind=kind),self.assertRaisesRegex(ValueError,'fixed local endpoint'):
                self.check()


class ResourceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.names=['r1-c1','r1-c8'];self.pid=90204;self.created=1000;self.owner_pid=91000
        self.children=[dict(phase=p,pid=92000+i,spawned_at=s,released_at=e,inner_released_at=e-10,returncode=-15)
            for i,(p,s,e) in enumerate(zip(self.names,(1010,1120),(1100,1210)))]
        labels=[('before-prepare',1001,None),('after-prepare',1002,None)]
        for child in self.children:
            s,e,pid=child['spawned_at'],child['released_at'],child['pid'];stage=child['phase']
            parts=[('pre-spawn',s-1,None),('spawned',s+1,pid),('ready',s+3,pid),('warm',s+4,pid),
                   ('after-warm',s+5,pid),('formal',s+6,pid),('after-formal',e-11,pid),
                   ('after-reap',e-9,None),('released',e+1,None)]
            labels.extend((stage+':'+label,wall,p) for label,wall,p in parts)
            write(self.root/'phases'/stage/'pre-spawn-absence.json',absence(self.pid))
        labels.append(('postflight',1220,None));self.rows=[]
        for i,(label,wall,pid) in enumerate(labels):
            entry=dict(pid=pid,ppid=self.owner_pid,owned=True,authorized_suspended=False)
            exact=dict(status='unknown',pid=None,raw=None) if pid is None else dict(status='known',pid=pid,ppid=self.owner_pid,
                raw=dict(returncode=0,stdout=f'{pid} {self.owner_pid} 1024 python'))
            self.rows.append(dict(label=label,recorded_at=datetime.fromtimestamp(wall,timezone.utc).isoformat(),monotonic_s=wall,
                owned_model_pid=pid,pressure_level=dict(returncode=0,text='1'),pressure_level_value=1,
                memory_pressure=dict(returncode=0,text='free percentage: 70'),free_percent=70,
                swap=dict(returncode=0,text='used = 12.00M'),swap_used_mb=12.,
                vm_stat=dict(returncode=0,text='Swapouts: 0\nPageouts: 0\nSwapins: 10\n'),vm_counters=dict(Swapouts=0,Pageouts=0,Swapins=10),
                power_metadata=dict(source='AC Power',raw=dict(returncode=0,stdout='AC Power')),
                processinfo_thermal_metadata=dict(code=0,raw=dict(returncode=0,stdout='0')),
                process_inventory_ok=True,native_process_inventory_ok=True,no_active_foreign_native_processes=True,
                only_owned_known_inference_processes=True,authorized_suspended_pids=[],
                native_inference_processes=[entry] if pid else [],known_inference_processes=[entry] if pid else [],exact_owned_pid_metadata=exact))
            write(self.root/f'absence-{i:03d}.json',dict(label=label,time=wall-.01,resource_index_if_sampled=i,**absence(self.pid)))
        self.complete=dict(owner_pid=self.owner_pid,samples=len(self.rows),absence_probes=len(self.rows));self.save()
    def save(self):
        totals=[]
        for i,row in enumerate(self.rows):
            write(self.root/f'resource-{i:03d}.json',row);item=copy.deepcopy(row)
            if i:
                item['swap_used_mb_delta']=row['swap_used_mb']-self.rows[i-1]['swap_used_mb']
                item['vm_counter_delta']={k:v-self.rows[i-1]['vm_counters'][k] for k,v in row['vm_counters'].items()}
            totals.append(item)
        write(self.root/'resources.json',totals)
    def check(self):return a.resources(self.root,self.complete,self.children,self.pid,self.created,self.names)

    def test_real_warm_formal_labels_with_exact_current_original_pid(self):
        self.assertEqual(set(self.check()['known_owned_samples']),set(self.names))
        with self.assertRaises(ValueError):a.absence(absence(74384),self.pid)

    def test_wrong_pid_stale_time_foreign_or_lost_server_identity_rejected(self):
        i=next(i for i,r in enumerate(self.rows) if r['label']=='r1-c8:formal');original=copy.deepcopy(self.rows[i])
        for kind in ('pid','ppid','time','foreign','missing'):
            row=self.rows[i]=copy.deepcopy(original)
            if kind=='pid':row['owned_model_pid']-=1
            if kind=='ppid':row['exact_owned_pid_metadata']['ppid']-=1
            if kind=='time':row['recorded_at']=datetime.fromtimestamp(1001,timezone.utc).isoformat()
            if kind=='foreign':row['native_inference_processes'][0]['owned']=False
            if kind=='missing':row['exact_owned_pid_metadata']=dict(status='unknown',pid=row['owned_model_pid'],raw={'returncode':1,'stdout':''})
            self.save()
            with self.subTest(kind=kind),self.assertRaises(ValueError):self.check()

    def test_original_pressure_thermal_sustained_swap_and_empty_suspension_gates(self):
        original=copy.deepcopy(self.rows)
        for kind in ('pressure','thermal','swap','suspended','legacy-label'):
            self.rows=copy.deepcopy(original)
            if kind=='pressure':self.rows[0]['pressure_level_value']=2
            if kind=='thermal':self.rows[0]['processinfo_thermal_metadata']['code']=2
            if kind=='swap':
                for i in (1,2):self.rows[i]['swap_used_mb']=12+70*i;self.rows[i]['swap']['text']=f'used = {12+70*i}.00M'
            if kind=='suspended':self.rows[0]['authorized_suspended_pids']=[self.pid]
            if kind=='legacy-label':self.rows[5]['label']='c1-capture:capturing'
            self.save()
            with self.subTest(kind=kind),self.assertRaises(ValueError):self.check()


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
    def test_legitimate_empty_sources_and_changed_prepost_model_proof(self):
        owner,wrapper,numeric,digest,empty=pins_cpu.fixture(self.root)
        support=a.legacy()
        with patch.object(a,'legacy',return_value=support),patch.object(support,'BASE_PLAN_SHA',digest):
            self.assertEqual(a.preparation_pins(owner,wrapper,numeric)['verified_entries_each'],157)
            proof=owner/'postflight/verified-model-pins.json';row=a.read(proof)
            next(iter(row.values()))['mtime_ns']+=1;write(proof,row)
            with self.assertRaises(ValueError):a.preparation_pins(owner,wrapper,numeric)
    def test_missing_or_failed_terminal_never_opens_protocol_or_model_evidence(self):
        owner=self.root/'owner';journal=self.root/'journal'
        for state,status,expected in (('PREPARED','INCOMPLETE','INCOMPLETE'),('RESTORED','FAIL','FAIL')):
            write(journal/'journal.json',dict(state=state));write(owner/'complete.json',dict(status=status,errors=['CPU only']))
            with patch.object(a,'load',side_effect=AssertionError('no audit backend before terminal')):
                result=a.audit(owner,journal,'0'*64)
            self.assertEqual(result['status'],expected)
            self.assertFalse(result['qualified'] or result['performance_qualified'] or result['http_raw_opened'])


if __name__=='__main__':unittest.main()
