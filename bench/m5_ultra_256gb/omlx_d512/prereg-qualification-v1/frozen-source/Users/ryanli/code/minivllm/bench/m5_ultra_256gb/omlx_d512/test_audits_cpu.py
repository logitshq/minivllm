"""Targeted offline mechanisms only; fake server + real frozen client/raw join.

No service/listener/MLX/model. Tests are written during the device-run embargo;
execute only after the root explicitly releases that embargo.
"""
import base64
import copy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bench.m5_ultra_256gb.omlx_d512 import audit_qualification as q, audit_resources as r
from bench.m5_ultra_256gb.omlx_d512 import trial as t, join_v2 as join
from bench.m5_ultra_256gb.omlx_d512.test_v2_cpu import Backend,PID,fake_spec,sealed,fixture


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,allow_nan=False)+'\n')


def refresh_artifacts(folder):
    complete=q.read(folder/'complete.json')
    complete['artifacts']={str(p.relative_to(folder)):dict(sha256=q.sha(p),bytes=p.stat().st_size)
                          for p in folder.rglob('*') if p.is_file() and p!=folder/'complete.json'}
    write(folder/'complete.json',complete)


class RecordingBackend(Backend):
    """Same fake process/MockTransport as producer tests; preserve actual raw commands."""
    async def port(self,port,pid=None):
        result=await super().port(port,pid)
        command=['/usr/sbin/lsof','-nP','-a']
        if pid is not None:command+=['-p',str(pid)]
        result['command']=command+[f'-iTCP:{port}','-sTCP:LISTEN','-Fpn'];return result
    async def identity(self,pid):
        result=await super().identity(pid)
        result['command']=['/bin/ps','-p',str(pid),'-o','pid=,ppid=,lstart=,comm='];return result
    async def absent(self,pid):
        result=await super().absent(pid)
        result['command']=['/bin/ps','-p',str(pid),'-o','pid='];return result


class QualificationTests(unittest.IsolatedAsyncioTestCase):
    async def make_trial(self,C=1):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        folder=Path(tmp.name).resolve()/'trial';spec,data=fake_spec(folder,C)
        spec['sources'][str(t.HERE/'serve_probe_v2.py')]=q.PROBE_SHA
        data[1][0]['probe_sha256']=q.PROBE_SHA
        backend=RecordingBackend(data)
        with patch.object(t,'validate'):
            produced=await t.run(spec,folder,lambda *_:None,backend=backend)
        self.assertEqual(produced['status'],'PASS',produced['errors'])
        release=q.read(folder/'release.json');outer={**release,'time':release['time']+.001}
        return folder,spec,outer

    def check(self,folder,spec,outer):
        cap=t.load('../pareto_realistic/http_capture_v2.py')
        return q.trial_evidence(folder,spec,PID,os.getpid(),outer,t,join,cap,lambda ids:'a'*len(ids))

    async def test_real_producer_artifacts_wire_and_explained_term_close_both_C(self):
        for C in (1,8):
            folder,spec,outer=await self.make_trial(C)
            value=self.check(folder,spec,outer)
            self.assertEqual(value['requests'],8 if C==1 else 16)
            self.assertEqual(value['max_http_outstanding'],C)
            self.assertEqual(value['exit_evidence'],dict(server_term_events=1,server_returncode=-15))
            self.assertFalse(value['observer_closed'] or value['generator_eof_proven'])
            self.assertEqual(value['raw_output_count'],2*value['requests'])
            self.assertEqual(value['committed_output_count'],value['requests'])

    async def test_missing_or_forged_preterm_seal_rejected_after_artifact_rehash(self):
        folder,spec,outer=await self.make_trial()
        seal=q.read(folder/'request-seal.json');seal['prefix_sha256']='0'*64
        write(folder/'request-seal.json',seal);refresh_artifacts(folder)
        with self.assertRaisesRegex(ValueError,'pre-TERM request seal'):self.check(folder,spec,outer)

    async def test_wrong_server_identity_and_unexplained_sigterm_rejected(self):
        folder,spec,outer=await self.make_trial()
        with self.assertRaises(ValueError):self.check(folder,spec,{**outer,'pid':PID+1})
        release=q.read(folder/'release.json');release['term_sent']=False
        write(folder/'release.json',release);refresh_artifacts(folder)
        with self.assertRaisesRegex(ValueError,'trial-owned SIGTERM'):self.check(folder,spec,outer)

    async def test_raw_usage_cannot_be_replaced_by_claimed_join_pass(self):
        folder,spec,outer=await self.make_trial()
        group=q.read(folder/'group.json');row=group['requests'][0]
        raw=base64.b64decode(row['raw_sse_chunks'][0]['body_bytes_base64'])
        frames=raw.decode().split('\n\n');usage=json.loads(frames[-3][6:])
        usage['usage']['completion_tokens']+=1;usage['usage']['total_tokens']+=1
        frames[-3]='data: '+json.dumps(usage);raw='\n\n'.join(frames).encode()
        row['raw_sse_chunks'][0]['body_bytes_base64']=base64.b64encode(raw).decode();row['raw_body_bytes']=len(raw)
        write(folder/'group.json',group);write(folder/'client/group.json',group);refresh_artifacts(folder)
        with self.assertRaises(ValueError):self.check(folder,spec,outer)

    async def test_endpoint_deadline_and_client_group_binding_are_real(self):
        folder,spec,outer=await self.make_trial()
        original=q.read(folder/'group.json')
        for change in ('endpoint','deadline','cleanup'):
            group=copy.deepcopy(original)
            if change=='endpoint':group['base_url']='http://127.0.0.1:19999'
            elif change=='deadline':group['request_timeout_s']+=1
            else:group['cleanup']['client_close_started']=False
            write(folder/'group.json',group);write(folder/'client/group.json',group);refresh_artifacts(folder)
            with self.subTest(change=change),self.assertRaises(ValueError):self.check(folder,spec,outer)

    def test_immediate_eos_and_eos_at_cap_exclude_once_not_lithos_usage_rule(self):
        for at_cap in (False,True):
            data=fixture(1,at_cap);group,events,jobs,fixtures=data
            batches=[x for x in events if x['event']=='batch']
            for event in list(batches):
                if not at_cap and not event['outputs'][0]['finished']:
                    events.remove(event);continue
                if event['outputs'][0]['finished']:
                    for token,output in zip(event['tokens'],event['outputs']):
                        token.update(token=99,finish_reason='stop')
                        output.update(finish_reason='stop',completion_tokens=511 if at_cap else 0,
                                      committed_ids=[7]*511 if at_cap else [])
            for row in group['requests']:
                raw=base64.b64decode(row['raw_sse_chunks'][0]['body_bytes_base64'])
                frames=raw.decode().split('\n\n')
                first=json.loads(frames[0][6:]);first['choices'][0].update(text='a'*511 if at_cap else '',finish_reason='stop')
                last=json.loads(frames[1][6:]);last['usage'].update(completion_tokens=511 if at_cap else 0,total_tokens=513 if at_cap else 2)
                raw=('data: '+json.dumps(first)+'\n\ndata: '+json.dumps(last)+'\n\ndata: [DONE]\n\n').encode()
                row['raw_sse_chunks'][0]['body_bytes_base64']=base64.b64encode(raw).decode();row['raw_body_bytes']=len(raw)
                wire=next(x for x in events if x['event']=='wire' and x['response_id']==first['id'])
                wire.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
            value=join.join(group,sealed(data),jobs,fixtures,lambda ids:'a'*len(ids),'a'*64,
                expected_official_args=['serve','--cpu-mock'],expected_model_path=events[0]['model_path'],expected_pid=PID)
            self.assertEqual(value['committed_output_count'],8*(511 if at_cap else 0))
            self.assertEqual(value['raw_output_count']-value['committed_output_count'],8)

    def test_missing_terminal_evidence_does_not_open_runtime_or_requests(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(q,'support',side_effect=AssertionError('must not load')):
            value=q.audit(directory)
        self.assertEqual(value['status'],'INCOMPLETE');self.assertFalse(value['qualified'] or value['raw_requests_opened'])

    def test_kill_bool_or_mismatched_returncode_never_qualifies(self):
        for inner,outer in ((dict(returncode=-9,term_sent=True),dict(returncode=-9)),
                            (dict(returncode=False,term_sent=False),dict(returncode=0)),
                            (dict(returncode=-15,term_sent=True),dict(returncode=0))):
            with self.assertRaises(ValueError):q.explained_release(inner,outer)

    def test_four_real_supplied_groups_zero_byte_source_and_changed_postflight(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'empty.py';source.write_bytes(b'')
            pin=dict(sha256=q.sha(source),bytes=0,mtime_ns=source.stat().st_mtime_ns)
            groups={group:{str(source):dict(pin)} for group in q.GROUPS}
            plan=dict(verification=dict(python='/cpu-fixture-only',groups=groups),execution=dict(pins={str(source):pin['sha256']}))
            verified={group:{str(source):dict(**pin,inode=1,device=1)} for group in q.GROUPS}
            for phase in ('preflight','postflight'):
                folder=root/phase
                write(folder/'verification-plan.json',plan['verification']);write(folder/'verified-pins.json',verified)
                write(folder/'complete.json',dict(status='PASS',groups=list(q.GROUPS),files=4,recursive_runtime_coverage_verified=False))
                copied=folder/'frozen-sources'/str(source).lstrip('/');copied.parent.mkdir(parents=True);copied.write_bytes(b'')
            self.assertFalse(q.preparations(root,plan)['recursive_runtime_coverage_verified'])
            changed=copy.deepcopy(verified);changed['target_model'][str(source)]['inode']=2
            write(root/'postflight/verified-pins.json',changed)
            with self.assertRaisesRegex(ValueError,'pre/post pin identities'):q.preparations(root,plan)


def absence(pid):
    commands=[(['/bin/ps','-p',str(pid),'-o','pid='],1),(['/bin/launchctl','print',r.JOB],113),
              (['/usr/sbin/lsof','-nP','-a','-iTCP:8001','-sTCP:LISTEN','-Fpn'],1)]
    return dict(status='PASS',checks=[dict(command=c,returncode=rc,stdout_sha256=hashlib.sha256(b'').hexdigest(),
        stdout_bytes=0,absent=True) for c,rc in commands])


class ResourceTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)
        self.owner_pid=91000;self.original=90204;self.created=1000
        self.children=[dict(phase=p,pid=92000+i,spawned_at=s,released_at=e,inner_released_at=e-10)
                       for i,(p,s,e) in enumerate(zip(r.PHASES,(1010,1120),(1100,1210)))]
        labels=[('before-prepare',1001,None),('after-prepare',1002,None)]
        for child in self.children:
            s,e,pid=child['spawned_at'],child['released_at'],child['pid'];stage=child['phase']
            labels.extend((stage+':'+name,wall,p) for name,wall,p in (
                ('pre-spawn',s-1,None),('spawned',s+1,pid),('ready',s+3,pid),('qualification',s+6,pid),
                ('after-qualification',e-11,pid),('after-reap',e-9,None),('released',e+1,None)))
            write(self.root/'phases'/stage/'pre-spawn-absence.json',absence(self.original))
        labels.append(('postflight',1220,None));self.rows=[]
        for i,(label,wall,pid) in enumerate(labels):
            exact=dict(status='unknown',pid=None,raw=None) if pid is None else dict(status='known',pid=pid,ppid=self.owner_pid,
                raw=dict(command=['/bin/ps','-p',str(pid),'-o','pid=,ppid=,rss=,comm='],returncode=0,stdout=f'{pid} {self.owner_pid} 1024 python',stderr=''))
            entry=dict(pid=pid,ppid=self.owner_pid,owned=True,authorized_suspended=False)
            self.rows.append(dict(label=label,recorded_at=datetime.fromtimestamp(wall,timezone.utc).isoformat(),monotonic_s=wall,
                owned_model_pid=pid,pressure_level=dict(returncode=0,text='1'),pressure_level_value=1,
                memory_pressure=dict(returncode=0,text='free percentage: 70'),free_percent=70,
                swap=dict(returncode=0,text='used = 12.00M'),swap_used_mb=12.,vm_stat=dict(returncode=0,text='Swapouts: 0\nPageouts: 0\n'),
                vm_counters=dict(Swapouts=0,Pageouts=0),power_metadata=dict(source='AC Power',raw=dict(returncode=0,stdout='AC Power')),
                processinfo_thermal_metadata=dict(code=0,raw=dict(returncode=0,stdout='0')),process_inventory_ok=True,
                native_process_inventory_ok=True,no_active_foreign_native_processes=True,only_owned_known_inference_processes=True,
                authorized_suspended_pids=[],native_inference_processes=[entry] if pid else [],known_inference_processes=[entry] if pid else [],
                exact_owned_pid_metadata=exact))
            write(self.root/f'absence-{i:03d}.json',dict(label=label,time=wall-.01,resource_index_if_sampled=i,**absence(self.original)))
        self.complete=dict(owner_pid=self.owner_pid,samples=len(self.rows),absence_probes=len(self.rows));self.save()
    def save(self):
        aggregate=[]
        for i,row in enumerate(self.rows):
            write(self.root/f'resource-{i:03d}.json',row);item=copy.deepcopy(row)
            if i:item.update(swap_used_mb_delta=row['swap_used_mb']-self.rows[i-1]['swap_used_mb'],
                             vm_counter_delta={k:v-self.rows[i-1]['vm_counters'][k] for k,v in row['vm_counters'].items()})
            aggregate.append(item)
        write(self.root/'resources.json',aggregate)
    def check(self):return r.resources(self.root,self.complete,self.children,self.original,self.created,r.PHASES)
    def test_two_actual_server_lifetimes_and_current_original_pid(self):
        self.assertEqual(set(self.check()['known_owned_samples']),set(r.PHASES))
        with self.assertRaises(ValueError):r.absence(absence(74384),self.original)
    def test_wrong_server_stale_phase_foreign_or_pressure_rejected(self):
        original=copy.deepcopy(self.rows);i=5
        for kind in ('pid','ppid','time','foreign','pressure','suspension'):
            self.rows=copy.deepcopy(original);row=self.rows[i]
            if kind=='pid':row['owned_model_pid']+=1
            if kind=='ppid':row['exact_owned_pid_metadata']['ppid']+=1
            if kind=='time':row['recorded_at']=datetime.fromtimestamp(1000,timezone.utc).isoformat()
            if kind=='foreign':row['native_inference_processes'][0]['owned']=False
            if kind=='pressure':row['pressure_level_value']=True
            if kind=='suspension':row['authorized_suspended_pids']=[self.original]
            self.save()
            with self.subTest(kind=kind),self.assertRaises(ValueError):self.check()
    def test_filename_sequence_beyond_999_and_missing_duplicate_index(self):
        names=[SimpleNamespace(name=f'resource-{i:03d}.json') for i in range(1002)]
        folder=SimpleNamespace(glob=lambda _:names)
        r.sequence_names(folder,'resource',1002)
        names[-1]=SimpleNamespace(name='resource-1000.json')
        with self.assertRaises(ValueError):r.sequence_names(folder,'resource',1002)


if __name__=='__main__':unittest.main()
