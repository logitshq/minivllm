"""Focused synthetic CPU audit mechanisms; no service/socket/model/MLX actions."""
import base64
import copy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bench.m5_ultra_256gb.omlx_d512_formal import audit_formal as a, audit_resources as r


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,allow_nan=False)+'\n')


def fixture(C=8,phase='formal'):
    fixtures=[dict(fixture_id=f'D{i}',prompt_text=f'完整 prompt {i}',prompt_ids=[i+1,99],normal_eos=True,max_output_tokens=512) for i in range(8)]
    count=8 if phase=='warm' else 64;jobs=[];rows=[]
    for i in range(count):
        f=fixtures[i%8];body=dict(model='Qwen3.8-27B-nvfp4',prompt=f['prompt_text'],temperature=0.0,top_p=1.0,
             repetition_penalty=1.0,max_tokens=512,stream=True,stream_options={'include_usage':True})
        job=dict(request_id=f'{phase}-{i}',fixture_id=f['fixture_id'],path='/v1/completions',body=body);jobs.append(job)
        n=(0,511,512)[i%3];finish='length' if n==512 else 'stop';rid=f'cmpl-{i:08x}'
        usage=dict(prompt_tokens=2,completion_tokens=n,total_tokens=2+n,prompt_tokens_details={'cached_tokens':0},
            total_time=1.0,time_to_first_token=0.1,prompt_eval_duration=0.1,generation_duration=0.9,
            prompt_tokens_per_second=20.0,generation_tokens_per_second=round(n/0.9,2))
        first=dict(id=rid,object='text_completion',model=body['model'],choices=[dict(index=0,text='a'*n,finish_reason=finish)])
        last=dict(id=rid,object='text_completion',model=body['model'],choices=[],usage=usage)
        payload=('data: '+json.dumps(first)+'\n\ndata: '+json.dumps(last)+'\n\ndata: [DONE]\n\n').encode()
        start=10.+2*(i//C);done=start+1
        rows.append(dict(**copy.deepcopy(job),status='done',http_status=200,done=True,error=None,
            validation_errors=[],count_qualification_errors=[],transport_valid=True,native_fields={},answer='a'*n,reasoning='',
            response_ids=[rid],usage=usage,finish_reason=finish,qualified_output_tokens=n,
            reported_prompt_tokens=2,reported_completion_tokens=n,reported_cached_prompt_tokens=0,
            start_perf_counter_s=start,done_perf_counter_s=done,terminal_perf_counter_s=done,e2e_s=1.,
            absolute_deadline_perf_counter_s=start+600,raw_body_bytes=len(payload),raw_body_truncated=False,
            raw_sse_chunks=[dict(client_perf_counter_s=done,body_bytes_base64=base64.b64encode(payload).decode())]))
    q=dict(status='qualified',receipt_sha256='a'*64,concurrency=C,normal_eos=True,
        usage_semantics='committed_target_output_excluding_terminal_including_reasoning',usage_count_rule='identity',raw_ids_count_rule='none',
        request_fixture_map={j['request_id']:j['fixture_id'] for j in jobs},
        fixture_contracts={j['fixture_id']:dict(path=j['path'],body_sha256=a.canonical_sha(j['body'])) for j in jobs})
    spec=dict(concurrency=C,repeat=1,port=18029,served_model='Qwen3.8-27B-nvfp4',fixtures=fixtures,
              jobs={phase:jobs},deadlines=dict(request=600,warm=720,formal=2400,close=10))
    end=rows[-1]['done_perf_counter_s'];total=sum(x['qualified_output_tokens'] for x in rows);window=end-10
    group=dict(schema='E023-http-group-v1',phase=phase,concurrency=C,qualification=q,qualification_errors=[],group_error=None,
        base_url='http://127.0.0.1:18029',request_timeout_s=600,group_timeout_s=spec['deadlines'][phase],
        barrier_release_perf_counter_s=9.,group_deadline_perf_counter_s=9.+spec['deadlines'][phase],requests=rows,
        connection_policy=dict(max_connections=2*C,max_keepalive_connections=0,retries=0,follow_redirects=False),
        cleanup=dict(complete=True,deadline_exceeded=False,errors=[],pending_tasks=[],cancelled_again=False,client_close_started=True,
            adapter='native-close-cancellation-v1',frozen_client_sha256=a.CLIENT_SHA,budget_s=10,
            start_perf_counter_s=end+.1,end_perf_counter_s=end+.2),
        summary=dict(transport_complete=True,eligible_under_supplied_qualification=True,formal_point_eligible=phase=='formal',
            planned_requests=count,submitted_requests=count,done_requests=count,successful_requests=count,count_qualified_requests=count,
            group_window_s=window,aggregate_output_tok_s=total/window,e2e_s=dict(p50=1.,p95=1.),max_http_outstanding=C))
    return group,spec,q


def change_wire(row,change):
    raw=base64.b64decode(row['raw_sse_chunks'][0]['body_bytes_base64']);parts=raw.decode().split('\n\n')
    first,last=(json.loads(p[6:]) for p in parts[:2]);change(first,last)
    raw=('data: '+json.dumps(first)+'\n\ndata: '+json.dumps(last)+'\n\ndata: [DONE]\n\n').encode()
    row['raw_sse_chunks'][0]['body_bytes_base64']=base64.b64encode(raw).decode();row['raw_body_bytes']=len(raw)


class RawTests(unittest.TestCase):
    def test_complete_eos_zero_and_at_cap_usage_identity_no_subtraction(self):
        for C in (1,8):
            for phase in ('warm','formal'):
                group,spec,q=fixture(C,phase);result=a.raw_group(group,spec,phase,q)
                self.assertEqual([x['committed_output_tokens'] for x in result['requests'][:3]],[0,511,512])
                self.assertEqual(result['point']['max_http_outstanding'],C)
                self.assertFalse(result['generator_eof_proven']);self.assertEqual(result['raw_ids_count_rule'],'none')

    def test_changed_raw_usage_rejects_even_when_claimed_point_pass(self):
        group,spec,q=fixture();change_wire(group['requests'][1],lambda f,u:u['usage'].update(completion_tokens=510,total_tokens=512))
        with self.assertRaisesRegex(ValueError,'usage equals'):a.raw_group(group,spec,'formal',q)

    def test_omitted_cache_nonzero_cache_bool_and_invalid_count_reject(self):
        for variant in ('missing','cached','bool','stop512','length511','timing-negative','timing-bool','timing-unknown','timing-null'):
            group,spec,q=fixture();row=group['requests'][1]
            def mutate(first,last):
                u=last['usage']
                if variant=='missing':u.pop('prompt_tokens_details')
                elif variant=='cached':u['prompt_tokens_details']['cached_tokens']=1
                elif variant=='bool':u['prompt_tokens']=True
                elif variant=='stop512':u.update(completion_tokens=512,total_tokens=514)
                elif variant=='length511':first['choices'][0]['finish_reason']='length'
                elif variant=='timing-negative':u['total_time']=-1.0
                elif variant=='timing-bool':u['total_time']=True
                elif variant=='timing-unknown':u['unreported_timing']=1.0
                else:u['total_time']=None
            change_wire(row,mutate)
            with self.subTest(variant=variant),self.assertRaises((ValueError,KeyError)):a.raw_group(group,spec,'formal',q)

    def test_original_prompt_binding_rejects_double_template_even_matching_body_hash(self):
        group,spec,q=fixture();job=spec['jobs']['formal'][0]
        job['body']['prompt']='<user>'+job['body']['prompt']+'</user>'
        group['requests'][0]['body']=copy.deepcopy(job['body'])
        q['fixture_contracts'][job['fixture_id']]['body_sha256']=a.canonical_sha(job['body'])
        with self.assertRaisesRegex(ValueError,'frozen full prompt'):a.raw_group(group,spec,'formal',q)

    def test_retained_done_tail_duplicate_id_and_malformed_bytes_reject(self):
        for variant in ('tail','badutf8','duplicate'):
            group,spec,q=fixture();row=group['requests'][1]
            if variant=='duplicate':change_wire(row,lambda f,u:(f.update(id='cmpl-00000000'),u.update(id='cmpl-00000000')))
            else:
                raw=base64.b64decode(row['raw_sse_chunks'][0]['body_bytes_base64'])+(b'x' if variant=='tail' else b'\xff')
                row['raw_sse_chunks'][0]['body_bytes_base64']=base64.b64encode(raw).decode();row['raw_body_bytes']=len(raw)
            with self.subTest(variant=variant),self.assertRaises((ValueError,UnicodeError)):a.raw_group(group,spec,'formal',q)

    def test_serial_mislabeled_C8_and_bad_deadline_rejected(self):
        group,spec,q=fixture(1);group['concurrency']=spec['concurrency']=q['concurrency']=8
        group['connection_policy'].update(max_connections=16)
        with self.assertRaisesRegex(ValueError,'concurrency did not occur'):a.raw_group(group,spec,'formal',q)
        group,spec,q=fixture();group['requests'][0]['absolute_deadline_perf_counter_s']+=1
        with self.assertRaisesRegex(ValueError,'request clocks'):a.raw_group(group,spec,'formal',q)

    def test_close_cancel_error_or_close_before_done_rejected(self):
        for variant in ('cancel','pending','clock'):
            group,spec,q=fixture();clean=group['cleanup']
            if variant=='cancel':clean['errors']=[dict(type='CancelledError')]
            elif variant=='pending':clean['pending_tasks']=['close:response']
            else:clean['start_perf_counter_s']=10
            with self.subTest(variant=variant),self.assertRaises(ValueError):a.raw_group(group,spec,'formal',q)

    def test_unexplained_term_and_kill_reject(self):
        for code,own in ((-15,False),(-9,True),(True,True)):
            with self.assertRaises(ValueError):a.legacy.explained_release(dict(returncode=code,term_sent=own),dict(returncode=code))
        a.legacy.explained_release(dict(returncode=-15,term_sent=True),dict(returncode=-15))

    def test_missing_terminal_never_reads_raw_or_grants_performance(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=a.audit(Path(tmp)/'owner',Path(tmp)/'journal','0'*64)
            self.assertEqual(result['status'],'INCOMPLETE');self.assertFalse(result['raw_requests_opened'] or result['performance_qualified'])


def absence(pid):
    commands=[(['/bin/ps','-p',str(pid),'-o','pid='],1),(['/bin/launchctl','print',r.raw.JOB],113),
              (['/usr/sbin/lsof','-nP','-a','-iTCP:8001','-sTCP:LISTEN','-Fpn'],1)]
    return dict(status='PASS',checks=[dict(command=c,returncode=rc,stdout_sha256=hashlib.sha256(b'').hexdigest(),
        stdout_bytes=0,absent=True) for c,rc in commands])


class ResourceTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)
        self.owner_pid=91000;self.original=90204;self.created=1000
        self.children=[dict(phase=p,pid=92000+i,spawned_at=1010+i*110,released_at=1100+i*110,inner_released_at=1090+i*110) for i,p in enumerate(r.PHASES)]
        labels=[('before-prepare',1001,None),('after-prepare',1002,None)]
        for child in self.children:
            s,e,pid=child['spawned_at'],child['released_at'],child['pid'];stage=child['phase']
            labels.extend((stage+':'+name,wall,p) for name,wall,p in (
                ('pre-spawn',s-1,None),('spawned',s+1,pid),('ready',s+3,pid),('warm',s+6,pid),('after-warm',s+7,pid),
                ('formal',s+8,pid),('after-formal',e-11,pid),('after-reap',e-9,None),('released',e+1,None)))
            write(self.root/'phases'/stage/'pre-spawn-absence.json',absence(self.original))
        labels.append(('postflight',1660,None));self.rows=[]
        for i,(label,wall,pid) in enumerate(labels):
            exact=dict(status='unknown',pid=None,raw=None) if pid is None else dict(status='known',pid=pid,ppid=self.owner_pid,
                raw=dict(returncode=0,stdout=f'{pid} {self.owner_pid} 1024 python'))
            entry=dict(pid=pid,ppid=self.owner_pid,owned=True,authorized_suspended=False)
            self.rows.append(dict(label=label,recorded_at=datetime.fromtimestamp(wall,timezone.utc).isoformat(),monotonic_s=wall,
                owned_model_pid=pid,pressure_level=dict(returncode=0,text='1'),pressure_level_value=1,
                memory_pressure=dict(returncode=0,text='free percentage: 70'),free_percent=70,
                swap=dict(returncode=0,text='used = 12.00M'),swap_used_mb=12.,vm_stat=dict(returncode=0,text='Swapouts: 0\nPageouts: 0\n'),
                vm_counters=dict(Swapouts=0,Pageouts=0),power_metadata=dict(source='AC Power',raw=dict(returncode=0,stdout='AC Power')),
                processinfo_thermal_metadata=dict(code=0,raw=dict(returncode=0,stdout='0')),process_inventory_ok=True,
                native_process_inventory_ok=True,no_active_foreign_native_processes=True,only_owned_known_inference_processes=True,
                authorized_suspended_pids=[],native_inference_processes=[entry] if pid else [],known_inference_processes=[entry] if pid else [],exact_owned_pid_metadata=exact))
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
    def test_six_real_phase_labels_and_negative_pid_pressure_suspension(self):
        self.assertEqual(set(self.check()['known_owned_samples']),set(r.PHASES));original=copy.deepcopy(self.rows)
        for variant in ('pid','pressure','suspension','missingformal'):
            self.rows=copy.deepcopy(original);row=self.rows[7]
            if variant=='pid':row['owned_model_pid']+=1
            elif variant=='pressure':row['pressure_level_value']=True
            elif variant=='suspension':row['authorized_suspended_pids']=[self.original]
            else:row['label']=row['label'].replace(':formal',':warm')
            self.save()
            with self.subTest(variant=variant),self.assertRaises(ValueError):self.check()
    def test_numeric_indices_over_999_pass_sequence_not_lexicographic_reject(self):
        class StopAfterSequence(Exception):pass
        with patch.object(Path,'glob',side_effect=lambda pattern:[SimpleNamespace(name=f'{pattern.split("-")[0]}-{i:03d}.json') for i in range(1002)]), \
             patch.object(r.raw,'read',side_effect=StopAfterSequence):
            with self.assertRaises(StopAfterSequence):r.resources(self.root,dict(samples=1002,absence_probes=1002),[],1,0,[])


class ActualProducerShapeTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_formal_run_and_frozen_client_into_independent_trial_audit(self):
        from bench.m5_ultra_256gb.omlx_d512_formal import formal
        from bench.m5_ultra_256gb.omlx_d512_formal import test_formal_cpu as fixture_module

        class RawCommandBackend(fixture_module.Backend):
            async def port(self,port,pid=None):
                value=await super().port(port,pid)
                value['command']=['/usr/sbin/lsof','-nP','-a']+(['-p',str(pid)] if pid else [])+[f'-iTCP:{port}','-sTCP:LISTEN','-Fpn']
                return value
            async def identity(self,pid):
                return dict(**await super().identity(pid),command=['/bin/ps','-p',str(pid),'-o','pid=,ppid=,lstart=,comm='])
            async def absent(self,pid):
                return dict(**await super().absent(pid),command=['/bin/ps','-p',str(pid),'-o','pid='])

        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve()
            for C in (1,8):
                folder=root/f'c{C}'/'trial';spec,bound=fixture_module.fake_spec(folder,C);backend=RawCommandBackend()
                async def checkpoint(label,pid):pass
                with patch.object(formal,'validate'),patch.object(formal,'bind_qualification',return_value=bound):
                    complete=await formal.run(spec,folder,checkpoint,backend=backend)
                    self.assertEqual(complete['status'],'PASS',complete['errors'])
                    own=dict(phase=f'r1-c{C}',pid=backend.child.pid,owner_pid=os.getpid(),
                             spawned_at=a.read(folder/'child.json')['time'])
                    write(folder.parent/'child.json',own);release=a.read(folder/'release.json')
                    write(folder.parent/'release.json',{**release,'time':complete['ended_at']+.001})
                    actual=a.trial_evidence(folder,spec,dict(phase=f'r1-c{C}',pid=backend.child.pid),os.getpid(),formal)
                    self.assertEqual(actual['requests'],64);self.assertEqual(actual['warm_requests_checked'],8)
                    self.assertEqual(actual['max_http_outstanding'],C)
                    self.assertEqual(actual['aggregate_output_tok_s'],complete['point']['aggregate_output_tok_s'])
                    self.assertEqual(actual['exit_evidence'],dict(server_term_events=1,server_returncode=-15))
                    self.assertFalse(actual['closed_logs_semantically_verified'] or actual['generator_eof_proven'])
                    (folder/'unrecorded.json').write_text('{}')
                    with self.assertRaisesRegex(ValueError,'artifact inventory'):
                        a.trial_evidence(folder,spec,dict(phase=f'r1-c{C}',pid=backend.child.pid),os.getpid(),formal)


if __name__=='__main__':unittest.main()
