"""Offline audit deltas over real LogWindows and frozen-client CPU wire fixtures.

Qualification binding is explicitly mocked. No process, socket, model, service
or resource action is performed; these are not whole-cycle performance proofs.
"""
import base64
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bench.m5_ultra_256gb.lithos_d512_formal import audit_formal as audit, contract, formal
from bench.m5_ultra_256gb.lithos_d512_formal import test_formal_cpu as fixture


class EvidenceBackend(fixture.Backend):
    """Fake direct child with the exact command-bearing OS receipt schema."""
    async def identity(self,pid):
        return dict(**await super().identity(pid),command=['/bin/ps','-p',str(pid),'-o','pid=,ppid=,lstart=,comm='])
    async def absent(self,pid):
        return dict(**await super().absent(pid),command=['/bin/ps','-p',str(pid),'-o','pid='])
    async def port(self,port,pid=None):
        command=['/usr/sbin/lsof','-nP','-a']+(['-p',str(pid)] if pid else [])+[f'-iTCP:{port}','-sTCP:LISTEN','-Fpn']
        return dict(**await super().port(port,pid),command=command)


class ActualSpecBackend(EvidenceBackend):
    """Actual closed metadata/spec; only child/HTTP responses are synthetic."""
    async def group(self,spec,phase,output):
        client=formal.load('../pareto_realistic/lithos_client.py');submitted=[]
        async def handler(request):
            body=json.loads(request.content);i=len(submitted);submitted.append(body)
            f=next(f for f in spec['fixtures'] if f['messages']==body['messages'])
            stop=i%3!=2;n=1 if i%3==0 else 512;text='a'*(n-int(stop))
            final=dict(model=spec['served_model'],created_at=f'CPU-{phase}-{i}',message=dict(role='assistant',content=''),
                done=True,done_reason='stop' if stop else 'length',total_duration=100,load_duration=0,
                prompt_eval_count=len(f['prompt_ids']),prompt_eval_duration=10,eval_count=n,eval_duration=20)
            raw=(json.dumps(dict(model=spec['served_model'],created_at='CPU',message=dict(role='assistant',content=text),done=False))+'\n'+json.dumps(final)+'\n').encode()
            self.stderr.write(f"INFO:__main__:Generation: {{'cached_prompt_tokens': 0, 'setup_ms': {1.0 if phase=='warm' and i==0 else 0.0}, 'verify_tokens': 1}}\n".encode());self.stderr.flush()
            return fixture.httpx.Response(200,stream=fixture.Stream(raw),headers={'content-type':'application/x-ndjson'})
        d=spec['deadlines']
        return await client.run_group(f"http://127.0.0.1:{spec['port']}",spec['jobs'][phase],spec['concurrency'],phase=phase,
            qualification=contract.count_contract(spec,phase),headers={'Authorization':'Bearer '+spec['api_key']},
            transport=fixture.httpx.MockTransport(handler),request_timeout_s=d['request'],group_timeout_s=d[phase],
            cleanup_timeout_s=d['close'],output_dir=output)


class Tests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.out=self.root/'trial'
        self.spec,group,events=fixture.fake_spec(self.out)
        self.spec['log_contract']={'max_bytes':formal.MAX_LOG_BYTES}
        self.backend=fixture.Backend(self.spec,group,events)
        real_load=formal.load
        self.patches=[patch.object(formal,'validate',return_value=None),
            patch.object(formal,'load',side_effect=lambda name:contract if name=='contract.py' else real_load(name)),
            patch.object(contract,'bind',side_effect=lambda *args:({},list(fixture.fixtures.fixture()[3].values()),{}, {'scope':'CPU-mock'}))]
        for p in self.patches:p.start();self.addCleanup(p.stop)
        async def checkpoint(label,pid):pass
        self.result=await formal.run(self.spec,self.out,checkpoint,backend=self.backend)
        self.assertEqual(self.result['status'],'PASS',self.result)
        self.groups={phase:audit.read(self.out/(phase+'-group.json')) for phase in ('warm','formal')}
        self.join=real_load('../lithos_d512_retry/join.py')

    def verify_logs(self):return audit.log_windows(self.out,self.spec,self.groups,self.join)

    async def test_real_producer_windows_reparse_and_committed_eos_metrics(self):
        checked=self.verify_logs()
        self.assertEqual(checked['warm']['count'],8);self.assertEqual(checked['formal']['count'],64)
        self.assertEqual(checked['all_generations']['count'],72)
        self.assertFalse(checked['per_wire_request_id_bound'])
        for phase,group in self.groups.items():
            raw=contract.check_group(group,self.spec,phase)
            actual=audit.metrics(group,self.join.wire)
            self.assertEqual(actual,{key:raw['point'][key] for key in actual})
            self.assertEqual([row['committed_output_tokens'] for row in raw['requests'][:3]],[0,511,512])
        self.assertFalse(self.result['performance_qualified'])

    async def test_missing_log_and_different_original_slice_reject(self):
        path=self.out/'formal-generation.log';original=path.read_bytes();path.unlink()
        with self.assertRaises(FileNotFoundError):self.verify_logs()
        path.write_bytes(original.replace(b"'setup_ms': 0.0",b"'setup_ms': 1.0",1))
        with self.assertRaisesRegex(ValueError,'original byte slice'):self.verify_logs()

    async def test_changed_prefix_boundary_and_clock_reject(self):
        path=self.out/'warm-end-log-boundary.json';original=audit.read(path)
        changed=copy.deepcopy(original);changed['prefix_sha256']='0'*64;formal.write(path,changed)
        with self.assertRaisesRegex(ValueError,'unchanged full line prefix'):self.verify_logs()
        formal.write(path,original)
        start=audit.read(self.out/'warm-start-log-boundary.json')
        changed=copy.deepcopy(original);changed['perf_counter_s']=start['perf_counter_s']
        formal.write(path,changed)
        with self.assertRaisesRegex(ValueError,'brackets complete POST/DONE/drain'):self.verify_logs()

    async def test_raw_post_done_tail_rejected_independent_of_cached_metrics(self):
        group=copy.deepcopy(self.groups['formal']);row=group['requests'][0]
        chunk=row['raw_ndjson_chunks'][-1]
        raw=base64.b64decode(chunk['body_bytes_base64'])+b'x'
        chunk['body_bytes_base64']=base64.b64encode(raw).decode();row['raw_body_bytes']+=1
        with self.assertRaisesRegex(ValueError,'tail bytes after DONE'):audit.metrics(group,self.join.wire)
        with self.assertRaises(ValueError):contract.check_group(group,self.spec,'formal')

    async def test_missing_terminal_owner_does_not_open_request_evidence(self):
        root=self.root/'not-closed';root.mkdir()
        result=audit.audit(root,self.root/'not-restored','0'*64)
        self.assertEqual(result['status'],'INCOMPLETE')
        self.assertFalse(result['raw_requests_opened'] or result['qualified'] or result['performance_qualified'])


class ClosedTrialTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_closed_metadata_run_to_full_trial_evidence(self):
        from bench.m5_ultra_256gb.lithos_d512_formal.test_contract_cpu import binding
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'trial';C=8;key='CPU-closed-trial'
            env=formal.load('../lithos_d512_retry/serve_probe.py').specification(out,C,18029,key,observed=False)['environment']
            deadlines=dict(ready=180,request=600,warm=720,formal=2400,close=10,term=15,kill=15,overall=3600,resource_interval=.01)
            spec=formal.prepare(out,C,1,18029,key,qualification=binding(),deadlines=deadlines,environment=env)
            backend=ActualSpecBackend(spec,None,None)
            async def checkpoint(label,pid):pass
            complete=await formal.run(spec,out,checkpoint,backend=backend)
            self.assertEqual(complete['status'],'PASS',complete)
            own=dict(phase='r1-c8',pid=backend.child.pid,owner_pid=os.getpid(),
                spawned_at=audit.read(out/'child.json')['time'])
            formal.write(root/'child.json',own)
            release=audit.read(out/'release.json')
            formal.write(root/'release.json',dict(**{k:v for k,v in release.items() if k!='time'},time=complete['ended_at']+.001))
            child={'phase':'r1-c8','pid':backend.child.pid}
            checked=audit.trial_evidence(out,spec,child,os.getpid(),formal,contract)
            self.assertEqual(checked['requests'],64);self.assertEqual(checked['warm_requests_checked'],8)
            self.assertEqual(checked['max_http_outstanding'],8);self.assertEqual(checked['n_fresh_processes'],1)
            self.assertEqual(checked['aggregate_output_tok_s'],complete['point']['aggregate_output_tok_s'])
            self.assertEqual(checked['log_windows']['all_generations']['count'],72)
            (out/'unlisted-evidence.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'complete retained artifact inventory'):
                audit.trial_evidence(out,spec,child,os.getpid(),formal,contract)


if __name__=='__main__':unittest.main()
