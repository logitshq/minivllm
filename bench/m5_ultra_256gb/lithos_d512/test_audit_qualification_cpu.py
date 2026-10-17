"""Closed synthetic receipts + real frozen raw join. No model/process/network.

Only plan admission/tokenizer are explicit fakes; raw NDJSON, observer/log join,
source-preparation receipt comparisons and direct-child evidence use the auditor.
These tests grant no device or actual runtime qualification.
"""
import base64
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from bench.m5_ultra_256gb.lithos_d512 import audit_qualification as a, test_cpu as f, trial


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,sort_keys=True)+'\n')
def reference(root,path): return dict(path=str(path.relative_to(root)),sha256=a.sha(path))
def raw(command,stdout='',code=1):return dict(command=command,returncode=code,stdout=stdout,stderr='')
def absence(pid):return raw(['/bin/ps','-p',str(pid),'-o','pid='])
def port_command(port,pid=None):
    return ['/usr/sbin/lsof','-nP','-a']+([] if pid is None else ['-p',str(pid)])+[f'-iTCP:{port}','-sTCP:LISTEN','-Fpn']


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.port=18029;self.pid=9900
        self.join=trial.load('join.py');self.cap=trial.load('../pareto_realistic/http_capture_v2.py')
        self.decode=lambda ids:''.join('a' if token==7 else '?' for token in ids)
        self.plan=dict(execution=dict(owner=dict(path='/CPU/owner.py',sha256=a.OWNER_SHA),pins={}),
            cold_restart=dict(predecessor={'CPU':'immutable'}),http=dict(port=self.port),
            verification=dict(python='/CPU/python',groups={g:{'/CPU/'+g:dict(sha256='a'*64,bytes=0,mtime_ns=0)} for g in a.GROUPS}))
        write(self.root/'owner-plan.json',self.plan);plan_sha=a.sha(self.root/'owner-plan.json')
        identity=dict(pid=self.pid,ppid=1,start='CPU owner',comm='python')
        stopped=dict(schema='E023-lithos-D512-cold-journal-v1',state='STOPPED',owner_directory=str(self.root),
            owner_plan=dict(path='/CPU/plan.json',sha256=plan_sha),owner_source=self.plan['execution']['owner'],
            id='CPU-journal',removed_at=1,trials=list(a.PHASES),predecessor=self.plan['cold_restart']['predecessor'])
        write(self.root/'stopped-journal.json',stopped)
        attempt=dict(schema='cold-service-owner-attempt-v1',owner_pid=self.pid,owner_identity=identity,
            owner_plan_sha256=plan_sha,owner_source=self.plan['execution']['owner'],journal_id='CPU-journal',
            removed_at=1,created_at=2,stopped_journal_sha256=a.sha(self.root/'stopped-journal.json'))
        write(self.root/'attempt.json',attempt)
        bound={k:attempt[k] for k in ('owner_pid','owner_identity','owner_source','owner_plan_sha256','journal_id','removed_at')}
        bound['attempt_sha256']=a.sha(self.root/'attempt.json')
        write(self.root/'spawn-intent.json',bound)
        for name in ('preflight','postflight'):
            folder=self.root/name
            write(folder/'verification-plan.json',self.plan['verification'])
            pins={g:{p:dict(**row,inode=10,device=1) for p,row in rows.items()} for g,rows in self.plan['verification']['groups'].items()}
            write(folder/'verified-pins.json',pins)
            write(folder/'complete.json',dict(status='PASS',groups=list(a.GROUPS),files=5))
        self.specs=[];children=[];stages=[]
        for i,(phase,C) in enumerate(zip(a.PHASES,(1,8))):
            cpid=9910+i;folder=self.root/'phases'/phase;out=folder/'trial';out.mkdir(parents=True)
            group,events,jobs,fixtures=f.fixture(C);events[0]['probe_sha256']=trial.FIXED['serve_probe.py']
            spec=dict(concurrency=C,port=self.port,served_model='CPU-model',command=['CPU-server',str(C)],
                cwd='/CPU/lithos',environment={'CPU':'ONLY'},jobs=jobs,fixtures=list(fixtures.values()),
                official_args=f.ARGS,pack_dir=f.PACK)
            self.specs.append(spec)
            write(out/'spec.json',spec);write(out/'launch.json',{k:spec[k] for k in ('command','cwd','environment')})
            write(out/'spawn-intent.json',dict(command=spec['command'],cwd=spec['cwd'],time=3+i*10))
            write(out/'child.json',dict(pid=cpid,time=4+i*10))
            write(out/'child-identity.json',raw(['/bin/ps','-p',str(cpid),'-o','pid=,ppid=,lstart=,comm='],f'{cpid} {self.pid} CPU time Python\n',0))
            write(out/'child-absence.json',absence(cpid))
            for name in ('port-before.json','port-after.json'):write(out/name,raw(port_command(self.port)))
            write(out/'listener.json',raw(port_command(self.port,cpid),f'p{cpid}\nf4\nn127.0.0.1:{self.port}\n',0))
            models=dict(object='list',data=[dict(id='CPU-model',object='model',created=1,owned_by='lithos-metal',context_window=2048)])
            for name in ('ready.json','ready-after.json'):
                write(out/name,dict(status=200,body_base64=base64.b64encode(json.dumps(models).encode()).decode()))
            write(out/'release.json',dict(pid=cpid,returncode=-15,waited_reaped=True,strict_absence=True,port_absent=True,time=8+i*10))
            write(out/'group.json',group);(out/'probe').mkdir()
            (out/'probe/observer.jsonl').write_bytes(b''.join(f.lines(events)))
            joined=self.join.join(group,f.lines(events),jobs,fixtures,self.decode,trial.FIXED['serve_probe.py'],
                expected_official_args=f.ARGS,expected_pack_dir=f.PACK)
            write(out/'raw-join.json',joined)
            log=b"INFO:monolith.serve:Generation: {'cached_prompt_tokens': 0, 'setup_ms': 1.0, 'verify_tokens': 1}\n"*len(jobs)
            (out/'server.stderr').write_bytes(log)
            write(out/'generation-log-check.json',self.join.generation_log_window(log,len(jobs),require_setup_zero=False,expected_logger='monolith.serve'))
            done=dict(schema='E023-lithos-D512-trial-result-v1',status='PASS',concurrency=C,errors=[],
                route_count_text_qualified=True,server_reaped=True,cleanup_complete=True,cleanup_task_resolved=True,
                client_task_resolved=True,small_pins_rechecked=True,performance_qualified=False,full_state_qualified=False)
            write(out/'complete.json',done);self.refresh_trial(out)
            intent=dict(**bound,phase=phase,time=3+i*10);write(folder/'intent.json',intent)
            child_identity=dict(pid=cpid,ppid=self.pid,start='CPU child',comm='python')
            own=dict(phase=phase,pid=cpid,owner_pid=self.pid,identity=child_identity,ownership='direct-Popen-returned',spawned_at=4+i*10)
            write(folder/'child.json',own);write(folder/'child-absence-00.json',absence(cpid))
            write(folder/'port-absence-00.json',raw(port_command(self.port)))
            release=dict(phase=phase,pid=cpid,returncode=-15,waited_reaped=True,strict_absence=True,
                absence=reference(self.root,folder/'child-absence-00.json'),time=9+i*10)
            write(folder/'release.json',release)
            children.append(dict(phase=phase,pid=cpid,identity=child_identity,
                **{k:reference(self.root,folder/(k+'.json')) for k in ('intent','child','release')}))
            stages.append(dict(phase=phase,status='PASS',concurrency=C,result=reference(self.root,out/'complete.json'),
                raw_join=reference(self.root,out/'raw-join.json'),log_check=reference(self.root,out/'generation-log-check.json')))
        write(self.root/'children.json',dict(schema='cold-service-owner-children-v1',children=children))
        write(self.root/'release.json',dict(schema='cold-service-owner-release-v2',**bound,children=children,
            children_sha256=a.sha(self.root/'children.json'),all_children_reaped=True,time=20))
        write(self.root/'trial-specs.json',self.specs)
        write(self.root/'qualifications.json',dict(schema='E023-lithos-D512-qualifications-v1',trials=stages,performance_qualified=False,full_state_qualified=False))
        write(self.root/'complete.json',dict(schema='E023-lithos-D512-owner-result-v1',status='PASS',errors=[],
            owner_pid=self.pid,owner_plan_sha256=plan_sha,qualification_trials_complete=True,all_children_reaped=True,
            planned_trials=list(a.PHASES),phases=stages,performance_qualified=False))
        owner=NS(validate_plan=lambda args:(None,self.plan,None,None,self.specs))
        self.patcher=patch.object(a,'support',return_value=(owner,trial,self.join,self.cap));self.patcher.start();self.addCleanup(self.patcher.stop)
        decoder=patch.object(a,'decoder',return_value=self.decode);decoder.start();self.addCleanup(decoder.stop)

    def refresh_trial(self,out):
        row=a.read(out/'complete.json')
        row['artifacts']={str(p.relative_to(out)):dict(sha256=a.sha(p),bytes=p.stat().st_size)
            for p in out.rglob('*') if p.is_file() and p!=out/'complete.json'}
        write(out/'complete.json',row)

    def refresh_published(self):
        done=a.read(self.root/'complete.json');published=a.read(self.root/'qualifications.json')
        for row in published['trials']:
            folder=self.root/'phases'/row['phase']/'trial'
            for k,n in (('result','complete.json'),('raw_join','raw-join.json'),('log_check','generation-log-check.json')):
                row[k]=reference(self.root,folder/n)
        done['phases']=published['trials'];write(self.root/'complete.json',done);write(self.root/'qualifications.json',published)

    def test_closed_c1_c8_raw_recomputed_without_qualification_scope_inflation(self):
        result=a.audit(self.root);self.assertEqual(result['status'],'PASS',result)
        self.assertEqual([x['requests'] for x in result['phases']],[8,16]);self.assertEqual(result['requests'],24)
        self.assertEqual([x['max_http_outstanding'] for x in result['phases']],[1,8])
        for field in ('performance_qualified','full_state_qualified','independent_resource_verified','restoration_verified','model_weights_rehashed','pack_payload_rehashed'):
            self.assertFalse(result[field])

    def test_incomplete_terminal_does_not_open_request_evidence(self):
        (self.root/'release.json').unlink()
        with patch.object(a,'support',side_effect=AssertionError('no support or requests before terminal')):
            result=a.audit(self.root)
        self.assertEqual(result['status'],'INCOMPLETE');self.assertFalse(result['raw_requests_opened'])

    def test_changed_saved_join_rejected_even_with_consistent_all_file_hashes(self):
        out=self.root/'phases/lithos-c1/trial';row=a.read(out/'raw-join.json');row['raw_output_count']+=1
        write(out/'raw-join.json',row);self.refresh_trial(out);self.refresh_published()
        result=a.audit(self.root);self.assertEqual(result['status'],'FAIL',result)
        self.assertIn('independent recomputation',result['error']['message'])

    def test_modified_raw_tokens_rejected_even_with_refreshed_artifact_hashes(self):
        out=self.root/'phases/lithos-c1/trial';path=out/'probe/observer.jsonl'
        lines=[json.loads(x) for x in path.read_text().splitlines()]
        next(x for x in lines if x['event']=='generation_return')['generation']['tokens'][0]=8
        path.write_bytes(b''.join(f.lines(lines)));self.refresh_trial(out);self.refresh_published()
        result=a.audit(self.root);self.assertEqual(result['status'],'FAIL',result)
        self.assertIn('decoded text',result['error']['message'])

    def test_missing_phase_or_extra_unresolved_spawn_cannot_pass(self):
        path=self.root/'phases/lithos-c8/child.json';path.unlink()
        result=a.audit(self.root);self.assertEqual(result['status'],'FAIL');self.assertFalse(result['raw_requests_opened'])

    def test_unknown_nested_spawn_is_rejected(self):
        write(self.root/'phases/extra/child.json',dict(pid=123))
        result=a.audit(self.root);self.assertEqual(result['status'],'FAIL');self.assertIn('omitted phase',result['error']['message'])

    def test_bad_runtime_receipt_is_rejected_before_raw_data(self):
        path=self.root/'postflight/verified-pins.json';row=a.read(path)
        row['pack']['/CPU/pack']['sha256']='b'*64;write(path,row)
        result=a.audit(self.root);self.assertEqual(result['status'],'FAIL');self.assertFalse(result['raw_requests_opened'])

    def test_raw_process_absence_is_required_not_just_boolean_release(self):
        path=self.root/'phases/lithos-c1/child-absence-00.json';row=a.read(path);row['stdout']='9910\n';row['returncode']=0;write(path,row)
        release_path=self.root/'phases/lithos-c1/release.json';release=a.read(release_path);release['absence']=reference(self.root,path);write(release_path,release)
        index=a.read(self.root/'children.json');index['children'][0]['release']=reference(self.root,release_path);write(self.root/'children.json',index)
        final=a.read(self.root/'release.json');final['children']=index['children'];final['children_sha256']=a.sha(self.root/'children.json');write(self.root/'release.json',final)
        result=a.audit(self.root);self.assertEqual(result['status'],'FAIL');self.assertIn('PID absence',result['error']['message'])


if __name__=='__main__': unittest.main()
