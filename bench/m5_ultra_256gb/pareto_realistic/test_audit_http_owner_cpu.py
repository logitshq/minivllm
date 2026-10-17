"""Synthetic resource/provenance mechanism tests; no OS/model/service probes."""
import copy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bench.m5_ultra_256gb.pareto_realistic import audit_http_owner as a


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')


def absent(pid):
    commands=[(['/bin/ps','-p',str(pid),'-o','pid='],1),(['/bin/launchctl','print',a.JOB],113),
        (['/usr/sbin/lsof','-nP','-a','-iTCP:8001','-sTCP:LISTEN','-Fpn'],1)]
    return dict(status='PASS',checks=[dict(command=c,returncode=rc,stdout_sha256=hashlib.sha256(b'').hexdigest(),stdout_bytes=0,absent=True) for c,rc in commands])


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()
        self.pid=80000;self.owner_pid=81000;self.created=1000
        self.children=[dict(phase=p,pid=82000+i,spawned_at=s+1000,released_at=e+1000,inner_released_at=e+990,returncode=0)
            for i,(p,s,e) in enumerate(zip(a.PHASES,(10,120,220,330),(100,200,310,410)))]
        labels=[('before-prepare',1,None),('after-prepare',2,None)]
        for child in self.children:
            phase=child['phase'];s=child['spawned_at']-1000;e=child['released_at']-1000;pid=child['pid']
            parts=[('pre-spawn',s-1,None)]
            if phase.endswith('capture'):
                parts += [('spawned',s+1,pid),('ready',s+10,pid),('capturing',s+30,pid),('after-client',e-30,pid),('after-reap',e-9,None)]
            else:parts += [('running',s+10,pid),('running',s+20,pid)]
            parts += [('released',e+1,None)]
            labels.extend((phase+':'+name,t,p) for name,t,p in parts)
            write(self.root/'phases'/phase/'pre-spawn-absence.json',absent(self.pid))
        labels.append(('postflight',420,None));self.rows=[]
        for i,(label,t,pid) in enumerate(labels):
            wall=t+1000;entry=dict(pid=pid,ppid=self.owner_pid,owned=True,authorized_suspended=False)
            exact=dict(status='unknown',pid=None,raw=None) if pid is None else dict(status='known',pid=pid,ppid=self.owner_pid,
                raw=dict(returncode=0,stdout=f'{pid} {self.owner_pid} 1024 python'))
            row=dict(label=label,recorded_at=datetime.fromtimestamp(wall,timezone.utc).isoformat(),monotonic_s=wall,
                owned_model_pid=pid,pressure_level=dict(returncode=0,text='1'),pressure_level_value=1,
                memory_pressure=dict(returncode=0,text='free percentage: 70'),free_percent=70,
                swap=dict(returncode=0,text='used = 12.00M'),swap_used_mb=12.0,
                vm_stat=dict(returncode=0,text='Swapouts: 0\nPageouts: 0\nSwapins: 10\n'),vm_counters=dict(Swapouts=0,Pageouts=0,Swapins=10),
                power_metadata=dict(source='AC Power',raw=dict(returncode=0,stdout='AC Power')),
                processinfo_thermal_metadata=dict(code=0,raw=dict(returncode=0,stdout='0')),
                process_inventory_ok=True,native_process_inventory_ok=True,no_active_foreign_native_processes=True,
                only_owned_known_inference_processes=True,authorized_suspended_pids=[],
                native_inference_processes=[entry] if pid else [],known_inference_processes=[entry] if pid else [],exact_owned_pid_metadata=exact)
            self.rows.append(row)
            write(self.root/f'absence-{i:03d}.json',dict(label=label,time=wall-.01,resource_index_if_sampled=i,**absent(self.pid)))
        self.complete=dict(owner_pid=self.owner_pid,samples=len(self.rows),absence_probes=len(self.rows));self.save()

    def save(self):
        aggregate=[]
        for i,row in enumerate(self.rows):
            write(self.root/f'resource-{i:03d}.json',row);item=copy.deepcopy(row)
            if i:
                item['swap_used_mb_delta']=row['swap_used_mb']-self.rows[i-1]['swap_used_mb']
                item['vm_counter_delta']={k:v-self.rows[i-1]['vm_counters'][k] for k,v in row['vm_counters'].items()}
            aggregate.append(item)
        write(self.root/'resources.json',aggregate)

    def check(self):return a.resources(self.root,self.complete,self.children,self.pid,self.created)

    def test_four_real_phase_labels_and_dynamic_original_pid(self):
        result=self.check();self.assertEqual(set(result['known_owned_samples']),set(a.PHASES))
        self.assertEqual(result['pressure'],1);self.assertEqual(result['swap_delta_mb'],0)
        bad=absent(74384)
        with self.assertRaises(ValueError):a.absence(bad,self.pid)

    def test_wrong_owned_pid_lifetime_foreign_or_exemption_rejected(self):
        index=next(i for i,r in enumerate(self.rows) if r['label']=='c1-replay:running')
        original=copy.deepcopy(self.rows[index])
        for kind in ('pid','ppid','foreign','suspended','time'):
            row=self.rows[index]=copy.deepcopy(original)
            if kind=='pid':row['owned_model_pid']=self.children[0]['pid']
            if kind=='ppid':row['exact_owned_pid_metadata']['ppid']+=1
            if kind=='foreign':row['native_inference_processes'][0]['pid']+=1
            if kind=='suspended':row['authorized_suspended_pids']=[self.pid]
            if kind=='time':row['recorded_at']=datetime.fromtimestamp(1001,timezone.utc).isoformat()
            self.save()
            with self.subTest(kind=kind),self.assertRaises(ValueError):self.check()

    def test_raw_pressure_thermal_and_sustained_swap_gates(self):
        original=copy.deepcopy(self.rows)
        for kind in ('pressure','thermal','swap','counters'):
            self.rows=copy.deepcopy(original)
            if kind=='pressure':self.rows[0]['pressure_level_value']=2
            if kind=='thermal':self.rows[0]['processinfo_thermal_metadata']['code']=2
            if kind=='swap':
                for i in (1,2):self.rows[i]['swap_used_mb']=12+70*i;self.rows[i]['swap']['text']=f'used = {12+70*i}.00M'
            if kind=='counters':self.rows[0]['vm_counters']['Swapouts']=True
            self.save()
            with self.subTest(kind=kind),self.assertRaises(ValueError):self.check()

    def test_final_replay_exit_probe_can_be_unknown_but_not_earlier(self):
        indices=[i for i,r in enumerate(self.rows) if r['label']=='c1-replay:running']
        for index in reversed(indices):
            row=self.rows[index];row['exact_owned_pid_metadata']=dict(status='unknown',pid=row['owned_model_pid'],raw=dict(returncode=1,stdout=''))
            row['known_inference_processes']=[];row['native_inference_processes']=[];self.save()
            if index==indices[-1]:self.check()
            else:
                with self.assertRaises(ValueError):self.check()

    def test_aggregate_delta_or_missing_phase_proof_rejected(self):
        aggregate=a.read(self.root/'resources.json');aggregate[2]['swap_used_mb_delta']=1;write(self.root/'resources.json',aggregate)
        with self.assertRaises(ValueError):self.check()
        self.save();(self.root/'phases/c8-replay/pre-spawn-absence.json').unlink()
        with self.assertRaises(FileNotFoundError):self.check()

    def test_prepared_and_failed_owner_do_not_load_auditors_or_ledgers(self):
        owner=self.root/'owned';journal=self.root/'journal'
        for state,status,expected in [('PREPARED','INCOMPLETE','INCOMPLETE'),('RESTORED','FAIL','FAIL')]:
            write(journal/'journal.json',dict(state=state));write(owner/'complete.json',dict(status=status,errors=['CPU']))
            with patch.object(a,'load',side_effect=AssertionError('no support execution')):
                result=a.audit(owner,journal,'0'*64,{})
            self.assertEqual(result['status'],expected);self.assertFalse(result['qualified'] or result['ledger_opened'])

    def test_current_pinned_support_hashes_without_importing_backends(self):
        self.assertEqual(a.sha(a.HERE/'audit_http_replay_v2.py'),a.NUMERICAL_AUDIT_SHA)
        self.assertEqual(a.sha(a.HERE.parent/'omlx_cold_restart/audit_restore_v3.py'),a.RESTORE_SHA)
        self.assertEqual(a.sha(a.HERE/'http_capture_v2.py'),a.CAPTURE_SHA)
        self.assertEqual(a.sha(a.HERE/'http_owner_v2.py'),a.OWNER_SHA)

    def test_capture_artifact_set_cannot_omit_or_tamper_raw_files(self):
        folder=self.root/'capture';folder.mkdir();write(folder/'complete.json',{})
        (folder/'raw.jsonl').write_bytes(b'CPU raw evidence\n');(folder/'server.stdout').write_bytes(b'')
        files={p.name:dict(sha256=a.sha(p),bytes=p.stat().st_size) for p in folder.iterdir() if p.name!='complete.json'}
        a.capture_artifacts(folder,files)
        with self.assertRaises(ValueError):a.capture_artifacts(folder,{})
        (folder/'raw.jsonl').write_bytes(b'different\n')
        with self.assertRaises(ValueError):a.capture_artifacts(folder,files)


if __name__=='__main__':unittest.main()
