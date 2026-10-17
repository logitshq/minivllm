"""Synthetic raw resource corruption tests; no OS/service/GPU operations."""
import copy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from bench.m5_ultra_256gb.lithos_d512_retry import audit_resources as a

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False)+'\n')

def absence(pid):
    commands=[(['/bin/ps','-p',str(pid),'-o','pid='],1),(['/bin/launchctl','print',a.JOB],113),
              (['/usr/sbin/lsof','-nP','-a','-iTCP:8001','-sTCP:LISTEN','-Fpn'],1)]
    return dict(status='PASS',checks=[dict(command=c,returncode=rc,stdout_sha256=hashlib.sha256(b'').hexdigest(),
        stdout_bytes=0,absent=True) for c,rc in commands])

class ResourceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.names=['lithos-c1','lithos-c8'];self.pid=90204;self.created=1000;self.owner_pid=91000
        self.children=[dict(phase=p,pid=92000+i,spawned_at=s,released_at=e,inner_released_at=e-10,returncode=-15)
            for i,(p,s,e) in enumerate(zip(self.names,(1010,1120),(1100,1210)))]
        labels=[('before-prepare',1001,None),('after-prepare',1002,None)]
        for child in self.children:
            s,e,pid=child['spawned_at'],child['released_at'],child['pid'];stage=child['phase']
            parts=[('pre-spawn',s-1,None),('spawned',s+1,pid),('ready',s+3,pid),('qualification',s+6,pid),('after-qualification',e-11,pid),
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

    def test_real_qualification_labels_with_exact_current_original_pid(self):
        self.assertEqual(set(self.check()['known_owned_samples']),set(self.names))
        with self.assertRaises(ValueError):a.absence(absence(74384),self.pid)

    def test_wrong_pid_stale_time_foreign_or_lost_server_identity_rejected(self):
        i=next(i for i,r in enumerate(self.rows) if r['label']=='lithos-c8:qualification');original=copy.deepcopy(self.rows[i])
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

if __name__=='__main__':unittest.main()
