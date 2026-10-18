"""Small retained-byte corruption fixtures. Not a real GLM/device test."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
HERE=Path(__file__).resolve().parent

def module(path,name):
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
au=module(HERE/'audit_closed.py','glm_closed_fake_test')
g=module(HERE.parent/'glm_flash/physical_guard.py','glm_guard_fake_test')
def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve()
        self.limits=g.policy(child_limit_bytes=200<<30,work_seconds=1800)
        self.path=self.root/'physical-0000/sample.json';raw=self.path.parent/'footprint.raw.json'
        write(raw,{'unit':'byte','bytes per unit':1,'errors':[],'warnings':[],
            'processes':[{'pid':20003,'footprint':10,'auxiliary':{'phys_footprint':9,'phys_footprint_peak':12}}],
            'auxiliary':{'sys_footprint':20<<30}})
        argv=[['/usr/bin/footprint','-p','20003','--noCategories','--sysFootprint','-f','bytes','-j',str(raw)],
            ['/usr/sbin/sysctl','-n','kern.memorystatus_vm_pressure_level'],['/usr/sbin/sysctl','vm.swapusage'],
            ['/usr/sbin/sysctl','-n','hw.memsize'],['/usr/bin/vm_stat']]
        stdout=['','1','total = 0.00M used = 0.00M free = 0.00M',str(256<<30),'Swapouts: 0.\n']
        self.sample=dict(schema='GLM-physical-sample-v1',pid=20003,owner_pid=20002,status='MEASURED',started_at=100.,ended_at=100.1,
            started_monotonic=5.,ended_monotonic=5.1,commands=[dict(command=c,returncode=0,stdout=v,stderr='') for c,v in zip(argv,stdout)],
            raw=dict(path=str(raw),bytes=raw.stat().st_size,sha256=au.sha(raw)),parsed=g.parse_raw(au.read(raw),20003,*stdout[1:]))
        write(self.path,self.sample)

    def test_original_three_readings_unequal_are_reparsed(self):
        row=au.raw_sample(self.path,20003,20002,g,self.limits)
        self.assertEqual(row['parsed']['current_bytes'],10);self.assertEqual(row['parsed']['peak_bytes'],12)

    def test_tampered_raw_and_declared_parse_cannot_pass(self):
        changed=copy.deepcopy(self.sample);changed['parsed']['top_bytes']=11;write(self.path,changed)
        with self.assertRaisesRegex(ValueError,'original bytes'):au.raw_sample(self.path,20003,20002,g,self.limits)
        write(self.path,self.sample);Path(self.sample['raw']['path']).write_text('{}')
        with self.assertRaisesRegex(ValueError,'byte binding'):au.raw_sample(self.path,20003,20002,g,self.limits)

    def test_bool_returncode_and_wrong_pid_rejected(self):
        changed=copy.deepcopy(self.sample);changed['commands'][0]['returncode']=False;write(self.path,changed)
        with self.assertRaises(ValueError):au.raw_sample(self.path,20003,20002,g,self.limits)
        write(self.path,self.sample)
        with self.assertRaises(ValueError):au.raw_sample(self.path,20004,20002,g,self.limits)

    def test_preserved_resources_do_not_override_pressure_or_swap_guard(self):
        changed=copy.deepcopy(self.sample);changed['commands'][1]['stdout']='2';changed['parsed']['pressure']=2;write(self.path,changed)
        with self.assertRaises(ValueError):au.raw_sample(self.path,20003,20002,g,self.limits)
        row=self.sample;bad=copy.deepcopy(row);bad['parsed']['swap_bytes']=1
        cold=dict(owner_pid=20002,last_sample_monotonic=4.9,baseline_swap_bytes=0,baseline_swapouts=0)
        with self.assertRaisesRegex(ValueError,'swap growth'):g.observe(bad,self.limits,cold,child_pid=20003,child_started_monotonic=5.)

    def test_missing_restore_is_incomplete_without_inspecting_model(self):
        restore=SimpleNamespace(audit=lambda journal:dict(status='INCOMPLETE',qualified=False))
        with patch.object(au,'module',return_value=restore):result=au.audit(self.root/'missing','a'*64)
        self.assertEqual(result['status'],'INCOMPLETE');self.assertFalse(result['observed_diagnostic_complete'])

    def counts_fixture(self):
        probe=SimpleNamespace(EOS=[99],CAPS=[8,32,32,32],SAMPLING={'temperature':0.0},require_target_route=lambda route:None)
        rows=[];events=[]
        write(self.root/'input.json',dict(prompt_ids=[1,2],normal_eos=[99]))
        for i,cap in enumerate(probe.CAPS):
            raw=[99] if i==0 else [7]*(cap-1)+[99];rid=f'case{i}'
            output=dict(request_id=rid,committed_ids=raw[:-1],finish_reason='stop',error=None,cached_tokens=0,prompt_tokens=2,completion_tokens=len(raw)-1)
            row=dict(index=i,cap=cap,prompt_ids=[1,2],raw_ids=raw,committed_ids=raw[:-1],request_id=rid,scheduler_output=output,
                reset_after={k:0 for k in au.STATE_KEYS},performance_qualified=False);rows.append(row);write(self.root/f'request-{i}.json',row)
            events.append(dict(kind='reset_before',case=i,pid=20003,counts={k:0 for k in au.STATE_KEYS}))
            for j,token in enumerate(raw):
                events.append(dict(kind='batch_before',case=i,pid=20003,route={},responses=[dict(request_id=rid,
                    token=token,finish_reason='stop' if j==len(raw)-1 else None,prompt_ids=[1,2],sampling={'temperature':0.0,'max_tokens':cap})]))
                events.append(dict(kind='batch_after',case=i,pid=20003,finals=[output] if j==len(raw)-1 else []))
            events.append(dict(kind='memory',phase='request_after',case=i,pid=20003))
        (self.root/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
        return probe,dict(pid=20003,completed_requests=rows)

    def test_immediate_eos_and_eos_at_cap_subtract_exactly_once(self):
        probe,complete=self.counts_fixture();value=au.request_counts(self.root,complete,probe)
        self.assertEqual(value['requests'],4);self.assertEqual(value['committed_tokens'],93)
        self.assertFalse(value['full_state_reference'] or value['independent_text_decode'])
        complete['completed_requests'][1]['raw_ids'][-1]=8
        write(self.root/'request-1.json',complete['completed_requests'][1])
        with self.assertRaises(ValueError):au.request_counts(self.root,complete,probe)

    def test_final_cannot_move_before_its_response_or_into_next_request(self):
        probe,complete=self.counts_fixture();path=self.root/'events.jsonl';original=path.read_bytes()
        events=[json.loads(line) for line in original.splitlines()]
        final=next(i for i,e in enumerate(events) if e['kind']=='batch_after' and e['case']==0)
        bad=copy.deepcopy(events);value=bad.pop(final);bad.insert(1,value)
        path.write_text(''.join(json.dumps(e)+'\n' for e in bad))
        with self.assertRaisesRegex(ValueError,'matching before'):au.request_counts(self.root,complete,probe)
        bad=copy.deepcopy(events);value=bad.pop(final)
        next_reset=next(i for i,e in enumerate(bad) if e['kind']=='reset_before' and e['case']==1)
        bad.insert(next_reset+1,value);path.write_text(''.join(json.dumps(e)+'\n' for e in bad))
        with self.assertRaises(ValueError):au.request_counts(self.root,complete,probe)

    def test_empty_or_boolean_drain_maps_are_not_full_drain_evidence(self):
        probe,complete=self.counts_fixture();path=self.root/'events.jsonl';events=[json.loads(line) for line in path.read_bytes().splitlines()]
        events[0]['counts']={};path.write_text(''.join(json.dumps(e)+'\n' for e in events))
        with self.assertRaisesRegex(ValueError,'exact drained reset'):au.request_counts(self.root,complete,probe)
        probe,complete=self.counts_fixture();complete['completed_requests'][0]['reset_after']['running']=False
        write(self.root/'request-0.json',complete['completed_requests'][0])
        with self.assertRaisesRegex(ValueError,'final drain'):au.request_counts(self.root,complete,probe)


if __name__=='__main__':unittest.main()
