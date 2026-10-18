"""Prepared stdlib fake tests only; not run during the parent's GPU experiment.

No MLX/model import, real subprocess, signal, service or physical measurement.
"""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

spec=importlib.util.spec_from_file_location('glm_physical_guard_cpu',Path(__file__).with_name('physical_guard.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def footprint(pid,top=100,current=90,peak=95,system=30*m.GIB):
    return {'unit':'byte','bytes per unit':1,'errors':[],'warnings':[],
        'processes':[{'pid':pid,'footprint':top,'auxiliary':{'phys_footprint':current,'phys_footprint_peak':peak}}],
        'auxiliary':{'sys_footprint':system}}


def sample(pid,at,**changes):
    value=m.parse_raw(footprint(pid),pid,'1','total = 100.00M used = 1.00M free = 99.00M',str(m.RAM),'Swapouts: 7.\n')
    value.update(changes)
    value.update(current_bytes=max(value['top_bytes'],value['aux_current_bytes']),
                 peak_bytes=max(value['top_bytes'],value['aux_peak_bytes']))
    return dict(status='MEASURED',pid=pid,owner_pid=os.getpid(),started_monotonic=at,
        ended_monotonic=at+.1,parsed=value)


def absent(pid):
    return dict(command=['/bin/ps','-p',str(pid),'-o','pid='],returncode=1,stdout='',stderr='')


class Child:
    def __init__(self,stubborn=False):
        self.pid=123456;self.returncode=None;self.stubborn=stubborn;self.actions=[]
    def poll(self):return self.returncode
    def terminate(self):
        self.actions.append('TERM')
        if not self.stubborn:self.returncode=-15
    def kill(self):self.actions.append('KILL');self.returncode=-9
    def wait(self,timeout):
        self.actions.append(('wait',timeout))
        if self.returncode is None:raise subprocess.TimeoutExpired('fake',timeout)
        return self.returncode


class Tests(unittest.TestCase):
    def setUp(self):
        self.limits=m.policy(child_limit_bytes=220*m.GIB,work_seconds=120)
        self.cold=[sample(os.getpid(),10),sample(os.getpid(),15)]
        self.admission=m.cold_admission(self.cold,self.limits,owner_pid=os.getpid())

    def observe(self,row,previous=None):
        return m.observe(row,self.limits,self.admission,child_pid=123456,child_started_monotonic=15.2,previous=previous)

    def test_real_unequal_top_auxiliary_semantics_and_strict_units(self):
        value=m.parse_raw(footprint(123),123,'1','used = 0.00M',str(m.RAM),'Swapouts: 0.')
        self.assertEqual((value['current_bytes'],value['peak_bytes']),(100,100))
        for field,bad in [('unit','KiB'),('bytes per unit',True),('errors',['warning'])]:
            raw=footprint(123);raw[field]=bad
            with self.subTest(field=field),self.assertRaises(ValueError):m.parse_raw(raw,123,'1','used = 0M',str(m.RAM),'Swapouts: 0.')
        with self.assertRaises(ValueError):m.parse_raw(footprint(124),123,'1','used = 0M',str(m.RAM),'Swapouts: 0.')

    def test_cold_is_stable_prerequisite_not_logical_weight_admission(self):
        self.assertEqual(self.admission['cold_system_bytes'],30*m.GIB)
        for field,value in [('system_bytes',41*m.GIB),('pressure',2),('swap_bytes',2<<20),('swapouts',8),('physical_ram',128*m.GIB)]:
            cold=copy.deepcopy(self.cold);cold[-1]['parsed'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):m.cold_admission(cold,self.limits,owner_pid=os.getpid())

    def test_each_child_reading_and_system_budget_reject_without_equality_assumption(self):
        self.assertTrue(self.observe(sample(123456,15.3))['observed_within_budget'])
        for field in ('top_bytes','aux_current_bytes','aux_peak_bytes','system_bytes'):
            value=241*m.GIB if field=='system_bytes' else 221*m.GIB
            row=sample(123456,15.3,**{field:value})
            if field=='aux_current_bytes':row['parsed'].update(aux_peak_bytes=value,peak_bytes=value)
            with self.subTest(field=field),self.assertRaises(ValueError):self.observe(row)

    def test_auxiliary_peak_monotonic_but_derived_max_may_decrease(self):
        first=sample(123456,15.3,top_bytes=200,aux_peak_bytes=100)
        second=sample(123456,15.5,top_bytes=110,aux_peak_bytes=105)
        self.assertGreater(first['parsed']['peak_bytes'],second['parsed']['peak_bytes'])
        self.observe(first);self.observe(second,first)
        second['parsed'].update(aux_peak_bytes=99,peak_bytes=110)
        with self.assertRaisesRegex(ValueError,'historical peak'):self.observe(second,first)

    def test_work_gap_identity_unknown_and_swap_fail_closed(self):
        cases=[sample(123456,18),sample(123456,136),sample(123457,15.3),
            sample(123456,15.3,swapouts=8),sample(123456,15.3,swap_bytes=(1<<20)+1),sample(123456,15.3,pressure=True)]
        bad=sample(123456,15.3);bad['status']='FAIL';cases.append(bad)
        for row in cases:
            with self.subTest(row=row),self.assertRaises(ValueError):self.observe(row)

    def test_measure_records_real_command_shape_with_fake_executor(self):
        with tempfile.TemporaryDirectory() as temp:
            calls=[]
            def run(argv,**kwargs):
                calls.append((argv,kwargs['timeout']))
                if argv[0]=='/usr/bin/footprint':Path(argv[-1]).write_text(json.dumps(footprint(123)));text=''
                elif argv[-1]=='kern.memorystatus_vm_pressure_level':text='1'
                elif argv[-1]=='hw.memsize':text=str(m.RAM)
                elif argv[-1]=='vm.swapusage':text='used = 1.00M'
                elif argv==['/usr/bin/vm_stat']:text='Swapouts: 7.'
                else:raise AssertionError('unknown command')
                return SimpleNamespace(returncode=0,stdout=text,stderr='')
            path=Path(temp)/'sample'
            row=m.measure(path,123,run=run)
            self.assertEqual(row['status'],'MEASURED');self.assertEqual(len(calls),5)
            self.assertTrue(all(0<x[1]<=5 for x in calls))
            self.assertEqual(json.loads((path/'sample.json').read_bytes()),row)
            with self.assertRaises(FileExistsError):m.measure(path,123,run=run)

    def test_failed_footprint_never_becomes_successful_exit_race(self):
        with tempfile.TemporaryDirectory() as temp:
            def run(argv,**kwargs):raise subprocess.TimeoutExpired(argv,kwargs['timeout'],output=b'partial',stderr=b'unknown')
            path=Path(temp)/'failure'
            with self.assertRaises(subprocess.TimeoutExpired):m.measure(path,123,run=run)
            row=json.loads((path/'sample.json').read_bytes())
            self.assertEqual(row['status'],'FAIL');self.assertEqual(row['commands'][0]['stdout'],'partial')
            self.assertNotIn('parsed',row)

    def test_direct_wait_term_kill_and_exact_absence_only(self):
        with tempfile.TemporaryDirectory() as temp:
            child=Child(stubborn=True)
            row=m.reap_owned(child,expected_pid=child.pid,owner_pid=os.getpid(),absence=absent,out=Path(temp)/'release.json')
            self.assertEqual(row['status'],'REAPED');self.assertEqual(row['returncode'],-9)
            self.assertEqual(child.actions,['TERM',('wait',15),'KILL',('wait',15)])
            self.assertFalse(row['physical_resource_safety_qualified'])
            wrong=lambda pid:absent(pid+1)
            row=m.reap_owned(child,expected_pid=child.pid,owner_pid=os.getpid(),absence=wrong,out=Path(temp)/'wrong.json')
            self.assertEqual(row['status'],'UNRESOLVED')
            with self.assertRaises(ValueError):m.reap_owned(None,expected_pid=child.pid,owner_pid=os.getpid(),absence=absent,out=Path(temp)/'none.json')

    def test_shutdown_failure_and_changed_sample_cannot_pass_report(self):
        release=dict(status='REAPED',pid=123456,owner_pid=os.getpid(),ownership='direct-Popen-returned',
            returncode=0,waited_reaped=True,strict_absence=True,absence=absent(123456))
        probe=dict(schema='GLM-official-load-result-v1',status='PASS',pid=123456,owner_pid=os.getpid(),
            short_lifecycle_diagnostic_complete=True,shutdown={'complete':True},completed_requests=[{'cap':n} for n in (8,32,32,32)])
        def report(rows,**kwargs):return m.diagnostic_report(self.cold,rows,self.limits,release,probe,child_started_monotonic=15.2,**kwargs)
        good=sample(123456,15.3);result=report([good]);self.assertEqual(result['status'],'PASS')
        self.assertFalse(result['physical_resource_safety_qualified'] or result['performance_qualified'])
        probe['shutdown']['complete']=False;self.assertEqual(report([good])['status'],'FAIL');probe['shutdown']['complete']=True
        self.assertEqual(report([])['status'],'FAIL')
        self.assertEqual(report([sample(123456,15.3,system_bytes=241*m.GIB)])['status'],'FAIL')
        self.assertEqual(report([good],observation_error={'message':'missed measurement'})['status'],'FAIL')


if __name__=='__main__':unittest.main()
