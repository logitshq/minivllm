"""E026 real-weight layer owner: stdlib mocks only, no GPU/service queries."""
import copy
from contextlib import ExitStack
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('layer_owner_cpu_review', HERE / 'layer_owner.py')
o = importlib.util.module_from_spec(spec); spec.loader.exec_module(o)


def sample(pressure=2, pid=None, used=324.0):
    native = [{'pid': 74384, 'ppid': 1, 'state': 'T', 'owned': False, 'authorized_suspended': True}]
    if pid: native.append({'pid': pid, 'ppid': os.getpid(), 'state': 'R', 'owned': True})
    return {'pressure_level_value': pressure, 'pressure_level': {'returncode': 0, 'text': str(pressure)},
        'vm_stat': {'returncode': 0, 'text': 'Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 4194304.\nSwapouts: 5.'},
        'swap': {'returncode': 0, 'text': f'total = 512.00M used = {used}M free = 188.00M'},
        'swap_used_mb': used, 'vm_counters': {'Swapouts': 5},
        'power_metadata': {'status': 'known', 'source': 'AC Power', 'raw': {'returncode': 0, 'stdout': "Now drawing from 'AC Power'"}},
        'processinfo_thermal_metadata': {'status': 'known', 'code': 0, 'raw': {'returncode': 0, 'stdout': '0'}},
        'native_process_inventory_ok': True, 'process_inventory_ok': True, 'authorized_suspended_pids': [74384],
        'native_inference_processes': native, 'known_inference_processes': [],
        'exact_owned_pid_metadata': {'status': 'known', 'pid': pid, 'ppid': os.getpid()}}


def physical(pid=12345, current=64 << 20, peak=80 << 20):
    return {'unit': 'byte', 'bytes per unit': 1, 'errors': [], 'warnings': [],
            'processes': [{'pid': pid, 'footprint': current, 'auxiliary': {'phys_footprint': current, 'phys_footprint_peak': peak}}]}


class FakeChild:
    pid = 12345
    def __init__(self, mode='done'):
        self.mode, self.returncode, self.actions = mode, None, []
    def poll(self): return self.returncode
    def wait(self, timeout):
        self.actions.append(('wait', timeout))
        if self.returncode is not None: return self.returncode
        if self.mode == 'done': self.returncode = 0; return 0
        if self.mode == 'childfail': self.returncode = 7; return 7
        if timeout < 2: raise TimeoutError('synthetic arm deadline')
        raise subprocess.TimeoutExpired('fake', timeout)
    def terminate(self): self.actions.append(('terminate',))
    def kill(self): self.actions.append(('kill',)); self.returncode = -9


class OwnerTests(unittest.TestCase):
    def test_budget_arm_work_total_boundaries(self):
        now = [10.0]
        b = o.Budget(lambda: now[0])
        with patch.object(o.signal, 'setitimer'):
            b.arm(); self.assertEqual(b.arm_end, 130)
            now[0] = 129.999; b.check()
            now[0] = 130
            with self.assertRaises(TimeoutError): b.check()
            b.arm_end = None; now[0] = 459.999; b.check()
            now[0] = 460
            with self.assertRaises(TimeoutError): b.check()
            b.cleanup_alarm(); now[0] = 490
            with self.assertRaisesRegex(RuntimeError, 'cleanup deadline'): b.cleanup_alarm()

    def test_pressure_recovery_cannot_rebound(self):
        o.guard([sample(2), sample(2)])
        o.guard([sample(2), sample(1)])
        for values in ((2,1,2), (1,2), (2,4), (2,True)):
            with self.subTest(values=values), self.assertRaises(RuntimeError): o.guard([sample(x) for x in values])

    def test_actual_pages_raw_consistency_swap_and_ownership(self):
        base = sample()
        for edit in ('free','swapouts','rawswapouts','used','rawused','foreign','lease','authorized','parent','identity','power','thermal','rawpressure'):
            row = sample(pid=12345)
            if edit == 'free': row['vm_stat']['text'] = row['vm_stat']['text'].replace('4194304', '1'); row['free_percent'] = 100
            if edit == 'swapouts': row['vm_counters']['Swapouts'] += 1
            if edit == 'rawswapouts': row['vm_stat']['text'] = row['vm_stat']['text'].replace('Swapouts: 5', 'Swapouts: 6')
            if edit == 'used': row.update(swap_used_mb=340.01); row['swap']['text'] = 'used = 340.01M'
            if edit == 'rawused': row['swap']['text'] = 'used = 325.0M'
            if edit == 'foreign': row['native_inference_processes'].append({'pid': 4444, 'ppid': os.getpid(), 'state': 'T', 'owned': True})
            if edit == 'lease': row['native_inference_processes'][0]['state'] = 'S'
            if edit == 'authorized': row['native_inference_processes'][0]['authorized_suspended'] = False
            if edit == 'parent': row['native_inference_processes'][1]['ppid'] += 1
            if edit == 'identity': row['exact_owned_pid_metadata']['pid'] += 1
            if edit == 'power': row['power_metadata']['raw']['returncode'] = 1
            if edit == 'thermal': row['processinfo_thermal_metadata']['code'] = 2
            if edit == 'rawpressure': row['pressure_level']['text'] = '1'
            with self.subTest(edit=edit), self.assertRaises(RuntimeError): o.guard([base,row],12345)
        row=sample(pid=12345,used=340); o.guard([base,row],12345)
        self.assertEqual(row['real_free_bytes'],64 << 30)
        with self.assertRaisesRegex(RuntimeError,'missing'): o.guard([base,sample()],12345)
        o.guard([base,sample()],12345,terminal_exit0=True)

    def test_footprint_units_pid_current_peak_and_error_guard(self):
        self.assertEqual(o.parse_footprint(physical(),12345)['peak_bytes'],80 << 20)
        for edit in ('units','boolunit','pid','current','peak','errors','warnings','missing'):
            row=physical()
            if edit=='units': row['unit']='MiB'
            if edit=='boolunit': row['bytes per unit']=True
            if edit=='pid': row['processes'][0]['pid'] += 1
            if edit=='current': row['processes'][0]['auxiliary']['phys_footprint'] = (80 << 20) + 1
            if edit=='peak': row['processes'][0]['auxiliary']['phys_footprint_peak']=2*o.GIB+1
            if edit=='errors': row['errors']=['failure']
            if edit=='warnings': row['warnings']=['partial']
            if edit=='missing': row['processes']=[]
            with self.subTest(edit=edit), self.assertRaises(RuntimeError): o.parse_footprint(row,12345)

    def test_unequal_os_readings_preserved_with_conservative_derived_values(self):
        # Values from the failed v1 probe; fixture only, no OS command is run.
        row=physical(current=65323704,peak=65323704)
        row['processes'][0]['footprint']=65389240
        parsed=o.parse_footprint(row,12345)
        self.assertEqual(parsed,dict(pid=12345,current_bytes=65389240,peak_bytes=65389240,
            top_bytes=65389240,aux_current_bytes=65323704,aux_peak_bytes=65323704))
        lower=physical(current=1024,peak=2048);lower['processes'][0]['footprint']=512
        self.assertEqual(o.parse_footprint(lower,12345)['current_bytes'],1024)
        self.assertEqual(o.parse_footprint(lower,12345)['peak_bytes'],2048)
        for field in ('top','current','peak'):
            for bad in (-1,True,2*o.GIB+1,1.0):
                corrupt=copy.deepcopy(row)
                if field=='top':corrupt['processes'][0]['footprint']=bad
                else:corrupt['processes'][0]['auxiliary']['phys_footprint'+('_peak' if field=='peak' else '')]=bad
                with self.subTest(field=field,bad=bad),self.assertRaises(RuntimeError):o.parse_footprint(corrupt,12345)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            def successful_unequal(cmd,**kw):
                Path(cmd[-1]).write_text(json.dumps(row));return NS(returncode=0,stdout='unequal OS fixture',stderr='')
            with patch.object(o.subprocess,'run',side_effect=successful_unequal):receipt=o.footprint(FakeChild(),out,0)
            self.assertEqual(receipt['status'],'measured')
            self.assertEqual(receipt['parsed'],parsed)
            self.assertEqual(json.loads((out/'footprint-000.raw.json').read_text()),row)

    def test_physical_probe_raw_preserved_and_live_error_aborts(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp); child=FakeChild()
            def success(cmd,**kw):
                Path(cmd[-1]).write_text(json.dumps(physical()))
                return NS(returncode=0,stdout='measurement',stderr='')
            with patch.object(o.subprocess,'run',side_effect=success): row=o.footprint(child,out,0)
            self.assertEqual(row['status'],'measured'); self.assertTrue(row['physical_pass'])
            self.assertEqual(row['raw_sha256'],o.sha(out/'footprint-000.raw.json'))
            with patch.object(o.subprocess,'run',return_value=NS(returncode=1,stdout='',stderr='exited')):
                with self.assertRaisesRegex(RuntimeError,'live or failed'): o.footprint(child,out,1)
                child.returncode=0; row=o.footprint(child,out,2)
            self.assertEqual(row['status'],'terminal-exit0-race-not-measured')
            self.assertIsNone(row['physical_pass'])
            self.assertEqual(json.loads((out/'footprint-001.json').read_text())['stderr'],'exited')
            def over_budget(cmd,**kw):
                Path(cmd[-1]).write_text(json.dumps(physical(peak=2*o.GIB+1)))
                return NS(returncode=0,stdout='',stderr='')
            with patch.object(o.subprocess,'run',side_effect=over_budget), self.assertRaisesRegex(RuntimeError,'2GiB'):
                o.footprint(child,out,3)

    def prepare_fixture(self, root):
        root=root.resolve(); here=root/'bench/layer'; base=here.parent; here.mkdir(parents=True)
        files=[here/n for n in ('candidate.py','integration.py','layer-preregistration.md','layer_screen.py','layer_weights.py','test_layer_cpu.py','test_layer_owner_cpu.py')]
        files += [base/n for n in ('resources_v2.py','resources.py','resource_base.py','suspended_service.py')]
        runtime=root/'runtime'
        files += [runtime/n for n in ('mlx/core.fake.so','mlx/lib/libmlx.dylib','mlx/lib/mlx.metallib','numpy/__init__.py','numpy/_core/_multiarray_umath.fake.so')]
        package=root/'frozen'; files += [package/'tutorial_code/miniengine/layers.py',package.parent/'native-source.json']
        bundle=root/'bundle'; files += [bundle/n for n in ('weights.bin','config.json','model.safetensors.index.json','shard-header.json','preparation.json')]
        python=root/'python'; files += [python]
        for p in files: p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# CPU fixture\n')
        python.chmod(0o700)
        manifest=bundle/'manifest.json';manifest.write_text(json.dumps({'bundle':{'filename':'weights.bin','sha256':o.sha(bundle/'weights.bin')}}));files.append(manifest)
        evidence=here/'source-pins.json';evidence.write_text('{"sources":{}}');files.append(evidence)
        files.append(Path(o.__file__).resolve())
        pins={str(p):o.sha(p) for p in files}
        plan={'schema':'E026-layer-plan-v2','arms':o.ARMS,'arm_seconds':120,'work_seconds':450,'cleanup_seconds':30,'total_seconds':480,
              'resource_policy':o.POLICY,'package_root':str(package),'bundle':{'path':str(manifest),'sha256':o.sha(manifest)},
              'execution':{'python':str(python),'screen':{'path':str(here/'layer_screen.py'),'sha256':pins[str(here/'layer_screen.py')]},'pins':pins,'environment':o.ENV}}
        p=root/'plan.json';p.write_text(json.dumps(plan));out=root/'out';out.mkdir()
        return here,base,NS(plan=p,plan_sha256=o.sha(p),out=out,suspended_pid=74384)

    def test_pins_and_budget_policy_cannot_be_silently_relaxed(self):
        with tempfile.TemporaryDirectory() as tmp:
            here,base,args=self.prepare_fixture(Path(tmp))
            with patch.object(o,'HERE',here),patch.object(o,'BASE',base),patch.dict(os.environ,{'MLX_EVIL':'1'}):
                plan,env=o.prepare(args,lambda:None)
                self.assertNotIn('MLX_EVIL',env)
                self.assertEqual({k:v for k,v in env.items() if k.startswith('MLX_')},o.ENV)
                (here/'integration.py').write_text('# changed\n')
                with self.assertRaisesRegex(RuntimeError,'pin mismatch'): o.prepare(args,lambda:None)
                plan['resource_policy']['mlx_boundary_max_bytes'] += 1
                args.plan.write_text(json.dumps(plan));args.plan_sha256=o.sha(args.plan)
                with self.assertRaisesRegex(RuntimeError,'budgets/policy'): o.prepare(args,lambda:None)

    def test_post_pin_mutation_and_hash_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'source.py';p.write_text('first');pins={str(p):o.sha(p)}
            o.verify_pins(pins,root,'before.json',lambda:None,freeze=True)
            p.write_text('second')
            with self.assertRaisesRegex(RuntimeError,'pin mismatch'): o.verify_pins(pins,root,'after.json',lambda:None)
            self.assertFalse((root/'after.json').exists())
            def deadline(): raise TimeoutError('deadline')
            with self.assertRaises(TimeoutError): o.sha(p,deadline)

    def descriptor(self, shape, dtype='bfloat16', file='raw.bin'):
        width,encoding,finite={'bfloat16':(2,'uint16-le',True),'float32':(4,'uint32-le',True)}[dtype]
        return dict(file=file,sha256='a'*64,bytes=math.prod(shape)*width,dtype=dtype,shape=shape,encoding=encoding,finite=finite)

    def synthetic_child(self, root, arm='off'):
        """Metadata-only mock for receipt logic; raw_record is separately tested."""
        root=root.resolve()
        out=root/'screen';out.mkdir();package=root/'frozen';package.mkdir()
        core=package/'tutorial_code/miniengine/model/gdn.py'
        core_manifest=package.parent/'native-source.json';core_manifest.write_text(json.dumps({'files':{'model/gdn.py':'b'*64}}))
        runtime=root/'runtime';runtime.write_text('CPU fixture')
        sources={str(core):'b'*64,str(core_manifest):o.sha(core_manifest)}
        sources.update({str(HERE/n):'b'*64 for n in ('layer_weights.py','integration.py','candidate.py','layer_screen.py')})
        pins={**sources,str(runtime):o.sha(runtime)}
        bundle={'bundle':{'sha256':'c'*64},'tensors':[{'parameter':f'p{i}','dtype':'BF16','shape':[1]} for i in range(14)]}
        manifest=root/'manifest.json';manifest.write_text(json.dumps(bundle))
        plan={'package_root':str(package),'bundle':{'path':str(manifest),'sha256':o.sha(manifest)},'execution':{'pins':pins}}
        contract=dict(schema='E026-real-weight-layer-v2',arm=arm,layer_index=0,lengths=o.LENGTHS,cache_kinds=['fresh','continuing'],cases=8,case_ids=o.CASES,
                      cache_limit_bytes=0,mlx_observation_budget_bytes=o.GIB,physical_observation_budget_bytes=2*o.GIB,child_alarm_seconds=120,
                      phase_count=68,case_phases=o.CASE_PHASES,continuing_offset=7,continuing_prefix_forward=False)
        params={f'p{i}':self.descriptor([1],file=f'p{i}.bin') for i in range(14)}
        result=dict(status='PASS',device_run=True,arm=arm,pid=12345,ppid=os.getpid(),source_pins_rechecked=True,bundle_rechecked=True,
                    performance_qualified=False,cross_arm_equality='PENDING_INDEPENDENT_RAW_AUDIT',elapsed_wall_s=90,contract=contract,
                    sources=sources,runtime={'mlx_file':str(runtime),'numpy_file':str(runtime)},bundle_manifest_sha256=o.sha(manifest),
                    bundle_sha256='c'*64,parameters_before=params,parameters_after=copy.deepcopy(params),cases=[],physical_receipts=[],
                    initial_active_bytes=0,release={'layer_weakref_dead':True,'initial_active_bytes':0,'final_active_bytes':1024,
                    'active_delta_bytes':1024,'known_persistent_buffer_bytes':4})
        witnesses=[]
        for index,case in enumerate(o.CASES):
            length=o.LENGTHS[index//2];kind=('fresh','continuing')[index%2];offset=7 if index%2 else 0
            inputs={'x':self.descriptor([length,5120]),'conv_seed':self.descriptor([3,10240]) if offset else None,
                    'S_seed':self.descriptor([48,128,128],'float32') if offset else None}
            hook=dict(sequence=1,layer=0,route='single-layer-eager',input_shape=[length,10240],tail_shape=[3,10240],input_dtype='mlx.core.bfloat16',tail_dtype='mlx.core.bfloat16',
                      offset=offset,state_present=bool(offset),eligible=True,enabled=arm!='off',materialize=arm=='on-materialize',replaced=arm!='off',y_identity_preserved=True)
            row=dict(case=case,status='PASS',length=length,kind=kind,layer_index=0,offset_before=offset,offset_after=offset+length,input=inputs,
                     input_after={k:None if v is None else o.nofile(v) for k,v in inputs.items()},
                     arrays={'output':self.descriptor([length,5120]),'conv_y':self.descriptor([length,10240]),'conv_state':self.descriptor([3,10240]),
                             'S':self.descriptor([48,128,128],'float32'),'original_conv_tail':self.descriptor([3,10240])},
                     parameters_before={k:o.nofile(v) for k,v in params.items()},parameters_after={k:o.nofile(v) for k,v in params.items()},hook_witness=hook,
                     profiler={'count':1,'restored':True,'original_source_sha256':'b'*64,'observer':'exact-original-code-return'})
            result['cases'].append(row);directory=out/'cases'/case;directory.mkdir(parents=True);o.write(directory,'case.json',row)
            witnesses.append({'case':case,**hook})
        (out/'witness.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in witnesses))
        phases=[];expected=[('global','after_weights_load'),('global','after_parameters_before')]
        expected += [(case,phase) for case in o.CASES for phase in o.CASE_PHASES]
        expected += [('global','after_parameters_after'),('global','after_layer_release')]
        for index,(case,phase) in enumerate(expected):
            raw=out/f'physical-{index:03d}.raw.json';raw.write_text(json.dumps(physical()))
            command={'command':['/usr/bin/footprint','-p','12345','--noCategories','-f','bytes','-j',str(raw)],'returncode':0,'stdout':'','stderr':''}
            o.write(out,f'physical-{index:03d}.command.json',command)
            receipt=dict(path=raw.name,sha256=o.sha(raw),pid=12345,case=case,phase=phase);result['physical_receipts'].append(receipt)
            phases.append(dict(index=index,case=case,phase=phase,pid=12345,ppid=os.getpid(),groups_are_evaluated=True,monotonic_s=index,elapsed_s=index,
                active_bytes=1024,peak_bytes=1024,cache_bytes=0,unique_buffer_bytes=4,group_buffer_bytes={'persistent':4},physical_receipt=receipt,
                physical_current_bytes=64<<20,physical_peak_bytes=80<<20,
                physical_top_bytes=64<<20,physical_aux_current_bytes=64<<20,physical_aux_peak_bytes=80<<20))
        (out/'phases.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in phases));o.write(out,'complete.json',result)
        return out,plan,result,phases

    def test_complete_receipt_and_partial_corruptions(self):
        for change in ('none','missing-phase','wrong-physical-pid','missing-case','seed-change','route','release'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as tmp:
                out,plan,result,phases=self.synthetic_child(Path(tmp))
                if change=='missing-phase': phases.pop()
                if change=='wrong-physical-pid':
                    raw=out/'physical-000.raw.json';raw.write_text(json.dumps(physical(pid=12346)))
                if change=='missing-case': result['cases'].pop()
                if change=='seed-change': result['cases'][1]['input_after']['S_seed']['sha256']='d'*64
                if change=='route': result['cases'][0]['hook_witness']['eligible']=False
                if change=='release': result['release']['layer_weakref_dead']=False
                for row in result['cases']: o.write(out/'cases'/row['case'],'case.json',row)
                (out/'phases.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in phases));o.write(out,'complete.json',result)
                with patch.object(o,'raw_record',return_value=out/'unused'):
                    if change=='none': self.assertEqual(o.validate_child(out,'off',12345,plan)['physical_phases'],68)
                    else:
                        with self.assertRaises(RuntimeError): o.validate_child(out,'off',12345,plan)

    def test_derived_peak_may_fall_but_auxiliary_historical_peak_may_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            out,plan,result,phases=self.synthetic_child(Path(tmp))
            def replace_physical(index,top,current,peak):
                raw=physical(current=current,peak=peak);raw['processes'][0]['footprint']=top
                path=out/f'physical-{index:03d}.raw.json';path.write_text(json.dumps(raw))
                phases[index]['physical_receipt']['sha256']=o.sha(path)
                result['physical_receipts'][index]=copy.deepcopy(phases[index]['physical_receipt'])
                phases[index].update(physical_top_bytes=top,physical_aux_current_bytes=current,physical_aux_peak_bytes=peak,
                                     physical_current_bytes=max(top,current),physical_peak_bytes=max(top,peak))
                (out/'phases.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in phases));o.write(out,'complete.json',result)
            replace_physical(0,90<<20,64<<20,80<<20)
            with patch.object(o,'raw_record',return_value=out/'unused'):
                self.assertEqual(o.validate_child(out,'off',12345,plan)['physical_phases'],68)
                replace_physical(1,64<<20,64<<20,79<<20)
                with self.assertRaisesRegex(RuntimeError,'auxiliary peak history'):
                    o.validate_child(out,'off',12345,plan)

    def test_actual_raw_bytes_sha_shape_and_path_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);p=out/'raw.bin';p.write_bytes(b'\x00\x00\x80\x3f')
            row=self.descriptor([2]);row['sha256']=o.sha(p)
            self.assertEqual(o.raw_record(out,row,'bfloat16',[2]),p.resolve())
            for field,value in [('shape',[1]),('sha256','0'*64),('file','../raw.bin'),('bytes',True),('finite',False)]:
                corrupted={**row,field:value}
                with self.subTest(field=field),self.assertRaises(RuntimeError):o.raw_record(out,corrupted,'bfloat16',[2])
            p.write_bytes(b'\x00\x00\x81\x3f')
            with self.assertRaisesRegex(RuntimeError,'SHA'):o.raw_record(out,row,'bfloat16',[2])

    def test_cross_arm_actual_bytes_and_metadata_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);off=root/'off';on=root/'on';off.mkdir();on.mkdir()
            for path,arm in ((off,'off'),(on,'on-lazy')):
                raw=path/'raw.bin';raw.write_bytes(b'\x00\x00\x80\x3f')
                row=self.descriptor([2]);row['sha256']=o.sha(raw)
                cases=[dict(case=case,offset_before=0,offset_after=1,input={'x':row,'conv_seed':None,'S_seed':None},arrays={'output':row}) for case in o.CASES]
                o.write(path,'complete.json',dict(arm=arm,parameters_before={'p':row},parameters_after={'p':row},cases=cases))
            self.assertEqual(o.compare_arms(off,on,'on-lazy')['status'],'PASS')
            (on/'raw.bin').write_bytes(b'\x00\x00\x81\x3f')
            with self.assertRaisesRegex(RuntimeError,'SHA'):o.compare_arms(off,on,'on-lazy')
            self.assertTrue((on/'first-difference.json').exists())

    def run_mock(self, root, mode='done', fail_probe=False, fail_reap=False, partial=False, difference=False):
        p=root/'plan.json';p.write_text('{}')
        args=NS(out=root/'out',plan=p,plan_sha256=o.sha(p),suspended_pid=74384)
        children=[]
        def popen(*a,**kw): child=FakeChild(mode);children.append(child);return child
        def probe(label,pid=None,**kw):
            if fail_probe and label.endswith('/running'): raise RuntimeError('probe failed')
            return sample(pid=pid)
        plan={'execution':{'python':'/not-executed','pins':{}},'bundle':{'path':'/not-read/manifest.json','sha256':'0'*64},'package_root':'/not-loaded'}
        with ExitStack() as stack:
            for name,value in (('getitimer',lambda *a:(0.0,0.0)),('signal',lambda *a:None),('setitimer',lambda *a:None)):
                stack.enter_context(patch.object(o.signal,name,value))
            stack.enter_context(patch.object(o,'prepare',return_value=(plan,{})))
            stack.enter_context(patch.object(o.subprocess,'Popen',side_effect=popen))
            stack.enter_context(patch.object(o.subprocess,'run',return_value=NS(returncode=0 if fail_reap else 1,stdout='12345' if fail_reap else '',stderr='')))
            stack.enter_context(patch.object(o,'footprint',return_value={'status':'measured','physical_pass':True}))
            v=stack.enter_context(patch.object(o,'validate_child'))
            v.side_effect=RuntimeError('partial child evidence') if partial else lambda out,arm,pid,plan,**kw:{'arm':arm,'pid':pid,'status':'PASS'}
            compare=stack.enter_context(patch.object(o,'compare_arms',return_value={'status':'PASS'}))
            if difference:compare.side_effect=RuntimeError('raw cross-arm difference')
            status=o.run(args,NS(snapshot=probe))
        return status,children,json.loads((args.out/'complete.json').read_text()),args.out

    def test_timeout_probe_and_partial_fail_stop_remaining_arms(self):
        for mode,fail_probe,partial,needle in [('timeout',False,False,'synthetic arm deadline'),('timeout',True,False,'probe failed'),('done',False,True,'partial child evidence')]:
            with self.subTest(mode=mode,probe=fail_probe,partial=partial),tempfile.TemporaryDirectory() as tmp:
                status,children,result,out=self.run_mock(Path(tmp),mode,fail_probe,partial=partial)
                self.assertEqual(status,1);self.assertEqual(len(children),1)
                self.assertIn(needle,result['errors'][0]['traceback'])
                self.assertTrue(json.loads((out/'off/release.json').read_text())['strict_absence'])
                if mode=='timeout':
                    self.assertIn(('terminate',),children[0].actions);self.assertIn(('kill',),children[0].actions)

    def test_child_failure_and_failed_reap_keep_first_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            status,children,result,out=self.run_mock(Path(tmp),'childfail',fail_reap=True)
            self.assertEqual(status,1);self.assertEqual(len(children),1)
            self.assertIn('did not exit zero',result['errors'][0]['traceback'])
            self.assertTrue(any(r['phase']=='cleanup' for r in result['errors']))
            self.assertEqual(json.loads((out/'first-error.json').read_text()),result['errors'][0])

    def test_all_three_direct_children_reaped_with_final_pins(self):
        with tempfile.TemporaryDirectory() as tmp:
            status,children,result,out=self.run_mock(Path(tmp))
            self.assertEqual(status,0);self.assertEqual(len(children),3)
            self.assertEqual([r['arm'] for r in result['arms']],o.ARMS)
            self.assertTrue(result['pins_rechecked'])
            for arm in o.ARMS:
                r=json.loads((out/arm/'release.json').read_text());self.assertTrue(r['waited_reaped'] and r['strict_absence'])

    def test_cross_arm_difference_reaps_second_and_never_starts_third(self):
        with tempfile.TemporaryDirectory() as tmp:
            status,children,result,out=self.run_mock(Path(tmp),difference=True)
            self.assertEqual(status,1);self.assertEqual(len(children),2)
            self.assertFalse((out/'on-materialize').exists())
            self.assertIn('raw cross-arm difference',result['errors'][0]['traceback'])
            self.assertTrue(json.loads((out/'on-lazy/release.json').read_text())['strict_absence'])

    def test_help_blocks_runtime_and_socket_imports(self):
        code='''
import importlib.abc,runpy,socket,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if name.split('.')[0] in {'mlx','numpy','torch','resources','resources_v2','resource_base'}: raise AssertionError(name)
sys.meta_path.insert(0,Block());socket.socket=lambda *a,**k: (_ for _ in ()).throw(AssertionError('socket'))
sys.argv=[sys.argv[1],'--help'];runpy.run_path(sys.argv[0],run_name='__main__')
'''
        r=subprocess.run([sys.executable,'-I','-B','-c',code,str(HERE/'layer_owner.py')],capture_output=True,text=True,timeout=5)
        self.assertEqual(r.returncode,0,r.stderr)


if __name__ == '__main__': unittest.main()
