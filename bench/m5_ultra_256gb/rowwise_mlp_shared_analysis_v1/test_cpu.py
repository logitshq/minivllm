"""Small synthetic stdlib evidence; no actual E056 result/model/backend reads."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
from bench.m5_ultra_256gb.rowwise_mlp_shared_analysis_v1 import audit as a


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n');return path


def parameters():
    return {f'p{i}':dict(shape=[1],dtype='mlx.core.uint8',bytes=1,sha256='0'*64,finite=True,handle_id=i+1) for i in range(1349)}


def receipts():
    seed=dict(used=1023,position=1022,status='PASS',tokens=list(range(8)),original_fixture=dict(eos_token_ids=[100]))
    return dict(schema='C1-MLP-shared-activation-canary-result-v1',status=a.SUCCESS,seed=seed,contract=a.producer_contract(),
        projection_pairs=12,raw_files=30,callback_count=256,native_m1_projection_calls=1536,shadow_projection_calls=12,shared_outputs_returned=0,
        projection_canary_complete=True,native_returning_provenance_complete=True,device_cleanup_completed=True,source_model_pins_rechecked=True,
        full_model_qualified=False,http_qualified=False,performance_qualified=False,engine_pareto_point=False,
        traversals=[dict(T=w,callback_count=128,native_m1_projection_calls=128*w,shadow_projection_calls=6,shared_outputs_returned=0,status='PASS') for w in (4,8)],
        provenance=[dict(T=w,k=w,status='PASS',labels=[f'T{w}/provenance/{arm}' for arm in ('native-causal','frozen-verifier')],
            full_logits=True,selected_state_arrays=128,raw_state_saved=False,shared_outputs_returned=0) for w in (4,8)],
        checks=a.expected_checks(),teacher=dict(inputs=list(range(9)),output_ids=list(range(1,10)),native_steps=9),
        device=dict(name='Apple M5 Ultra',architecture='applegpu_g17d',mlx_version='0.32.3'),elapsed_wall_s=10,
        memory_after_cleanup=dict(active_bytes=0,peak_bytes=1,cache_bytes=0))


def saved(root,name,width,size,key,*,last_word=0):
    raw=b'\0\0'*(width*size-1)+struct.pack('<H',last_word);shape=[width,size]
    header=json.dumps({'root/'+key:dict(dtype='BF16',shape=shape,data_offsets=[0,len(raw)])}).encode()
    path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(struct.pack('<Q',len(header))+header+raw)
    desc=dict(path=name,sha256=a.sha(path),finite=True,metadata={key:{'array':'root/'+key}},tensors={'root/'+key:dict(shape=shape,
        dtype=a.BF16,encoding='BF16',bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),finite=True)})
    put(path.with_suffix('.json'),desc);return desc


def captures(root):
    rows=[]
    for w in (4,8):
        for i in (0,32,63):
            tag=f'T{w}/layer{i}'
            rows.append(dict(T=w,layer=i,input=saved(root,tag+'/input.safetensors',w,5120,'x'),projections=[
                dict(name=n,status='PASS',files=[saved(root,tag+'/'+n+'-'+arm+'.safetensors',w,17408,'output') for arm in ('native','shadow')]) for n in ('gate','up')]))
    return rows


def closure_fixture(root):
    canary=root/'code/canary.py';canary.parent.mkdir();canary.write_text('# fake canary\n')
    ownercode=root/'owner-code';ownercode.mkdir()
    for name in ('execute.py','runner.py','terminal_hold.py'):(ownercode/name).write_text('# fixture '+name+'\n')
    execution=root/'execution';owner=root/'owner';lease=root/'lease';child=root/'numerical';complete=receipts()
    pp54=put(root/'prior/child-plan.json',{})
    rp54=put(root/'prior/complete.json',dict(schema='C1-verifier-cost-result-v1',status='PASS_LOCAL_COST_ONLY',plan_sha256=a.sha(pp54),seed=complete['seed']))
    param54=put(root/'prior/parameters-before.json',parameters())
    ap54=put(root/'prior/independent.json',dict(schema='E054-local-cost-independent-v2',status='PASS',local_cost_audited=True,
        complete_sha256=a.sha(rp54),parent_plan_sha256=a.E054_PARENT_SHA,source_and_small_evidence={str(param54):a.sha(param54)}))
    refs={key:dict(path=str(path),sha256=a.sha(path)) for key,path in (('e054_plan',pp54),('e054_result',rp54),('e054_audit',ap54))}
    cp=put(ownercode/'prereg-v1/child-plan.json',dict(schema='C1-MLP-shared-activation-plan-v1',canary_sha256=a.sha(canary),deadline_s=900,**refs))
    put(child/'plan.json',a.read(cp));numcmd=[str(a.R/'.venv/bin/python'),'-I','-B',str(canary),'--run','--plan',str(cp),'--plan-sha256',a.sha(cp),'--out',str(child)]
    prior55=root/'prior55';lp=prior55/'lease';ep=prior55/'execution'
    plan55=put(prior55/'plan.json',dict(outputs=dict(lease=str(lp),execution=str(ep))))
    closure55=[put(lp/'lease.json',dict(closed=True,child_pid=77,jobs=[['prior']],service_pid=52818,service_identity='Mon Oct19 09:22:08')),
        put(lp/'job-0.json',dict(waited=True,returncode=0,pid=77,command=['prior'])),put(lp/'resume.json',dict(identity_match=True,state_after='S')),
        put(ep/'final.json',dict(status='PASS_CAPTURE_AND_RESTORE',service_identity_after='Mon Oct19 09:22:08'))]
    paths=[plan55,*closure55,canary,*[ownercode/n for n in ('execute.py','runner.py','terminal_hold.py')],pp54,rp54,ap54,cp]
    pins={str(p):a.sha(p) for p in paths};sourcepins={str(canary):a.sha(canary)}
    parent=dict(schema=a.PARENT_SCHEMA,outputs=dict(execution=str(execution),screen=str(owner),lease=str(lease),numerical=str(child)),pins=pins,
        runtime_pins={'original':'receipt'},child_environment={},screen_deadline_seconds=1650,driver_deadline_seconds=1800,max_observation_gap_seconds=12,physical={},scope='fixture',service=dict(pid=52818,identity='Mon Oct19 09:22:08',process={}),numerical_command=numcmd,numerical_plan_sha256=a.sha(cp))
    pp=put(root/'parent.json',parent);put(execution/'plan.json',parent)
    actual=dict(parent,physical_baseline={'swap_bytes':0});ep=put(execution/'execution-plan.json',actual);put(owner/'plan.json',actual)
    cmd=[str(a.R/'.venv/bin/python'),'-I','-B',str(ownercode/'runner.py'),'--plan',str(ep),'--plan-sha256',a.sha(ep),'--out',str(owner)]
    put(execution/'final.json',dict(status='PASS_CAPTURE_AND_RESTORE',errors=[],lease_returncode=0,service_identity_after=parent['service']['identity']))
    put(lease/'lease.json',dict(closed=True,service_pid=52818,service_identity=parent['service']['identity'],jobs=[cmd],child_pid=222,owner_pid=111))
    put(lease/'resume.json',dict(identity_match=True,state_after='S'));put(lease/'job-0.json',dict(command=cmd,waited=True,returncode=0,pid=222))
    put(execution/'runner-identity.json',dict(pid=222,ppid=111))
    put(owner/'result.json',dict(schema=a.OWNER_SCHEMA,status='PASS_CAPTURE_ONLY',errors=[],child_reaped=True,child_returncode=0,physical_samples=2,resource_samples=2))
    put(owner/'child-reap.json',dict(pid=333,waited=True,absent=True,returncode=0,errors=[]))
    complete.update(pid=333,ppid=222,plan_sha256=a.sha(cp),sources=sourcepins)
    cpath=put(child/'complete.json',complete)
    done=put(owner/'numerical-done.json',dict(schema=a.TERMINAL_SCHEMA,pid=333,ppid=222,qualifier_returncode=0,wrapper_errors=[],command=numcmd,
        qualifier_complete=dict(path=str(cpath),sha256=a.sha(cpath),status=a.SUCCESS)))
    physical=put(owner/'resources/physical-0001/sample.json',{});metadata=put(owner/'resources/resource-0001.json',{})
    put(owner/'numerical-exit-ack.json',dict(command=numcmd,pid=333,owner_pid=222,done_sha256=a.sha(done),final_physical_and_metadata_complete=True,
        physical_sha256=a.sha(physical),metadata_sha256=a.sha(metadata)))
    copies=[]
    for i,(name,digest) in enumerate(pins.items()):
        target=execution/'sources'/f'{i}.txt';target.parent.mkdir(exist_ok=True);target.write_bytes(Path(name).read_bytes())
        copies.append(dict(original=name,copy=str(target),sha256=digest,bytes=target.stat().st_size))
    put(execution/'source-copies.json',copies);put(execution/'runtime-pins-verified.json',parent['runtime_pins'])
    return dict(CANARY=canary,OWNER=ownercode,SOURCE_PINS=sourcepins,DEPENDENCY_PINS={},E055_PLAN=plan55,E055_PLAN_SHA=a.sha(plan55),E054={k:v['sha256'] for k,v in refs.items()}),pp,child


def rebind_complete(root,complete):
    path=put(root/'numerical/complete.json',complete);done=root/'owner/numerical-done.json';d=a.read(done);d['qualifier_complete']['sha256']=a.sha(path);put(done,d)
    ack=root/'owner/numerical-exit-ack.json';d=a.read(ack);d['done_sha256']=a.sha(done);put(ack,d)


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()

    def test_default_off_no_live_or_raw_access(self):
        with patch.object(a,'audit',side_effect=AssertionError('audit default')),contextlib.redirect_stdout(io.StringIO()) as out:self.assertEqual(a.main([]),0)
        value=json.loads(out.getvalue());self.assertEqual((value['raw_files'],value['projection_pairs'],value['raw_state_files']),(30,12,0));self.assertFalse(value['raw_full_state_checked'])

    def test_all30_real_size_synthetic_files_and12_pairs(self):
        complete=dict(captures=captures(self.root));result=dict(raw_files=0,input_files=0,tensor_pairs=0,pairs=[])
        a.raw_captures(self.root,complete,result,{},{});self.assertEqual((result['raw_files'],result['input_files'],result['tensor_pairs']),(30,6,12))
        self.assertTrue(all(p['bit_equal'] for p in result['pairs']))
        (self.root/'hidden.safetensors').write_bytes(b'extra')
        with self.assertRaisesRegex(ValueError,'exact30'):a.raw_captures(self.root,complete,dict(raw_files=0,input_files=0,tensor_pairs=0,pairs=[]),{},{})

    def test_last_word_signed_zero_and_nonfinite_not_tolerated(self):
        reader=a.h.raw_reader();left=saved(self.root,'a.safetensors',1,2,'output');right=saved(self.root,'b.safetensors',1,2,'output',last_word=0x8000)
        with self.assertRaisesRegex(ValueError,'original-bit mismatch'):a.h.compare_pair(reader,reader.snapshot_index(self.root,left),reader.snapshot_index(self.root,right),'signedzero')
        for word in (0x7f80,0xff80,0x7fc1):
            desc=saved(self.root,'bad.safetensors',1,2,'output',last_word=word)
            with self.assertRaisesRegex(ValueError,'finite'):reader.snapshot_index(self.root,desc)
        path=self.root/left['path'];path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError,'SHA'):reader.snapshot_index(self.root,left)

    def test_exact_full_shape_and_sidecar_cannot_hide_reduced_input(self):
        rows=captures(self.root);rows[0]['input']=saved(self.root,'T4/layer0/input.safetensors',4,1,'x')
        with self.assertRaisesRegex(ValueError,'full saved original shape'):a.raw_captures(self.root,dict(captures=rows),dict(raw_files=0,input_files=0,tensor_pairs=0,pairs=[]),{},{})
        rows[0]['input']=saved(self.root,'T4/layer0/input.safetensors',4,5120,'x');side=self.root/'T4/layer0/input.json';d=a.read(side);d['finite']=False;put(side,d)
        with self.assertRaisesRegex(ValueError,'sidecar'):a.raw_captures(self.root,dict(captures=rows),dict(raw_files=0,input_files=0,tensor_pairs=0,pairs=[]),{},{})

    def test_exact1183_receipts_and_double_root_canonical_keys(self):
        complete=receipts();a.producer_receipts(complete);self.assertEqual(len(complete['checks']),1183)
        canonical=[r for r in complete['checks'] if r['check'].startswith('canonical-preserved/')];self.assertEqual(len(canonical),655)
        self.assertTrue(all(r['name'].startswith('root/root/') for r in canonical))
        for mutate in ('late-state','canonical-path','fallback','unqualified','teacher'):
            bad=copy.deepcopy(complete)
            if mutate=='late-state':bad['checks']=[r for r in bad['checks'] if not (r['check']=='T8/provenance/frozen-verifier' and r['name']=='root/state/63/V')]
            elif mutate=='canonical-path':next(r for r in bad['checks'] if r['check'].startswith('canonical-preserved/'))['name']='root/state/0/S'
            elif mutate=='fallback':bad['traversals'][1]['shared_outputs_returned']=1
            elif mutate=='unqualified':bad['full_model_qualified']=True
            else:bad['teacher']['output_ids'][7]=77
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):a.producer_receipts(bad)

    def test_closed_source_pid_ack_chain_and_changed_receipts_reject(self):
        changes,pp,child=closure_fixture(self.root)
        with patch.multiple(a,**changes):
            a.closed_owner(pp,a.sha(pp),child,{})
            for name,key,value in [('lease/lease.json','closed',False),('owner/child-reap.json','absent',False),('owner/numerical-exit-ack.json','done_sha256','0'*64),
                                   ('owner/numerical-exit-ack.json','metadata_sha256','0'*64),('execution/final.json','status','FAIL')]:
                path=self.root/name;raw=path.read_bytes();d=a.read(path);d[key]=value;put(path,d)
                with self.subTest(name=name,key=key),self.assertRaises(ValueError):a.closed_owner(pp,a.sha(pp),child,{})
                path.write_bytes(raw)
            (changes['OWNER']/'runner.py').write_text('# changed\n')
            with self.assertRaisesRegex(ValueError,'source copy|source pin'):a.closed_owner(pp,a.sha(pp),child,{})

    def test_closed_gate_before_any_raw_file_is_opened(self):
        changes,pp,child=closure_fixture(self.root);path=self.root/'execution/final.json';d=a.read(path);d['status']='FAIL';put(path,d)
        with patch.multiple(a,**changes),patch.object(a,'raw_captures',side_effect=AssertionError('raw access')):result=a.audit(child,pp,a.sha(pp))
        self.assertEqual(result['status'],'FAIL');self.assertEqual(result['raw_files'],0);self.assertIn('restore',result['error']['message'])

    def test_complete_synthetic_audit_then_parameter_change_retains_failure(self):
        changes,pp,child=closure_fixture(self.root);complete=a.read(child/'complete.json');complete['captures']=captures(child);rebind_complete(self.root,complete)
        for n in ('parameters-before.json','parameters-after.json'):put(child/n,parameters())
        with patch.multiple(a,**changes):
            result=a.audit(child,pp,a.sha(pp));self.assertEqual(result['status'],'PASS',result.get('error'))
            self.assertTrue(result['projection_bytes_audited']);self.assertFalse(result['raw_full_state_checked']);self.assertEqual(result['tensor_pairs'],12)
            d=parameters();d['p1348']['sha256']='1'*64;put(child/'parameters-after.json',d)
            result=a.audit(child,pp,a.sha(pp));self.assertEqual(result['status'],'FAIL');self.assertFalse(result['projection_bytes_audited']);self.assertEqual(result['tensor_pairs'],12)
            self.assertIn('parameter value',result['error']['message'])

    def test_failed_cli_saved_and_never_overwritten_strict_json(self):
        out=self.root/'failed.json';args=['--run','--child',str(self.root/'absent'),'--parent-plan',str(self.root/'missing.json'),'--parent-plan-sha256','0'*64,'--out',str(out)]
        with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(a.main(args),1)
        self.assertEqual(a.read(out)['status'],'FAIL')
        with self.assertRaisesRegex(ValueError,'fresh output'):a.main(args)
        for value in ('{"a":1,"a":2}','{"a":NaN}'):
            with self.assertRaises(ValueError):a.parse(value)


if __name__=='__main__':unittest.main()
