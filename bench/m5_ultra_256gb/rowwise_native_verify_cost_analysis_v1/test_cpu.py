"""Tiny stdlib fixtures only: no actual E054 payload/result reads or backend."""
import ast
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
from bench.m5_ultra_256gb.rowwise_native_verify_cost_analysis_v1 import audit as a


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value)+'\n'); return path


def saved(root,name,dtype,words,shape=None):
    raw=struct.pack('<'+str(len(words))+('H' if dtype=='BF16' else 'I'),*words); shape=shape or [len(words)]
    header=json.dumps({'root/logits':dict(dtype=dtype,shape=shape,data_offsets=[0,len(raw)])}).encode()
    path=root/name; path.write_bytes(struct.pack('<Q',len(header))+header+raw)
    return dict(path=name,sha256=a.sha(path),finite=True,metadata={},tensors={'root/logits':dict(shape=shape,
        dtype=a.DTYPES[dtype],encoding=dtype,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),finite=True)})


def timing_fixture(root,native=100,candidate=80):
    cases=[]; ledger=[]; ids=list(range(8))
    for width,baseline in ((4,'device'),(4,'host'),(8,'device'),(8,'host')):
        samples=[]
        for group in range(5):
            for pair in range(10):
                order=['native','candidate'] if pair%2==0 else ['candidate','native']
                for ordinal,arm in enumerate(order):
                    sample=dict(group=group,pair=pair,order=order,ordinal=ordinal,arm=arm,
                                elapsed_ns=native if arm=='native' else candidate,ids=ids[:width],head=None)
                    samples.append(sample); ledger.append(dict(T=width,baseline=baseline,**sample))
        # Manually expected known numbers, independent of auditor summarize.
        ratio=native/candidate
        summary=dict(groups=[dict(group=g,median_paired_ratio=ratio,ratios=[ratio]*10) for g in range(5)],median_group_ratio=ratio,
                     totals_ns={arm:dict(p50=float(n),p90=n) for arm,n in (('native',native),('candidate',candidate))})
        cases.append(dict(T=width,k=width,baseline=baseline,samples=samples,summary=summary))
    path=root/'timings.jsonl'; path.write_text(''.join(json.dumps(r)+'\n' for r in ledger)); return path,cases,ids,ledger


def qualifications(ids):
    return [dict(T=w,k=k,status='PASS',committed_lens=[1022+k],pending=ids[k-1],
        head=dict(calls=1,status='RETURNED',x=dict(shape=[w,5120],dtype='mlx.core.bfloat16'),
        weight=dict(shape=[248320,640],dtype='mlx.core.uint32'),scales=dict(shape=[248320,320],dtype='mlx.core.uint8'),
        output=dict(shape=[w,248320],dtype='mlx.core.bfloat16'))) for w in (4,8) for k in (1,w//2,w)]


def closure_fixture(root):
    screen=root/'code/screen.py'; screen.parent.mkdir(); screen.write_text('# fixture\n')
    execution=root/'execution'; owner=root/'owner'; lease=root/'lease'; child=root/'numerical'
    e042plan=put(root/'prior/plan.json',{}); seed=dict(used=1023,position=1022,status='PASS',tokens=list(range(8)))
    e042result=put(root/'prior/result.json',dict(status='PASS',forward_prefix_qualified=True,plan_sha256=a.sha(e042plan),reference_blocks=[seed]))
    child_plan=dict(schema='C1-verifier-cost-plan-v1',screen_sha256=a.sha(screen),deadline_s=1500,
                    **{key:dict(path=str(path),sha256=a.sha(path)) for key,path in (('e042_plan',e042plan),('e042_result',e042result))})
    cpp=put(screen.parent/'prereg-v1/child-plan.json',child_plan); put(child/'plan.json',child_plan)
    numerical=[str(a.R/'.venv/bin/python'),'-I','-B',str(screen),'--run','--plan',str(cpp),'--plan-sha256',a.sha(cpp),'--out',str(child)]
    pins={str(p):a.sha(p) for p in (screen,e042plan,e042result,cpp)}
    parent=dict(schema='E054-c1-verifier-cost-owner-plan-v1',outputs=dict(execution=str(execution),screen=str(owner),lease=str(lease),numerical=str(child)),
                pins=pins,runtime_pins={'receipt':'unchanged'},service=dict(pid=52818,identity='Mon Oct19 09:22:08'),numerical_command=numerical,numerical_plan_sha256=a.sha(cpp))
    pp=put(root/'parent.json',parent); put(execution/'plan.json',parent)
    actual=dict(parent,physical_baseline={'swap_bytes':0}); ep=put(execution/'execution-plan.json',actual); put(owner/'plan.json',actual)
    command=[str(a.R/'.venv/bin/python'),'-I','-B',str(screen.with_name('runner.py')),'--plan',str(ep),'--plan-sha256',a.sha(ep),'--out',str(owner)]
    put(execution/'final.json',dict(status='PASS_CAPTURE_AND_RESTORE',errors=[],lease_returncode=0,service_identity_after=parent['service']['identity']))
    put(lease/'lease.json',dict(closed=True,service_pid=52818,service_identity=parent['service']['identity'],jobs=[command],child_pid=222,owner_pid=111))
    put(lease/'resume.json',dict(identity_match=True,state_after='S'))
    put(lease/'job-0.json',dict(command=command,waited=True,returncode=0,pid=222))
    put(execution/'runner-identity.json',dict(pid=222,ppid=111))
    put(owner/'result.json',dict(schema='E054-c1-cost-owner-v1',status='PASS_CAPTURE_ONLY',errors=[],child_reaped=True,child_returncode=0,physical_samples=2,resource_samples=2))
    put(owner/'child-reap.json',dict(pid=333,waited=True,absent=True,returncode=0,errors=[]))
    complete=put(child/'complete.json',dict(pid=333,ppid=222,plan_sha256=a.sha(cpp),seed=seed))
    done=put(owner/'numerical-done.json',dict(schema='E054-terminal-hold-v1',pid=333,ppid=222,qualifier_returncode=0,wrapper_errors=[],command=numerical,
        qualifier_complete=dict(path=str(complete),sha256=a.sha(complete),status='PASS_LOCAL_COST_ONLY')))
    physical=put(owner/'resources/physical-0001/sample.json',{}); metadata=put(owner/'resources/resource-0001.json',{})
    put(owner/'numerical-exit-ack.json',dict(command=numerical,pid=333,owner_pid=222,done_sha256=a.sha(done),final_physical_and_metadata_complete=True,
        physical_sha256=a.sha(physical),metadata_sha256=a.sha(metadata)))
    copies=[]
    for i,(name,digest) in enumerate(pins.items()):
        target=execution/'sources'/f'{i}.txt'; target.parent.mkdir(exist_ok=True); target.write_bytes(Path(name).read_bytes())
        copies.append(dict(original=name,copy=str(target),sha256=digest,bytes=target.stat().st_size))
    put(execution/'source-copies.json',copies); put(execution/'runtime-pins-verified.json',parent['runtime_pins'])
    return screen,pp,child


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); self.root=Path(self.tmp.name).resolve()

    def test_default_off_and_stdlib_only(self):
        with patch.object(a,'audit',side_effect=AssertionError('must remain inert')),contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(a.main([]),0)
        value=json.loads(output.getvalue()); self.assertEqual((value['raw_files'],value['tensor_pairs'],value['timing_count']),(32,2064,400))
        tree=ast.parse(Path(a.__file__).read_text()); imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        imports += [x.name for n in ast.walk(tree) if isinstance(n,ast.Import) for x in n.names]
        self.assertFalse(any(x.split('.')[0] in ('mlx','numpy','torch','safetensors','subprocess','socket') for x in imports))

    def test_raw_original_bits_finite_hash_shape_and_signed_zero(self):
        reader=a.raw_reader(); left=saved(self.root,'a.safetensors','BF16',[0,0]); right=saved(self.root,'b.safetensors','BF16',[0,0x8000])
        l,r=reader.snapshot_index(self.root,left),reader.snapshot_index(self.root,right)
        self.assertEqual(a.compare_pair(reader,l,l,'same')['tensor_pairs'],1)
        with self.assertRaisesRegex(ValueError,'original-bit mismatch'): a.compare_pair(reader,l,r,'zero')
        for dtype,word in (('BF16',0x7f80),('BF16',0xffc0),('F32',0x7f800001)):
            desc=saved(self.root,'nan.safetensors',dtype,[word])
            with self.assertRaisesRegex(ValueError,'finite'): reader.snapshot_index(self.root,desc)
        desc=saved(self.root,'short.safetensors','F32',[0],[2])
        with self.assertRaisesRegex(ValueError,'bounds/shape'): reader.snapshot_index(self.root,desc)
        path=self.root/left['path']; path.write_bytes(path.read_bytes()[:-1]+b'\x01')
        with self.assertRaisesRegex(ValueError,'SHA'): reader.snapshot_index(self.root,left)

    def test_raw_late_state_word_and_post_index_mutation(self):
        reader=a.raw_reader(); left=saved(self.root,'a.safetensors','F32',[0,0]); right=saved(self.root,'b.safetensors','F32',[0,1])
        l,r=reader.snapshot_index(self.root,left),reader.snapshot_index(self.root,right)
        with self.assertRaisesRegex(ValueError,'element_index'): a.compare_pair(reader,l,r,'last-state')
        path=self.root/right['path']; path.write_bytes(path.read_bytes()[:-1]+b'\x01')
        with self.assertRaisesRegex(ValueError,'changed after indexing'): reader.tensor_bytes(r,'root/logits')

    def test_complete_padding_original_layout_and_full_vocab_argmax(self):
        spec=a.layout(8); self.assertEqual(len(spec),129)
        index=dict(tensors={k:dict(shape=s,encoding=d) for k,(s,d) in spec.items()}); a.check_layout(index,8)
        for key,field,value in [('root/state/63/K','shape',[1,4,1023,256]),('root/state/62/S','encoding','BF16'),('root/logits','shape',[8,1])]:
            bad=copy.deepcopy(index); bad['tensors'][key][field]=value
            with self.assertRaises(ValueError): a.check_layout(bad,8)
        words=[0]*248320; words[-1]=0x3f80; self.assertEqual(a.greedy(struct.pack('<248320H',*words),1),[248319])
        self.assertEqual(a.greedy(struct.pack('<4H',0x8000,0,0xbf80,0),1,4),[0])

    def test_six_selected_commit_and_actual_returned_head_receipts(self):
        ids=list(range(8)); rows=qualifications(ids); a.qualification(rows,ids)
        for field,value in [('pending',999),('committed_lens',[1024]),('status','FAIL')]:
            bad=copy.deepcopy(rows); bad[0][field]=value
            with self.assertRaises(ValueError): a.qualification(bad,ids)
        bad=copy.deepcopy(rows); bad[-1]['head']['status']='EVALUATED'
        with self.assertRaisesRegex(ValueError,'head'): a.qualification(bad,ids)
        with self.assertRaises(ValueError): a.qualification(rows[:-1],ids)

    def test_all400_ordered_durations_independent_known_statistics(self):
        path,cases,ids,rows=timing_fixture(self.root); result,total=a.timing_ledger(path,cases,ids,1500)
        self.assertEqual(total,200*(100+80)); self.assertEqual(len(result),4)
        for case in result:
            self.assertEqual(case['summary']['median_group_ratio'],1.25)
            self.assertEqual(case['break_even']['median_group_remaining_ns'],20)
            self.assertEqual(case['break_even']['positive_pairs'],50)

    def test_timing_rejects_order_truncation_wrong_ids_bool_and_summary(self):
        path,cases,ids,rows=timing_fixture(self.root)
        for mutate in ('order','id','bool','head','extra','truncated'):
            bad=copy.deepcopy(rows)
            if mutate=='order': bad[0],bad[1]=bad[1],bad[0]
            elif mutate=='id': bad[399]['ids'][0]=99
            elif mutate=='bool': bad[0]['elapsed_ns']=True
            elif mutate=='head': bad[0]['head']={}
            elif mutate=='extra': bad.append(bad[-1])
            text=''.join(json.dumps(row)+'\n' for row in bad)
            path.write_text(text[:-1] if mutate=='truncated' else text)
            with self.subTest(mutate=mutate),self.assertRaises(ValueError): a.timing_ledger(path,cases,ids,1500)
        path,cases,ids,_=timing_fixture(self.root); cases[0]['summary']['median_group_ratio']=2
        with self.assertRaisesRegex(ValueError,'recomputed'): a.timing_ledger(path,cases,ids,1500)

    def test_negative_draft_budget_and_slow_group_are_preserved(self):
        path,cases,ids,_=timing_fixture(self.root,native=80,candidate=100); result,_=a.timing_ledger(path,cases,ids,1500)
        self.assertTrue(all(r['break_even']['median_group_remaining_ns']==-20 and r['break_even']['positive_pairs']==0 for r in result))
        samples=copy.deepcopy(cases[0]['samples']); samples[0]['elapsed_ns']=1
        summary,budget=a.summarize(samples); self.assertLess(summary['groups'][0]['ratios'][0],.8)
        self.assertEqual(budget['groups'][0]['remaining_draft_and_overhead_ns'][0],-99)

    def test_complete_producer_receipts_missing_late_array_rejected(self):
        def checks(label,spec): return [dict(check=label,name=k,shape=s,dtype=a.DTYPES[d],reference_finite=True,candidate_finite=True,bit_equal=True) for k,(s,d) in spec.items()]
        rows=[]
        for label in a.expected_labels(): rows += checks(label,a.layout(1 if label.endswith('/continuation') else int(label[1])))
        rows += [r for arm in ('canonical','oracle','host','device','candidate') for r in checks('canonical-preserved/'+arm,a.layout(1,True))]
        post=[r for arm in ('canonical','oracle','host','device','candidate') for r in checks('post-timing-canonical/'+arm,a.layout(1,True))]
        complete=dict(checks=rows,post_checks=post); a.producer_checks(complete)
        post.pop()
        with self.assertRaisesRegex(ValueError,'post-timing'): a.producer_checks(complete)

    def test_closed_source_and_pid_chain_then_tamper_refused(self):
        screen,pp,child=closure_fixture(self.root)
        with patch.object(a,'SCREEN',screen),patch.object(a,'SOURCE_PINS',{str(screen):a.sha(screen)}):
            a.closure(pp,a.sha(pp),child,{})
            mutations=[('lease/lease.json','closed',False),('lease/job-0.json','waited',False),('owner/numerical-exit-ack.json','done_sha256','0'*64),
                       ('owner/numerical-exit-ack.json','physical_sha256','0'*64),('owner/child-reap.json','absent',False),('execution/final.json','status','FAIL')]
            for relative,key,value in mutations:
                path=self.root/relative; original=path.read_bytes(); changed=a.read(path); changed[key]=value; put(path,changed)
                with self.subTest(relative=relative,key=key),self.assertRaises(ValueError): a.closure(pp,a.sha(pp),child,{})
                path.write_bytes(original)
            screen.write_text('# changed\n')
            with self.assertRaisesRegex(ValueError,'source pin mismatch'): a.closure(pp,a.sha(pp),child,{})

    def test_closed_gate_precedes_any_raw_reader(self):
        screen,pp,child=closure_fixture(self.root); final=self.root/'execution/final.json'; value=a.read(final); value['status']='FAIL'; put(final,value)
        with patch.object(a,'SCREEN',screen),patch.object(a,'SOURCE_PINS',{str(screen):a.sha(screen)}),patch.object(a,'raw_reader',side_effect=AssertionError('must not read any payload')):
            result=a.audit(child,pp,a.sha(pp))
        self.assertEqual(result['status'],'FAIL'); self.assertEqual(result['raw_files'],0); self.assertIn('restore',result['error']['message'])

    def test_failed_attempt_is_saved_no_overwrite_and_strict_json(self):
        out=self.root/'fail.json'
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(a.main(['--run','--child',str(self.root/'missing'),'--parent-plan',str(self.root/'missing.json'),'--parent-plan-sha256','0'*64,'--out',str(out)]),1)
        self.assertEqual(a.read(out)['status'],'FAIL')
        with self.assertRaisesRegex(ValueError,'fresh output'): a.main(['--run','--child',str(self.root),'--parent-plan',str(out),'--parent-plan-sha256','0'*64,'--out',str(out)])
        for raw in ('{"a":1,"a":2}','{"a":NaN}'):
            with self.assertRaises(ValueError): a.parse(raw)


if __name__=='__main__': unittest.main()
