"""Bounded synthetic metadata/timing fixtures; no actual E057/model data."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from bench.m5_ultra_256gb.rowwise_mlp_shared_cost_analysis_v1 import audit as a
from bench.m5_ultra_256gb.rowwise_mlp_shared_analysis_v1 import test_cpu as f56
from bench.m5_ultra_256gb.rowwise_native_verify_cost_analysis_v1 import test_cpu as f54

put=f56.put


def mlp_rows():
    return [dict(T=w,k=k,status='PASS',call_count=128,scope='Python qualification calls; not GPU dispatch counts or timed observer',calls=[
        dict(layer=i,projection=n,T=w,K=5120,N=17408,weight_shape=[17408,640],scales_shape=[17408,320],input_dtype='mlx.core.bfloat16',output_dtype='mlx.core.bfloat16')
        for i in range(64) for n in ('gate','up')]) for w,k in ((4,1),(4,2),(4,4),(8,1),(8,4),(8,8))]


def checks():
    def rows(label,spec):return [dict(check=label,name=k,shape=s,dtype=a.h.DTYPES[d],reference_finite=True,candidate_finite=True,bit_equal=True) for k,(s,d) in spec.items()]
    before=[]
    for label in a.h.expected_labels():before+=rows(label,a.h.layout(1 if label.endswith('/continuation') else int(label[1])))
    canonical={'root/'+k:v for k,v in a.h.layout(1,True).items()}
    before += [r for arm in ('canonical','oracle','host','device','candidate') for r in rows('canonical-preserved/'+arm,canonical)]
    after=[r for arm in ('canonical','oracle','host','device','candidate') for r in rows('post-timing-canonical/'+arm,canonical)]
    return before,after


def fixture(root):
    priorroot=root/'e056';priorroot.mkdir();oldchanges,oldparent,oldchild=f56.closure_fixture(priorroot)
    priorparameters=put(oldchild/'parameters-before.json',f56.parameters())
    # E056 fixture canonical metadata and records are complete; its independent
    # result is a synthetic accepted prerequisite, not an actual GPU audit.
    oldplan=oldchanges['OWNER']/'prereg-v1/child-plan.json'
    prioraudit=put(priorroot/'independent.json',dict(schema='E056-MLP-canary-independent-v1',status='PASS',projection_bytes_audited=True,
        audit_source_sha256=a.CANARY_AUDIT_SHA,complete_sha256=a.sha(oldchild/'complete.json'),parent_plan_sha256=a.sha(oldparent),
        raw_files=30,input_files=6,tensor_pairs=12,provenance_receipt_pairs=4,provenance_receipt_arrays=516,raw_full_state_checked=False,
        source_and_small_evidence={str(priorparameters):a.sha(priorparameters)}))
    target=root/'e057';target.mkdir();changes,parentpath,child=f56.closure_fixture(target)
    screen,ownercode=changes['CANARY'],changes['OWNER'];planpath=ownercode/'prereg-v1/child-plan.json'
    refs={k:dict(path=str(p),sha256=a.sha(p)) for k,p in [('e054_plan',priorroot/'prior/child-plan.json'),('e056_plan',oldplan),('e056_result',oldchild/'complete.json'),('e056_audit',prioraudit)]}
    childplan=dict(schema='C1-MLP-shared-cost-plan-v1',adapter_sha256=a.sha(screen),deadline_s=900,**refs)
    put(planpath,childplan);put(child/'plan.json',childplan)
    parent=a.read(parentpath);parent['schema']=a.PARENT_SCHEMA;parent['numerical_plan_sha256']=a.sha(planpath);parent['numerical_command'][8]=a.sha(planpath)
    for path in [planpath,oldparent,*[Path(ref['path']) for ref in refs.values()]]:parent['pins'][str(path)]=a.sha(path)
    put(parentpath,parent);put(target/'execution/plan.json',parent)
    actual=dict(parent,physical_baseline={'swap_bytes':0});ep=put(target/'execution/execution-plan.json',actual);put(target/'owner/plan.json',actual)
    lease=a.read(target/'lease/lease.json');lease['jobs'][0][7]=a.sha(ep);put(target/'lease/lease.json',lease)
    job=a.read(target/'lease/job-0.json');job['command']=lease['jobs'][0];put(target/'lease/job-0.json',job)
    owner=a.read(target/'owner/result.json');owner['schema']=a.OWNER_SCHEMA;put(target/'owner/result.json',owner)
    done=a.read(target/'owner/numerical-done.json');done['schema']=a.TERMINAL_SCHEMA;done['command']=parent['numerical_command'];done['qualifier_complete']['status']=a.SUCCESS;put(target/'owner/numerical-done.json',done)
    ack=a.read(target/'owner/numerical-exit-ack.json');ack['command']=parent['numerical_command'];put(target/'owner/numerical-exit-ack.json',ack)
    sourcepins={str(screen):a.sha(screen)};complete=a.read(child/'complete.json');before,after=checks()
    complete.update(schema='C1-MLP-shared-cost-result-v1',status=a.SUCCESS,plan_sha256=a.sha(planpath),sources=sourcepins,
        contract=a.producer_contract(),inherited_contract=a.base.contract(),inherited_plan=refs['e054_plan'],
        adaptation='Original E054 device code with private candidate_run/write/helper dependencies; actual shared MLP+head candidate, not original E054 run.',
        numerical_screen_qualified=True,pre_timing_parameters_unchanged=True,mlp_qualifications=mlp_rows(),checks=before,post_checks=after,
        qualification=f54.qualifications(list(range(8))))
    complete['seed']=a.read(oldchild/'complete.json')['seed'];complete['seed']['tokens']=[999,*range(7)]
    # The same actual seed must flow through the prerequisite too.
    oldcomplete=a.read(oldchild/'complete.json');oldcomplete['seed']=copy.deepcopy(complete['seed']);oldcomplete['teacher']['inputs']=[999,*range(8)];oldcomplete['teacher']['output_ids']=list(range(9))
    put(oldchild/'complete.json',oldcomplete);pri=a.read(prioraudit);pri['complete_sha256']=a.sha(oldchild/'complete.json');put(prioraudit,pri)
    for key,path in [('e056_result',oldchild/'complete.json'),('e056_audit',prioraudit)]:
        childplan[key]['sha256']=a.sha(path);parent['pins'][str(path)]=a.sha(path)
    put(planpath,childplan);put(child/'plan.json',childplan);parent['pins'][str(planpath)]=a.sha(planpath)
    parent['numerical_plan_sha256']=a.sha(planpath);parent['numerical_command'][8]=a.sha(planpath);complete['plan_sha256']=a.sha(planpath)
    put(parentpath,parent);put(target/'execution/plan.json',parent);actual=dict(parent,physical_baseline={'swap_bytes':0});put(ep,actual);put(target/'owner/plan.json',actual)
    lease['jobs'][0][7]=a.sha(ep);put(target/'lease/lease.json',lease);job['command']=lease['jobs'][0];put(target/'lease/job-0.json',job)
    done['command']=parent['numerical_command'];put(target/'owner/numerical-done.json',done);ack['command']=parent['numerical_command'];put(target/'owner/numerical-exit-ack.json',ack)
    _,timings,ids,_=f54.timing_fixture(child);complete['timings']=timings;f56.rebind_complete(target,complete)
    for name in ('parameters-before.json','parameters-before-timing.json','parameters-after.json'):put(child/name,f56.parameters())
    copies=[]
    for i,(name,digest) in enumerate(parent['pins'].items()):
        path=target/'execution/sources'/f'{i}.txt';path.write_bytes(Path(name).read_bytes());copies.append(dict(original=name,copy=str(path),sha256=digest,bytes=path.stat().st_size))
    put(target/'execution/source-copies.json',copies)
    patches=dict(SCREEN=screen,OWNER=ownercode,CANARY=oldchanges['CANARY'],E056_PARENT=oldparent,SOURCE_PINS=sourcepins,DEPENDENCY_PINS={})
    priorpatches=dict(SOURCE_PINS=oldchanges['SOURCE_PINS'],E054={'e054_plan':refs['e054_plan']['sha256']})
    return patches,priorpatches,parentpath,child,target


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()

    def test_default_off_and_exact_frozen_helpers(self):
        with patch.object(a,'audit',side_effect=AssertionError('audit')),contextlib.redirect_stdout(io.StringIO()) as out:self.assertEqual(a.main([]),0)
        self.assertEqual(json.loads(out.getvalue())['qualified_mlp_calls'],768)
        self.assertEqual(a.sha(a.COST_AUDIT),a.COST_AUDIT_SHA);self.assertEqual(a.h.__file__,str(a.B/'rowwise_native_verify_cost_analysis_v1/audit.py'))
        self.assertIs(a.h.timing_ledger,a.cost.base.timing_ledger);self.assertIs(a.h.compare_pair,a.cost.base.compare_pair)

    def test_six128_exact_callbacks_and_reject_any_late_or_reordered_call(self):
        rows=mlp_rows();a.mlp_receipts(rows);self.assertEqual(sum(len(r['calls']) for r in rows),768)
        for what in ('missing','order','dtype','shape','count','width','scope'):
            bad=copy.deepcopy(rows)
            if what=='missing':bad[-1]['calls'].pop()
            elif what=='order':bad[-1]['calls'][-1],bad[-1]['calls'][-2]=bad[-1]['calls'][-2],bad[-1]['calls'][-1]
            elif what=='dtype':bad[-1]['calls'][-1]['output_dtype']='mlx.core.float32'
            elif what=='shape':bad[-1]['calls'][-1]['weight_shape']=[17408,768]
            elif what=='count':bad[-1]['call_count']=127
            elif what=='width':bad[-1]['k']=4
            else:bad[-1]['scope']='GPU dispatch counts'
            with self.subTest(what=what),self.assertRaises(ValueError):a.mlp_receipts(bad)

    def test_frozen_raw_reader_retains_lateword_finite_and_signedzero(self):
        reader=a.h.raw_reader();left=f54.saved(self.root,'a.safetensors','BF16',[0,0]);right=f54.saved(self.root,'b.safetensors','BF16',[0,0x8000])
        with self.assertRaisesRegex(ValueError,'original-bit mismatch'):a.h.compare_pair(reader,reader.snapshot_index(self.root,left),reader.snapshot_index(self.root,right),'last')
        bad=f54.saved(self.root,'nan.safetensors','BF16',[0x7fc1])
        with self.assertRaisesRegex(ValueError,'finite'):reader.snapshot_index(self.root,bad)

    def test_frozen_v2_complete_checks_preserve_double_root_paths(self):
        before,after=checks();a.cost.producer_checks(dict(checks=before,post_checks=after));self.assertEqual((len(before),len(after)),(2719,655))
        before.pop()
        with self.assertRaisesRegex(ValueError,'complete source-bound'):a.cost.producer_checks(dict(checks=before,post_checks=after))

    def test_frozen400_timing_order_negative_budget_and_wrong_id(self):
        path,cases,ids,rows=f54.timing_fixture(self.root,native=80,candidate=100);summaries,total=a.h.timing_ledger(path,cases,ids,900)
        self.assertEqual(total,36000);self.assertTrue(all(s['break_even']['median_group_remaining_ns']==-20 for s in summaries))
        rows[-1]['ids'][0]=999;path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with self.assertRaises(ValueError):a.h.timing_ledger(path,cases,ids,900)

    def test_actual_new_schema_closed_chain_e056_refs_and_ack_tamper(self):
        patches,priorpatches,pp,child,target=fixture(self.root)
        with patch.multiple(a,**patches),patch.multiple(a.c56,**priorpatches):
            a.closed_owner(pp,a.sha(pp),child,{})
            for name,key,value in [('lease/lease.json','closed',False),('owner/numerical-exit-ack.json','done_sha256','0'*64),('owner/child-reap.json','absent',False)]:
                path=target/name;raw=path.read_bytes();d=a.read(path);d[key]=value;put(path,d)
                with self.assertRaises(ValueError):a.closed_owner(pp,a.sha(pp),child,{})
                path.write_bytes(raw)
            changed=patches['E056_PARENT'];raw=changed.read_bytes();put(changed,dict(a.read(changed),schema='wrong'))
            with self.assertRaises(ValueError):a.closed_owner(pp,a.sha(pp),child,{})
            changed.write_bytes(raw)

    def test_synthetic_cost_coordinator_and_changed_pretiming_parameters(self):
        patches,priorpatches,pp,child,target=fixture(self.root)
        def bounded_raw(child,complete,result,remember):
            result.update(raw_files=32,tensor_pairs=2064);return list(range(8)),{}
        with patch.multiple(a,**patches),patch.multiple(a.c56,**priorpatches),patch.object(a,'numerical_files',side_effect=bounded_raw):
            result=a.audit(child,pp,a.sha(pp));self.assertEqual(result['status'],'PASS',result.get('error'))
            self.assertTrue(result['shared_mlp_cost_audited']);self.assertEqual(result['qualified_mlp_calls'],768);self.assertEqual(result['timing_count'],400)
            changed=f56.parameters();changed['p1348']['sha256']='1'*64;put(child/'parameters-before-timing.json',changed)
            result=a.audit(child,pp,a.sha(pp));self.assertEqual(result['status'],'FAIL');self.assertFalse(result['shared_mlp_cost_audited']);self.assertEqual(result['timings'],[])
            self.assertIn('before-timing',result['error']['message'])

    def test_unclosed_or_missing_pretiming_gate_prevents_raw_access(self):
        patches,priorpatches,pp,child,target=fixture(self.root)
        complete=a.read(child/'complete.json');complete['pre_timing_parameters_unchanged']=False;f56.rebind_complete(target,complete)
        with patch.multiple(a,**patches),patch.multiple(a.c56,**priorpatches),patch.object(a,'numerical_files',side_effect=AssertionError('must not read raw')):
            result=a.audit(child,pp,a.sha(pp));self.assertEqual(result['status'],'FAIL');self.assertEqual(result['raw_files'],0);self.assertIn('pre-timing gate',result['error']['message'])

    def test_failed_cli_preserved_fresh_output(self):
        out=self.root/'failed.json';args=['--run','--child',str(self.root/'none'),'--parent-plan',str(self.root/'missing'),'--parent-plan-sha256','0'*64,'--out',str(out)]
        with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(a.main(args),1)
        self.assertEqual(a.read(out)['status'],'FAIL')
        with self.assertRaisesRegex(ValueError,'fresh output'):a.main(args)


if __name__=='__main__':unittest.main()
