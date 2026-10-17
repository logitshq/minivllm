"""Synthetic receipt corruption and real CPU scheduler regressions only.

No model/MLX import or device execution. Full descriptor fixtures exercise the
auditor's receipt contract, not the underlying arithmetic. These tests do not
establish serving or performance qualification.
"""
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bench.m5_ultra_256gb.route_decode_graph import audit_serving as a
from bench.m5_ultra_256gb.identity_refill import test_audit_model_v2_cpu as slots

LAYOUT=['full_attention' if i%4==3 else 'linear_attention' for i in range(64)]
FIXTURES=[dict(fixture_id='fixture-'+str(i),prompt_ids=[10+i,20+i],normal_eos=True,
               eos_token_ids=[99],max_output_tokens=512) for i in range(8)]


def cursor(rows):return a.RouteEvents(io.BytesIO(b''.join((json.dumps(r)+'\n').encode() for r in rows)))


def route(phase,step,label,B,used,L=1536,functions=2):
    graph=L>=1536
    return dict(event='route-call',phase=phase,step=step,direct_label=label,status='SUBMITTED',
        B=B,used=used,capacities=[[L,L]]*16,expected_graph=graph,
        actual_invocations=int(graph),functions=functions)


def parameter_receipts():
    """Explicit mocked device output; all true bits below are synthetic."""
    rows=[dict(event='parameter-inputs',prompts=[{**f,'fixture_id':'diagnostic-repeat-'+f['fixture_id'],
        'prompt_ids':(f['prompt_ids']*512)[:1023],'source_fixture_id':f['fixture_id'],
        'diagnostic_only':True,'max_output_tokens':4} for f in FIXTURES],
        history='diagnostic repeated D IDs, not generated history')]
    def compare(label,arrays):rows.append(dict(event='parameter-comparison',label=label,arrays=arrays))
    def direct(label,B,used,extra=()):
        rows.append(route('parameter-diagnostic',None,label,B,used,functions=1 if used==1023 else 2))
        arrays=list(extra)+a.state_rows(label+'/input',LAYOUT,B,1536,'root/caches')
        arrays += [a.array_row(label+'/input','root/'+n,[B],'mlx.core.int32') for n in ('positions','lens')]
        arrays += [a.array_row(label+'/logits','root',[B,248320])]+a.state_rows(label+'/state',LAYOUT,B,1536)
        compare(label,arrays)
    for used in (1023,1024):
        for i in range(8):
            label=f'parameter-prefill-{used-1}-{i}'
            compare(label,[a.array_row(label+'/logits','root',[1,248320])]+a.state_rows(label+'/state',LAYOUT,1,1024,single=True))
        for n,B in enumerate((1,8,1)):direct(f'parameter-baseline-{used}-{n}-B{B}',B,used)
        for path,shape,dtype in a.PARAMETERS:
            for n,B in enumerate((1,8,1)):
                label=f'parameter-changed-{path}-{used}-{n}-B{B}'
                extra=[a.array_row(path+'/'+suffix,'root',shape,dtype) for suffix in ('original-values','independent-new-values')] if n==0 else []
                direct(label,B,used,extra)
                rows.append(dict(event='parameter-eager-change',label=label,parameter=path,used=used,B=B,changed_eager_logits=True))
            rows.append(dict(event='parameter-original-handles-restored',parameter=path,both=True))
            for n,B in enumerate((1,8,1)):
                direct(f'parameter-restored-{path}-{used}-{n}-B{B}',B,used)
                compare('restored-original',[a.array_row(f'parameter-restored-original-{path}-{used}-{n}','root',[B,248320])])
        compare('parameter-prefill-retained',[item for _ in range(2) for i in range(8)
            for item in a.state_rows('parameter-prefill-retained',LAYOUT,1,1024,f'root/{i}',single=True)])
    return rows


class Tests(unittest.TestCase):
    def test_fixed_parameter_matrix_complete_and_original_descriptor_multiplicity(self):
        c=cursor(parameter_receipts());c.parameters(FIXTURES,LAYOUT);result=c.finish()
        self.assertEqual((result['parameter_direct_calls'],result['meaningful_eager_change_receipts']),(42,18))
        # 42*(130 input +129 output),16*129 prefill,12 parameter-value
        # comparisons,18 restored logits,2*2*8*128 retained cache comparisons.
        self.assertEqual(result['parameter_array_comparisons'],17068)
        self.assertEqual(result['route_hits'],{'parameter-diagnostic':{'1023':21,'1536':21}})

    def test_parameter_mutation_restoration_full_vocabulary_state_and_input_corruption_reject(self):
        original=parameter_receipts()
        for corruption in ('unchanged','wrong-parameter','restore-handle','vocab','state-rank',
                           'input-dtype','bit','finite','missing-state','extra-state'):
            rows=copy.deepcopy(original)
            changed=next(x for x in rows if x['event']=='parameter-eager-change')
            restored=next(x for x in rows if x['event']=='parameter-original-handles-restored')
            compared=next(x for x in rows if x['event']=='parameter-comparison' and x['label']=='parameter-baseline-1023-0-B1')
            arrays=compared['arrays']
            if corruption=='unchanged':changed['changed_eager_logits']=False
            elif corruption=='wrong-parameter':changed['parameter']='other.weight'
            elif corruption=='restore-handle':restored['both']=False
            elif corruption=='vocab':next(x for x in arrays if x['check'].endswith('/logits'))['shape']=[1,1]
            elif corruption=='state-rank':next(x for x in arrays if x['name'].endswith('/S'))['shape']=[48,128,128]
            elif corruption=='input-dtype':next(x for x in arrays if x['name']=='root/positions')['dtype']='mlx.core.float32'
            elif corruption=='bit':arrays[0]['bit_equal']=False
            elif corruption=='finite':arrays[0]['candidate_finite']=False
            elif corruption=='missing-state':arrays.pop(0)
            elif corruption=='extra-state':arrays.append(copy.deepcopy(arrays[0]))
            with self.subTest(corruption=corruption),self.assertRaises(ValueError):cursor(rows).parameters(FIXTURES,LAYOUT)

    def test_parameter_event_omission_reordering_and_unknown_tail_reject(self):
        rows=parameter_receipts();at=next(i for i,x in enumerate(rows) if x['event']=='parameter-eager-change')
        for changed in (rows[:at]+rows[at+1:],rows[:at]+[rows[at+1],rows[at]]+rows[at+2:]):
            with self.assertRaises(ValueError):cursor(changed).parameters(FIXTURES,LAYOUT)
        c=cursor(rows+[dict(event='extra')]);c.parameters(FIXTURES,LAYOUT)
        with self.assertRaisesRegex(ValueError,'extra'):c.finish()

    def test_actual_callable_domain_step_label_and_count_are_not_inferred_from_metadata_only(self):
        expected=route('D-C8',7,None,8,1024)
        for key,value in [('actual_invocations',0),('actual_invocations',2),('step',8),('step',True),
                          ('direct_label','wrong'),('functions',99),('expected_graph',False),('used',1023),('B',1)]:
            bad=copy.deepcopy(expected);bad[key]=value;c=cursor([bad]);c.families={1023}
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):c.route('D-C8',7,None,8,1024,1536)
        fallback=route('lifecycle',2,None,3,978,L=1024)
        c=cursor([fallback]);c.families={1023,1536};c.route('lifecycle',2,None,3,978,1024)
        self.assertEqual(c.fallbacks,1);self.assertEqual(c.hits,{})
        with self.assertRaises(ValueError):cursor([]).route('D-C8',7,None,8,1024,1792)

    def test_consumed_ahead_does_not_count_current_as_an_executed_graph(self):
        base=a.base_auditor();counts={};current=['decode',['x'],[1022],[7]];future=['decode',['x'],[1023],[8]]
        row=dict(schedule=dict(decode=['x'],requests=[dict(rid='x',prompt_ids=[1]*978,sampling=dict(max_tokens=512))]),
            arrays=[],ahead={'candidate':dict(consumed_prior_future=current,discarded_prior_future=None)},
            after={'engine':{'batch_rids':['x']}})
        for extra_current in (False,True):
            events=([route('D-C1',9,None,1,1023,functions=1)] if extra_current else [])+[route('D-C1',9,None,1,1024)]
            c=cursor(events);c.families={1023}
            # This one test isolates added route accounting; full numerical
            # validation is independently covered by inherited checker tests.
            with patch.object(base.CaseAudit,'check_arrays') as inherited:
                cls=a.extended_case(base,c,counts);case=cls('D-C1',slots.FIXTURES,slots.CONFIG,[99])
                case.old_ids=['x'];case.capacity=1536;case.step=9
                if extra_current:
                    with self.assertRaises(ValueError):case.check_arrays(row,[current,future],set(),True,1)
                else:
                    case.check_arrays(row,[current,future],set(),True,1);c.finish()
                    self.assertEqual(counts['D-C1']['consumed'],1)
                inherited.assert_called_once()

    def pair(self):
        base=a.base_auditor();cls=a.extended_case(base,cursor([]),{})
        c8=cls('D-C8',slots.FIXTURES,slots.CONFIG,[99])
        for row in slots.toy_rows()[:3]:c8.accept(row)
        return base,cls,c8

    def test_real_scheduler_nontrivial_stack_carries_by_copy_and_wrong_slot_rejects(self):
        base,cls,c8=self.pair();saved=list(c8.free)
        self.assertEqual(saved,[1,3,5,7,0,2,4,6])
        lifecycle=cls('lifecycle',slots.FIXTURES,slots.CONFIG,[99],c8.finished)
        self.assertIsNot(lifecycle.free,c8.free);lifecycle.accept(slots.toy_rows()[3]);self.assertEqual(c8.free,saved)
        base,cls,c8=self.pair();lifecycle=cls('lifecycle',slots.FIXTURES,slots.CONFIG,[99],c8.finished)
        bad=slots.toy_rows()[3];bad['schedule']['requests'][0]['slot']=0
        with self.assertRaisesRegex(ValueError,'FCFS/metadata'):lifecycle.accept(bad)
        self.assertIs(lifecycle.accept.__func__,base.CaseAudit.accept)
        self.assertIs(lifecycle.finish.__func__,base.CaseAudit.finish)

    def test_invalid_predecessor_and_pending_reject_but_new_c8_resets(self):
        for mutation in ('live','duplicate-slot','bool-slot','pending'):
            base,cls,c8=self.pair();pending=list(c8.finished)
            if mutation=='live':c8.staging.add('leak')
            elif mutation=='duplicate-slot':c8.free[0]=c8.free[1]
            elif mutation=='bool-slot':c8.free[c8.free.index(1)]=True
            else:pending.reverse()
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):cls('lifecycle',slots.FIXTURES,slots.CONFIG,[99],pending)
        base=a.base_auditor();routes=cursor([]);routes.families={1023,1536};cls=a.extended_case(base,routes,{})
        c1=cls('D-C1',slots.FIXTURES,slots.CONFIG,[99]);c1.finished=['c1-7']
        c8=cls('D-C8',slots.FIXTURES,slots.CONFIG,[99])
        self.assertEqual(c8.free,list(range(7,-1,-1)));self.assertEqual(c8.finished,[]);self.assertEqual(routes.families,set())
        bad=slots.toy_rows()[0];bad['arrays'][0]['bit_equal']=False
        with self.assertRaisesRegex(ValueError,'bit/finite'):c8.accept(bad)

    def test_missing_or_failed_terminal_never_opens_big_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);result=a.audit(root,'not-an-authorized-plan')
            self.assertEqual(result['status'],'INCOMPLETE');self.assertFalse(result['qualified'] or result['ledger_opened'])
            (root/'base').mkdir()
            for name,contents in [('plan.json',{}),('complete.json',{'status':'FAIL'}),
                                   ('base/complete.json',{'status':'FAIL'}),('base/plan.json',{})]:
                (root/name).write_text(json.dumps(contents))
            for name in ('base/steps.jsonl','route-events.jsonl'):(root/name).write_bytes(b'must not parse\n')
            result=a.audit(root,'not-an-authorized-plan')
            self.assertEqual(result['status'],'FAIL');self.assertFalse(result['qualified'] or result['ledger_opened'])
            self.assertFalse(result['offline_tensor_bytes_rechecked'])


if __name__=='__main__':unittest.main()
