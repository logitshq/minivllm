"""Slot transition regression using the real CPU Scheduler; no arrays/device.

The zero-layer toy rows exercise scheduling and constructor adaptation only.
Real-model identity/64-layer/full-state gates remain in the pinned v1 auditor.
"""
import copy
import io
import tempfile
import unittest

from bench.m5_ultra_256gb.identity_refill import audit_model as v1
from bench.m5_ultra_256gb.identity_refill import audit_model_v2 as v2
from tutorial_code.miniengine.engine.request import Request
from tutorial_code.miniengine.engine.scheduler import Scheduler
from tutorial_code.miniengine.sampling_params import SamplingParams

FIXTURES = [dict(fixture_id='f'+str(i), prompt_ids=[1]) for i in range(8)]
CONFIG = dict(vocab_size=128, layer_types=[])


def metadata(req):
    return dict(rid=req.req_id, prompt_ids=list(req.prompt_token_ids), output_ids=list(req.output_token_ids),
        computed=req.num_computed_tokens, slot=req.slot, status=req.status.name,
        sampling=dict(max_tokens=req.sampling.max_tokens, temperature=0, top_p=1.0,
                      top_k=0, ignore_eos=False, stop_token_ids=[]))


def toy_rows():
    """Produce a nontrivial stack via real schedule/update, not auditor state."""
    scheduler = Scheduler(8, 2048, 2048, {99})
    requests = {}
    rows = []
    for case, step, first, count in [('D-C8', 1, 0, 8), ('D-C8', 2, 8, 4),
                                    ('D-C8', 3, 12, 4), ('lifecycle', 1, 0, 8)]:
        for i in range(first, first+count):
            rid = ('c8-' if case == 'D-C8' else 'probe-') + str(i)
            cap = 512 if case == 'D-C8' else (1, 2, 3, 5, 8, 13, 21, 34)[i]
            req = Request(rid, [1], SamplingParams(max_tokens=cap, temperature=0))
            requests[rid] = req; scheduler.add(req)
        out = scheduler.schedule()
        decode = [r.req_id for r in out.decode]
        schedule = dict(decode=decode, prefill=[[r.req_id,n] for r,n in out.prefill],
                        finished_ids=list(out.finished_ids), requests=[metadata(r) for r in requests.values()])
        calls = ([['decode', decode, [r.num_computed_tokens for r in out.decode],
                   [r.output_token_ids[-1] for r in out.decode]]] if decode else [])
        calls += [['prefill',r.req_id,r.num_computed_tokens,n] for r,n in out.prefill]
        sampled = {r.req_id: (5 if case=='D-C8' and step==1 and int(r.req_id[3:])%2 else 99)
                   for r in out.decode + [r for r,n in out.prefill]}
        scheduler.update(out, sampled)
        delivered = [[rid,token,requests[rid].is_finished] for rid,token in sampled.items()]
        for rid,token,done in delivered:
            if done: requests.pop(rid)
        surviving = [rid for rid in decode if rid in requests]
        arrays = []
        for key in calls:
            literal = tuple(key) if key[0]=='prefill' else ('decode',tuple(key[1]),tuple(key[2]),tuple(key[3]))
            arrays.append(dict(check='logits '+repr(literal),name='root',shape=[1 if key[0]=='prefill' else len(decode),128],
                               dtype='mlx.core.bfloat16',bit_equal=True,reference_finite=True,candidate_finite=True))
        ahead = dict(carried_future=None, consumed_prior_future=None, discarded_prior_future=None, discarded_current_calls=[])
        rows.append(dict(case=case,step=step,status='PASS',schedule=schedule,reference_delivered=delivered,
            candidate_delivered=copy.deepcopy(delivered),call_keys=calls,reference_discarded_ahead=None,
            ahead=dict(reference=copy.deepcopy(ahead),candidate=copy.deepcopy(ahead)),arrays=arrays,
            after=dict(state={},engine=dict(requests=[metadata(r) for r in requests.values()],
                waiting=[r.req_id for r in scheduler.waiting],running=[r.req_id for r in scheduler.running],
                free_slots=list(scheduler.free_slots),finished_pending=list(scheduler._finished_since_last_step),
                batch_rids=surviving,batch_lens=[requests[r].num_computed_tokens for r in surviving],
                has_work=scheduler.has_work(),ahead_key=None))))
    return rows


def pair(module):
    base = module.source_auditor()
    cls = module.extended_case(base, module.Refills(io.BytesIO()), 3)
    c8 = cls('D-C8', FIXTURES, CONFIG, [99])
    for row in toy_rows()[:3]: c8.accept(row)
    return cls,c8


class Tests(unittest.TestCase):
    def test_real_scheduler_sequence_old_rejects_new_accepts_without_mutating_predecessor(self):
        cls,old = pair(v1)
        self.assertEqual(old.free, [1,3,5,7,0,2,4,6])
        with self.assertRaisesRegex(ValueError, 'FCFS/metadata'):
            cls('lifecycle', FIXTURES, CONFIG, [99], old.finished).accept(toy_rows()[3])
        cls,old = pair(v2.private_v1())
        saved = list(old.free)
        new = cls('lifecycle', FIXTURES, CONFIG, [99], old.finished)
        self.assertEqual(new.free,saved);self.assertIsNot(new.free,old.free)
        new.accept(toy_rows()[3])
        self.assertEqual(old.free,saved)
        self.assertNotEqual(new.free,saved)  # Real pop/retire append order changed.

    def test_wrong_slot_is_still_rejected(self):
        cls,old = pair(v2.private_v1())
        new = cls('lifecycle', FIXTURES, CONFIG, [99], old.finished)
        row = toy_rows()[3];row['schedule']['requests'][0]['slot']=0
        with self.assertRaisesRegex(ValueError,'FCFS/metadata'):new.accept(row)

    def test_corrupt_predecessor_and_pending_ids_are_not_masked(self):
        for corruption in ('live','duplicate-slot','bool-slot','pending','missing-predecessor','wrong-predecessor'):
            module=v2.private_v1();cls,old=pair(module);pending=list(old.finished)
            if corruption=='live':old.staging.add('leaked')
            if corruption=='duplicate-slot':old.free[0]=old.free[1]
            if corruption=='bool-slot':old.free[old.free.index(1)]=True
            if corruption=='pending':pending.reverse()
            if corruption=='missing-predecessor':cls=module.extended_case(module.source_auditor(),module.Refills(io.BytesIO()),3)
            if corruption=='wrong-predecessor':cls('D-C1',FIXTURES,CONFIG,[99])
            with self.subTest(corruption=corruption),self.assertRaises(ValueError):
                cls('lifecycle',FIXTURES,CONFIG,[99],pending)

    def test_c1_to_c8_is_a_fresh_pair_and_original_checkers_are_inherited(self):
        module=v2.private_v1();base=module.source_auditor()
        cls=module.extended_case(base,module.Refills(io.BytesIO()),3)
        c1=cls('D-C1',FIXTURES,CONFIG,[99]);c1.finished=['c1-7']
        c8=cls('D-C8',FIXTURES,CONFIG,[99])
        self.assertEqual(c8.free,list(range(7,-1,-1)));self.assertEqual(c8.finished,[])
        self.assertIs(c8.accept.__func__,base.CaseAudit.accept)
        self.assertIs(c8.finish.__func__,base.CaseAudit.finish)
        row=toy_rows()[0];row['arrays'][0]['bit_equal']=False
        with self.assertRaisesRegex(ValueError,'bit/finite'):c8.accept(row)

    def test_missing_evidence_stays_incomplete_and_source_provenance_is_explicit(self):
        with tempfile.TemporaryDirectory() as path:
            result=v2.audit(path)
        self.assertEqual(result['status'],'INCOMPLETE');self.assertFalse(result['qualified'] or result['ledger_opened'])
        self.assertEqual(result['schema'],'E028-model-independent-audit-v2')
        self.assertEqual(result['frozen_v1_source_sha256'],v2.V1_SHA)


if __name__=='__main__':unittest.main()
