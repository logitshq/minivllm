"""Tiny synthetic sidecar checks, not device evidence; no active files read."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('_e053_route_test',HERE/'routes.py')
r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
a=r.original


def pair(B=8,used=1023,before=0,delta=1,step=3,kind=None,fallback=0):
    after=[dict(candidate=before+delta,fallback=fallback) for _ in range(48)]
    combined=dict(event='combined-graph-call',status='SUBMITTED',installation=0,phase='HTTP-C8',step=step,
        direct_label=None,B=B,used=used,before=[dict(candidate=before,fallback=0) for _ in range(48)],after=after,
        actual_invocations=1,kind=kind or ('compiled-tracing' if delta or fallback else 'compiled-reuse'),
        bound=1023 if used<1024 else 1536,per_layer_candidate_delta=delta,per_layer_fallback_delta=fallback)
    route=dict(event='route-call',phase='HTTP-C8',step=step,direct_label=None,status='SUBMITTED',B=B,used=used,
        capacities=[[1536,1536] for _ in range(16)],expected_graph=True,actual_invocations=1,functions=1 if used<1024 else 2)
    return [combined,route]


def dual(rows,extra=None):
    header=dict(event='route-startup',schema='GDN-graph-combined-http-events-v1',pid=123,phase='HTTP-C8',identity={})
    ref=dict(path='/cpu-only/routes.jsonl',sha256='0'*64)
    replay=dict(event='replay-route-startup',schema='GDN-graph-combined-http-replay-events-v1',pid=124,phase='HTTP-C8',captured_route_events=ref)
    def stream(items):return io.BytesIO(b''.join((json.dumps(x)+'\n').encode() for x in items))
    return r.DualRoutes(a.load(a.SERVING,a.SERVING_SHA),stream([header,*rows]),stream([replay,*rows,*(extra or [])]),header,ref)


class BatchedRouteTests(unittest.TestCase):
    def test_each_batch_traces_candidate_and_reuses_without_python_delta(self):
        for B in (1,2,4,6,8):
            d=dual(pair(B=B)+pair(B=B,before=1,delta=0,step=4))
            d.route('HTTP-C8',3,None,B,1023,1536);d.route('HTTP-C8',4,None,B,1023,1536)
            out=d.finish();self.assertEqual(out['combined']['compiled-tracing'],1)
            self.assertEqual(out['combined']['compiled-reuse'],1)
            self.assertTrue(all(x==dict(candidate=1,fallback=0) for x in d.cursors[0].counters[0]))

    def test_new_bound_and_new_batch_require_their_own_trace(self):
        for used,B in ((1024,8),(1023,4)):
            d=dual(pair()+pair(B=B,used=used,before=1,delta=0,step=4))
            d.route('HTTP-C8',3,None,8,1023,1536)
            with self.assertRaises(ValueError):d.route('HTTP-C8',4,None,B,used,1536)
        d=dual(pair()+pair(used=1024,before=1,delta=1,step=4))
        d.route('HTTP-C8',3,None,8,1023,1536);d.route('HTTP-C8',4,None,8,1024,1536)
        self.assertEqual(d.finish()['combined']['compiled-tracing'],2)

    def test_native_fallback_rejected_even_if_both_streams_agree(self):
        d=dual(pair(B=8,delta=0,fallback=1))
        with self.assertRaises(ValueError):d.route('HTTP-C8',3,None,8,1023,1536)

    def test_nonuniform_missing_or_negative_layer_counts_rejected(self):
        for mutate in (lambda x:x['after'].pop(),lambda x:x['after'][0].update(candidate=2),
                       lambda x:x['after'][0].update(candidate=-1)):
            rows=pair();mutate(rows[0]);d=dual(rows)
            with self.assertRaises(ValueError):d.route('HTTP-C8',3,None,8,1023,1536)

    def test_external_decode_accumulates_candidate_for_all_batches(self):
        d=dual(pair(B=8,before=1,delta=1));d.external_decode('HTTP-C8',8)
        d.route('HTTP-C8',3,None,8,1023,1536);d.finish()
        self.assertTrue(all(x==dict(candidate=2,fallback=0) for x in d.cursors[0].counters[0]))
        for B in (0,9,True):
            with self.assertRaises(ValueError):dual([]).external_decode('HTTP-C8',B)

    def test_full_stream_rejects_extra_or_missing_route_event(self):
        d=dual(pair(),extra=pair());d.route('HTTP-C8',3,None,8,1023,1536)
        with self.assertRaises(ValueError):d.finish()
        d=dual(pair()[:1])
        with self.assertRaises(ValueError):d.route('HTTP-C8',3,None,8,1023,1536)

    def test_domain_and_same_source_but_wrong_original_route_are_rejected(self):
        rows=pair();rows[1]['capacities'][0][0]=1280;d=dual(rows)
        with self.assertRaises(ValueError):d.route('HTTP-C8',3,None,8,1023,1536)
        for B in (0,9,True):
            with self.assertRaises(ValueError):dual([]).route('HTTP-C8',3,None,B,1023,1536)


if __name__=='__main__':unittest.main(verbosity=2)
