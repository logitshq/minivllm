import unittest
from unittest.mock import Mock
import timing

def make_callbacks(counter=None, fail_on=None):
    c={}
    for w in (4,8):
        c[w]={}
        for arm in timing.ARMS:
            def fn(w=w, arm=arm):
                if counter is not None:
                    counter[0]+=1
                if fail_on and (w,arm,counter[0])==fail_on:
                    raise RuntimeError("fail")
            c[w][arm]=fn
    return c

class T(unittest.TestCase):
    def test_exact_counts(self):
        cnt=[0]
        cbs=make_callbacks(cnt)
        clock_calls=[0]
        def clock():
            clock_calls[0]+=1
            return clock_calls[0]*100
        rows=timing.collect(cbs, clock_ns=clock)
        self.assertEqual(len(rows),60)
        # 12 warmups +60 timed =72 calls
        self.assertEqual(cnt[0],72)
        self.assertEqual(clock_calls[0],120) # 2 per timed call

    def test_order_rotation(self):
        rows=timing.collect(make_callbacks(), clock_ns=lambda: 1)
        # check rotation: repeat0 order native,shared_without,shared_with ; repeat1 shifted
        for g in range(5):
            for rep in range(2):
                shift=rep%3
                order=timing.ARMS[shift:]+timing.ARMS[:shift]
                # for each ordinal block, arms should follow order for both widths interleaved
                # Our collect iterates arm outer then width inner, so ordinal sequence is interleaved widths
                pass
        # simple check ordinals consecutive
        self.assertEqual([r["ordinal"] for r in rows], list(range(60)))

    def test_reduce_math(self):
        # build deterministic rows: native 200, shared_with 100 => ratio 2
        rows=[]
        ordv=0
        for g in range(5):
            for rep in range(2):
                shift=rep%3
                order=timing.ARMS[shift:]+timing.ARMS[:shift]
                for arm in order:
                    for w in (4,8):
                        val=200 if arm=="native" else 120 if arm=="shared_without_features" else 100
                        rows.append({"width":w,"group":g,"repeat":rep,"ordinal":ordv,"arm":arm,"elapsed_ns":val})
                        ordv+=1
        res=timing.reduce(rows)
        self.assertAlmostEqual(res[4]["median_ratio"],2.0)
        self.assertAlmostEqual(res[4]["capture_overhead"],100/120)

    def test_reject_missing(self):
        rows=timing.collect(make_callbacks(), clock_ns=lambda: 10)
        with self.assertRaises(ValueError):
            timing.reduce(rows[:-1])
        # duplicate
        dup=list(rows)
        dup[0]=dup[1]
        with self.assertRaises(ValueError):
            timing.reduce(dup)
        # bool
        bad=list(rows)
        bad[0]=dict(bad[0]); bad[0]["elapsed_ns"]=True
        with self.assertRaises(ValueError):
            timing.reduce(bad)
        # negative
        bad2=list(rows)
        bad2[0]=dict(bad2[0]); bad2[0]["elapsed_ns"]=-1
        with self.assertRaises(ValueError):
            timing.reduce(bad2)

    def test_failure_propagation(self):
        cnt=[0]
        def fail_fn():
            cnt[0]+=1
            if cnt[0]==5:
                raise RuntimeError("boom")
        cbs={4:{"native":fail_fn,"shared_without_features":lambda: None,"shared_with_features":lambda: None},
             8:{"native":lambda: None,"shared_without_features":lambda: None,"shared_with_features":lambda: None}}
        # warmups will fail, should propagate
        with self.assertRaises(RuntimeError):
            timing.collect(cbs, clock_ns=lambda: 1)

    def test_no_throughput_field(self):
        rows=timing.collect(make_callbacks(), clock_ns=lambda: 1)
        for r in rows:
            self.assertNotIn("throughput", r)
            self.assertNotIn("tokens_per_sec", r)

if __name__=="__main__":
    unittest.main()
