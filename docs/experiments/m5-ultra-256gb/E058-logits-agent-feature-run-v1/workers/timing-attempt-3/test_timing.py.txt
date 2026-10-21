import unittest
import timing

def make_callbacks(counter=None):
    c={}
    for w in (4,8):
        c[w]={}
        for arm in timing.ARMS:
            def fn(w=w, arm=arm):
                if counter is not None:
                    counter[0]+=1
            c[w][arm]=fn
    return c

def fake_rows():
    rows=[]
    for w,g,rep,ordv,arm in timing.SCHEDULE:
        val=200 if arm=="native" else 120 if arm=="shared_without_features" else 100
        rows.append({"width":w,"group":g,"repeat":rep,"ordinal":ordv,"arm":arm,"elapsed_ns":val})
    return rows

class T(unittest.TestCase):
    def test_exact_counts(self):
        cnt=[0]
        cbs=make_callbacks(cnt)
        cc=[0]
        def clock(): cc[0]+=1; return cc[0]*100
        rows=timing.collect(cbs, clock_ns=clock)
        self.assertEqual(len(rows),60)
        self.assertEqual(cnt[0],72)
        self.assertEqual(cc[0],120)

    def test_all3_rotations(self):
        rows=timing.collect(make_callbacks(), clock_ns=lambda: 1)
        shifts=set()
        for g in range(5):
            for rep in range(2):
                shift=(g*2+rep)%3
                shifts.add(shift)
                base=g*2*6+rep*6
                arms=[rows[base+i]["arm"] for i in range(0,6,2)]  # w=4 arm per step
                expected=list(timing.ARMS[shift:]+timing.ARMS[:shift])
                self.assertEqual(arms, expected)
        self.assertEqual(shifts,{0,1,2})

    def test_schedule_tamper_rejected(self):
        rows=fake_rows()
        # swap two rows
        rows[0],rows[1]=rows[1],rows[0]
        with self.assertRaises(ValueError): timing.reduce(rows)
        rows=fake_rows()
        # alter arm
        rows[0]=dict(rows[0]); rows[0]["arm"]="native" if rows[0]["arm"]!="native" else "shared_with_features"
        with self.assertRaises(ValueError): timing.reduce(rows)
        # duplicate ordinal
        rows=fake_rows()
        rows[1]=dict(rows[1]); rows[1]["ordinal"]=rows[0]["ordinal"]
        rows[1]["width"]=rows[0]["width"]; rows[1]["arm"]=rows[0]["arm"]; rows[1]["group"]=rows[0]["group"]; rows[1]["repeat"]=rows[0]["repeat"]
        with self.assertRaises(ValueError): timing.reduce(rows)

    def test_on_sample_propagates(self):
        delivered=[]
        def on_sample(r):
            delivered.append(r["ordinal"])
            if r["ordinal"]==2:
                raise RuntimeError("on_sample boom")
        cnt=[0]
        orig_make=make_callbacks(cnt)
        # wrap to count timed callbacks after failure
        with self.assertRaises(RuntimeError):
            timing.collect(orig_make, clock_ns=lambda: 1, on_sample=on_sample)
        self.assertEqual(delivered, [0,1,2])
        self.assertEqual(cnt[0], 12+3)  # warmups + 3 timed attempted (0,1,2 third fails in on_sample but callback already done)

    def test_reduce_math(self):
        res=timing.reduce(fake_rows())
        self.assertAlmostEqual(res[4]["median_ratio"],2.0)
        self.assertAlmostEqual(res[4]["capture_overhead"],100/120)

    def test_reject_bool_negative_missing(self):
        rows=fake_rows()
        bad=list(rows); bad[0]=dict(bad[0]); bad[0]["elapsed_ns"]=True
        with self.assertRaises(ValueError): timing.reduce(bad)
        bad2=list(rows); bad2[0]=dict(bad2[0]); bad2[0]["elapsed_ns"]=-1
        with self.assertRaises(ValueError): timing.reduce(bad2)
        with self.assertRaises(ValueError): timing.reduce(rows[:-1])

    def test_failure_propagation(self):
        cnt=[0]
        def fail_fn():
            cnt[0]+=1
            if cnt[0]==5: raise RuntimeError("boom")
        cbs={4:{"native":fail_fn,"shared_without_features":lambda: None,"shared_with_features":lambda: None},
             8:{"native":lambda: None,"shared_without_features":lambda: None,"shared_with_features":lambda: None}}
        with self.assertRaises(RuntimeError): timing.collect(cbs, clock_ns=lambda: 1)

    def test_no_throughput(self):
        rows=timing.collect(make_callbacks(), clock_ns=lambda: 1)
        for r in rows: self.assertNotIn("throughput", r)

if __name__=="__main__": unittest.main()
