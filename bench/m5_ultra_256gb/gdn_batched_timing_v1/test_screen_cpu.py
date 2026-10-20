"""Tiny CPU-only contracts; injected fake arrays, no backend/process execution."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

P=Path(__file__).with_name('screen.py')
spec=importlib.util.spec_from_file_location('_e048_screen_cpu',P)
s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)


def captures():
    out=[]
    for b in (2,8):
        for age,start in (('early',1),('late',129)):
            rows=[f'row{i}' for i in range(b)]; first={}
            for layer in s.LAYERS:
                first[str(layer)]=dict(layer=layer,batch_size=b,decode_step=start,row_ids=rows,
                    **{'reference_'+kind:dict(finite=True,path=f'captures/b{b}-{age}/step-{start}/layer-{layer}/reference-{kind}.safetensors')
                       for kind in ('input','output')})
            out.append(dict(case_id=f'b{b}-{age}',batch_size=b,start=start,row_ids=rows,steps=[first,{},{}]))
    return out


class Contracts(unittest.TestCase):
    def test_default_off_does_not_read_evidence_or_import_qualifier(self):
        with patch.object(s,'source',side_effect=AssertionError('source')),patch.object(s,'evidence',side_effect=AssertionError('evidence')):
            self.assertEqual(s.main([]),0)
            with self.assertRaises(ValueError): s.main(['--out','/cpu-unused'])

    def test_selects_twelve_first_actual_captures_without_mutating(self):
        source=captures(); before=copy.deepcopy(source); result=s.select_captures(source)
        self.assertEqual(source,before); self.assertEqual(len(result),12)
        self.assertEqual([(r['batch_size'],r['decode_step'],r['layer']) for r in result],
                         [(b,step,l) for b in (2,8) for step in (1,129) for l in (0,32,62)])

    def test_wrong_capture_step_or_path_is_rejected(self):
        for field,value in (('decode_step',2),('row_ids',['only-one-row'])):
            rows=captures(); rows[0]['steps'][0]['0'][field]=value
            with self.assertRaises(ValueError): s.select_captures(rows)
        rows=captures(); rows[0]['steps'][0]['0']['reference_input']['path']='../../elsewhere'
        with self.assertRaises(ValueError): s.select_captures(rows)

    def test_schedule_has_balanced_abba_pairs_and_exact_count(self):
        rows=list(s.paired_order()); self.assertEqual(len(rows),100)
        self.assertEqual([r['arm'] for r in rows[:4]],['native','candidate','candidate','native'])
        for group in range(5):
            for arm in ('native','candidate'):
                self.assertEqual(sum(r['group']==group and r['arm']==arm for r in rows),10)
        self.assertEqual(len(rows)*12,1200)

    def test_timed_region_omits_reset_and_materializes_all_three_outputs(self):
        events=[]
        class Array:
            def __init__(self,name): self.name=name
            def __getitem__(self,key): events.append(('view',self.name)); return Array(self.name+'-view')
        class State:
            def __init__(self,conv,S): events.append(('state',)); self.conv,self.S=conv,S
        class MX:
            def eval(self,*values): events.append(('eval',*[v.name for v in values]))
            def synchronize(self): events.append(('sync',))
        seed={f'root/{name}':Array(name) for name in ('x','conv','S')}; seen=[]
        def fn(x,state,**kw):
            events.append(('forward',)); seen.append(state)
            self.assertEqual(kw,dict(_compile_gates=True,_norm_reuse_normal=True))
            state.conv,state.S=Array('conv-out'),Array('S-out'); return Array('output')
        clocks=iter((100,150,200,250))
        def clock(): events.append(('clock',)); return next(clocks)
        for _ in range(2):
            elapsed,value=s.measured(MX(),SimpleNamespace(BatchState=State),fn,seed,clock=clock)
            self.assertEqual(elapsed,50); self.assertEqual(set(value),{'output','conv','S'})
        self.assertIsNot(seen[0],seen[1]); self.assertEqual(seed['root/conv'].name,'conv')
        first=events.index(('clock',)); end=events.index(('clock',),first+1)
        self.assertEqual(events[first+1:end],[('forward',),('eval','output','conv-out','S-out'),('sync',)])
        self.assertEqual(events[first-2:first],[('eval','x-view','conv-view','S-view'),('sync',)])

    def test_native_dispatch_unchanged_candidate_exactly_once_no_fallback(self):
        before=dict(candidate=3,fallback=0)
        s.dispatch(SimpleNamespace(calls=dict(before)),before,'native')
        s.dispatch(SimpleNamespace(calls=dict(candidate=4,fallback=0)),before,'candidate')
        for arm,counts in (('candidate',before),('candidate',dict(candidate=4,fallback=1)),('native',dict(candidate=4,fallback=0))):
            with self.assertRaises(ValueError): s.dispatch(SimpleNamespace(calls=counts),before,arm)


if __name__=='__main__': unittest.main()
