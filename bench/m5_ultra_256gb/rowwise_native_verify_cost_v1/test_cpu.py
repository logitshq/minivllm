"""Tiny control-flow checks; no backend, model, network or process operations."""
import contextlib
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

P=Path(__file__).with_name('screen.py')
spec=importlib.util.spec_from_file_location('cost_screen_test_target',P)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class Array:
    def __init__(self,values):self.values=list(values)
    def __getitem__(self,key):return Array(self.values[key] if isinstance(key,slice) else [self.values[key]])
    def astype(self,dtype):return self
    def item(self):return self.values[0]
    def tolist(self):return list(self.values)


class CPU(unittest.TestCase):
    def test_default_inert_and_explicit_run(self):
        with patch.object(m,'device',side_effect=AssertionError('device access')), patch.object(m,'validate',side_effect=AssertionError('source/model inspection')):
            with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(m.main([]),0)
            with contextlib.redirect_stderr(io.StringIO()):
                for argv in (['--run'],['--out','/not-created']):
                    with self.assertRaises(SystemExit) as error:m.main(argv)
                    self.assertEqual(error.exception.code,2)
        self.assertFalse(m.contract()['engine_pareto_point'])

    def test_schedule_is_complete_balanced_and_alternating(self):
        schedule=m.schedule();self.assertEqual(len(schedule),50)
        self.assertEqual(len({(g,p) for g,p,_ in schedule}),50)
        self.assertEqual(sum(order[0]=='native' for _,_,order in schedule),25)
        self.assertTrue(all(a[2]!=b[2] for a,b in zip(schedule,schedule[1:])))

    def test_commit_keeps_selected_state_and_next_position(self):
        holder=NS();selected=[object() for _ in range(64)]
        self.assertEqual(m.commit(holder,selected,1022,8,42),42)
        self.assertEqual(holder.lens,[1030]);self.assertEqual(holder.req_ids,['c1-cost-local'])
        self.assertIsNot(holder.layers,selected);self.assertTrue(all(a is b for a,b in zip(holder.layers,selected)))
        with self.assertRaises(ValueError):m.commit(holder,selected[:63],1022,8,42)
        with self.assertRaises(ValueError):m.commit(holder,selected,1022,0,42)

    def test_native_device_input_is_causal_and_submits_each_step(self):
        submitted=[];calls=[]
        mx=NS(int32='int32',array=lambda x,dtype:Array(x),argmax=lambda a,axis:Array(a.values),
            concatenate=lambda xs,axis:Array([v for a in xs for v in a.values]),
            eval=lambda *args:None,async_eval=lambda token,*states:submitted.append(token.item()))
        state=lambda conv,S:NS(conv=conv,S=S)
        context=lambda mode,caches,**kw:NS(mode=mode,caches=caches,**kw)
        r=NS(mx=mx,BatchState=state,BatchKV=lambda K,V:NS(K=K,V=V),ForwardContext=context,DecodeBatch=lambda model:NS())
        def decode(token,ctx):calls.append((token.item(),ctx.positions.item()));return Array([token.item()+1])
        engine=NS(model=object(),runner=NS(decode_model=decode))
        helper=NS(context_from_input=lambda *args:context('decode',[state(Array([0]),Array([0])) for _ in range(64)],positions=Array([1022])))
        result=m.native(r,engine,helper,[False]*64,{'root/tokens':Array([7])},1022,4,False)
        self.assertEqual(calls,[(7,1022),(8,1023),(9,1024),(10,1025)])
        self.assertEqual(submitted,[8,9,10,11]);self.assertEqual(result['ids'],[8,9,10,11])
        self.assertEqual(result['holder'].lens,[1026]);self.assertEqual(result['pending'],11)

    def test_timed_candidate_does_not_call_pure_audit_hooks(self):
        forbidden=lambda *args:(_ for _ in ()).throw(AssertionError('pure audit in timing'))
        q=NS(counted_head=forbidden,prefix_ownership=forbidden)
        states=[NS(conv=Array([0]),S=Array([0])) for _ in range(64)]
        ctx=NS(caches=states);project=object();passed=[]
        def verify(*args,**kwargs):
            passed.append(kwargs['head_project']);return NS(logits=Array([11,12,13,14]),prefixes=[states]*4)
        mx=NS(int32='int32',array=lambda x,dtype:Array(x),eval=lambda *args:None,argmax=lambda a,axis:a)
        runtime=NS(mx=mx,DecodeBatch=lambda model:NS())
        result=m.candidate_run(runtime,NS(model=object()),q,NS(verify=verify),
            NS(context_from_input=lambda *args:ctx),[False]*64,{},1022,[10,11,12,13],4,project)
        self.assertEqual(passed,[project]);self.assertIsNone(result['head'])
        self.assertEqual(result['holder'].lens,[1026]);self.assertEqual(result['pending'],14)

    def test_short_prefixes_do_not_get_timed_speed_claim(self):
        self.assertEqual(m.selected_prefixes(4),(1,2,4));self.assertEqual(m.selected_prefixes(8),(1,4,8))
        with self.assertRaises(ValueError):m.selected_prefixes(2)
        self.assertEqual(m.contract()['timed_prefix'],'k=T only')
        self.assertFalse(m.contract()['short_prefix_cost_measured'])

    def test_summary_rejects_missing_duplicate_or_nonpositive_samples(self):
        rows=[dict(group=g,pair=p,arm=arm,elapsed_ns=200 if arm=='native' else 100)
              for g,p,order in m.schedule() for arm in order]
        value=m.summarize(rows);self.assertEqual(value['median_group_ratio'],2)
        for broken in (rows[:-1],[dict(x,elapsed_ns=0) if i==0 else x for i,x in enumerate(rows)],
                       [rows[1] if i==0 else x for i,x in enumerate(rows)]):
            with self.assertRaises(ValueError):m.summarize(broken)


if __name__=='__main__':unittest.main()
