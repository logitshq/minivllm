"""Source-only CPU mechanisms; fake engine, no model/backend or service."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_combined_c24_domain', HERE/'domain.py')
d = importlib.util.module_from_spec(spec); spec.loader.exec_module(d)


def numeric():
    return dict(schema='E023-native-qualification-plan-v1', model='/fake/model', deadline_s=1800,
        candidate=dict(d.CANDIDATE), configuration=dict(max_len=2048,
        max_seqs_by_case={'D-C1':1,'D-C8':8,'lifecycle':8}, max_batched_tokens=2048, cache_limit_gib=2))


def fixtures():
    return [dict(fixture_id=f'D{i}', prompt_text=f'original prompt {i}',
        max_output_tokens=512, normal_eos=True, eos_token_ids=[99]) for i in range(8)]


class DomainTests(unittest.TestCase):
    def test_domain_rejects_old_future_and_bool(self):
        self.assertEqual([d.request_count(c) for c in (2,4)], [8,8])
        for value in (True,False,1,8,16,32,2.0,'4'):
            with self.subTest(value=value), self.assertRaises(ValueError): d.concurrency(value)

    def test_original_body_bytes_only_request_ids_change(self):
        s=importlib.util.spec_from_file_location('_c24_original_jobs',d.JOBS)
        old=importlib.util.module_from_spec(s);s.loader.exec_module(old)
        base=old.jobs_for(fixtures(),1)
        for C in (2,4):
            actual=d.jobs_for(fixtures(),C)
            self.assertEqual(len(actual),8)
            self.assertEqual([x['request_id'] for x in actual],[f'qual-c{C}-r{i:02d}' for i in range(8)])
            self.assertEqual([{k:v for k,v in j.items() if k!='request_id'} for j in actual],
                             [{k:v for k,v in j.items() if k!='request_id'} for j in base])

    def test_new_capacity_does_not_relabel_base_plan(self):
        base=numeric();before=copy.deepcopy(base)
        for C in (2,4):
            self.assertEqual(d.configuration(base,C)['max_seqs'],C)
        self.assertEqual(base,before)
        for key,value in (('max_len',1536),('cache_limit_gib',4),('max_batched_tokens',4096)):
            bad=copy.deepcopy(base);bad['configuration'][key]=value
            with self.assertRaises(ValueError):d.configuration(bad,2)
        bad=copy.deepcopy(base);bad['candidate']['norm_reuse']=True
        with self.assertRaises(ValueError):d.configuration(bad,2)

    def environment(self, shared=False, fail_observer=False):
        calls=[]; phases=[]; observed=[]; shared_array=object()
        class Gate:
            def __init__(self):self.flags=[]
            def set_gates_compile(self,v):self.flags.append(('gates',v))
            def set_norm_reuse(self,v):self.flags.append(('norm',v))
        class Engine:
            def __init__(self,model,**kwargs):
                calls.append((model,kwargs));self.scheduler=NS(max_seqs=kwargs['max_seqs'])
                self.runner=NS(max_len=kwargs['max_len'],spec=None);self.eos_ids={99}
                self.gates=[Gate() for _ in range(48)]
                layers=[NS(is_attn=False,linear_attn=g) for g in self.gates]+[NS(is_attn=True) for _ in range(16)]
                weight=shared_array if shared else object()
                self.model=NS(layers=layers,parameters=lambda:{'weight':weight})
        def observer(engine,cmp,arm):
            self.assertEqual([g.flags for g in engine.gates],
                [[('gates',arm=='candidate'),('norm',False)]]*48)
            if fail_observer and arm=='candidate':raise RuntimeError('observer construction failed')
            handle=NS(engine=engine,arm=arm);observed.append(handle);return handle
        def meta(label,left,right):self.assertEqual(left,right,label)
        args=dict(numeric=numeric(),fixtures=fixtures(),engine_type=Engine,
            mx=NS(bfloat16='BF16',synchronize=lambda:calls.append('sync')),
            tree_flatten=lambda x:list(x.items()),observer_type=observer,
            comparator=NS(meta=meta),deadline=lambda:None,
            phase=lambda name,**kw:phases.append((name,kw)))
        return args,calls,phases,observed

    def test_pair_constructor_uses_actual_c_and_independent_flags(self):
        for C in (2,4):
            args,calls,phases,seen=self.environment();engines=[];observers=[]
            d.load_pair_into(engines,observers,value=C,**args)
            self.assertEqual([c[1]['max_seqs'] for c in calls[:-1]],[C,C])
            self.assertEqual([c[1]['compile_decode'] for c in calls[:-1]],[False,True])
            self.assertEqual([c[1]['fuse_decode'] for c in calls[:-1]],[False,True])
            self.assertEqual([o.arm for o in observers],['reference','candidate'])
            self.assertEqual([r[0] for r in phases],['load-pair-start','load-pair-complete'])
            self.assertEqual([r[1]['max_seqs'] for r in phases],[C,C])
            self.assertIsNot(engines[0],engines[1])

    def test_shared_weights_rejected_and_returned_models_remain_owned(self):
        args,_,phases,_=self.environment(shared=True);engines=[];observers=[]
        with self.assertRaisesRegex(ValueError,'share array'):
            d.load_pair_into(engines,observers,value=2,**args)
        self.assertEqual(len(engines),2);self.assertEqual(observers,[])
        self.assertEqual(len(phases),1)

    def test_partial_observer_error_retains_cleanup_handles(self):
        args,_,phases,seen=self.environment(fail_observer=True);engines=[];observers=[]
        with self.assertRaisesRegex(RuntimeError,'observer'):
            d.load_pair_into(engines,observers,value=4,**args)
        self.assertEqual(len(engines),2);self.assertEqual(len(observers),1)
        self.assertIs(observers[0],seen[0]);self.assertEqual(len(phases),1)
        with self.assertRaisesRegex(ValueError,'fresh'):
            d.load_pair_into(engines,observers,value=4,**args)


if __name__=='__main__':unittest.main()
