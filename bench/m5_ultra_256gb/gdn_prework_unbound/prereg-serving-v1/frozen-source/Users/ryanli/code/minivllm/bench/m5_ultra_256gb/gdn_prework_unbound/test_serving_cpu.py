"""Prepared stdlib fake contracts; NOT RUN during formal device measurement.

The actual frozen Observer/CallProxy is used, with tiny fake models/counters.
These prove lifecycle and receipt rejection mechanisms, never numeric results.
"""
import builtins
from contextlib import redirect_stdout, redirect_stderr
import copy
import gc
import io
import json
from pathlib import Path
import runpy
import tempfile
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

from bench.m5_ultra_256gb.gdn_prework_unbound import serving as s


class Tensor:
    dtype='fake';ndim=1
    def __init__(self,values):self.values=list(values);self.shape=(len(values),)
    def tolist(self):return list(self.values)
    def view(self,*args):return self
    def __getitem__(self,key):return Tensor(self.values[key])


class Layer:
    def _decode(self,x):return x


class Model:
    def __init__(self):
        self.layers=[SimpleNamespace(is_attn=True) if i%4==3 else
                     SimpleNamespace(is_attn=False,linear_attn=Layer()) for i in range(64)]
    def __call__(self,tokens,ctx,**kwargs):
        if ctx.mode=='decode' and ctx.max_len>256:
            for layer in self.layers:
                if not layer.is_attn:layer.linear_attn._decode(tokens)
        return Tensor([7]*tokens.shape[0])


class Runner:
    def __init__(self,model):
        self.model=self.decode_model=model;self.batch=SimpleNamespace(req_ids=['r0']);self._ahead=None
    def run(self,out):return {}


class Scheduler:
    def schedule(self):return None


class Engine:
    def __init__(self):self.model=Model();self.runner=Runner(self.model);self.scheduler=Scheduler()


class Adapter:
    def __init__(self,*,fail_at=None,fail_restore=None):
        self.handles=[];self.closed=[];self.fail_at=fail_at;self.fail_restore=fail_restore;self.on_restore=None
    def install(self,layer,module,*,enabled):
        if enabled is not True:raise AssertionError('expected explicit candidate ON')
        index=len(self.handles)
        if index==self.fail_at:raise RuntimeError('partial install')
        handle=SimpleNamespace(layer=layer,calls=dict(candidate=0,fallback=0),closed=False)
        def call(this,x):
            handle.calls['candidate' if x.shape[0]==1 else 'fallback']+=1;return x
        layer._decode=MethodType(call,layer)
        def restore():
            if self.on_restore:self.on_restore()
            self.closed.append(index);delattr(handle.layer,'_decode');handle.layer=None;handle.closed=True
            if index==self.fail_restore:raise RuntimeError('restore failure after removing own hook')
        handle.restore=restore;self.handles.append(handle);return handle


def base_module():return s.load(s.BASE,'_gdn_serving_cpu_actual_observer')
def ctx(B=1,used=900):
    return SimpleNamespace(mode='decode',max_len=used,caches=[],positions=Tensor([used-1]*B),lens=Tensor([used]*B))


def canary_receipts():
    audit=dict(schema='GDN-prework-real-layer-independent-v1',status='PASS',canary_qualified=True,
        audit_source_sha256=s.AUDIT_SHA,raw_files=60,full_output_state_pairs=54,causal_state_pairs=24,
        full_vocabulary_greedy_checks=6,device_comparison_arrays=162,
        full_model_qualified=False,http_qualified=False,performance_qualified=False)
    complete=dict(schema='GDN-prework-real-layer-result-v1',status='PASS',canary_qualified=True,
        candidate_calls=18,candidate_compared_calls=18,reference_chains=2,candidate_chains=6,
        device_run=True,instance_hooks_restored=True,source_model_pins_rechecked=True,
        dispatch_counters={str(i):dict(candidate=6,fallback=0) for i in (0,32,62)},
        full_model_qualified=False,http_qualified=False,performance_qualified=False)
    return audit,complete


def complete_fixture():
    result=dict(status='PASS',source_model_pins_rechecked=True,total_requests=37,
        cases=[dict(name=name,completed=[None]*n) for name,n in zip(s.CASES,(8,16,13))])
    rows={name:dict(candidate=48 if name=='D-C1' else 0,fallback=48 if name=='D-C8' else 0,
            b1_calls=int(name=='D-C1'),multirow_calls=int(name=='D-C8'),prefill_calls=1,short_bypass_calls=0) for name in s.CASES}
    coverage=dict(enabled=True,layer_installs=96,layer_restores=96,observer_restores=4,per_case=rows,
        final_counters=[[dict(candidate=1,fallback=0) for _ in range(48)],
                        [dict(candidate=0,fallback=1) for _ in range(48)]])
    return result,coverage


class Tests(unittest.TestCase):
    def test_import_default_off_and_exact_owner_cli_selection(self):
        original=builtins.__import__
        def guarded(name,*args,**kwargs):
            if name=='mlx' or name.startswith(('mlx.','numpy','tutorial_code')):raise AssertionError('runtime import')
            return original(name,*args,**kwargs)
        with patch('builtins.__import__',side_effect=guarded):runpy.run_path(s.__file__)
        with patch.object(s,'validate',side_effect=AssertionError('should not validate describe')) as validate,redirect_stdout(io.StringIO()) as out:
            self.assertEqual(s.main([]),0);validate.assert_not_called()
        self.assertFalse(json.loads(out.getvalue())['enabled'])
        for argv in (['--run'],['--plan','/tmp/x'],['--plan-sha256','a','--out','/tmp/y']):
            with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):s.main(argv)
        with patch.object(s,'validate',side_effect=ValueError('real owner argv reached validation')) as validate:
            with self.assertRaisesRegex(ValueError,'real owner'):s.main(['--plan','/tmp/p','--plan-sha256','a','--out','/tmp/o'])
            validate.assert_called_once_with(Path('/tmp/p'),'a')

    def test_canary_requires_actual_independent_fixed_counts_and_no_scope_inflation(self):
        a,c=canary_receipts();s.canary_gate(a,c)
        for key,value in (('status','INCOMPLETE'),('canary_qualified',1),('raw_files',59),('full_output_state_pairs',53),
                          ('causal_state_pairs',23),('audit_source_sha256','other'),('performance_qualified',True)):
            wrong=copy.deepcopy(a);wrong[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):s.canary_gate(wrong,c)
        for key,value in (('candidate_calls',17),('candidate_compared_calls',True),('device_run',False),('instance_hooks_restored',False)):
            wrong=copy.deepcopy(c);wrong[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):s.canary_gate(a,wrong)
        wrong=copy.deepcopy(c);wrong['dispatch_counters']['32']['fallback']=1
        with self.assertRaises(ValueError):s.canary_gate(a,wrong)

    def test_pinned_prerequisite_checks_real_bytes_not_claimed_status(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory).resolve()/'audit.json';s.write(path,{'status':'PASS'})
            ref=dict(path=str(path),sha256=s.sha(path));self.assertEqual(s.pinned(ref),{'status':'PASS'})
            s.write(path,{'status':'FAIL'})
            with self.assertRaisesRegex(ValueError,'changed prerequisite'):s.pinned(ref)
            with self.assertRaisesRegex(ValueError,'absolute'):s.pinned(dict(path='audit.json',sha256=ref['sha256']))

    def test_plan_self_sources_and_deadline_fail_before_loading_any_runtime(self):
        path=Path('/p/serving.json');original_path=Path('/p/original.json');own=str(Path(s.__file__).resolve())
        sources={own:'self'};plan=dict(schema='GDN-prework-serving-plan-v1',qualifier_sha256='self',
            qualification=dict(path=str(original_path),sha256=s.BASE_PLAN_SHA),canary={'real':'binding'},sources=sources,deadline_s=1800)
        original=dict(deadline_s=1800,candidate=dict(compile_decode=True,fuse_decode=True,gates_compile=True,norm_reuse=False))
        digests={str(path):'plan',str(original_path):s.BASE_PLAN_SHA,own:'self'}
        def validate(value):
            with patch.object(s,'sha',side_effect=lambda p:digests[str(p)]),patch.object(s,'read',side_effect=lambda p:value if Path(p)==path else original), \
                 patch.object(s,'source_pins',return_value=sources),patch.object(s,'bind_canary',return_value={'proof':True}):
                return s.validate(path,'plan')
        self.assertEqual(validate(plan),(plan,{'proof':True}))
        for key,value in (('deadline_s',1800.0),('deadline_s',True),('deadline_s',1801),('qualifier_sha256','wrong'),('unknown',0)):
            bad=copy.deepcopy(plan);bad[key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):validate(bad)

    def test_default_off_and_reference_leave_layers_and_private_base_untouched(self):
        base=base_module();observer,run=base.Observer,base.run_case;adapter=Adapter();e=Engine()
        with s.adapt(base,adapter,lambda r:None) as counts:
            self.assertIs(base.Observer,observer);self.assertIs(base.run_case,run);self.assertFalse(counts['enabled'])
        with s.adapt(base,adapter,lambda r:None,enabled=True):
            seen=base.Observer(e,object(),'reference');seen.restore()
        self.assertEqual(adapter.handles,[]);self.assertIs(base.Observer,observer)

    def test_real_observer_install_order_reverse_cleanup_and_no_retained_engine(self):
        base=base_module();adapter=Adapter();e=Engine();model=e.model;wr=weakref.ref(e);wm=weakref.ref(model)
        def restored_proxy():self.assertIs(e.runner.model,model)
        adapter.on_restore=restored_proxy
        with s.adapt(base,adapter,lambda r:None,enabled=True) as counts:
            observed=base.Observer(e,object(),'candidate')
            self.assertIsInstance(e.runner.model,base.CallProxy)
            self.assertEqual(len(adapter.handles),48)
            self.assertTrue(all('_decode' in vars(layer.linear_attn) for layer in model.layers if not layer.is_attn))
            observed.restore();observed.restore()
            self.assertEqual(adapter.closed,list(reversed(range(48))))
            self.assertEqual((counts['layer_installs'],counts['layer_restores']),(48,48))
            self.assertNotIn('run',vars(e.runner));self.assertNotIn('schedule',vars(e.scheduler))
            self.assertIsNone(observed.engine);self.assertEqual(observed.handles,[])
        adapter.on_restore=None;del restored_proxy,e,model;gc.collect()
        self.assertIsNone(wr());self.assertIsNone(wm())

    def test_partial_install_and_partial_observer_construction_restore_all_hooks(self):
        for failure in ('adapter','observer'):
            base=base_module();e=Engine();model=e.model;adapter=Adapter(fail_at=9 if failure=='adapter' else None)
            if failure=='observer':
                original=base.Observer
                class Broken(original):
                    def __init__(self,*args):super().__init__(*args);raise RuntimeError('base constructor')
                base.Observer=Broken
            with s.adapt(base,adapter,lambda r:None,enabled=True):
                with self.assertRaises(RuntimeError):base.Observer(e,object(),'candidate')
            self.assertIs(e.runner.model,model);self.assertIs(e.runner.decode_model,model)
            self.assertNotIn('run',vars(e.runner));self.assertNotIn('schedule',vars(e.scheduler))
            self.assertTrue(all('_decode' not in vars(layer.linear_attn) for layer in model.layers if not layer.is_attn))
            self.assertEqual(adapter.closed,list(reversed(range(len(adapter.handles)))))

    def test_restore_exception_still_attempts_every_remaining_handle(self):
        base=base_module();e=Engine();adapter=Adapter(fail_restore=41)
        with s.adapt(base,adapter,lambda r:None,enabled=True):
            observed=base.Observer(e,object(),'candidate')
            with self.assertRaisesRegex(RuntimeError,'restore failure'):observed.restore()
            self.assertEqual(adapter.closed,list(reversed(range(48))))
            self.assertIsNone(observed.engine);self.assertEqual(observed.handles,[])

    def test_real_semantic_calls_can_be_multiple_per_step_without_extra_token_reads(self):
        base=base_module();e=Engine();adapter=Adapter();events=[]
        with s.adapt(base,adapter,events.append,enabled=True) as coverage:
            observed=base.Observer(e,object(),'candidate');observed.case='D-C1';observed.begin()
            for used in (900,901):e.runner.model(Tensor([8]),ctx(used=used))
            self.assertEqual([x['step'] for x in events],[1,1])
            self.assertEqual([x['semantic_key'][2] for x in events],[[899],[900]])
            self.assertEqual([x['candidate'] for x in events],[48,48])
            self.assertEqual(coverage['per_case']['D-C1']['b1_calls'],2)
            observed.case='D-C8';observed.begin();e.runner.batch.req_ids=[f'r{i}' for i in range(8)]
            e.runner.model(Tensor([8]*8),ctx(8))
            self.assertEqual(events[-1]['fallback'],48);self.assertEqual(events[-1]['path'],'B>1-fallback')
            observed.restore()

    def test_silent_fallback_stale_mixed_layers_and_short_bypass_are_distinct(self):
        zero=[dict(candidate=0,fallback=0) for _ in range(48)]
        hit=[dict(candidate=1,fallback=0) for _ in range(48)]
        fallback=[dict(candidate=0,fallback=1) for _ in range(48)]
        self.assertEqual(s.dispatch_delta(zero,hit,'decode',1,900)['candidate'],48)
        self.assertEqual(s.dispatch_delta(zero,fallback,'decode',8,900)['fallback'],48)
        self.assertEqual(s.dispatch_delta(zero,zero,'decode',1,200)['path'],'short-packed-bypass')
        self.assertEqual(s.dispatch_delta(zero,zero,'prefill',19,None)['path'],'prefill-untouched')
        for after in (zero,fallback,hit[:-1]+zero[-1:],hit[:-1]+[dict(candidate=True,fallback=0)]):
            with self.assertRaises(ValueError):s.dispatch_delta(zero,after,'decode',1,900)
        with self.assertRaises(ValueError):s.dispatch_delta(zero,hit,'prefill',1,None)

    def test_original_case_function_is_called_unchanged_and_hooks_restore_on_error(self):
        base=base_module();calls=[]
        def original(case,engines,observers,*args,**kwargs):
            calls.append((case,engines,observers,args,kwargs));raise RuntimeError('same original case')
        base.run_case=original;old_observer=base.Observer
        with self.assertRaisesRegex(RuntimeError,'same original case'):
            with s.adapt(base,Adapter(),lambda r:None,enabled=True):
                obs=[SimpleNamespace(case=None,step=99)];case={'name':'lifecycle'}
                base.run_case(case,['real engine placeholder'],obs,1,2,pinned=True)
        self.assertIs(base.run_case,original);self.assertIs(base.Observer,old_observer)
        self.assertEqual(calls[0][2][0].case,'lifecycle');self.assertEqual(calls[0][2][0].step,0)
        self.assertEqual(calls[0][3:],((1,2),{'pinned':True}))

    def test_base_pass_alone_missing_actual_domain_or_unobserved_dispatch_rejects(self):
        result,counts=complete_fixture();s.require_complete(result,counts)
        for corrupt in ('no_c1','no_c8','unrestored','extra_counter','missing_case','base_fail'):
            r,c=copy.deepcopy(result),copy.deepcopy(counts)
            if corrupt=='no_c1':c['per_case']['D-C1']['b1_calls']=0
            elif corrupt=='no_c8':c['per_case']['D-C8']['multirow_calls']=0
            elif corrupt=='unrestored':c['layer_restores']-=1
            elif corrupt=='extra_counter':c['final_counters'][0][0]['candidate']+=1
            elif corrupt=='missing_case':r['cases'].pop()
            else:r['status']='FAIL'
            with self.subTest(corrupt=corrupt),self.assertRaises(ValueError):s.require_complete(r,c)


if __name__=='__main__':unittest.main()
