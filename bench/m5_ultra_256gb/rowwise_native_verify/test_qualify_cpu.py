"""Unexecuted stdlib fake tests for the owned-child qualifier's narrow seams.

No test imports MLX or loads model/runtime payloads. These are source-prepared
mechanism tests, not a device oracle or a performance measurement.
"""
from contextlib import ExitStack, redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bench.m5_ultra_256gb.rowwise_native_verify import qualify as q
from bench.m5_ultra_256gb.exact_verifier_bridge import qualify as bridge
from bench.m5_ultra_256gb.pareto_realistic import qualify_native as support


LAYOUT = tuple(i%4==3 for i in range(64))


class Array:
    dtype = 'fake-original-dtype'
    def __init__(self, shape): self.shape=shape; self.ndim=len(shape)
    def view(self, *args): raise AssertionError('CPU seam must not inspect tensor payloads')
    def __getitem__(self, index):
        if index!=slice(None): raise AssertionError('full backing slice only')
        return Array(self.shape)


def states(capacity=1536):
    return [SimpleNamespace(K=Array((1,4,capacity,256)),V=Array((1,4,capacity,256))) if attn else
            SimpleNamespace(conv=Array((1,3,10240)),S=Array((1,48,128,128))) for attn in LAYOUT]


class Tests(unittest.TestCase):
    def test_default_description_is_inert_and_does_not_grant_qualification(self):
        output=io.StringIO()
        with patch.object(q,'validate',side_effect=AssertionError('unexpected validation')), \
             patch.object(q,'device',side_effect=AssertionError('unexpected device')), \
             patch.object(q,'BRIDGE_SHA','UNSEALED'),patch.object(q,'CANDIDATE_SHA','UNSEALED'),redirect_stdout(output):
            self.assertEqual(q.main([]),0)
        result=json.loads(output.getvalue())
        self.assertFalse(result['enabled_by_default']); self.assertFalse(result['sources_sealed'])
        self.assertEqual(result['used'],[257,1023,1024]); self.assertEqual(result['compared_prefixes'],42)
        for key in ('compiled_qualified','rollback_qualified','full_model_qualified','http_qualified','performance_qualified'):
            self.assertFalse(result[key])

    def test_partial_cli_rejects_before_output_or_device(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'must-not-exist'
            with patch.object(q,'device',side_effect=AssertionError('unexpected device')), \
                 redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as error:
                q.main(['--out',str(out)])
            self.assertEqual(error.exception.code,2); self.assertFalse(out.exists())

    def test_unsealed_sources_fail_before_source_or_model_load(self):
        plan=dict(schema='Rowwise-native-verifier-plan-v1',qualifier_sha256='pending',candidate={},
                  bridge={},e024_plan={},deadline_s=1800)
        with patch.object(q,'BRIDGE_SHA','UNSEALED'),patch.object(q,'sha',side_effect=AssertionError('no hash/load yet')), \
             self.assertRaisesRegex(ValueError,'UNSEALED'):
            q.validate(plan)

    def test_seed_domain_and_teacher_positions_are_explicit_not_full_fixture_claim(self):
        fixtures=[dict(fixture_id=f'D{i}',normal_eos=True,max_output_tokens=512,
                       prompt_ids=list(range(886+i)),eos_token_ids=[999999]) for i in range(8)]
        before=json.dumps(fixtures,sort_keys=True)
        cases=q.seed_cases(bridge,fixtures)
        self.assertEqual([(c['used'],len(c['prompt_ids']),c['start'],c['position']) for c in cases],
                         [(257,256,1,256),(1023,886,137,1022),(1024,886,138,1023)])
        self.assertEqual(cases[0]['domain'],'diagnostic-prefix256')
        self.assertIs(cases[0]['original_fixture'],fixtures[0])
        self.assertEqual(cases[0]['prompt_ids'],fixtures[0]['prompt_ids'][:256])
        self.assertTrue(all(c['backing']==1536 for c in cases))
        self.assertEqual(json.dumps(fixtures,sort_keys=True),before)

    def test_anchor_accepts_position256_only_with_full_fixed1536_backing(self):
        case=dict(position=256,used=257)
        q.seed_admission(case,[256],256,states(),LAYOUT)
        for position,lens,capacity in ((255,[255],1536),(256,[255],1536),(256,[256],768),(256,[256],1537)):
            with self.subTest(position=position,lens=lens,capacity=capacity),self.assertRaises(ValueError):
                q.seed_admission(case,lens,position,states(capacity),LAYOUT)
        q.seed_admission(dict(position=1023,used=1024),[1023],1023,states(),LAYOUT)

    def test_actual_seed_glue_requests1536_and_reaches_position256_anchor(self):
        events=[]; sentinel=RuntimeError('stop after observed anchor; no decode in this fake')
        class Batch:
            def __init__(self,model): self.layers=[]; self.lens=[]
            def rebuild(self,keep,new):
                events.append(('rebuild',keep,new[0][0],new[0][2]))
                self.layers=states(new[0][2]); self.lens=[256]
            def arrays(self): return [a for state in self.layers for a in vars(state).values()]
            def clear(self): events.append(('clear',)); self.layers.clear()
        def teacher(first,eos,start,advance,anchor,emit):
            self.assertEqual((first,eos,start),(123,{999},1))
            anchor(first,1); raise sentinel
        def dump(mx,out,relative,snap,deadline):
            self.assertEqual(len(snap['arrays']),131)
            self.assertEqual(snap['arrays']['root/state/3/K'].shape,(1,4,1536,256))
            events.append(('anchor',relative))
            return dict(path=relative,sha256='fake',metadata=snap['metadata'])
        mx=SimpleNamespace(int32='i32',array=lambda values,dtype:Array((len(values),)),
                           eval=lambda *values:None,argmax=lambda *args,**kwargs:SimpleNamespace(item=lambda:123))
        fake_bridge=SimpleNamespace(canonical_state=bridge.canonical_state,reference_tokens=teacher)
        helper=SimpleNamespace(write=lambda *args:None,dump=dump)
        cache=SimpleNamespace(make_cache=lambda model:states(),DecodeBatch=Batch)
        context=SimpleNamespace(ForwardContext=lambda *args,**kwargs:None)
        engine=SimpleNamespace(model=lambda *args,**kwargs:Array((1,248320)),eos_ids={999})
        case=dict(prompt_ids=list(range(256)),fixture_id='diagnostic',used=257,position=256,start=1)
        with self.assertRaises(RuntimeError) as error:
            q.acquire_seed(mx,cache,context,fake_bridge,support,helper,None,engine,LAYOUT,case,
                           Path('/not-opened'),lambda:None,lambda value:None)
        self.assertIs(error.exception,sentinel)
        self.assertEqual(events,[('rebuild',[],'teacher-reference',1536),
                                 ('anchor','seeds/used257/canonical.safetensors'),('clear',)])

    def test_every_prefix_snapshot_keeps_full128_arrays_and_one_full_vocab_row(self):
        caches=states(); logits=Array((1,248320))
        snap=q.output_snapshot(support,bridge,logits,caches,LAYOUT,1023,42,2)
        self.assertEqual(len(snap['arrays']),129)
        self.assertEqual(snap['arrays']['root/logits'].shape,(1,248320))
        self.assertEqual(snap['metadata']['position'],1023)
        self.assertEqual(snap['metadata']['lens'],1024)
        for i,attn in enumerate(LAYOUT):
            keys=('K','V') if attn else ('conv','S')
            for name in keys:
                self.assertEqual(snap['arrays'][f'root/state/{i}/{name}'].shape,getattr(caches[i],name).shape)
        ctx=SimpleNamespace(caches=caches,positions=Array((1,)),lens=Array((1,)),max_len=1024,mode='decode',prefill_shape=None)
        self.assertEqual(len(q.input_snapshot(support,bridge,Array((8,)),ctx,LAYOUT)['arrays']),131)

    def test_prefix_wrapper_and_kv_aliases_or_missing_states_reject(self):
        ctx=SimpleNamespace(caches=states()); output=SimpleNamespace(prefixes=[states(),states()])
        q.prefix_ownership(output,ctx,LAYOUT,2)
        output.prefixes[1][3].K=output.prefixes[0][3].K
        with self.assertRaisesRegex(ValueError,'KV descriptors'): q.prefix_ownership(output,ctx,LAYOUT,2)
        output.prefixes=[states(),states()]; output.prefixes[1][0]=ctx.caches[0]
        with self.assertRaisesRegex(ValueError,'cache wrappers'): q.prefix_ownership(output,ctx,LAYOUT,2)
        output.prefixes=[states(),states()[:-1]]
        with self.assertRaisesRegex(ValueError,'complete candidate prefix'): q.prefix_ownership(output,ctx,LAYOUT,2)

    def test_input_after_is_saved_after_evaluation_and_on_single_failure_without_retry(self):
        for failing in (None,'verify','evaluate'):
            with self.subTest(failing=failing):
                events=[]; output=object(); failure=RuntimeError('sentinel')
                def verify(*args,**kwargs):
                    self.assertEqual(kwargs,{'enabled':True}); events.append('verify')
                    if failing=='verify': raise failure
                    return output
                def evaluate(value):
                    self.assertIs(value,output); events.append('evaluate')
                    if failing=='evaluate': raise failure
                call=lambda:q.checked_call(SimpleNamespace(verify=verify),None,None,None,None,
                    before=lambda:events.append('before'),evaluate=evaluate,
                    after=lambda:events.append('after'),compare=lambda:events.append('compare'))
                if failing:
                    with self.assertRaises(RuntimeError) as error: call()
                    self.assertIs(error.exception,failure)
                else: self.assertIs(call(),output)
                self.assertEqual(events,['before','verify']+([] if failing=='verify' else ['evaluate'])+['after','compare'])

    def test_failed_child_retains_case_and_raw_failure_without_false_qualification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); plan=root/'plan.json'; plan.write_text('{"qualifier_sha256":"fake","deadline_s":100}')
            out=root/'out'
            fake_bridge=SimpleNamespace(HELPER=Path('helper.py'),HELPER_SHA='helper',OLD=Path('old.py'),OLD_SHA='old',
                CORE_PINS=Path('pins.json'),IncompleteCoverage=bridge.IncompleteCoverage)
            old=SimpleNamespace(MODEL_SOURCE_SHA='core',SUPPORT_SHA='comparator')
            def failed(*args):
                directory,result=args[-4],args[-1]
                result['cases'].append(dict(used=257,width=2,status='FAIL',prefixes=[dict(prefix=1,status='FAIL')]))
                (directory/'failure.json').write_text('{"label":"prefix1/root/logits"}')
                raise RuntimeError('original-dtype mismatch')
            with ExitStack() as stack:
                stack.enter_context(patch.object(q,'validate',return_value=(fake_bridge,None,None,old,None,
                    {'e023_qualifier':{'path':'comparator.py'}},{})))
                stack.enter_context(patch.object(q,'device',side_effect=failed))
                stack.enter_context(patch.object(q.signal,'getitimer',return_value=(0.0,0.0)))
                stack.enter_context(patch.object(q.signal,'getsignal',return_value=None))
                stack.enter_context(patch.object(q.signal,'signal'))
                stack.enter_context(patch.object(q.signal,'setitimer'))
                stack.enter_context(redirect_stdout(io.StringIO()))
                self.assertEqual(q.main(['--plan',str(plan),'--plan-sha256',q.sha(plan),'--out',str(out)]),1)
            result=json.loads((out/'complete.json').read_text())
            self.assertEqual(result['status'],'FAIL'); self.assertEqual(len(result['cases']),1)
            self.assertTrue((out/'failure.json').is_file())
            for flag in ('forward_prefix_qualified','compiled_qualified','rollback_qualified','http_qualified','performance_qualified'):
                self.assertFalse(result[flag])
            self.assertFalse(result['source_model_pins_rechecked'])


if __name__=='__main__': unittest.main()
