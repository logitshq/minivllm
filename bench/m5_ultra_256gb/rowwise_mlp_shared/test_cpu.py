"""Scalar fingerprint mechanisms, not native numerical/device qualification."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from bench.m5_ultra_256gb.rowwise_mlp_shared import candidate as c
from bench.m5_ultra_256gb.rowwise_native_verify import test_candidate_cpu as f

P=1000003


class Array(f.Array):
    def __getitem__(self,index):
        value=super().__getitem__(index); return Array(value.shape,value.dtype,value.values)
    def __add__(self,other):
        values=other.values if hasattr(other,'values') else [other]*len(self.values)
        return Array(self.shape,self.dtype,[(x+y)%P for x,y in zip(self.values,values)])
    def __mul__(self,other):
        if self.shape!=other.shape: raise AssertionError('original rowwise multiply')
        return Array(self.shape,self.dtype,[(x*y)%P for x,y in zip(self.values,other.values)])


def convert(value): return Array(value.shape,value.dtype,value.values)


class MX(f.MX):
    @staticmethod
    def concatenate(arrays,axis=0): return convert(f.MX.concatenate(arrays,axis))


def silu(x):
    if x.shape!=(1,17408): raise AssertionError('SiLU must remain original M1')
    return Array(x.shape,x.dtype,[(v*v+3)%P for v in x.values])


RUNTIME=SimpleNamespace(mx=MX,silu=silu,ForwardContext=f.ForwardContext,BatchKV=f.BatchKV,BatchState=f.BatchState)


class Linear:
    def __init__(self,model,index,name,K,N):
        self.model,self.index,self.name,self.in_features,self.out_features=model,index,name,K,N
        self.q=model.cfg.quantization; self.weight=Array((N,K//8),'u32',[index+{'gate':2,'up':7,'down':11,'head':29}[name]])
        self.scales=Array((N,K//16),'u8',[3])
    def __call__(self,x):
        if x.shape!=(1,self.in_features): raise AssertionError('native projection must retain M1')
        self.model.events.append(('native',self.index,self.name,x.shape))
        return Array((1,self.out_features),'bf16',[(v*self.weight.values[0]+self.scales.values[0])%P for v in x.values])


class MLP:
    def __init__(self,model,index):
        self.gate_proj=Linear(model,index,'gate',5120,17408); self.up_proj=Linear(model,index,'up',5120,17408)
        self.down_proj=Linear(model,index,'down',17408,5120)
    def __call__(self,x): return self.down_proj(silu(self.gate_proj(x))*self.up_proj(x))


class Mixer:
    def __init__(self,model,index): self.native=f.Layer(model,index); self.model,self.index=model,index
    def __call__(self,h,ctx,index,**kwargs):
        expected={} if index%4==3 else {'_norm_reuse_normal':True}
        if kwargs!=expected: raise AssertionError('mixer route changed or GDN _projection injected')
        return convert(self.native(h,ctx,index,_norm_reuse_normal=True))


class Layer:
    def __init__(self,model,index):
        self.model,self.index,self.is_attn=model,index,index%4==3; self.mlp=MLP(model,index)
        if self.is_attn:self.self_attn=Mixer(model,index)
        else:self.linear_attn=Mixer(model,index)
    def input_layernorm(self,x): return self.norm(x,'input',5)
    def post_attention_layernorm(self,x): return self.norm(x,'post',7)
    def norm(self,x,label,multiplier):
        if x.shape!=(1,5120): raise AssertionError('original M1 norm')
        self.model.events.append((label,self.index,x.shape))
        return Array(x.shape,x.dtype,[(v*multiplier+1)%P for v in x.values])
    def __call__(self,x,ctx,index,*,_norm_reuse_normal=False):
        if not _norm_reuse_normal:raise AssertionError('normal context proof')
        h=self.input_layernorm(x)
        y=self.self_attn(h,ctx,index) if self.is_attn else self.linear_attn(h,ctx,index,_norm_reuse_normal=True)
        mixed=x+y; return mixed+self.mlp(self.post_attention_layernorm(mixed))


class Embed(f.Embed):
    def __call__(self,tokens):return convert(super().__call__(tokens))


class Model(f.Model):
    def __init__(self):
        super().__init__(); self.cfg.intermediate_size=17408; self.cfg.vocab_size=248320
        self.embed_tokens=Embed(self); self.layers=[Layer(self,i) for i in range(64)]
    def norm(self,x):return convert(super().norm(x))
    def projector(self,x,w,s):
        found=[(i,name,op) for i,layer in enumerate(self.layers) for name,op in (('gate',layer.mlp.gate_proj),('up',layer.mlp.up_proj))
               if w is op.weight and s is op.scales]
        if len(found)!=1:raise AssertionError('actual live per-layer weights')
        index,name,op=found[0]
        if x.shape not in ((2,5120),(4,5120),(8,5120)):raise AssertionError('supported T,K')
        self.events.append(('shared',index,name,x.shape))
        return Array((x.shape[0],17408),'bf16',[(v*w.values[0]+s.values[0])%P for v in x.values])


def tokens(width):return Array((width,),'i32',range(41,41+width))
def verify(model,width=4,used=1023,**kwargs):
    ctx=f.seed(used); output=c.verify(model,tokens(width),ctx,RUNTIME,enabled=True,mlp_input_project=model.projector,**kwargs)
    return output,ctx


class Tests(unittest.TestCase):
    def test_off_inert_and_enabled_bool(self):
        class Poison:
            def __getattribute__(self,name):raise AssertionError('OFF inspected '+name)
        self.assertIsNone(c.verify(Poison(),Poison(),Poison(),Poison(),head_project=Poison(),mlp_input_project=Poison()))
        for flag in (0,1,'on',None):
            with self.assertRaises(TypeError):c.verify(Poison(),Poison(),Poison(),Poison(),enabled=flag)

    def test_no_projector_calls_frozen_original_unchanged(self):
        model=Model(); ctx=f.seed(1023)
        original=c.frozen(); sentinel=object()
        with patch.object(original,'verify',return_value=sentinel) as fn:
            self.assertIs(c.verify(model,tokens(4),ctx,RUNTIME,enabled=True),sentinel)
            fn.assert_called_once_with(model,unittest.mock.ANY,ctx,RUNTIME,enabled=True,head_project=None)
        self.assertEqual(model.events,[])

    def test_all_prefixes_equal_independent_token_major_for_9_domains(self):
        for used in (257,1023,1024):
            for width in (2,4,8):
                model=Model(); actual,ctx=verify(model,width,used); before=f.state_value(ctx.caches)
                expected=f.token_major_reference(Model(),tokens(width),ctx)
                with self.subTest(used=used,width=width):
                    self.assertEqual(actual.logits.values,expected.logits.values)
                    self.assertEqual([f.state_value(p) for p in actual.prefixes],[f.state_value(p) for p in expected.prefixes])
                    self.assertEqual(f.state_value(ctx.caches),before)
                    self.assertEqual(sum(e[0]=='shared' for e in model.events),128)

    def test_mixer_recurrence_before_sharing_and_row_arithmetic_order(self):
        model=Model(); verify(model,4)
        for i in range(64):
            rows=[e for e in model.events if len(e)>1 and e[1]==i and e[0] in ('input','layer','post','shared','native')]
            expected=[]
            for t in range(4): expected.extend([('input',i,(1,5120)),('layer',i,1022+t,1023+t,1023+t,(1,5120)),('post',i,(1,5120))])
            expected.extend([('shared',i,'gate',(4,5120)),('shared',i,'up',(4,5120))])
            expected.extend([('native',i,'down',(1,17408))]*4)
            self.assertEqual(rows,expected)
        self.assertEqual(sum(e[0]=='norm' for e in model.events),4)

    def test_prefix_cache_ownership_and_full_padding_remain_private(self):
        model=Model(); actual,ctx=verify(model,8); before=f.state_value(ctx.caches)
        for i in range(64):
            wrappers=[ctx.caches[i],*(p[i] for p in actual.prefixes)]
            self.assertEqual(len({id(v) for v in wrappers}),9)
            for field in (('K','V') if i%4==3 else ('conv','S')):
                arrays=[getattr(v,field) for v in wrappers];self.assertEqual(len({id(v) for v in arrays}),9)
            if i%4==3:
                self.assertTrue(all(p[i].K.cells[1535]==ctx.caches[i].K.cells[1535] for p in actual.prefixes))
        actual.prefixes[-1][63].K.cells[1535]=-1
        self.assertEqual(f.state_value(ctx.caches),before)
        self.assertNotEqual(actual.prefixes[-2][63].K.cells[1535],-1)

    def test_all_layer_metadata_admitted_before_embedding_or_state(self):
        for mutate in (lambda m:setattr(m.cfg,'intermediate_size',4096),lambda m:setattr(m.layers[63].mlp.gate_proj,'in_features',6144),
                       lambda m:setattr(m.layers[63].mlp.up_proj.scales,'dtype','f32'),lambda m:setattr(m.layers[0].mlp.down_proj,'biases',object()),
                       lambda m:setattr(m.layers[0].mlp.down_proj.weight,'shape',(5120,640))):
            model=Model(); mutate(model); ctx=f.seed(); before=f.state_value(ctx.caches)
            with self.assertRaises(ValueError):c.verify(model,tokens(4),ctx,RUNTIME,enabled=True,mlp_input_project=model.projector)
            self.assertEqual(model.events,[]); self.assertEqual(f.state_value(ctx.caches),before)
        model=Model();self.assertIsNone(c.verify(model,tokens(3),f.seed(),RUNTIME,enabled=True,mlp_input_project=model.projector));self.assertEqual(model.events,[])

    def test_explicit_projector_failure_once_no_retry_no_canonical_mutation(self):
        for returned in (None,Array((4,3),'bf16',[1]*4),Array((4,17408),'f32',[1]*4),RuntimeError('projection failed')):
            model=Model(); ctx=f.seed(); before=f.state_value(ctx.caches); calls=[]
            def project(*args):
                calls.append(1)
                if isinstance(returned,Exception):raise returned
                return returned
            with self.assertRaises((ValueError,RuntimeError)):c.verify(model,tokens(4),ctx,RUNTIME,enabled=True,mlp_input_project=project)
            self.assertEqual(calls,[1]);self.assertEqual(f.state_value(ctx.caches),before)
            self.assertFalse(any(e[0]=='native' for e in model.events))

    def test_mixer_failure_propagates_before_any_projection(self):
        model=Model(); model.fail_at=(0,257); ctx=f.seed(); before=f.state_value(ctx.caches)
        with self.assertRaises(RuntimeError) as caught:c.verify(model,tokens(4),ctx,RUNTIME,enabled=True,mlp_input_project=model.projector)
        self.assertIs(caught.exception,model.failure);self.assertFalse(any(e[0]=='shared' for e in model.events));self.assertEqual(f.state_value(ctx.caches),before)

    def test_live_weight_replacement_changes_correct_branch_without_packing(self):
        model=Model(); first,ctx=verify(model,2); op=model.layers[17].mlp.gate_proj
        op.weight=Array(op.weight.shape,op.weight.dtype,[89]); before=op.weight
        actual,_=verify(model,2); reference=Model(); reference.layers[17].mlp.gate_proj.weight=Array(op.weight.shape,op.weight.dtype,[89])
        expected=f.token_major_reference(reference,tokens(2),ctx)
        self.assertEqual(actual.logits.values,expected.logits.values);self.assertNotEqual(first.logits.values,actual.logits.values);self.assertIs(op.weight,before)

    def test_shared_head_receives_only_original_m1_final_norm_rows(self):
        model=Model();model.lm_head=Linear(model,0,'head',5120,248320);calls=[]
        def head(x,w,s):
            self.assertIs(w,model.lm_head.weight);self.assertIs(s,model.lm_head.scales);calls.append(x.shape)
            return Array((4,248320),'bf16',[(v*w.values[0]+s.values[0])%P for v in x.values])
        actual,ctx=verify(model,4,head_project=head); reference=Model();reference.lm_head=Linear(reference,0,'head',5120,248320)
        expected=f.token_major_reference(reference,tokens(4),ctx)
        self.assertEqual(actual.logits.values,expected.logits.values);self.assertEqual(calls,[(4,5120)]);self.assertEqual(sum(e[0]=='norm' for e in model.events),4)

    def test_missing_original_silu_rejected_before_work(self):
        model=Model(); runtime=SimpleNamespace(**{k:v for k,v in vars(RUNTIME).items() if k!='silu'})
        with self.assertRaisesRegex(ValueError,'runtime.silu'):c.verify(model,tokens(4),f.seed(),runtime,enabled=True,mlp_input_project=model.projector)
        self.assertEqual(model.events,[])

    def test_frozen_baseline_source_change_refuses_and_restores_cache(self):
        c.frozen.cache_clear();self.addCleanup(c.frozen.cache_clear)
        with tempfile.TemporaryDirectory() as directory:
            wrong=Path(directory)/'wrong.py';wrong.write_text('# wrong\n');model=Model()
            with patch.object(c,'ORIGINAL',wrong),self.assertRaisesRegex(ValueError,'frozen original'):
                c.verify(model,tokens(4),f.seed(),RUNTIME,enabled=True,mlp_input_project=model.projector)
            self.assertEqual(model.events,[])


if __name__=='__main__':unittest.main()
