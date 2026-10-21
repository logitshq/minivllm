"""Root independent integration checks for the agent's reference and wrapper.

Reuse the previously tested scalar model; compare values and ownership rather
than requiring a particular number of internal concatenations. No MLX imports.
"""
import sys
sys.path.insert(0,'/Users/ryanli/code/minivllm')
from types import SimpleNamespace
import unittest
import implementation as impl
from bench.m5_ultra_256gb.rowwise_mlp_shared_features import test_cpu as ef
from bench.m5_ultra_256gb.rowwise_mlp_shared_features.round import select_greedy
f=ef.f
n=f.f


def model():
    value=f.Model();value.lm_head=f.Linear(value,0,'head',5120,248320)
    return value


def projector(m):
    def run(x,w,s):
        if w is m.lm_head.weight:
            assert s is m.lm_head.scales
            return f.Array((x.shape[0],248320),'bf16',[(v*w.values[0]+s.values[0])%f.P for v in x.values])
        return m.projector(x,w,s)
    return run


def original(tokens,ctx):
    m=model();captures={i:[] for i in impl.CAPTURE}
    for i in captures:m.layers[i]=ef.Recorder(m.layers[i],captures[i])
    output=n.token_major_reference(m,tokens,ctx)
    return output,tuple(zip(*(captures[i] for i in impl.CAPTURE)))


def context(layers,used):
    return n.ForwardContext('decode',layers,n.Array((1,),'i32',[used-1]),n.Array((1,),'i32',[used]),used)


class Integration(unittest.TestCase):
    def equivalent(self,output,expected,features):
        self.assertEqual(output.logits.values,expected.logits.values)
        self.assertEqual(output.features.shape,(len(features),25600))
        self.assertEqual(output.features.rows,features)
        self.assertEqual([n.state_value(p) for p in output.prefixes], [n.state_value(p) for p in expected.prefixes])

    def test_reference_matches_independent_complete_layer_capture(self):
        for width in (1,2,4,8):
            for used in (257,1023,1024):
                with self.subTest(width=width,used=used):
                    ctx=n.seed(used);before=n.state_value(ctx.caches);tokens=f.tokens(width)
                    expected,features=original(tokens,ctx)
                    actual=impl.reference_with_features(model(),tokens,ctx,ef.RUNTIME)
                    self.equivalent(actual,expected,features)
                    self.assertEqual(n.state_value(ctx.caches),before)

    def test_candidate_matches_independent_features_logits_and_all_prefixes(self):
        for width in (2,4,8):
            ctx=n.seed(1023);before=n.state_value(ctx.caches);tokens=f.tokens(width);m=model()
            expected,features=original(tokens,ctx)
            actual=impl.candidate_with_features(m,tokens,ctx,ef.RUNTIME,projector(m))
            self.equivalent(actual,expected,features)
            self.assertEqual(n.state_value(ctx.caches),before)

    def test_all_selected_prefixes_continue_from_equal_state(self):
        for width in (4,8):
            ctx=n.seed(1023);tokens=f.tokens(width);m=model()
            expected,features=original(tokens,ctx)
            actual=impl.candidate_with_features(m,tokens,ctx,ef.RUNTIME,projector(m))
            for count in range(1,width+1):
                with self.subTest(width=width,count=count):
                    token=f.Array((1,),'i32',[99]);left=context(expected.prefixes[count-1],1023+count);right=context(actual.prefixes[count-1],1023+count)
                    continuation,feats=original(token,left)
                    self.equivalent(impl.reference_with_features(model(),token,right,ef.RUNTIME),continuation,feats)

    def test_reference_prefix_wrappers_and_backings_are_independent(self):
        for width in (1,4,8):
            ctx=n.seed();actual=impl.reference_with_features(model(),f.tokens(width),ctx,ef.RUNTIME)
            states=[ctx.caches,*actual.prefixes]
            for i in range(64):
                self.assertEqual(len({id(p[i]) for p in states}),width+1)
                for key in (('K','V') if i%4==3 else ('conv','S')):
                    self.assertEqual(len({id(getattr(p[i],key)) for p in states}),width+1)
            before=n.state_value(ctx.caches);actual.prefixes[-1][63].K.cells[1535]=-1
            self.assertEqual(n.state_value(ctx.caches),before)
            if width>1:self.assertNotEqual(actual.prefixes[-2][63].K.cells[1535],-1)

    def test_every_rejection_selects_matching_features_and_cache(self):
        for width in (4,8):
            m=model();out=impl.candidate_with_features(m,f.tokens(width),n.seed(),ef.RUNTIME,projector(m));drafts=list(range(width-1))
            for accepted in range(width):
                post=drafts+[999];post[accepted]=999;chosen=select_greedy(drafts,post,out)
                self.assertEqual(chosen.accepted_drafts,accepted);self.assertIs(chosen.layers,out.prefixes[accepted])
                self.assertEqual(chosen.features.rows,out.features.rows[:accepted+1])
                self.assertEqual(chosen.produced,tuple(drafts[:accepted])+(999,))

    def test_invalid_width_and_rejected_candidate(self):
        with self.assertRaises(ValueError):impl.reference_with_features(model(),f.tokens(3),n.seed(),ef.RUNTIME)
        m=model();self.assertIsNone(impl.candidate_with_features(m,f.tokens(1),n.seed(),ef.RUNTIME,projector(m)))

if __name__=='__main__':unittest.main()
