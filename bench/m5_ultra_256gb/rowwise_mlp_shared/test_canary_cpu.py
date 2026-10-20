"""Tiny mechanism tests; no backend, payload, device or model runtime imports."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from bench.m5_ultra_256gb.rowwise_mlp_shared import canary as c
from bench.m5_ultra_256gb.rowwise_mlp_shared import test_cpu as f


class Tests(unittest.TestCase):
    def test_default_inert_and_exact_count_contract(self):
        with patch.object(c,'validate',side_effect=AssertionError('validation on default')),patch.object(c,'device',side_effect=AssertionError('device on default')),redirect_stdout(io.StringIO()) as out:
            self.assertEqual(c.main([]),0)
        value=json.loads(out.getvalue())
        self.assertFalse(value['enabled_by_default']);self.assertEqual((value['projection_pairs'],value['input_files'],value['raw_files']),(12,6,30))
        self.assertEqual((value['provenance_pairs'],value['provenance_tensors_per_pair']),(4,129))
        self.assertTrue(all(value[k] is False for k in ('compiled_qualified','full_model_qualified','http_qualified','performance_qualified','engine_pareto_point')))

    def test_partial_run_arguments_rejected_before_device(self):
        with patch.object(c,'device',side_effect=AssertionError('device')),self.assertRaises(ValueError):c.main(['--run'])
        with patch.object(c,'validate',side_effect=AssertionError('validation')),self.assertRaises(SystemExit):c.main(['--plan','missing'])

    def test_shadow_values_never_feed_later_layers_two_widths(self):
        all_counts=[]
        for width in c.WIDTHS:
            model=f.Model();ctx=f.f.seed(1023);before=f.f.state_value(ctx.caches);captures=[]
            def shadow(x,w,s):
                actual=model.projector(x,w,s)
                return f.Array(actual.shape,actual.dtype,[-1]*width)
            def emit(T,layer,name,x,native,shared):
                captures.append((T,layer,name,x.shape,list(native.values),list(shared.values)))
                self.assertNotEqual(native.values,shared.values)
            probe=c.NativeReturningProbe(model,f.MX,width,shadow,emit)
            output=f.c.verify(model,f.tokens(width),ctx,f.RUNTIME,enabled=True,mlp_input_project=probe)
            original=f.c.frozen().verify(f.Model(),f.tokens(width),f.f.seed(1023),f.RUNTIME,enabled=True)
            causal=f.f.token_major_reference(f.Model(),f.tokens(width),f.f.seed(1023))
            for expected in (original,causal):
                self.assertEqual(output.logits.values,expected.logits.values)
                self.assertEqual([f.f.state_value(p) for p in output.prefixes],[f.f.state_value(p) for p in expected.prefixes])
            self.assertEqual(f.f.state_value(ctx.caches),before)
            self.assertEqual([(a,b,d) for a,b,d,*_ in captures],[(width,i,n) for i in c.LAYERS for n in ('gate','up')])
            receipt=probe.finish();self.assertEqual(receipt['shared_outputs_returned'],0);all_counts.append(receipt)
        self.assertEqual([sum(x[k] for x in all_counts) for k in ('callback_count','native_m1_projection_calls','shadow_projection_calls')],[256,1536,12])

    def test_probe_actual_weight_order_and_gate_up_input_identity(self):
        model=f.Model();probe=c.NativeReturningProbe(model,f.MX,4,model.projector,lambda *a:None);x=f.Array((4,5120),'bf16',[1]*4)
        gate,up=model.layers[0].mlp.gate_proj,model.layers[0].mlp.up_proj
        with self.assertRaisesRegex(ValueError,'ordered actual'):probe(x,up.weight,up.scales)
        self.assertEqual(model.events,[])
        probe(x,gate.weight,gate.scales)
        with self.assertRaisesRegex(ValueError,'same actual'):probe(x[:],up.weight,up.scales)
        with self.assertRaisesRegex(ValueError,'complete native-returning'):probe.finish()

    def test_probe_duplicate_live_handles_and_domain_rejected(self):
        model=f.Model();model.layers[1].mlp.up_proj=model.layers[0].mlp.gate_proj
        with self.assertRaisesRegex(ValueError,'distinct actual'):c.NativeReturningProbe(model,f.MX,4,model.projector,lambda *a:None)
        for width in (2,3,16):
            with self.assertRaisesRegex(ValueError,'fixed probe domain'):c.NativeReturningProbe(f.Model(),f.MX,width,lambda *a:None,lambda *a:None)

    def test_probe_wrong_input_and_shadow_rejection_no_retry(self):
        model=f.Model();op=model.layers[0].mlp.gate_proj;calls=[]
        def shadow(*a):calls.append(a);return None
        probe=c.NativeReturningProbe(model,f.MX,4,shadow,lambda *a:self.fail('bad shadow emitted'))
        for x in (f.Array((4,5120),'f32',[1]*4),f.Array((4,6144),'bf16',[1]*4)):
            with self.assertRaisesRegex(ValueError,'normalized row'):probe(x,op.weight,op.scales)
        self.assertEqual(model.events,[])
        with self.assertRaisesRegex(ValueError,'shadow shared'):probe(f.Array((4,5120),'bf16',[1]*4),op.weight,op.scales)
        self.assertEqual(len(calls),1);self.assertEqual(sum(e[0]=='native' for e in model.events),4)

    def test_probe_shadow_exception_identity_preserved(self):
        model=f.Model();op=model.layers[0].mlp.gate_proj;failure=RuntimeError('failed kernel');calls=[]
        def shadow(*a):calls.append(1);raise failure
        probe=c.NativeReturningProbe(model,f.MX,8,shadow,lambda *a:self.fail('emit'))
        with self.assertRaises(RuntimeError) as caught:probe(f.Array((8,5120),'bf16',[1]*8),op.weight,op.scales)
        self.assertIs(caught.exception,failure);self.assertEqual(calls,[1])

    def provenance_args(self,width=4):
        def reference():return f.f.token_major_reference(f.Model(),f.tokens(width),f.f.seed(1023))
        native,baseline,output=reference(),reference(),reference();rows=[]
        def snapshot(value):
            logits=value['logits'];return dict(logits=(logits.shape,logits.dtype,list(logits.values)),state=value['state'])
        support=SimpleNamespace(snapshot=snapshot)
        bridge=SimpleNamespace(canonical_state=lambda layers,layout:f.f.state_value(layers))
        def compare(label,left,right):self.assertEqual(left,right);rows.append(label)
        cmp=SimpleNamespace(compare=compare)
        oracle=dict(logits=native.logits,selected={width:native.prefixes[-1]})
        return [support,bridge,cmp,width,tuple(i%4==3 for i in range(64)),oracle,baseline,output],rows

    def test_provenance_compares_both_full_logits_and_selected_states(self):
        for width in c.WIDTHS:
            args,rows=self.provenance_args(width);receipt=c.compare_provenance(*args)
            self.assertEqual(rows,[f'T{width}/provenance/native-causal',f'T{width}/provenance/frozen-verifier'])
            self.assertEqual(receipt['selected_state_arrays'],128);self.assertEqual(receipt['k'],width)
            self.assertTrue(receipt['full_logits']);self.assertFalse(receipt['raw_state_saved'])

    def test_provenance_rejects_changed_native_state_and_frozen_logits(self):
        args,_=self.provenance_args();args[5]['selected'][4][63].K.cells[1535]=-9
        with self.assertRaises(AssertionError):c.compare_provenance(*args)
        args,_=self.provenance_args();args[6].logits.values=tuple(-9 if i==2 else v for i,v in enumerate(args[6].logits.values))
        with self.assertRaises(AssertionError):c.compare_provenance(*args)
        args,_=self.provenance_args();args[7].prefixes[-1].pop()
        with self.assertRaisesRegex(ValueError,'complete selected'):c.compare_provenance(*args)

    def test_invalid_plan_rejected_without_loading_dependencies(self):
        valid=dict(schema='C1-MLP-shared-activation-plan-v1',canary_sha256='same',e054_plan={},e054_result={},e054_audit={},deadline_s=1800)
        with patch.object(c,'sha',return_value='same'),patch.object(c,'pinned',side_effect=AssertionError('evidence read')):
            for deadline in (0,-1,float('nan'),float('inf'),True,1801):
                with self.assertRaisesRegex(ValueError,'deadline'):c.validate(dict(valid,deadline_s=deadline))
            with self.assertRaisesRegex(ValueError,'exact canary plan'):c.validate(dict(valid,extra=True))
            with self.assertRaisesRegex(ValueError,'source SHA'):c.validate(dict(valid,canary_sha256='wrong'))


if __name__=='__main__':unittest.main()
