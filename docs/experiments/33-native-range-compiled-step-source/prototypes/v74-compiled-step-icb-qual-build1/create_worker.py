from pathlib import Path
import shutil,difflib
P=Path('/private/tmp/minivllm-engine-optimization-20261014');D=Path(__file__).resolve().parent;A=P/'v44-native-fullqual-build1';s=(A/'fullqual.py').read_text()
for n in ('candidate-manifest.json','baseline-source-proof.json'):shutil.copyfile(A/n,D/n)
s=s.replace('Root-only conv-SiLU to already-activated normreuse+held32 ICB versus original accepted25 GDN numerical qualification','Root-only V74 independently hypothesized compiled pure131-to132 step then native ICB strict qualification')
s=s.replace('CAP = 256','ACTUAL44 = P / "v23-root-control-build1/fullmodel-v44native-attempt1/child-output/receipt.json"\nACTUAL44_SHA = "1dcfc41e7779c575602bd6c73bae4e4800664db42e1ee87e75414335bdc45b6d"\nCAP = 256',1)
s=s.replace('    assert sha(BASELINE_QUAL)==BASELINE_QUAL_SHA','    assert sha(BASELINE_QUAL)==BASELINE_QUAL_SHA\n    assert sha(ACTUAL44)==ACTUAL44_SHA\n    assert json.loads(ACTUAL44.read_text())["status"]=="PASS_NATIVE_CONV_ACTIVATED_NORMREUSE_CACHE_ICB_BASE_RESET_CHANGED_VS_ACCEPTED25_STRICT"',1)
s=s.replace('    if args.inert:', '    actual44=json.loads(ACTUAL44.read_text())\n    actual44_cases={r["name"]:r for r in actual44["requests"]}\n    allowed_state_PSOs={u["PSO"] for p in actual44["capture"]["native"]["feedback"]["pingpong"]["pairs"] for u in p["actual_binding_uses"]}\n    assert len(allowed_state_PSOs)==1 and all(len(r["steps"])==31 for r in actual44_cases.values())\n    if args.inert:',1)
s=s.replace('"ICB_scope": "actual native ICB qualification; unrelated to mx.compile"','"ICB_scope": "whole pure functional decode mx.compile(shapeless=False),cold eval outsidecapture then warm compiled_fn invocation captured as native ICB"',1)
s=s.replace('"A_saved_baseline_receipt":str(BASELINE_QUAL),"A_saved_baseline_receipt_sha256":BASELINE_QUAL_SHA}', '"A_saved_baseline_receipt":str(BASELINE_QUAL),"A_saved_baseline_receipt_sha256":BASELINE_QUAL_SHA,\n               "actual44_gold_sha256":ACTUAL44_SHA,"origin":"independent current-task hypothesis; installed oMLX GenerationBatch._step has NO whole-step mx.compile",\n               "numeric_policy":"strict originalshape/dtype/finite/Cmemcmp93x132; first cold mismatch STOP, no tolerance widening"}',1)
s=s.replace('normreuse_cache_on=48, whole_functional_off=True','normreuse_cache_on=48, whole_functional_off=False, external_compiled_pure_step=True',1)
start=s.index('        capture_ctx = ForwardContext("decode"');end=s.index('        for value,label in zip(captured,labels_out):',start)
block='''        # Compile/evaluate outside native capture using only explicit131 seed arrays.
        pure_module=module("v74_functional_step",D/"functional_step.py")
        trace_counts={"pure_body":0}
        def observed_trace():trace_counts["pure_body"]+=1
        pure,detach_trace=pure_module.make_pure(model,backend=mx,kv_type=BatchKV,state_type=BatchState,context_type=ForwardContext,max_len=CAP,on_trace=observed_trace)
        forbidden_closures={"capture_seed","tokens","positions","lens","flat","ctx","capture_ctx","seed_holders"}
        assert not forbidden_closures.intersection(pure.__code__.co_freevars)
        compiled_fn=mx.compile(pure,shapeless=False)
        with capture_norm_engagement():
            cold_outputs=tuple(compiled_fn(*capture_seed));assert len(cold_outputs)==132
            mx.eval(*cold_outputs);mx.synchronize()
        cold_traces=trace_counts["pure_body"];cold_GDN_calls=len(norm_capture_calls)
        assert cold_traces>=1 and cold_GDN_calls==48*cold_traces and set(norm_capture_calls)==set(map(id,gdns))
        receipt["compiled_cold"]={"status":"CHECKING","inputs":131,"outputs":132,"shapeless":False,"fixed_cap":CAP,"trace_count":cold_traces,"GDN_python_calls":cold_GDN_calls,"pure_closure_names":list(pure.__code__.co_freevars),"seed_originals_readonly":originals_readonly(native,seed_holders,seed_baselines,"after-cold-compile-eval"),"outputs_checked":[],"scope":"additionalcoldqualification outside nativecapture/93steps/timing"}
        save()
        cold_holders=[shared(v,label) for v,label in zip(cold_outputs,labels_out)]
        cold_gold=actual44_cases["base"]["steps"][0]["outputs"]
        for index,(holder,gold) in enumerate(zip(cold_holders,cold_gold)):
            try:sig=metadata_finite(np,holder)
            except BaseException:
                (args.out/"cold-first-failure.raw").write_bytes(bytes(holder.bytes_view))
                write(args.out/"cold-first-failure.json",{"index":index,"label":holder.label,"reason":"nonfinite/metadata validation","actual":holder.meta,"expected_actual44":gold});raise
            exact=(sig["label"]==gold["label"] and sig["shape"]==gold["shape"] and sig["dtype"]==gold["dtype"] and sig["nbytes"]==gold["bytes"] and sig["sha256"]==gold["A_sha256"])
            row={"index":index,"actual":sig,"expected_actual44":gold,"strict_raw_SHA_match":exact}
            receipt["compiled_cold"]["outputs_checked"].append(row);save()
            if not exact:
                (args.out/"cold-first-failure.raw").write_bytes(bytes(holder.bytes_view));write(args.out/"cold-first-failure.json",row)
                raise AssertionError(("FIRST_COLD_NUMERIC_FAIL_STOP",index,holder.label))
        receipt["compiled_cold"]["status"]="PASS_ALL132_SAVED_ACTUAL44_A_RAW_HASHES_AND131_READONLY"
        # Release cold output owners before warm compiled invocation is captured.
        del cold_outputs,cold_holders,holder
        mx.synchronize();resources.sample("compiled-cold-all132-pass-before-capture",loaded=True,mx=mx);save()
        native.call("v33_begin");armed=True
        with capture_norm_engagement():
            captured=tuple(compiled_fn(*capture_seed));assert len(captured)==132
            mx.eval(*captured);mx.synchronize()
        native.call("v33_end_build");armed=False
        assert trace_counts["pure_body"]==cold_traces and len(norm_capture_calls)==cold_GDN_calls,"capture unexpectedly retraced Python; STOP"
        detach_trace()
        receipt["B_compiled_step_capture_engagement"]={"compiled_function_inputs":131,"compiled_function_outputs":132,"cold_trace_count":cold_traces,"capture_trace_delta":trace_counts["pure_body"]-cold_traces,"cold_GDN_python_calls":cold_GDN_calls,"capture_GDN_python_calls":len(norm_capture_calls)-cold_GDN_calls,"trace_observer_detached_before_native_replay":True,"GDN_observer_removed":GDN._decode_conv_norm_reuse_cache is candidate_norm,"nativeS48_actual_binding_eligibility_required":True,"no_new_S_whitelist":True}
        assert GDN._decode_conv_norm_reuse_cache is candidate_norm
'''
s=s[:start]+block+s[end:]
# Require actual S indices/function matching previously qualified40 graph, not a new inferred fused recurrence.
needle='''        assert (pp["requested_count"],pp["eligible_count"],pp["staged_link_count"])==(48,48,83)
        resources.sample("after-capture-seal",loaded=True,mx=mx); save()
'''
replacement='''        assert (pp["requested_count"],pp["eligible_count"],pp["staged_link_count"])==(48,48,83)
        assert len(pp["pairs"])==48
        for pair in pp["pairs"]:
            uses=pair["actual_binding_uses"];assert pair["eligible"] and pair["bytes"]==3145728 and len(uses)==2
            assert {u["PSO"] for u in uses}==allowed_state_PSOs and len({u["dispatch"] for u in uses})==1
            oldS=next(u for u in uses if u["argument"]=="S");newS=next(u for u in uses if u["argument"]=="S_new")
            assert oldS["index"]==6 and oldS["access"]==0 and newS["index"]==9 and newS["access"]!=0
        receipt["compiled_capture_S_actual_proof"]={"pairs":48,"S_index":6,"S_new_index":9,"exact_old40_PSO_functions":sorted(allowed_state_PSOs),"actual_eligible":48,"new_whitelist":False}
        resources.sample("after-capture-seal",loaded=True,mx=mx); save()
'''
assert needle in s;s=s.replace(needle,replacement,1)
s=s.replace('                    native.call("v34_compare_output",original.address,original.nbytes,ctypes.c_void_p(reference.address))','                    gold44=actual44_cases[name]["steps"][step-1]["outputs"][len(equal)]\n                    assert gold44["label"]==reference.label and gold44["shape"]==sig["shape"] and gold44["dtype"]==sig["dtype"] and gold44["bytes"]==sig["nbytes"] and gold44["A_sha256"]==sig["sha256"],("A differs from actual44",name,step,reference.label)\n                    native.call("v34_compare_output",original.address,original.nbytes,ctypes.c_void_p(reference.address))',1)
s=s.replace('        source_check(library)\n        receipt.update(status="PASS_NATIVE_CONV_ACTIVATED_NORMREUSE_CACHE_ICB_BASE_RESET_CHANGED_VS_ACCEPTED25_STRICT"','        assert total_steps==93 and all(len(r["steps"])==31 and r["A_ids"]==actual44_cases[r["name"]]["A_ids"] for r in receipt["requests"])\n        assert trace_counts["pure_body"]==cold_traces\n        receipt["compiled_trace_count_after_native_replay"]=trace_counts["pure_body"]\n        source_check(library)\n        receipt.update(status="PASS_V74_COMPILED_PURE_STEP_NATIVE_ICB_93X132_STRICT"',1)
s=s.replace('source_pins_verified_final=True,gates_callable_count=48)','source_pins_verified_final=True,gates_callable_count=48,strict_output_checks=93*132,additional_cold_hash_checks=132)',1)
(D/'fullqual.py').write_text(s);(D/'worker-from44.diff').write_text(''.join(difflib.unified_diff((A/'fullqual.py').read_text().splitlines(True),s.splitlines(True),fromfile=str(A/'fullqual.py'),tofile=str(D/'fullqual.py'))))
