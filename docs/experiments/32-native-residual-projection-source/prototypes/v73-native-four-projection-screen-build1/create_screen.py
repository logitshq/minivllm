from pathlib import Path
import difflib
P=Path('/private/tmp/minivllm-engine-optimization-20261014');D=Path(__file__).resolve().parent;A=P/'v44-native-decode-screen-build1';F=P/'v68-native-fullqual-build1'
s=(A/'screen.py').read_text().replace('Root-only B-only native exploratory timing','Root-only V73 four-original-pointer GDN B-only native exploratory timing')
s=s.replace('ROOT = P / "candidate-v44-conv-normreuse-build1-root"','ROOT = P / "candidate-v68-four-projection-build1-root"')
s=s.replace('fullmodel-v44native-attempt1/child-output/receipt.json','fullmodel-v68native-attempt1/child-output/receipt.json',1).replace('QUAL_SHA = "1dcfc41e7779c575602bd6c73bae4e4800664db42e1ee87e75414335bdc45b6d"','QUAL_SHA = "015d175ec1094926fa4f278505f3716c602e4e3926843be6a63a828011be548f"\nRELEASE = P / "v23-root-control-build1/fullmodel-v68native-attempt1/external-release.json"\nRELEASE_SHA = "1329e63363e79c793076e6203bbe1eb0c24fcef906c6d03f405f1cd448f85c7b"\nOLD44 = P / "v23-root-control-build1/fullmodel-v44native-attempt1/child-output/receipt.json"\nOLD44_SHA = "1dcfc41e7779c575602bd6c73bae4e4800664db42e1ee87e75414335bdc45b6d"')
s=s.replace('assert q["status"]=="PASS_NATIVE_CONV_ACTIVATED_NORMREUSE_CACHE_ICB_BASE_RESET_CHANGED_VS_ACCEPTED25_STRICT"','assert q["status"]=="PASS_V68_FOUR_ORIGINAL_POINTER_GDN_FULL_ICB_BASE_RESET_CHANGED_STRICT"\n    assert q["strict_output_checks"]==12276 and q["four_projection_target384_readonly_final"] and q["four_projection_ON"]==48\n    assert sha(RELEASE)==RELEASE_SHA and json.loads(RELEASE.read_text())["GPU_released"]\n    assert sha(OLD44)==OLD44_SHA')
s=s.replace('engagement=q["B_conv_normreuse_cache_capture_engagement"]\n    assert engagement["calls"]==engagement["distinct_GDN_objects"]==48 and engagement["counter_removed_after_capture"] is True','engagement=q["B_four_projection_capture_engagement"]\n    assert engagement["calls"]==engagement["distinct_GDN_objects"]==48 and engagement["observers_removed_before_replay"] is True\n    assert len(engagement["projection_calls"])==48 and all(v["outputs"]==4 for v in engagement["projection_calls"])')
s=s.replace('    assert all(x["strict_bytes_equal"] and x["finite"] for x in expected)','    assert all(x["strict_bytes_equal"] and x["finite"] for x in expected)\n    assert expected==json.loads(OLD44.read_text())["requests"][0]["steps"][-1]["outputs"]',1)
s=s.replace('P/"v44-native-fullqual-build1/candidate-manifest.json"','P/"v68-native-fullqual-build1/candidate-manifest.json"').replace('package.rglob("*.py")}==expected_source and len(expected_source)==27','package.rglob("*") if f.is_file()}==expected_source and sum(k.endswith(".py") for k in expected_source)==28')
s=s.replace('"qualification_receipt":str(QUAL),"qualification_sha256":QUAL_SHA,"requests":[],','"qualification_receipt":str(QUAL),"qualification_sha256":QUAL_SHA,"qualification_release_sha256":RELEASE_SHA,"old44_A_gold_sha256":OLD44_SHA,"requests":[],\n             "individual_argument_reflection":"NOT_OBSERVED; sourceABI13 pointers, native40 report lacks per-argument table",\n             "clockscope_comparison":"Same raw-native V44 t1 after seed uploads; do not compare directly to V67/V71 engine decode clocks including seed setup",')
s=s.replace('P/"v44-native-fullqual-build1/fullqual.py"','P/"v68-native-fullqual-build1/fullqual.py"')
s=s.replace('        gdns=[l.linear_attn','        initial_parameter_identity=[(name,id(v)) for name,v in tree_flatten(model.parameters())]\n        gdns=[l.linear_attn',1)
start=s.index('        candidate_norm=GDN._decode_conv_norm_reuse_cache');end=s.index('        receipt.update(stage="ready"',start)
source=F.joinpath('fullqual.py').read_text();cs=source.index('        four_capture_calls=[];projection_capture_calls=[]');ce=source.index('        receipt["A_baseline_source"]',cs)
block=source[cs:ce]
s=s[:start]+'''        candidate_norm=GDN._decode_conv_norm_reuse_cache
        candidate_decode=GDN._decode_dispatch
        candidate_four=GDN._decode_conv_norm_reuse_four_projection
        assert all(g._four_projection_enabled is False and g._four_projection_helper is None for g in gdns)
'''+block+s[end:]
start=s.index('        capture_seed,first=setup();assert first==w["continuation_ids"][0]')
prepstart=source.index('        # Additional single ordinary decode solely qualifies actual norm-output layout.')
prepend=source.index('        capture_seed, first = setup(base_prompt);',prepstart)
prep=source[prepstart:prepend]
# Replace only the baseline-context wrapping of this extra preparation decode;
# use candidate44 OFF branch here, without a new baseline numerical comparison.
begin=prep.index('        layout_seed,layout_first=setup(base_prompt);layout_x={}')
end=prep.index('        assert set(layout_x)',begin)
prep=prep[:begin]+'''        layout_seed,layout_first=setup();layout_x={}
        assert all(g._four_projection_enabled is False for g in gdns)
        def retain_layout_x(self,x,*a,**kw):
            assert id(self) not in layout_x
            layout_x[id(self)]=x
            return candidate_decode(self,x,*a,**kw)
        GDN._decode_dispatch=retain_layout_x
        try:
            layout_ctx=ForwardContext("decode",layers(layout_seed[3:]),positions=layout_seed[1],lens=layout_seed[2],max_len=20)
            layout_logits=model(layout_seed[0],layout_ctx)
            mx.eval(layout_logits,*states(layout_ctx),*layout_x.values());mx.synchronize()
            assert int(mx.argmax(layout_logits,axis=-1).item())==w["continuation_ids"][1]
        finally:GDN._decode_dispatch=candidate_decode
'''+prep[end:]
prep=prep.replace('resources.sample("after-four-projection-layout-prepare",loaded=True,mx=mx)','sample("after-four-projection-layout-prepare")')
s=s[:start]+prep+s[start:]
start=s.index('        assert len(norm_capture_calls)==48');end=s.index('        assert len(captured)==132',start)
blockstart=source.index('        assert len(four_capture_calls)==48');blockend=source.index('        for value,label in zip(captured,labels_out):',blockstart)
s=s[:start]+source[blockstart:blockend]+s[end:]
# Save original capture owners once, outside all clocks.
s=s.replace('        step_fn=native.lib.v34_replay_step;copy_fn=native.lib.v34_copy_output','        seed_baselines=[h.raw() for h in seed_holders]\n        output_baselines=[h.raw() for h in output_holders]\n        receipt["captured263_original_owners_readonly_baselines"]={"seed_arrays":131,"captured_outputs":132,"bytes":sum(map(len,seed_baselines))+sum(map(len,output_baselines)),"scope":"once-only backups outside all clocks; actual model owners retained"}\n        step_fn=native.lib.v34_replay_step;copy_fn=native.lib.v34_copy_output',1)
# Reuse full68 actual full-range weight binding assertions without adding native instrumentation.
start=source.index('        fused_pso=[v for v in receipt["capture"]');end=source.index('        assert_icb(receipt["capture"]["native"],[])',start)
binding=source[start:end].replace('receipt["capture"]["native"]','receipt["capture_native"]')
needle='        base.assert_icb(receipt["capture_native"],[])'
s=s.replace(needle,binding+needle,1)
s=s.replace('            report=native.report(f"native-request-{index+1}");f=report["feedback"]','            owners_readonly={"capture_seed":base.originals_readonly(native,seed_holders,seed_baselines,f"request{index+1}.capture_seed"),"captured_reference":base.originals_readonly(native,output_holders,output_baselines,f"request{index+1}.captured_reference")}\n            report=native.report(f"native-request-{index+1}");f=report["feedback"]',1)
s=s.replace('"final_hash_cpu_copy_bytes_outside_timing":sum(h.nbytes for h in output_holders),"native":report,','"final_hash_cpu_copy_bytes_outside_timing":sum(h.nbytes for h in output_holders),"native":report,"captured263_original_owners_readonly":owners_readonly,',1)
s=s.replace('        assert GDN._decode_conv_norm_reuse_cache is candidate_norm\n        for name,mod','        assert GDN._decode_conv_norm_reuse_cache is candidate_norm and GDN._decode_conv_norm_reuse_four_projection is candidate_four\n        assert all(g._four_projection_enabled and g._four_projection_helper is not None for g in gdns)\n        assert initial_parameter_identity==[(name,id(v)) for name,v in tree_flatten(model.parameters())] and len(initial_parameter_identity)==1349\n        assert target_fingerprints()==target_before\n        receipt["target384_original_parameters_readonly_final"]=True\n        receipt["same1349_parameter_owners_final"]=True\n        for name,mod',1)
s=s.replace('PASS_CONV_ACTIVATED_NORMREUSE_CACHE_ICB_S_PINGPONG_B_ONLY_EXPLORATORY_SCREEN','PASS_V73_FOUR_ORIGINAL_POINTER_GDN_NATIVE_B_ONLY_EXPLORATORY_SCREEN')
(D/'screen.py').write_text(s);(D/'screen-from44.diff').write_text(''.join(difflib.unified_diff((A/'screen.py').read_text().splitlines(True),s.splitlines(True),fromfile=str(A/'screen.py'),tofile=str(D/'screen.py'))))
(D/'layout_probe.metal').write_bytes((F/'layout_probe.metal').read_bytes())
