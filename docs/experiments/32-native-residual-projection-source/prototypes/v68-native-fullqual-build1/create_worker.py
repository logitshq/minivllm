from pathlib import Path
import difflib
P=Path('/private/tmp/minivllm-engine-optimization-20261014');D=Path(__file__).resolve().parent;A=P/'v44-native-fullqual-build1';s=(A/'fullqual.py').read_text()
s=s.replace('Root-only conv-SiLU to already-activated normreuse+held32 ICB versus original accepted25 GDN numerical qualification','Root-only V68 original-pointer four-GDN-projection+convnorm ICB versus accepted25 full numeric qualification')
s=s.replace('ROOT = P / "candidate-v44-conv-normreuse-build1-root"','ROOT = P / "candidate-v68-four-projection-build1-root"')
s=s.replace('CAP = 256','ACTUAL44 = P / "v23-root-control-build1/fullmodel-v44native-attempt1/child-output/receipt.json"\nACTUAL44_SHA = "1dcfc41e7779c575602bd6c73bae4e4800664db42e1ee87e75414335bdc45b6d"\nNUM66 = P / "v23-root-control-build1/numeric-v66b2-attempt1/child-output/receipt.json"\nNUM66_SHA = "af5d31fa9aeff992c36df169ebdd1d14a7ea2b04102c91ae95fade50582bf305"\nCAP = 256',1)
s=s.replace('    actual = {str(f.relative_to(package)): sha(f) for f in package.rglob("*.py")}\n    assert actual == expected and len(actual) == 27','    actual = {str(f.relative_to(package)): sha(f) for f in package.rglob("*") if f.is_file()}\n    assert actual == expected and sum(k.endswith(".py") for k in actual) == 28\n    assert sha(ACTUAL44)==ACTUAL44_SHA and sha(NUM66)==NUM66_SHA\n    n66=json.loads(NUM66.read_text())\n    assert n66["status"]=="PASS_V66_FOUR_ORIGINAL_POINTER_GDN_PROJECTION_LOCAL_STRICT" and n66["strict_output_checks"]==396 and n66["readonly_all35"]\n    assert json.loads(ACTUAL44.read_text())["status"]=="PASS_NATIVE_CONV_ACTIVATED_NORMREUSE_CACHE_ICB_BASE_RESET_CHANGED_VS_ACCEPTED25_STRICT"',1)
s=s.replace('    if args.inert:', '    actual44_cases={r["name"]:r for r in json.loads(ACTUAL44.read_text())["requests"]}\n    assert all(len(r["steps"])==31 for r in actual44_cases.values())\n    if args.inert:',1)
s=s.replace('"A_saved_baseline_receipt":str(BASELINE_QUAL),"A_saved_baseline_receipt_sha256":BASELINE_QUAL_SHA}', '"A_saved_baseline_receipt":str(BASELINE_QUAL),"A_saved_baseline_receipt_sha256":BASELINE_QUAL_SHA,\n               "actual44_gold_sha256":ACTUAL44_SHA,"local66_numeric_sha256":NUM66_SHA,\n               "four_projection_scope":"qualified original input-layernorm normalB1 only; defaultOFF; no per-token layout probe; possible loss of qkv/conv overlap",\n               "declared_fused_kernel_buffer_arguments":13,"actual_individual_argument_reflection":"NOT_OBSERVED: native40 report has no per-argument table; frozen source ABI9inputs+4outputs, no implicit descriptor inputs"}',1)
s=s.replace('        candidate_decode,candidate_norm=GDN._decode_dispatch,GDN._decode_conv_norm_reuse_cache','        candidate_decode,candidate_norm=GDN._decode_dispatch,GDN._decode_conv_norm_reuse_cache\n        candidate_four=GDN._decode_conv_norm_reuse_four_projection\n        assert all(g._four_projection_enabled is False and g._four_projection_helper is None for g in gdns)',1)
s=s.replace('            GDN._decode_dispatch=legacy_decode;GDN._decode_conv_norm_reuse_cache=legacy_decode\n            baseline_calls.append("oldaccepted25_decode_AST")', '            flags=[g._four_projection_enabled for g in gdns]\n            for g in gdns:g.set_four_projection(False)\n            assert all(g._four_projection_enabled is False for g in gdns)\n            GDN._decode_dispatch=legacy_decode;GDN._decode_conv_norm_reuse_cache=legacy_decode\n            baseline_calls.append({"method":"oldaccepted25_decode_AST","four_projection_OFF":48,"legacy_dispatch_observed":GDN._decode_dispatch is legacy_decode})',1)
s=s.replace('                GDN._decode_dispatch=candidate_decode;GDN._decode_conv_norm_reuse_cache=candidate_norm\n                assert GDN._decode_dispatch is candidate_decode and GDN._decode_conv_norm_reuse_cache is candidate_norm', '                GDN._decode_dispatch=candidate_decode;GDN._decode_conv_norm_reuse_cache=candidate_norm\n                for g,flag in zip(gdns,flags):g.set_four_projection(flag)\n                assert GDN._decode_dispatch is candidate_decode and GDN._decode_conv_norm_reuse_cache is candidate_norm',1)
start=s.index('        norm_capture_calls=[]');end=s.index('        receipt["A_baseline_source"]',start)
s=s[:start]+'''        four_capture_calls=[];projection_capture_calls=[]
        @contextmanager
        def capture_norm_engagement():
            from tutorial_code.miniengine.kernels.v66_four_projection.four_projection import FourProjection
            original_project=FourProjection.project
            assert GDN._decode_conv_norm_reuse_four_projection is candidate_four
            def counted(self,*a,**kw):
                four_capture_calls.append(id(self))
                return candidate_four(self,*a,**kw)
            def project_counted(self,x,**kw):
                out=original_project(self,x,**kw)
                assert len(out)==4 and len({id(v) for v in out})==4
                assert [v.shape for v in out]==[(1,n) for n in (10240,6144,48,48)] and all(v.dtype==mx.bfloat16 for v in out)
                projection_capture_calls.append({"helper_id":id(self),"x_id":id(x),"outputs":4,"output_shapes":[list(v.shape) for v in out],"output_dtypes":[str(v.dtype) for v in out]})
                return out
            GDN._decode_conv_norm_reuse_four_projection=counted;FourProjection.project=project_counted
            try:yield
            finally:
                GDN._decode_conv_norm_reuse_four_projection=candidate_four;FourProjection.project=original_project
        target_holders=[];target_before=[]
        def target_fingerprints():
            return [{**h.meta,"sha256":hashlib.sha256(h.bytes_view).hexdigest()} for h in target_holders]
''' +s[end:]
s=s.replace('        capture_seed, first = setup(base_prompt); assert first == w["continuation_ids"][0]', '''        # Additional single ordinary decode solely qualifies actual norm-output layout.
        # It is outside capture, the 93 numerical steps, and any future timing.
        layout_seed,layout_first=setup(base_prompt);layout_x={}
        with baseline_decode():
            def retain_layout_x(self,x,*a,**kw):
                assert id(self) not in layout_x
                layout_x[id(self)]=x
                return legacy_decode(self,x,*a,**kw)
            GDN._decode_dispatch=retain_layout_x
            try:
                layout_ctx=ForwardContext("decode",ctx_layers(layout_seed[3:]),positions=layout_seed[1],lens=layout_seed[2],max_len=len(base_prompt)+1)
                layout_logits=model(layout_seed[0],layout_ctx)
                mx.eval(layout_logits,*state_of(layout_ctx),*layout_x.values());mx.synchronize()
                assert int(mx.argmax(layout_logits,axis=-1).item())==w["continuation_ids"][1]
            finally:GDN._decode_dispatch=legacy_decode
        assert set(layout_x)==set(map(id,gdns)) and len(layout_x)==48
        from tutorial_code.miniengine.model.layers import RMSNorm
        assert all(type(layer.input_layernorm) is RMSNorm for layer in model.layers)
        layout_kernel=mx.fast.metal_kernel(name="v68_prepare_observed_live_strides",input_names=["value"],output_names=["descriptor"],source=(D/"layout_probe.metal").read_text(),ensure_row_contiguous=False)
        layout_rows=[]
        def observed_layout(value,label):
            descriptor,=layout_kernel(inputs=[value],grid=(1,1,1),threadgroup=(1,1,1),output_shapes=[(4,)],output_dtypes=[mx.uint64])
            mx.eval(descriptor);v=descriptor.tolist();assert v[:2]==list(value.shape)
            expected=[value.shape[1],1];contiguous=v[3]==1 and (value.shape[0]==1 or v[2]==expected[0])
            row={"label":label,"shape":v[:2],"dtype":str(value.dtype),"actual_strides_elements":v[2:],"row_contiguous_derived_singleton_aware":contiguous,"internal_MLX_flags_observed":False,"ensure_row_contiguous":False}
            layout_rows.append(row);assert contiguous,("layout STOP",row);return tuple(v[2:])
        prepared=[]
        for layer_index,layer in enumerate(model.layers):
            if layer.is_attn:continue
            g=layer.linear_attn;operand_strides=[]
            for projection,op in zip(("qkv","z","b","a"),g._four_projection_ops()):
                for suffix,value in (("weight",op.weight),("scales",op.scales)):
                    label=f"parameter.layers.{layer_index}.linear_attn.in_proj_{projection}.{suffix}"
                    holder=shared(value,label);target_holders.append(holder)
                    target_before.append({**holder.meta,"sha256":hashlib.sha256(holder.bytes_view).hexdigest()})
                    operand_strides.append(observed_layout(value,label))
            x=layout_x[id(g)];assert x.shape==(1,5120) and x.dtype==mx.bfloat16
            x_strides=observed_layout(x,f"layer.{layer_index}.actual_input_layernorm_X")
            xholder=shared(x,f"layer.{layer_index}.layout_X");x_before=hashlib.sha256(xholder.bytes_view).hexdigest()
            active=mx.get_active_memory()
            g.prepare_four_projection(observed_weight_scale_strides=operand_strides,observed_x_strides=x_strides,runtime_version=versions["mlx"],device_name=info["device_name"],architecture=info["architecture"],input_layout_contract="qualified-original-input-layernorm-B1")
            assert mx.get_active_memory()==active and g._four_projection_enabled is False
            helper=g._four_projection_helper;ops=tuple(v for op in g._four_projection_ops() for v in (op.weight,op.scales))
            assert all(a is b for a,b in zip(helper.operands,ops)) and hashlib.sha256(xholder.bytes_view).hexdigest()==x_before
            prepared.append({"layer":layer_index,"helper_id":id(helper),"original_operand_ids":[id(v) for v in ops],"x_strides":list(x_strides),"observed_weight_scale_strides":[list(v) for v in operand_strides],"mlx_active_before_prepare":active,"mlx_active_after_prepare":mx.get_active_memory(),"same_original_handles":True,"X_readonly":True})
        assert len(prepared)==48 and len(target_holders)==384 and len(layout_rows)==432
        assert sum(h.nbytes for h in target_holders)==2278195200 and target_fingerprints()==target_before
        assert initial_parameter_identity==[(name,id(value)) for name,value in tree_flatten(model.parameters())] and len(initial_parameter_identity)==1349
        for g in gdns:g.set_four_projection(True)
        receipt["four_projection_preparation"]={"extra_ordinary_decode_steps":1,"included_in_93_steps":False,"purpose":"independent actual48 original norm-output strides; no performance timing","layout_rows":layout_rows,"prepared":prepared,"original384_target_arrays":target_before,"readonly_after_prepare":True,"target_payload_bytes":2278195200,"extra_packed_weight_bytes":0,"parameter_owners":1349,"same_parameter_identity_after_prepare":True,"source_contract":"unchanged original RMSNorm→norm_metal output_shapes=[x.shape],linear row*D stores; restricted engine path, arbitrary externalX not qualified","target_parameter_holders_retained":True}
        save();del layout_seed,layout_ctx,layout_logits,layout_x,x,xholder
        mx.synchronize();resources.sample("after-four-projection-layout-prepare",loaded=True,mx=mx)
        capture_seed, first = setup(base_prompt); assert first == w["continuation_ids"][0]''',1)
start=s.index('        assert len(norm_capture_calls)==48');end=s.index('        for value,label in zip(captured,labels_out):',start)
s=s[:start]+'''        assert len(four_capture_calls)==48 and set(four_capture_calls)==set(map(id,gdns))
        assert len(projection_capture_calls)==48 and {r["helper_id"] for r in projection_capture_calls}=={id(g._four_projection_helper) for g in gdns}
        from tutorial_code.miniengine.kernels.v66_four_projection.four_projection import FourProjection
        receipt["B_four_projection_capture_engagement"]={"calls":48,"distinct_GDN_objects":48,"all48_enabled":all(g._four_projection_enabled for g in gdns),"projection_calls":projection_capture_calls,"four_output_consumption":"unchanged conv uses qkv; original output norm uses z; gates use b/a; whole132 numerical comparison","observers_removed_before_replay":GDN._decode_conv_norm_reuse_four_projection is candidate_four and FourProjection.project.__name__=="project"}
        assert receipt["B_four_projection_capture_engagement"]["observers_removed_before_replay"]
''' +s[end:]
needle='''        assert receipt["capture"]["native"]["status"] == "OK"
'''
replacement=needle+'''        fused_pso=[v for v in receipt["capture"]["native"]["feedback"]["icb"]["PSO_records"] if "v66_gdn_four_original_pointers_r4" in v["function"]]
        assert len(fused_pso)==1 and fused_pso[0]["actual_supports_ICB"]
        buffer_rows=receipt["capture"]["native"]["buffers"];binding_proof=[]
        parameter_by_name={v.label:v for v in parameter_holders}
        for holder in target_holders:
            owner=parameter_by_name[holder.label];assert owner.original is holder.original and owner.address==holder.address and owner.nbytes==holder.nbytes
            matches=[v for v in buffer_rows if v["used"] and v["zero_copy"] and v["original_contents"]<=holder.address and holder.address+holder.nbytes<=v["original_contents"]+v["length"] and v["role"].startswith(holder.label+f" full-range={holder.address}+{holder.nbytes} binding-delta=0 readonly:")]
            assert len(matches)==1 and "v66_gdn_four_original_pointers_r4" in matches[0]["role"],("unproven fused original weight range",holder.label,matches)
            binding_proof.append({"parameter":holder.label,"original_object_id":id(holder.original),"address":holder.address,"bytes":holder.nbytes,"actual_original_buffer":matches[0]["original_buffer"],"role":matches[0]["role"],"full_original_parameter_range_readonly":True})
        assert len(binding_proof)==384
        receipt["four_projection_native_binding_proof"]={"fused_PSO":fused_pso[0],"Python_capture_calls":48,"original_target_ranges":binding_proof,"buffer_argument_count_source_proof":13,"individual_argument_reflection":"NOT_OBSERVED; native40 report omits per-argument table; source9inputs+4outputs,noimplicitdescriptorbuffers"}
'''
assert needle in s;s=s.replace(needle,replacement,1)
s=s.replace('                    native.call("v34_compare_output",original.address,original.nbytes,ctypes.c_void_p(reference.address))','                    gold44=actual44_cases[name]["steps"][step-1]["outputs"][len(equal)]\n                    assert gold44["label"]==reference.label and gold44["shape"]==sig["shape"] and gold44["dtype"]==sig["dtype"] and gold44["bytes"]==sig["nbytes"] and gold44["A_sha256"]==sig["sha256"],("A differs from actual44",name,step,reference.label)\n                    native.call("v34_compare_output",original.address,original.nbytes,ctypes.c_void_p(reference.address))',1)
s=s.replace('        source_check(library)\n        receipt.update(status="PASS_NATIVE_CONV_ACTIVATED_NORMREUSE_CACHE_ICB_BASE_RESET_CHANGED_VS_ACCEPTED25_STRICT"','        assert total_steps==93 and all(len(r["steps"])==31 and r["A_ids"]==actual44_cases[r["name"]]["A_ids"] for r in receipt["requests"])\n        assert all(g._four_projection_enabled and g._four_projection_helper is not None for g in gdns)\n        assert target_fingerprints()==target_before\n        receipt["four_projection_target384_readonly_final"]=True\n        receipt["A_four_OFF_legacy_dispatch_observations"]=baseline_calls\n        source_check(library)\n        receipt.update(status="PASS_V68_FOUR_ORIGINAL_POINTER_GDN_FULL_ICB_BASE_RESET_CHANGED_STRICT"',1)
s=s.replace('source_pins_verified_final=True,gates_callable_count=48)','source_pins_verified_final=True,gates_callable_count=48,four_projection_ON=48,strict_output_checks=93*132)',1)
(D/'fullqual.py').write_text(s);(D/'worker-from44.diff').write_text(''.join(difflib.unified_diff((A/'fullqual.py').read_text().splitlines(True),s.splitlines(True),fromfile=str(A/'fullqual.py'),tofile=str(D/'fullqual.py'))))
