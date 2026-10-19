"""Tiny raw safetensors fixtures only. No model, backend, mmap or process imports."""
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PATH=Path(__file__).with_name('candidate.py')
spec=importlib.util.spec_from_file_location('glm_repack_cpu_candidate',PATH)
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)

class Tests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.root=Path(tmp.name).resolve()

    def fixture(self,experts=3):
        source=self.root/'source';source.mkdir();config=dict(model_type='glm5_next',text_config=dict(num_hidden_layers=1,num_nextn_predict_layers=1,n_routed_experts=experts,hidden_size=64,moe_intermediate_size=64,mlp_layer_types=['sparse']))
        q=dict(group_size=64,bits=4);q['model.layers.0.mlp.switch_mlp.down_proj']=dict(group_size=64,bits=5);q['model.layers.0.mlp.shared_experts.gate_proj']=dict(group_size=64,bits=6);q['model.layers.1.mlp.switch_mlp.down_proj']=dict(group_size=64,bits=5)
        config.update(quantization=q,quantization_config=q);(source/'config.json').write_bytes(c.encoded(config));(source/'tokenizer.json').write_bytes(b'{"fixture":"exact bytes  ","version":1}\n')
        raw={};descriptors={}
        for e in range(experts):
            for projection in ('gate_proj','up_proj','down_proj'):
                for suffix in ('weight','scales','biases'):
                    key=f'model.language_model.layers.0.mlp.experts.{e}.{projection}.{suffix}'
                    shape=[64,10 if projection=='down_proj' else 8] if suffix=='weight' else [64,1]
                    dtype='U32' if suffix=='weight' else 'F16';n=c.math.prod(shape)*c.DTYPES[dtype]
                    raw[key]=bytes([(e*17+len(raw))%256])*n;descriptors[key]=dict(dtype=dtype,shape=shape)
        for key in list(raw):
            mtp=key.replace('.layers.0.','.layers.1.')
            raw[mtp]=raw[key];descriptors[mtp]=dict(descriptors[key])
        extra={
            'model.language_model.layers.1.eh_proj.weight':('U32',[64,16]),
            'model.language_model.layers.1.enorm.weight':('F16',[64]),
            'model.language_model.layers.1.hnorm.weight':('F16',[64]),
            'model.language_model.layers.1.shared_head.norm.weight':('F16',[64]),
            'model.language_model.layers.1.shared_head.head.weight':('F16',[64]),
            'model.language_model.layers.1.embed_tokens.weight':('F16',[64]),
            'model.language_model.layers.1.mlp.experts.0.down_proj.weight':('U32',[64,10]),
            'model.language_model.layers.0.mlp.shared_experts.gate_proj.weight':('U32',[64,12]),
            'model.language_model.norm.weight':('BF16',[64]),'lm_head.weight':('F32',[64,2])}
        for key,(dtype,shape) in extra.items():raw[key]=bytes([173])*c.math.prod(shape)*c.DTYPES[dtype];descriptors[key]=dict(dtype=dtype,shape=shape)
        keys=list(reversed(sorted(raw)));batches=[keys[::2],keys[1::2]];download=[];inventory=[];weight_map={}
        for i,keys in enumerate(batches):
            name=f'model-{i}.safetensors';header={'__metadata__':{'format':'pt'}};payload=b''
            for key in keys:header[key]=dict(**descriptors[key],data_offsets=[len(payload),len(payload)+len(raw[key])]);payload+=raw[key];weight_map[key]=name
            h=c.encoded(header);h+=b' '*((-len(h))%8);prefix=struct.pack('<Q',len(h))+h;p=source/name;p.write_bytes(prefix+payload)
            download.append(dict(name=name,bytes=p.stat().st_size,sha256=c.sha_file(p),sha256_verified=True))
            inventory.append(dict(name=name,file_bytes=p.stat().st_size,header_bytes=len(prefix),header_sha256=c.digest(prefix)))
        (source/'model.safetensors.index.json').write_bytes(c.encoded(dict(metadata={'total_size':sum(len(v) for v in raw.values())},weight_map=weight_map)))
        d=self.root/'download.json';h=self.root/'headers.json';d.write_bytes(c.encoded(dict(status='PASS_PINNED_DOWNLOAD_AND_SHA_ONLY',snapshot=str(source),shards=download)));h.write_bytes(c.encoded(dict(status='PASS_HEADER_INVENTORY_ONLY',snapshot=str(source),shards=inventory)))
        runtime=self.root/'fake-reviewed-loader.py';runtime.write_bytes(b'# explicit CPU source fixture\n')
        args=(source,c.reference(d),c.reference(h),{str(runtime):c.sha_file(runtime)})
        return args,raw,descriptors,config

    def test_default_never_reads_or_executes(self):
        with patch.object(c,'inspect',side_effect=AssertionError('read')),patch.object(c,'execute',side_effect=AssertionError('copy')),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(c.main([]),0)
            with self.assertRaises(ValueError):c.main(['--out',str(self.root/'out')])

    def test_plan_is_metadata_only_and_original_stack_order(self):
        args,raw,descriptors,config=self.fixture(experts=12)
        with patch.object(c,'sha_file',side_effect=lambda p,*a:hashlib.sha256(PATH.read_bytes()).hexdigest() if p==PATH else (_ for _ in ()).throw(AssertionError('payload hashing during inspect'))):
            plan,rows=c.inspect(*args)
        self.assertEqual(plan['summary']['stacked_tensors'],9)
        target=next(r for r in rows if r['key'].endswith('switch_mlp.down_proj.weight'))
        self.assertEqual([int(p['key'].split('.')[6]) for p in target['parts']],list(range(12)))
        self.assertEqual(target['shape'],[12,64,10])

    def test_roundtrip_multiple_shards_and_all_unmodified_tensors_sidecars(self):
        args,raw,descriptors,config=self.fixture();plan,rows=c.inspect(*args);before={p.name:c.sha_file(p) for p in args[0].iterdir()};out=self.root/'output'
        result=c.execute(plan,out,buffer=73,shard_limit=10000,reserve=0)
        self.assertEqual(result['payload_io_policy'],c.payload_io_policy())
        self.assertEqual(result['status'],'PASS_BYTES_ONLY');self.assertFalse(result['runtime_load_executed']);self.assertGreater(len(result['files']),1)
        index=json.loads((out/'model.safetensors.index.json').read_bytes())['weight_map'];self.assertEqual(set(index),{r['key'] for r in rows})
        for row in rows:
            name=index[row['key']];head,meta=c.header(out/name);a,b=head[row['key']]['data_offsets']
            with (out/name).open('rb') as f:f.seek(meta['header_bytes']+a);actual=f.read(b-a)
            self.assertEqual(actual,b''.join(raw[p['key']] for p in row['parts']))
            self.assertEqual(head[row['key']]['shape'],row['shape']);self.assertEqual(head[row['key']]['dtype'],row['dtype'])
        for name in ('config.json','tokenizer.json'):self.assertEqual((out/name).read_bytes(),(args[0]/name).read_bytes())
        self.assertEqual(before,{p.name:c.sha_file(p) for p in args[0].iterdir()})
        self.assertIn('model.language_model.layers.1.mlp.experts.0.down_proj.weight',index)

    def test_exact_original_dsv_sanitize_ast_on_fake_bytes_agrees(self):
        args,raw,descriptors,config=self.fixture();plan,rows=c.inspect(*args)
        path=c.SITE/'mlx_lm/models/deepseek_v32.py';self.assertEqual(c.digest(path.read_bytes()),c.REVIEWED['mlx_lm/models/deepseek_v32.py'])
        tree=ast.parse(path.read_text());cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='Model');fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='sanitize')
        class Array:
            def __init__(self,dtype,shape,data):self.dtype,self.shape,self.data=dtype,shape,data
        def stack(values):
            self.assertTrue(all(v.dtype==values[0].dtype and v.shape==values[0].shape for v in values))
            return Array(values[0].dtype,[len(values)]+values[0].shape,b''.join(v.data for v in values))
        env={'mx':SimpleNamespace(stack=stack)};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),env)
        def remap(key):
            return 'model.'+key[len('model.language_model.'):] if key.startswith('model.language_model.') else key
        before={remap(k):Array(descriptors[k]['dtype'],descriptors[k]['shape'],v) for k,v in raw.items()}
        after={remap(r['key']):Array(r['dtype'],r['shape'],b''.join(raw[p['key']] for p in r['parts'])) for r in rows}
        obj=SimpleNamespace(args=SimpleNamespace(num_hidden_layers=1,n_routed_experts=3))
        a=env['sanitize'](obj,before);b=env['sanitize'](obj,after)
        self.assertEqual(set(a),set(b))
        for k in a:self.assertEqual((a[k].dtype,a[k].shape,a[k].data),(b[k].dtype,b[k].shape,b[k].data))

    def pinned_tree(self,relative):
        path=c.SITE/relative
        self.assertEqual(c.digest(path.read_bytes()),c.REVIEWED[relative])
        return path,ast.parse(path.read_text())

    def test_actual_mtp_wrapper_container_language_stack_path_equivalence(self):
        # Extract only reviewed definitions: never import or instantiate an MLX model.
        args,raw,descriptors,config=self.fixture();plan,rows=c.inspect(*args)
        calls=[]
        class Array:
            def __init__(self,dtype,shape,data):self.dtype,self.shape,self.data=dtype,shape,data;self.ndim=len(shape)
        def stack(values):
            self.assertTrue(all(v.dtype==values[0].dtype and v.shape==values[0].shape for v in values))
            calls.append((values[0].dtype,values[0].shape))
            return Array(values[0].dtype,[len(values)]+values[0].shape,b''.join(v.data for v in values))
        mx=SimpleNamespace(stack=stack,float32='F32',floating='floating',issubdtype=lambda d,k:d in ('F16','BF16','F32'))
        def function(relative,owner,name,env):
            path,tree=self.pinned_tree(relative)
            scope=next(x for x in tree.body if isinstance(x,(ast.ClassDef,ast.FunctionDef)) and x.name==owner) if owner else tree
            fn=next(x for x in scope.body if isinstance(x,ast.FunctionDef) and x.name==name)
            exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),env)
            return env[name]
        dsv=function('mlx_lm/models/deepseek_v32.py','Model','sanitize',{'mx':mx})
        langrel='omlx/patches/mlx_vlm_glm5_next_compat/vendor/mlx_vlm/models/glm5_next/language.py'
        langenv={'mx':mx,'DSV32Model':SimpleNamespace(sanitize=dsv)}
        function(langrel,None,'_dequantize_router_gates',langenv)
        original=function(langrel,'LanguageModel','sanitize',langenv)
        mtprel='omlx/patches/mlx_vlm_mtp/glm5_next_vlm_runtime.py';path,tree=self.pinned_tree(mtprel)
        constants={'_NEXTN_PREFIXES','_MTP_FUSION_KEYS','_MTP_DROP_PREFIXES'}
        helpers={'_is_mtp_named','_match_nextn'}
        nodes=[n for n in tree.body if (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in constants for t in n.targets)) or (isinstance(n,ast.FunctionDef) and n.name in helpers)]
        env={'mx':mx,'original_sanitize':original,'logger':SimpleNamespace(info=lambda *a:None)}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
        patched=function(mtprel,'_patch_language_model','sanitize',env)
        lang=SimpleNamespace(args=SimpleNamespace(num_hidden_layers=1,num_nextn_predict_layers=1,n_routed_experts=3,hidden_size=64))
        lang.sanitize=lambda weights:patched(lang,weights)
        outer=SimpleNamespace(language_model=lang,vision_model=SimpleNamespace(sanitize=lambda weights:weights))
        container=function('omlx/patches/mlx_vlm_glm5_next_compat/vendor/mlx_vlm/models/glm5_next/glm5_next.py','Model','sanitize',{})
        before={k:Array(descriptors[k]['dtype'],descriptors[k]['shape'],v) for k,v in raw.items()}
        after={r['key']:Array(r['dtype'],r['shape'],b''.join(raw[p['key']] for p in r['parts'])) for r in rows}
        a=container(outer,before);original_stacks=len(calls);calls.clear();b=container(outer,after)
        self.assertEqual(original_stacks,18);self.assertEqual(len(calls),9)  # MTP is still stacked.
        self.assertEqual(set(a),set(b))
        for k in a:self.assertEqual((a[k].dtype,a[k].shape,a[k].data),(b[k].dtype,b[k].shape,b[k].data))
        self.assertEqual(len([k for k in b if k.startswith('language_model.mtp.0.block.mlp.switch_mlp.')]),9)
        for suffix in ('eh_proj.weight','enorm.weight','hnorm.weight','norm.weight'):
            self.assertIn('language_model.mtp.0.'+suffix,b)
        self.assertFalse(any('.layers.1.' in k or '.shared_head.head.' in k for k in b))

    def test_actual_nextn_quantization_alias_path_preserves_mixed_bits(self):
        args,raw,descriptors,config=self.fixture()
        path,tree=self.pinned_tree('omlx/utils/model_loading.py')
        constants={'_VLM_TEXT_PREFIX','_CKPT_TEXT_PREFIX','_RUNTIME_TEXT_PREFIX'}
        functions={'_add_glm5_next_nextn_quant_keys','_add_mla_split_quant_keys','expand_per_layer_quant_keys'}
        nodes=[n for n in tree.body if (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in constants for t in n.targets)) or (isinstance(n,ast.FunctionDef) and n.name in functions)]
        env={};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
        original=c.encoded(config);a=env['expand_per_layer_quant_keys'](json.loads(original));b=env['expand_per_layer_quant_keys'](json.loads(original))
        self.assertEqual(a,b);self.assertEqual(c.encoded(config),original)
        for key in ('quantization','quantization_config'):
            q=a[key];self.assertEqual(q['bits'],4)
            self.assertEqual(q['language_model.model.layers.0.mlp.switch_mlp.down_proj'],dict(group_size=64,bits=5))
            self.assertEqual(q['language_model.mtp.0.block.mlp.switch_mlp.down_proj'],dict(group_size=64,bits=5))
            self.assertEqual(q['language_model.model.layers.0.mlp.shared_experts.gate_proj'],dict(group_size=64,bits=6))

    def test_real_payload_open_unaligned_offsets_short_tail_and_full_sha(self):
        # On Darwin this calls the real F_NOCACHE, including an unaligned seek/read.
        path=self.root/'unaligned.bin';raw=bytes(range(251))*43+b'last-seven'
        with c.payload_open(path,'xb') as f:f.write(raw[:3]);f.write(raw[3:]);f.flush();c.os.fsync(f.fileno())
        with c.payload_open(path,'rb') as f:
            f.seek(3);parts=[]
            for chunk in iter(lambda:f.read(73),b''):parts.append(chunk)
        self.assertEqual(b''.join(parts),raw[3:])
        self.assertEqual(c.sha_file(path,73,payload=True),hashlib.sha256(raw).hexdigest())

    def test_payload_nocache_before_first_io_and_failure_closes_descriptor(self):
        path=self.root/'source';path.write_bytes(b'abc');events=[]
        class File:
            closed=False
            def __enter__(self):return self
            def __exit__(self,*args):self.closed=True
            def fileno(self):return 123
            def read(self):events.append('read');return b'abc'
        file=File()
        def control(fd,command,value):
            events.append('fcntl');self.assertEqual((fd,command,value),(123,48,1));return 0
        with patch.object(c.sys,'platform','darwin'),patch.object(Path,'open',return_value=file),patch.object(c.fcntl,'fcntl',side_effect=control):
            with c.payload_open(path,'rb') as f:self.assertEqual(f.read(),b'abc')
        self.assertEqual(events,['fcntl','read']);self.assertTrue(file.closed)
        file=File()
        with patch.object(c.sys,'platform','darwin'),patch.object(Path,'open',return_value=file),patch.object(c.fcntl,'fcntl',side_effect=OSError('unsupported fd')):
            with self.assertRaisesRegex(OSError,'unsupported fd'):
                with c.payload_open(path,'rb'):self.fail('must not yield after failure')
        self.assertTrue(file.closed)
        with patch.object(c.sys,'platform','linux'),patch.object(c.fcntl,'fcntl',side_effect=AssertionError('Darwin-only')):
            self.assertEqual(c.payload_io_policy()['name'],'not_applicable_linux_cpu_fixture')
            with c.payload_open(path,'rb') as f:self.assertEqual(f.read(),b'abc')

    def test_missing_expert_and_mismatched_dtype_reject(self):
        args,raw,desc,config=self.fixture();plan,rows=c.inspect(*args);tensors={p['key']:{k:v for k,v in p.items() if k!='key'} for r in rows for p in r['parts']}
        key=next(k for k in tensors if '.experts.1.' in k)
        missing=dict(tensors);missing.pop(key)
        with self.assertRaisesRegex(ValueError,'missing/out-of-range'):c.outputs_for(missing,config)
        wrong=dict(tensors);wrong[key]={**wrong[key],'dtype':'BF16'}
        with self.assertRaisesRegex(ValueError,'mixed expert'):c.outputs_for(wrong,config)

    def test_wrong_mixed_quantization_rejects(self):
        args,raw,desc,config=self.fixture();plan,rows=c.inspect(*args);tensors={p['key']:{k:v for k,v in p.items() if k!='key'} for r in rows for p in r['parts']}
        config['quantization']['model.layers.0.mlp.switch_mlp.down_proj']['bits']=4
        with self.assertRaisesRegex(ValueError,'mixed expert quantization'):c.outputs_for(tensors,config)

    def test_stale_sidecar_or_repacker_plan_rejects_before_output(self):
        args,*_=self.fixture();plan,_=c.inspect(*args);(args[0]/'tokenizer.json').write_bytes(b'changed');out=self.root/'out'
        with self.assertRaisesRegex(ValueError,'unchanged plan'):c.execute(plan,out,reserve=0)
        self.assertFalse(out.exists())

    def test_full_payload_hash_catches_header_preserving_corruption(self):
        args,*_=self.fixture();path=args[0]/'model-0.safetensors'
        with path.open('r+b') as f:f.seek(-1,2);old=f.read(1);f.seek(-1,2);f.write(bytes([old[0]^1]))
        plan,_=c.inspect(*args);out=self.root/'out'
        with self.assertRaisesRegex(ValueError,'original full payload SHA'):c.execute(plan,out,reserve=0)
        self.assertFalse(out.exists())

    def test_existing_output_and_disk_reserve_reject_before_copy(self):
        args,*_=self.fixture();plan,_=c.inspect(*args);out=self.root/'out';out.mkdir()
        with self.assertRaisesRegex(ValueError,'fresh'):c.execute(plan,out,reserve=0)
        with patch.object(c.os,'statvfs',return_value=SimpleNamespace(f_bavail=1,f_frsize=1)),self.assertRaisesRegex(ValueError,'insufficient disk'):
            c.execute(plan,self.root/'new',reserve=0)
        self.assertFalse((self.root/'new').exists())

    def test_failure_keeps_partial_files_and_never_finalizes_index(self):
        args,*_=self.fixture();plan,_=c.inspect(*args);out=self.root/'out'
        with patch.object(Path,'rename',side_effect=OSError('CPU injected rename failure')),self.assertRaisesRegex(OSError,'CPU injected'):
            c.execute(plan,out,buffer=53,shard_limit=10000,reserve=0)
        self.assertTrue((out/'failure.json').is_file());self.assertTrue(list(out.glob('*.partial')))
        self.assertFalse((out/'model.safetensors.index.json').exists());self.assertFalse((out/'complete.json').exists())

    def test_invalid_header_overlap_and_duplicate_key_reject(self):
        path=self.root/'bad.safetensors';h={'a':dict(dtype='U32',shape=[1],data_offsets=[0,4]),'b':dict(dtype='U32',shape=[1],data_offsets=[0,4])};raw=c.encoded(h);path.write_bytes(struct.pack('<Q',len(raw))+raw+b'1234')
        with self.assertRaisesRegex(ValueError,'gap/overlap'):c.header(path)
        with self.assertRaisesRegex(ValueError,'duplicate JSON'):c.loads(b'{"a":1,"a":2}')

if __name__=='__main__':unittest.main()
