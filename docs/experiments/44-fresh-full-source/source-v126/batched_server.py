"""V126 opt-in ordinary real-batch worker: original native15 work/Handler core.

No native replay library/adapter, no CAP64, no batching wait or serial-generate
shim. This new HTTP wrapper is CPU prework only, qualification pending.
"""
import argparse,json,queue,threading,time,sys,traceback,hashlib,inspect,builtins,ast,textwrap,os
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from pathlib import Path
from tokenizers import Tokenizer
P=Path('/private/tmp/minivllm-engine-optimization-20261014')
OLD=P/'m4-final-native15-official-attempt2'
sys.path.insert(0,str(OLD))
from common import actual_preimport_identity,write_new
ap=argparse.ArgumentParser()
ap.add_argument('--model',required=True);ap.add_argument('--port',type=int,required=True)
ap.add_argument('--batch',type=int,required=True);ap.add_argument('--trace',type=Path,required=True)
ap.add_argument('--package-root',type=Path,required=True);ap.add_argument('--source-manifest',type=Path,required=True)
ap.add_argument('--gates',choices=['on'],required=True);ap.add_argument('--run-dir',type=Path,required=True)
ap.add_argument('--owner-pid',type=int,required=True);ap.add_argument('--client-batch',type=int,choices=[2,4],required=True)
a=ap.parse_args()
assert os.getppid()==a.owner_pid and a.batch==4 and a.gates=='on'
assert not any(k.startswith(('DYLD_','METAL_')) for k in os.environ),'ordinary batched worker must not load native replay interposer'
assert a.package_root.resolve()==P/'candidate-v17-gdn-gates-build1-root'
assert a.source_manifest.resolve()==P/'source-manifest-v17-gdn-gates-build1.json'
assert a.model==json.loads((P/'workload.json').read_text())['model_dir']
assert {k:os.environ.get(k) for k in ('MLX_MAX_OPS_PER_BUFFER','MLX_MAX_MB_PER_BUFFER','MLX_BFS_MAX_WIDTH','MLX_METAL_FAST_SYNCH')}=={'MLX_MAX_OPS_PER_BUFFER':'50','MLX_MAX_MB_PER_BUFFER':'50','MLX_BFS_MAX_WIDTH':'20','MLX_METAL_FAST_SYNCH':'0'}
assert not any(n=='mlx' or n.startswith('mlx.') for n in sys.modules)
preimport_identity=actual_preimport_identity()
sys.path.insert(0,str(a.package_root))
pending=queue.Queue();ready=threading.Event();fail=[];live={};trace=[];runtime_identity={}
tok=Tokenizer.from_file(str(Path(a.model)/'tokenizer.json'))

def work():
 try:
  from runtime_fence import install_runtime
  verify_runtime=install_runtime()
  import mlx.core as mx
  import mlx.nn as nn
  import mlx.nn.layers.activations as activations
  actual_runtime=verify_runtime(mx,nn,activations)
  from tutorial_code.miniengine.engine.llm_engine import LLMEngine
  from tutorial_code.miniengine.sampling_params import SamplingParams
  import tutorial_code.miniengine as actual_package
  actual_root=Path(actual_package.__file__).resolve().parent
  assert actual_root==Path(preimport_identity['source']['root']),'actual package resolves outside frozen candidate'
  expected=json.loads(a.source_manifest.read_text())
  actual={str(f.relative_to(actual_root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in actual_root.rglob('*.py')}
  assert actual==expected['files'] and len(actual)==25,'V17 source identity mismatch'
  runtime_identity.update({'package_file':str(Path(actual_package.__file__).resolve()),'llm_engine_file':str(Path(inspect.getfile(LLMEngine)).resolve()),'actual_source_files_sha256':actual,'manifest_sha256':hashlib.sha256(a.source_manifest.read_bytes()).hexdigest(),'manifest_all_sources_match':True,'server_max_seqs':a.batch})
  e=LLMEngine(a.model,max_seqs=a.batch,max_len=2048,max_batched_tokens=512,dtype=mx.bfloat16)
  from tutorial_code.miniengine.cache import AttnCache, DecodeBatch
  assert (AttnCache.step,DecodeBatch.step)==(256,256)
  assert e.scheduler.max_seqs==4 and e.runner.spec is None
  runtime_identity.update(static_profile='legacy_batched256',dynamic_native_batch_migration=False,
    native_adapter_installed=False,physical_capacity_steps=[256,256],allocator_cache_limit_bytes=2<<30,
    required_client_concurrency=a.client_batch,qualification_pending=True)
  from tutorial_code.miniengine.model.gdn import GatedDeltaNet
  from mlx.utils import tree_flatten
  def parameter_metadata():
   rows=[]
   for key,value in tree_flatten(e.model.parameters()):
    assert isinstance(value,mx.array),'non-array parameter introduced: '+str(key)
    rows.append({'key':key,'shape':list(value.shape),'dtype':str(value.dtype),'python_array_identity':builtins.id(value)})
   return rows
  gdns=[layer.linear_attn for layer in e.model.layers if not layer.is_attn]
  assert len(gdns)==48 and all(type(g) is GatedDeltaNet for g in gdns)
  before_parameters=parameter_metadata();assert len(before_parameters)==1349
  assert all(object.__getattribute__(g,'_gates_compile_enabled') is False and object.__getattribute__(g,'_compiled_decode_gates') is None for g in gdns),'constructor default-off missing'
  setter_calls=0
  for g in gdns:g.set_gates_compile(True);setter_calls+=1
  assert setter_calls==48
  assert before_parameters==parameter_metadata(),'setter changed parameter tree'
  def gates_metadata():
   flags=[object.__getattribute__(g,'_gates_compile_enabled') for g in gdns]
   functions=[object.__getattribute__(g,'_compiled_decode_gates') for g in gdns]
   assert len(flags)==48 and all(type(v) is bool and v==(a.gates=='on') for v in flags)
   return {'enabled':a.gates=='on','count':len(flags),'flags':flags,'compiled_callable_count':sum(callable(fn) for fn in functions),'all_other_callable_values_none':all(fn is None or callable(fn) for fn in functions)}
  originals={name:getattr(GatedDeltaNet,name) for name in ('_decode','_decode_gates')}
  from tutorial_code.miniengine.cache import BatchState
  fn=ast.parse(textwrap.dedent(inspect.getsource(originals['_decode_gates']))).body[0]
  boundary=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.If) and isinstance(n.test,ast.UnaryOp) and isinstance(n.test.operand,ast.Name) and n.test.operand.id=='eligible')
  fn.name='actual_eligible';fn.body=fn.body[:boundary]+[ast.Return(value=ast.Name(id='eligible',ctx=ast.Load()))]
  ns=dict(originals['_decode_gates'].__globals__);exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'frozen-actual-warm-eligibility','exec'),ns)
  actual_eligible=ns['actual_eligible']
  route={'normal_decode_calls':0,'gates_route_calls':0,'compiled_eligible_calls':0,'fallback_calls':0,'decode_batch_counts':{},'setter_calls':48}
  def decode_hook(self,x,st,**kw):
   B=x.shape[0];assert 1<=B<=a.client_batch and x.shape==(B,5120) and x.dtype==mx.bfloat16 and type(st) is BatchState and kw.get('_compile_gates') is True
   assert st.conv.shape==(B,3,10240) and st.S.shape==(B,48,128,128)
   route['normal_decode_calls']+=1;route['decode_batch_counts'][str(B)]=route['decode_batch_counts'].get(str(B),0)+1
   return originals['_decode'](self,x,st,**kw)
  def gates_hook(self,b,aa,x,st):
   B=x.shape[0];assert b.shape==aa.shape==(B,48) and b.dtype==aa.dtype==mx.bfloat16
   route['gates_route_calls']+=1;eligible=actual_eligible(self,b,aa,x,st);assert eligible==(B==1),'unexpected actual frozen eligibility';key='compiled_eligible_calls' if eligible else 'fallback_calls';route[key]+=1
   result=originals['_decode_gates'](self,b,aa,x,st)
   if B==1:assert callable(object.__getattribute__(self,'_compiled_decode_gates'))
   return result
  GatedDeltaNet._decode=decode_hook;GatedDeltaNet._decode_gates=gates_hook
  def restore_routes():
   for n,fn in originals.items():setattr(GatedDeltaNet,n,fn)
   assert all(getattr(GatedDeltaNet,n) is fn for n,fn in originals.items())
   assert sum(int(b)*n for b,n in route['decode_batch_counts'].items())==48*31*3*a.client_batch
   assert route['normal_decode_calls']==route['gates_route_calls']
   assert route['decode_batch_counts'].get(str(a.client_batch),0)>0,'requested concurrent decode batch never formed'
   return {**route,'hooks_restored_before_measure':True,'scope':'three warm groups only; batch1 compile eligibility; original batch>1 fallback'}
  runtime_identity['preimport_identity']=preimport_identity
  runtime_identity['runtime_before']=actual_runtime
  runtime_identity['gates_startup']=gates_metadata()
  runtime_identity['parameter_tree_startup_sha256']=hashlib.sha256(json.dumps(before_parameters,sort_keys=True).encode()).hexdigest()
  runtime_identity['parameter_tree_array_count']=len(before_parameters)
  write_new(a.run_dir/'startup-gates-metadata.json',{'gates':gates_metadata(),'parameter_metadata':before_parameters,'parameter_tree_unchanged':True,'scope':'after model ready, before any requests; metadata only'})
  runtime_identity['timed_instrumentation']=False
  completed_requests=0;warm_metadata_written=False
  mx.synchronize();mx.clear_cache();ready.set()
  while True:
   if not e.has_work(): job=pending.get();rid=e.add_request(job['ids'],SamplingParams(max_tokens=job['max_tokens'],temperature=0,ignore_eos=False));live[rid]=job
   while True:
    try: job=pending.get_nowait()
    except queue.Empty: break
    rid=e.add_request(job['ids'],SamplingParams(max_tokens=job['max_tokens'],temperature=0,ignore_eos=False));live[rid]=job
   s=time.perf_counter();results=e.step();end=time.perf_counter()
   trace.append({'start_s':s,'end_s':end,'delivered_ids':len(results),'active_decode_batch':e.runner.batch.size,'pending_requests':len(e.requests)})
   for rid,id,done in results:
    job=live[rid];job['queue'].put((id,done,end))
    if done:live.pop(rid,None);completed_requests+=1
   if not e.has_work():
    assert not live and not e.requests and not e.runner.staging
    assert e.runner.batch.size==0 and e.runner._ahead is None
    if completed_requests==3*a.client_batch and not warm_metadata_written:
     current_parameters=parameter_metadata();assert current_parameters==before_parameters,'warm changed parameter tree/array identity'
     gates=gates_metadata();routes=restore_routes();runtime_identity['runtime_after_warm']=verify_runtime(mx,nn,activations)
     write_new(a.run_dir/'warm-metadata-pending.json',{'completed_requests':completed_requests,'gates':gates,'routes':routes,'parameter_tree_unchanged':True,'parameter_metadata':current_parameters,'scope':'idle after exactly three warm requests; metadata only, no array values/eval/logits/cache reads'})
     assert not (a.run_dir/'warm-metadata.json').exists()
     (a.run_dir/'warm-metadata-pending.json').rename(a.run_dir/'warm-metadata.json')
     warm_metadata_written=True
    a.trace.write_text(json.dumps({'runtime_source_identity':runtime_identity,'steps':trace,'mlx_active_bytes':mx.get_active_memory(),'mlx_peak_bytes':mx.get_peak_memory(),'scope':'MLX allocator incl weights/cache, not RSS/system memory'},indent=2))
 except BaseException:
  fail.append(traceback.format_exc());ready.set();print(fail[-1],flush=True)
threading.Thread(target=work,daemon=True).start()
class Handler(BaseHTTPRequestHandler):
 protocol_version='HTTP/1.1'
 def log_message(self,*args):pass
 def do_GET(self):
  if not ready.is_set() or fail:self.send_error(503);return
  body=json.dumps({'data':[{'id':'tutorial-core'}],'benchmark_identity':runtime_identity}).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
 def do_POST(self):
  body=json.loads(self.rfile.read(int(self.headers['Content-Length'])));ids=tok.encode(body['prompt'],add_special_tokens=False).ids
  q=queue.Queue();pending.put({'ids':ids,'max_tokens':body['max_tokens'],'queue':q})
  self.send_response(200);self.send_header('Content-Type','text/event-stream');self.send_header('Cache-Control','no-cache');self.send_header('Connection','close');self.end_headers()
  def send(d):self.wfile.write(('data: '+json.dumps(d,ensure_ascii=False)+'\n\n').encode());self.wfile.flush()
  out=[];previous=''
  while True:
   t,done,when=q.get(timeout=180);out.append(t);decoded=tok.decode(out,skip_special_tokens=True).rstrip('\ufffd')
   assert decoded.startswith(previous),(decoded,previous)
   piece=decoded[len(previous):];previous=decoded
   if piece:send({'choices':[{'text':piece,'finish_reason':None}],'benchmark_delivery':{'emitted_id_ordinal':len(out),'id':t,'engine_delivery_perf_counter':when}})
   if done:break
  eos=out[-1] in {248044,248046};n=len(out)-int(eos)
  send({'choices':[{'text':'','finish_reason':'stop' if eos else 'length'}],'usage':{'prompt_tokens':len(ids),'completion_tokens':n,'total_tokens':len(ids)+n,'prompt_tokens_details':{'cached_tokens':0}},'benchmark_output_ids':out})
  self.wfile.write(b'data: [DONE]\n\n');self.wfile.flush();self.close_connection=True
ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()
