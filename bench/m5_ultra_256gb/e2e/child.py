"""Benchmark-only loopback SSE shim for the unchanged token-ID LLMEngine."""
import argparse,json,queue,threading,time,sys,traceback,hashlib,inspect,builtins
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from pathlib import Path
from tokenizers import Tokenizer
ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--port',type=int,required=True);ap.add_argument('--batch',type=int,required=True);ap.add_argument('--trace',type=Path,required=True);ap.add_argument('--package-root',type=Path,required=True);ap.add_argument('--source-manifest',type=Path,required=True);ap.add_argument('--norm-reuse',choices=['off','on'],required=True);ap.add_argument('--owner-pid',type=int,required=True);ap.add_argument('--run-dir',type=Path,required=True);ap.add_argument('--go',type=Path,required=True);ap.add_argument('--attempt-dir',type=Path,required=True);ap.add_argument('--expected-power',choices=['AC','Battery'],required=True);a=ap.parse_args()
import os
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run import verify_frozen, actual_preimport_identity, write_new, child_environment
assert a.owner_pid==os.getppid(), "child owner PID differs"
marker=json.loads((a.attempt_dir/'attempt-consumed.json').read_text())
assert marker['owner_pid']==a.owner_pid and marker['GO_sha256']==hashlib.sha256(a.go.read_bytes()).hexdigest()
write_new(a.run_dir/'child-attempt.consumed.json',{'owner_pid':a.owner_pid,'child_pid':os.getpid()})
f=verify_frozen();assert child_environment(os.environ)==dict(os.environ)
preimport_identity=actual_preimport_identity()
write_new(a.run_dir/'preimport-child.json',preimport_identity)
pending=queue.Queue();ready=threading.Event();fail=[];live={};trace=[];runtime_identity={}
tok=Tokenizer.from_file(str(Path(a.model)/'tokenizer.json'))
def work():
 warm=None;runtime_finders=[]
 try:
  import runtime as local_runtime
  from worker import SourceFinder
  source_finder=SourceFinder(local_runtime,Path(f['candidate_root']),f['candidate_sources'])
  sys.meta_path.insert(0,source_finder);runtime_finders.append(source_finder)
  def runtime_fence(mx,nn,activations):return local_runtime.runtime_identity(f,mx,nn,activations)
  import mlx.core as mx
  import mlx.nn as nn
  import mlx.nn.layers.activations as activations
  runtime_identity['runtime_before']=runtime_fence(mx,nn,activations)
  from tutorial_code.miniengine.engine.llm_engine import LLMEngine
  from tutorial_code.miniengine.sampling_params import SamplingParams
  import tutorial_code.miniengine as actual_package
  actual_root=Path(actual_package.__file__).resolve().parent
  assert actual_root==Path(preimport_identity['source']['root']),'actual package resolves outside frozen candidate'
  expected=json.loads(a.source_manifest.read_text())
  actual={str(f.relative_to(actual_root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in actual_root.rglob('*.py')}
  assert actual==expected['files'] and len(actual)==26,'V19 source identity mismatch'
  runtime_identity.update({'package_file':str(Path(actual_package.__file__).resolve()),'llm_engine_file':str(Path(inspect.getfile(LLMEngine)).resolve()),'actual_source_files_sha256':actual,'manifest_sha256':hashlib.sha256(a.source_manifest.read_bytes()).hexdigest(),'manifest_all_sources_match':True,'server_max_seqs':a.batch})
  e=LLMEngine(a.model,max_seqs=a.batch,max_len=2048,max_batched_tokens=512,dtype=mx.bfloat16,cache_limit_gb=0,drafter_dir=None)
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
  before_parameters=parameter_metadata();assert len(before_parameters)==1349,'parameter array count differs'
  assert all(object.__getattribute__(g,'_gates_compile_enabled') is False and object.__getattribute__(g,'_compiled_decode_gates') is None for g in gdns),'constructor default-off missing'
  gates_setter_calls=0;norm_reuse_setter_calls=0
  assert all(object.__getattribute__(g,'_norm_reuse_enabled') is False for g in gdns),'norm_reuse default off missing'
  for g in gdns:
   g.set_gates_compile(True);gates_setter_calls+=1
   g.set_norm_reuse(a.norm_reuse=='on');norm_reuse_setter_calls+=1
  assert gates_setter_calls==norm_reuse_setter_calls==48
  assert before_parameters==parameter_metadata(),'setter changed parameter tree'
  def gates_metadata():
   flags=[object.__getattribute__(g,'_gates_compile_enabled') for g in gdns]
   functions=[object.__getattribute__(g,'_compiled_decode_gates') for g in gdns]
   assert len(flags)==48 and all(type(v) is bool and v is True for v in flags)
   return {'enabled':True,'count':len(flags),'flags':flags,'compiled_callable_count':sum(callable(fn) for fn in functions),'all_other_callable_values_none':all(fn is None or callable(fn) for fn in functions)}
  def norm_reuse_metadata():
   flags=[object.__getattribute__(g,'_norm_reuse_enabled') for g in gdns]
   assert len(flags)==48 and all(type(v) is bool and v==(a.norm_reuse=='on') for v in flags)
   return {'enabled':a.norm_reuse=='on','count':len(flags),'flags':flags}
  import tutorial_code.miniengine.model.gdn as gdnmod
  import tutorial_code.miniengine.kernels.gdn_norm_reuse_metal as normmod
  from tutorial_code.miniengine.cache import BatchState
  from warm_routes import WarmRoutes
  warm=WarmRoutes(gdnmod,normmod,mx,gdns,BatchState,a.norm_reuse=='on')
  runtime_identity['preimport_identity']=preimport_identity
  runtime_identity['gates_startup']=gates_metadata()
  runtime_identity['norm_reuse_startup']=norm_reuse_metadata()
  runtime_identity['actual_setter_calls']={'gates':gates_setter_calls,'norm_reuse':norm_reuse_setter_calls}
  runtime_identity['parameter_tree_startup_sha256']=hashlib.sha256(json.dumps(before_parameters,sort_keys=True).encode()).hexdigest()
  runtime_identity['parameter_tree_array_count']=len(before_parameters)
  write_new(a.run_dir/'startup-gates-metadata.json',{'gates':gates_metadata(),'norm_reuse':norm_reuse_metadata(),'actual_setter_calls':{'gates':gates_setter_calls,'norm_reuse':norm_reuse_setter_calls},'parameter_metadata':before_parameters,'parameter_tree_unchanged':True,'scope':'after model ready, before any requests; metadata only'})
  runtime_identity['timed_instrumentation']=False
  runtime_identity['instrumentation_scope']='measurement has no added hooks; three warm requests have actual route/shape validation hooks'
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
    if completed_requests==3 and not warm_metadata_written:
     current_parameters=parameter_metadata();assert current_parameters==before_parameters,'warm changed parameter tree/array identity'
     gates=gates_metadata();routes=warm.restore();assert gates['compiled_callable_count']==48,'expected lazy callable activation missing'
     runtime_identity['runtime_after_warm']=runtime_fence(mx,nn,activations)
     write_new(a.run_dir/'warm-metadata-pending.json',{'completed_requests':completed_requests,'gates':gates,'norm_reuse':norm_reuse_metadata(),'routes':routes,'parameter_tree_unchanged':True,'parameter_metadata':current_parameters,'scope':'idle after exactly three warm requests; metadata only, no array values/eval/logits/cache reads'})
     assert not (a.run_dir/'warm-metadata.json').exists()
     (a.run_dir/'warm-metadata-pending.json').rename(a.run_dir/'warm-metadata.json')
     warm_metadata_written=True
    a.trace.write_text(json.dumps({'runtime_source_identity':runtime_identity,'steps':trace,'mlx_active_bytes':mx.get_active_memory(),'mlx_peak_bytes':mx.get_peak_memory(),'scope':'MLX allocator incl weights/cache, not RSS/system memory'},indent=2))
 except BaseException:
  fail.append(traceback.format_exc());ready.set();print(fail[-1],flush=True)
 finally:
  if warm is not None:
   try:warm.restore(validate=False)
   except BaseException:fail.append(traceback.format_exc())
  for finder in runtime_finders:
   if finder in sys.meta_path:sys.meta_path.remove(finder)
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
