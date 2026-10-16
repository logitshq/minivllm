"""E010 prefill experiment: same loopback shim and instrumentation on both arms."""
import argparse,json,queue,threading,time,sys,traceback,hashlib,inspect
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from pathlib import Path
from tokenizers import Tokenizer
ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--port',type=int,required=True);ap.add_argument('--batch',type=int,required=True);ap.add_argument('--trace',type=Path,required=True);ap.add_argument('--package-root',type=Path,required=True);ap.add_argument('--source-manifest',type=Path,required=True);ap.add_argument('--graph',action='store_true');ap.add_argument('--fuse',action='store_true');ap.add_argument('--pool',type=int,default=0);ap.add_argument('--grouped',action='store_true');a=ap.parse_args()
sys.path.insert(0,str(a.package_root))
pending=queue.Queue();ready=threading.Event();fail=[];live={};trace=[];runtime_identity={}
tok=Tokenizer.from_file(str(Path(a.model)/'tokenizer.json'))
def work():
 try:
  import mlx.core as mx
  from tutorial_code.miniengine.engine.llm_engine import LLMEngine
  from tutorial_code.miniengine.sampling_params import SamplingParams
  import tutorial_code.miniengine as actual_package
  from tutorial_code.miniengine.engine import model_runner
  if not a.grouped:model_runner.prefill_batch_backend_supported=lambda *args:False
  from tutorial_code.miniengine.model.qwen3_5 import Qwen3_5ForCausalLM
  original_call=Qwen3_5ForCausalLM.__call__;prefill_calls=[]
  def observed(model,ids,ctx=None,*args,**kwargs):
   if ctx is not None and ctx.mode in ('prefill','prefill_batch'):prefill_calls.append({'mode':ctx.mode,'shape':list(ids.shape)})
   return original_call(model,ids,ctx,*args,**kwargs)
  Qwen3_5ForCausalLM.__call__=observed
  actual_root=Path(actual_package.__file__).resolve().parent
  expected=json.loads(a.source_manifest.read_text())
  actual={str(f.relative_to(actual_root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in actual_root.rglob('*.py')}
  assert actual==expected['files'],'source identity mismatch'
  runtime_identity.update({'gates_compile_enabled':True,'norm_reuse_enabled':False,'compile_decode':a.graph,'fuse_decode':a.fuse,'allocator_pool_gib':a.pool,'grouped_prefill':a.grouped,'integration_adapter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'server_max_seqs':a.batch,'package_file':str(Path(actual_package.__file__).resolve()),'llm_engine_file':str(Path(inspect.getfile(LLMEngine)).resolve()),'actual_source_files_sha256':actual,'manifest_sha256':hashlib.sha256(a.source_manifest.read_bytes()).hexdigest(),'manifest_all_sources_match':True,'server_max_seqs':a.batch})
  e=LLMEngine(a.model,max_seqs=a.batch,max_len=2048,max_batched_tokens=512,dtype=mx.bfloat16,cache_limit_gb=a.pool,drafter_dir=None,compile_decode=a.graph,**({'fuse_decode':True} if a.fuse else {}))
  for layer in e.model.layers:
   if not layer.is_attn:layer.linear_attn.set_gates_compile(True);layer.linear_attn.set_norm_reuse(False)
  mx.synchronize();mx.clear_cache();ready.set()
  while True:
   if not e.has_work(): job=pending.get();rid=e.add_request(job['ids'],SamplingParams(max_tokens=job['max_tokens'],temperature=0,ignore_eos=False));live[rid]=job
   while True:
    try: job=pending.get_nowait()
    except queue.Empty: break
    rid=e.add_request(job['ids'],SamplingParams(max_tokens=job['max_tokens'],temperature=0,ignore_eos=False));live[rid]=job
   s=time.perf_counter();results=e.step();end=time.perf_counter()
   trace.append({'start_s':s,'end_s':end,'delivered_ids':len(results),'active_decode_batch':e.runner.batch.size,'pending_requests':len(e.requests),'mlx_active_bytes':mx.get_active_memory(),'mlx_cache_bytes':mx.get_cache_memory()})
   for rid,id,done in results:
    job=live[rid];job['queue'].put((id,done,end))
    if done:live.pop(rid,None)
   if not e.has_work():
    a.trace.write_text(json.dumps({'runtime_source_identity':runtime_identity,'steps':trace,'prefill_calls':prefill_calls,'mlx_active_bytes':mx.get_active_memory(),'mlx_cache_bytes':mx.get_cache_memory(),'mlx_peak_bytes':mx.get_peak_memory(),'scope':'MLX allocator incl weights/cache, not RSS/system memory'},indent=2))
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
class BenchmarkHTTPServer(ThreadingHTTPServer):
 request_queue_size=128
 daemon_threads=True
BenchmarkHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()
