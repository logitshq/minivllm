"""Explicit C1 native ICB benchmark adapter; ordinary token-ID engine prefill."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

P = Path('/private/tmp/minivllm-engine-optimization-20261014')
ap = argparse.ArgumentParser()
ap.add_argument('--config', type=Path, required=True)
ap.add_argument('--model', required=True)
ap.add_argument('--port', type=int, required=True)
ap.add_argument('--trace', type=Path, required=True)
ap.add_argument('--run-dir', type=Path, required=True)
ap.add_argument('--owner-pid', type=int, required=True)
a = ap.parse_args()
assert os.getppid() == a.owner_pid
config = json.loads(a.config.read_text())
for path, digest in config['source_pins'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
assert not any(n == 'mlx' or n.startswith('mlx.') for n in sys.modules)
assert os.environ.get('DYLD_INSERT_LIBRARIES') == config['native_library']
assert {k: os.environ.get(k) for k in ('MLX_MAX_OPS_PER_BUFFER', 'MLX_MAX_MB_PER_BUFFER',
    'MLX_BFS_MAX_WIDTH', 'MLX_METAL_FAST_SYNCH')} == {
    'MLX_MAX_OPS_PER_BUFFER': '50', 'MLX_MAX_MB_PER_BUFFER': '50',
    'MLX_BFS_MAX_WIDTH': '20', 'MLX_METAL_FAST_SYNCH': '0'}
assert all(importlib.metadata.version(k) == '0.32.3' for k in ('mlx', 'mlx-metal'))
assert config['scheduler_max_seqs'] == 1
assert config['mode'] in ('conv', 'normreuse_cache', 'conv_normreuse_cache')
sys.path.insert(0, config['package_root'])
package = Path(config['package_root']) / 'tutorial_code/miniengine'
actual = {str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest() for p in package.rglob('*.py')}
assert actual == config['package_sources']
w = json.loads((P / 'workload.json').read_text())
assert a.model == w['model_dir']
from tokenizers import Tokenizer
tok = Tokenizer.from_file(str(Path(a.model) / 'tokenizer.json'))
pending = queue.Queue(); ready = threading.Event(); fail = []; live = {}; trace = []
runtime_identity = {}; adapter = None


def write_new(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def work():
    global adapter
    try:
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from tutorial_code.miniengine.engine.llm_engine import LLMEngine
        from tutorial_code.miniengine.sampling_params import SamplingParams
        import tutorial_code.miniengine as actual_package
        assert Path(actual_package.__file__).resolve().parent == package
        e = LLMEngine(a.model, max_seqs=1, max_len=2048, max_batched_tokens=512,
                      dtype=mx.bfloat16, cache_limit_gb=2)
        def parameter_metadata():
            return [(n, list(v.shape), str(v.dtype), id(v)) for n, v in tree_flatten(e.model.parameters())]
        parameters = parameter_metadata(); assert len(parameters) == 1349
        gdns = [l.linear_attn for l in e.model.layers if not l.is_attn]
        assert len(gdns) == 48
        for g in gdns:
            g.set_gates_compile(True)
            if config['mode'] == 'conv': g.set_conv_silu(True)
            elif config['mode'] == 'normreuse_cache': g.set_norm_reuse_cache(True)
            else: g.set_conv_norm_reuse_cache(True)
        assert parameter_metadata() == parameters
        # Qualifying one ordinary warm request primes kernels and is discarded.
        # It is explicitly recorded; every HTTP request still freshly prefills.
        warm = e.generate([w['prompt_ids']], SamplingParams(max_tokens=32, temperature=0, ignore_eos=False))
        assert warm == [w['continuation_ids']]
        assert e.runner.batch.size == 0 and not e.requests and e.runner._ahead is None
        spec = importlib.util.spec_from_file_location('qualified_native_runner', config['adapter_source'])
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        adapter = module.install(e, config['native_library'], config['native_library_sha256'])
        runtime_identity.update(package_file=str(Path(actual_package.__file__).resolve()),
            manifest_all_sources_match=True, mode=config['mode'], scheduler_max_seqs=1,
            model_loads=1, parameter_count=1349, parameter_tree_unchanged=True,
            gates_on=48, fixed_cap=256, native_library_sha256=config['native_library_sha256'],
            adapter_sha256=config['source_pins'][config['adapter_source']],
            prior_fullqual_sha256=config['fullqual_receipt_sha256'], prefix_cache=0,
            startup_ordinary_warm_requests=1, startup_ordinary_warm_tokens=32,
            client_warm_requests=3, measurement_hooks=False,
            cache_policy='fresh per-request prefill; all131seed ranges uploaded once; native token/state feedback')
        write_new(a.run_dir / 'startup-metadata.json', runtime_identity)
        mx.synchronize(); mx.clear_cache(); ready.set()
        completed = 0; wrote_warm = False
        while True:
            if not e.has_work():
                job = pending.get()
                rid = e.add_request(job['ids'], SamplingParams(max_tokens=job['max_tokens'], temperature=0, ignore_eos=False))
                live[rid] = job
            while True:
                try: job = pending.get_nowait()
                except queue.Empty: break
                rid = e.add_request(job['ids'], SamplingParams(max_tokens=job['max_tokens'], temperature=0, ignore_eos=False))
                live[rid] = job
            start = time.perf_counter(); results = e.step(); end = time.perf_counter()
            trace.append({'start_s': start, 'end_s': end, 'delivered_ids': len(results),
                          'active_decode_batch': e.runner.batch.size, 'pending_requests': len(e.requests)})
            for rid, token, done in results:
                live[rid]['queue'].put((token, done, end))
                if done: live.pop(rid); completed += 1
            if not e.has_work():
                if completed == 3 and not wrote_warm:
                    assert not e.requests and not live and adapter.active is None
                    assert e.runner.batch.size == 0 and e.runner._ahead is None
                    assert parameters == parameter_metadata()
                    native = adapter.report(); f = native['feedback']
                    assert native['status'] == 'OK' and adapter.native_steps == 93 and adapter.request_count == 3
                    assert f['total_steps'] == 93 and f['resets'] == 3
                    assert f['icb']['execution_count'] == 93 and f['icb']['bank_executions'] == [48, 45]
                    assert f['pingpong']['eligible_count'] == 48
                    write_new(a.run_dir / 'warm-metadata-pending.json', {
                        'completed_requests': 3, 'native_steps': 93, 'native': native,
                        'parameter_tree_unchanged': True, 'terminal_cleanup': True,
                        'measurement_hooks': False, 'startup_ordinary_warm_requests': 1})
                    (a.run_dir / 'warm-metadata-pending.json').rename(a.run_dir / 'warm-metadata.json')
                    wrote_warm = True
                a.trace.write_text(json.dumps({'runtime_source_identity': runtime_identity, 'steps': trace,
                    'mlx_active_bytes': mx.get_active_memory(), 'mlx_peak_bytes': mx.get_peak_memory()}))
    except BaseException:
        fail.append(traceback.format_exc()); ready.set(); print(fail[-1], flush=True)
        (a.run_dir / 'worker-failure.txt').write_text(fail[-1])


threading.Thread(target=work, daemon=True).start()


class Handler(BaseHTTPRequestHandler):
 protocol_version='HTTP/1.1'
 def log_message(self,*args):pass
 def do_GET(self):
  if self.path=='/benchmark-state':
   if not ready.is_set() or fail:self.send_error(503);return
   body=json.dumps({'native':adapter.report(),'native_steps':adapter.native_steps,'request_count':adapter.request_count,'active':adapter.active}).encode()
   self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body);return
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
