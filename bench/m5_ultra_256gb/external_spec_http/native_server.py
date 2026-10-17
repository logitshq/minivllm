"""E020 C1 loopback shim: ordinary engine calls, no step observer or profiler.

Derived from frontier/native_child.py's queue/SSE transport. Source identity is
checked before loading; raw output IDs are sent once, at request completion.
Only this script's main() imports the model runtime. --help is CPU-only.
"""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import queue
import sys
import threading
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--port', required=True, type=int)
    parser.add_argument('--package-root', required=True, type=Path)
    parser.add_argument('--source-manifest', required=True, type=Path)
    args = parser.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(args.model) / 'tokenizer.json'))
    pending, ready, failures, identity = queue.Queue(), threading.Event(), [], {}
    eos_ids = set()

    def worker():
        job = None
        try:
            sys.path.insert(0, str(args.package_root))
            import mlx.core as mx
            import tutorial_code.miniengine as package
            from tutorial_code.miniengine.engine.llm_engine import LLMEngine
            from tutorial_code.miniengine.sampling_params import SamplingParams
            root = Path(package.__file__).resolve().parent
            actual = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in root.rglob('*.py')}
            assert actual == json.loads(args.source_manifest.read_text())['files'], 'native source mismatch'
            engine = LLMEngine(args.model, max_seqs=4, max_len=2048, max_batched_tokens=512,
                dtype=mx.bfloat16, cache_limit_gb=2, drafter_dir=None,
                compile_decode=True, fuse_decode=True)
            for layer in engine.model.layers:
                if not layer.is_attn:
                    layer.linear_attn.set_gates_compile(True)
                    layer.linear_attn.set_norm_reuse(False)
            eos_ids.update(engine.eos_ids)
            identity.update(source_files=actual, manifest_all_sources_match=True,
                cache_limit_gib=2, compile_decode=True, fuse_decode=True, drafter=None,
                gates_compile=True, norm_reuse=False, grouped_prefill='unmodified core default',
                step_observer=False, max_seqs=4, normal_eos=True)
            mx.synchronize()
            mx.clear_cache()  # Once at load, matching prior native shim; never per request.
            ready.set()
            while True:
                job = pending.get()
                rid = engine.add_request(job['ids'], SamplingParams(max_tokens=32, temperature=0, ignore_eos=False))
                while engine.has_work():
                    for actual_rid, token, done in engine.step():
                        assert actual_rid == rid
                        job['queue'].put((token, done))
                job = None
        except BaseException:
            error = traceback.format_exc()
            failures.append(error)
            if job is not None:
                job['queue'].put(error)
            print(error, flush=True)
            ready.set()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'
        def log_message(self, *unused):
            pass
        def do_GET(self):
            if self.path != '/v1/models':
                self.send_error(404)
                return
            if not ready.is_set() or failures:
                self.send_error(503)
                return
            body = json.dumps({'data': [{'id': 'native-current-target-only'}],
                               'benchmark_identity': identity}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def do_POST(self):
            if self.path != '/v1/completions' or not ready.is_set() or failures:
                self.send_error(503)
                return
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if body.get('max_tokens') != 32 or body.get('temperature') != 0:
                self.send_error(400)
                return
            ids = tok.encode(body['prompt'], add_special_tokens=False).ids
            delivery = queue.Queue()
            pending.put({'ids': ids, 'queue': delivery})
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'close')
            self.end_headers()
            def send(value):
                self.wfile.write(('data: ' + json.dumps(value, ensure_ascii=False) + '\n\n').encode())
                self.wfile.flush()
            output, previous = [], ''
            while True:
                item = delivery.get(timeout=300)
                if isinstance(item, str):
                    send({'error': item})
                    self.close_connection = True
                    return
                token, done = item
                output.append(token)
                decoded = tok.decode(output, skip_special_tokens=True).rstrip('\ufffd')
                assert decoded.startswith(previous), 'non-monotone incremental decoding'
                piece, previous = decoded[len(previous):], decoded
                if piece:
                    send({'choices': [{'text': piece, 'finish_reason': None}]})
                if done:
                    break
            eos = output[-1] in eos_ids
            count = len(output) - int(eos)
            send({'choices': [{'text': '', 'finish_reason': 'stop' if eos else 'length'}],
                  'usage': {'prompt_tokens': len(ids), 'completion_tokens': count,
                            'total_tokens': len(ids) + count, 'prompt_tokens_details': {'cached_tokens': 0}},
                  'benchmark_output_ids': output})
            self.wfile.write(b'data: [DONE]\n\n')
            self.wfile.flush()
            self.close_connection = True

    threading.Thread(target=worker, daemon=True).start()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.daemon_threads = True
    server.serve_forever()


if __name__ == '__main__':
    main()
