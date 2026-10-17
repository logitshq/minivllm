"""V120 short-C1 loopback SSE draft. Root owns timeout/resources/PID cleanup. Inert by default."""
import argparse
import json
import os
from pathlib import Path
import queue
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from adapter import D, P, build, qualification


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inert', action='store_true')
    ap.add_argument('--execute-gpu', action='store_true')
    ap.add_argument('--mode', choices=('ordinary', 'speculative'), default='ordinary')
    ap.add_argument('--qualification-binding', type=Path, default=D / 'qualification-binding.json')
    ap.add_argument('--port', type=int)
    ap.add_argument('--owner-pid', type=int)
    ap.add_argument('--run-dir', type=Path)
    a = ap.parse_args(argv)
    if not a.execute_gpu:
        assert not any(n == 'mlx' or n.startswith(('mlx.', 'numpy')) for n in sys.modules)
        print(json.dumps({'status': 'CPU_READY_SHORT_C1_ACTUAL117_BOUND_GPU_UNRUN', 'GPU_used': False, 'mode': a.mode}))
        return 0
    qualification(a.qualification_binding)  # HOLD is checked before importing tokenizer or MLX.
    if not (a.owner_pid == os.getppid() and a.port and 1024 <= a.port <= 65535 and a.run_dir and a.run_dir.is_absolute()):
        raise ValueError('root-owned process, port and absolute new run-dir required')
    a.run_dir.mkdir(exist_ok=False)
    from tokenizers import Tokenizer
    w = json.loads((P / 'workload.json').read_text())
    tok = Tokenizer.from_file(str(Path(w['model_dir']) / 'tokenizer.json'))
    jobs = queue.Queue()
    ready = threading.Event()
    failed = []
    identity = {}

    def worker():
        try:
            engine, SamplingParams, meta = build(a.mode, a.qualification_binding, execute_gpu=True)
            identity.update(meta)
            (a.run_dir / 'startup-metadata.json').write_text(json.dumps(identity, indent=2) + '\n')
            ready.set()
            while True:
                job = jobs.get()
                emitted = []
                try:
                    rid = engine.add_request(job['ids'], SamplingParams(max_tokens=32, temperature=0, ignore_eos=False))
                    while engine.has_work():
                        for current, token, done in engine.step():
                            if current != rid:
                                raise RuntimeError('C1 received foreign delivery')
                            emitted.append(token)
                            job['queue'].put(('token', token, done))
                    if engine.requests or engine.runner.staging or engine.runner.batch.size or engine.runner._ahead is not None or engine.runner.feats:
                        raise RuntimeError('terminal engine lifecycle not empty')
                    spec = engine.runner.spec
                    if spec is not None:
                        if spec.state or any(l.self_attn.ctx_len or l.self_attn.ctx_k is not None or l.self_attn.ctx_v is not None for l in spec.drafter.layers):
                            raise RuntimeError('draft lifecycle not reset')
                    job['queue'].put(('end', emitted))
                except BaseException as error:
                    job['queue'].put(('error', repr(error)))
                    raise
        except BaseException as error:
            failed.append(repr(error))
            (a.run_dir / 'worker-failure.json').write_text(json.dumps({'error': repr(error)}) + '\n')
            ready.set()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'
        def log_message(self, *args):
            pass
        def do_GET(self):
            if self.path != '/v1/models':
                self.send_error(404)
                return
            if not ready.is_set() or failed:
                self.send_error(503)
                return
            body = json.dumps({'data': [{'id': 'tutorial-candidate116'}], 'benchmark_identity': identity}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def do_POST(self):
            if self.path != '/v1/completions' or not ready.is_set() or failed:
                self.send_error(503)
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                ids = tok.encode(body['prompt'], add_special_tokens=False).ids
                if ids != w['prompt_ids'] or len(ids) != 19 or body.get('max_tokens') != 32 or body.get('temperature') != 0 or body.get('stream') is not True:
                    raise ValueError('only qualified uncached19→32 greedy fixture is enabled')
                if body.get('repetition_penalty', 1.0) != 1.0:
                    raise ValueError('repetition penalty must remain1')
            except Exception:
                self.send_error(400)
                return
            q = queue.Queue()
            jobs.put({'ids': ids, 'queue': q})
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'close')
            self.end_headers()
            def send(value):
                self.wfile.write(('data: ' + json.dumps(value, ensure_ascii=False) + '\n\n').encode())
                self.wfile.flush()
            output = []
            previous = ''
            try:
                while True:
                    item = q.get(timeout=180)
                    if item[0] == 'error':
                        raise RuntimeError(item[1])
                    if item[0] == 'end':
                        assert item[1] == output
                        break
                    _, token, done = item
                    output.append(token)
                    decoded = tok.decode(output, skip_special_tokens=True).rstrip('\ufffd')
                    if not decoded.startswith(previous):
                        raise RuntimeError('non-monotonic text rendering')
                    piece = decoded[len(previous):]
                    previous = decoded
                    if piece:
                        send({'choices': [{'text': piece, 'finish_reason': None}]})
                # Awaiting the end message also includes engine drop/reset/lifecycle in full E2E.
                eos = bool(output and output[-1] in {248044, 248046})
                count = len(output) - int(eos)
                send({'choices': [{'text': '', 'finish_reason': 'stop' if eos else 'length'}],
                    'usage': {'prompt_tokens': len(ids), 'completion_tokens': count, 'total_tokens': len(ids) + count,
                              'prompt_tokens_details': {'cached_tokens': 0}}, 'benchmark_output_ids': output})
                self.wfile.write(b'data: [DONE]\n\n')
                self.wfile.flush()
            except BaseException as error:
                send({'error': str(error)})
            self.close_connection = True

    threading.Thread(target=worker, daemon=True).start()
    ThreadingHTTPServer(('127.0.0.1', a.port), Handler).serve_forever()


if __name__ == '__main__':
    raise SystemExit(main())
