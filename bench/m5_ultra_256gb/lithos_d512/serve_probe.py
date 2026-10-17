"""Qualification-only taps around the unchanged official Lithos server.

--run --out NEW --concurrency 1|8 -- <original monolith.serve arguments>
No installed/model imports before --run. Three scoped methods return the original
objects/strings: Session.generate, WireResponse.body and WireResponse.finish.
The actual final NDJSON line joins the wire to Generation.tokens; Ollama emits
no request ID. Thread-local generation ownership ends at body(), not at DONE.
The external owner controls deadlines, resources, source/model/pack pins and reap.
"""
import argparse
import dataclasses
import hashlib
import importlib
import json
from pathlib import Path
import sys
import threading

ROOT = Path('/Users/ryanli/code/lithos-metal')
REPO = Path('/Users/ryanli/code/minivllm')
E020_PLAN = REPO / 'bench/m5_ultra_256gb/external_spec_http/prereg-v1/plan.json'
E020_SHA = 'e3f0c3e24d4ea841d5e6c0b0094652e65e463c02367836e765ef6d9e3c6e2bdc'
PINS = {
    'monolith/serve.py': '9759ee14c4860ed43218118d6a3eafbb680e9e722e3ddad72bae52473579f419',
    'monolith/generate.py': '434ac1398c6010b37fa44f138505d5ca1757f4e7b747bc6b582d386e2e3ba3b4',
    'monolith/serving/protocol.py': '5f015e2c55037bd0ae73f8b33e7eb64adf8fcb67f19f0f1dc2d25157b1f83ada',
    'monolith/serving/events.py': 'aaa11acb7cb058f55916365e94a0b45b160f286356ffd1d610f08fbaf67b316d',
    'monolith/runtime/prefix_cache.py': '2ffa0f9bfed792765d27af811f8df3d08f9c6b5709c05d6b1e996ba0721297ad',
}
SCHEMA = 'E023-lithos-D512-observer-v1'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_sources():
    for name, digest in PINS.items():
        need(sha(ROOT / name) == digest, 'official source changed: ' + name)


def specification(out, concurrency, port, key, *, observed=True):
    """Pure original E020 Lithos-none launch; no private recipe or repacking."""
    need(type(concurrency) is int and concurrency in (1, 8), 'C1/C8 only')
    need(type(port) is int and 1024 <= port <= 65535 and type(observed) is bool, 'explicit port/mode')
    need(type(key) is str and key, 'API key required')
    need(sha(E020_PLAN) == E020_SHA, 'frozen E020 launch inputs changed')
    plan = json.loads(E020_PLAN.read_text())
    args = ['--model', plan['workload']['model_dir'], '--pack', plan['packs']['lithos-none']['target']['path'],
        '--local-files-only', '--host', '127.0.0.1', '--port', str(port), '--max-context', '2048',
        '--prefill-chunk-size', '128-exact', '--keep-warm', '0', '--no-warmup', '--no-draft']
    python = str(ROOT / '.venv/bin/python')
    command = ([python, '-B', str(Path(__file__).resolve()), '--run', '--out', str(Path(out).resolve()/'probe'),
        '--concurrency', str(concurrency), '--', *args] if observed else
        [python, '-B', '-m', 'monolith.serve', *args])
    return dict(command=command, cwd=str(ROOT), observer_enabled=observed,
        environment=dict(MLX_MAX_OPS_PER_BUFFER='50', MLX_MAX_MB_PER_BUFFER='50', MLX_BFS_MAX_WIDTH='20',
            MLX_METAL_FAST_SYNCH='0', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
            MONOLITH_ACCELERATOR='on', LITHOS_METAL_API_KEY=key),
        original_launch_plan_sha256=E020_SHA)


def jobs_for(fixtures, concurrency, model):
    need(type(concurrency) is int and concurrency in (1, 8) and len(fixtures) == 8, 'fixed D8/C1/C8')
    need(type(model) is str and model, 'actual served model required')
    jobs = []
    for i in range(8 if concurrency == 1 else 16):
        f = fixtures[i % 8]
        need(f['normal_eos'] is True and f['max_output_tokens'] == 512, 'D512 normal EOS')
        jobs.append(dict(request_id=f'qual-lithos-c{concurrency}-r{i:02d}', fixture_id=f['fixture_id'],
            path='/api/chat', body=dict(model=model, messages=[dict(role='user', content=f['prompt_text'])],
                stream=True, options=dict(num_predict=512, temperature=0, top_p=1, top_k=0, seed=0))))
    return jobs


def route(session):
    """Actual selected session and already executed program metadata; no arrays."""
    info = session.dev.info()
    return dict(device=dict(name=info.name, gpu_cores=info.gpu_cores, apple_family=info.apple_family),
        backend=session.profile.backend, pack_dir=str(session.pack.dir.resolve()),
        capacity=session.pack.manifest['options']['max_context'], has_drafter=session.drafter is not None,
        verify=session.verify, verify_length=session.verify_length, decoder_kernel_config=session.decoder_kernel_config,
        sampling=session._sampling, seed=session.seed, accelerator=session.accelerator,
        prefill_chunk_size=session.prefill_chunk_size, effective_prefill_exact=session.prefill_exact,
        prefix_cache=session.prefix_cache is not None, eos=session.eos,
        config_eos=session.model.config.eos_token_id,
        programs=[dict(key=str(k), backend=engine.program.backend_id,
                       capacity=engine.program.context_capacity) for k, engine in session.engines.items()])


class Observer:
    """Three original method calls, one generation per worker; no global current request."""
    def __init__(self, session_class, wire_class, emit, route_read=route, *, expected_count=None, source_check=None):
        self.session_class, self.wire_class = session_class, wire_class
        self.emit, self.route_read = emit, route_read
        self.local = threading.local()
        self.lock = threading.Lock()
        self.started = self.returned = self.bodies = self.terminals = 0
        self.active = 0
        self.originals = None
        self.expected_count, self.source_check = expected_count, source_check
        self.sealed = False

    def __enter__(self):
        need(self.originals is None, 'observer already installed')
        old_generate, old_body, old_finish = self.session_class.generate, self.wire_class.body, self.wire_class.finish
        self.originals = old_generate, old_body, old_finish

        def generate(session, prompt_ids, max_new_tokens, **kwargs):
            if self.sealed:
                self.emit(dict(event='error', message='generation after finite qualification was sealed'))
                raise ValueError('generation after finite qualification was sealed')
            need(getattr(self.local, 'pending', None) is None, 'recursive or unbound generation')
            with self.lock:
                need(self.active == 0, 'official single-generation route overlapped')
                self.active += 1; self.started += 1; index = self.started
            before = list(prompt_ids)
            self.emit(dict(event='generation_start', index=index, prompt_ids=before, cap=max_new_tokens,
                           streaming=callable(kwargs.get('on_tokens')), cache_prefix_tokens=kwargs.get('cache_prefix_tokens')))
            try:
                gen = old_generate(session, prompt_ids, max_new_tokens, **kwargs)
                need(list(prompt_ids) == before, 'generation mutated prompt input')
                record = dict(index=index, generation=dataclasses.asdict(gen), route=self.route_read(session))
                self.local.pending = record
                self.emit(dict(event='generation_return', **record))
                with self.lock:
                    self.returned += 1
                return gen
            except BaseException as exc:
                self.emit(dict(event='error', index=index, type=type(exc).__name__, message=str(exc)))
                raise
            finally:
                with self.lock:
                    self.active -= 1

        def body(wire, message, finish, prompt_tokens, completion_tokens):
            pending = getattr(self.local, 'pending', None)
            need(pending is not None and wire.protocol == 'ollama', 'unbound official body')
            result = old_body(wire, message, finish, prompt_tokens, completion_tokens)
            self.emit(dict(event='body', index=pending['index'], wire_id=wire.id,
                           metrics=dict(wire.metrics), body=result))
            self.local.pending = None
            with self.lock:
                self.bodies += 1
            return result

        def finish(wire, body_value, include_usage):
            for line in old_finish(wire, body_value, include_usage):
                need(wire.protocol == 'ollama' and isinstance(line, str), 'unexpected wire protocol')
                self.emit(dict(event='wire', wire_id=wire.id, line=line))
                if json.loads(line).get('done') is True:
                    with self.lock:
                        self.terminals += 1
                    if self.expected_count is not None and self.terminals == self.expected_count:
                        expected = dict(started=self.expected_count, returned=self.expected_count,
                            bodies=self.expected_count, terminals=self.expected_count, active=0)
                        need(self.counts() == expected and callable(self.source_check), 'finite observer counts/source check')
                        self.source_check()
                        self.emit(dict(event='sealed', status='COMPLETE', source_pins_rechecked=True, **expected))
                        self.sealed = True
                yield line

        self.session_class.generate, self.wire_class.body, self.wire_class.finish = generate, body, finish
        return self

    def __exit__(self, typ, exc, tb):
        self.session_class.generate, self.wire_class.body, self.wire_class.finish = self.originals
        self.originals = None

    def counts(self):
        return dict(started=self.started, returned=self.returned, bodies=self.bodies, terminals=self.terminals,
                    active=self.active)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--concurrency', type=int, choices=(1, 8))
    parser.add_argument('official_args', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if not args.run:
        print(json.dumps(dict(schema=SCHEMA, qualification_only=True, formal_observer=False))); return
    if args.out is None or args.concurrency is None:
        parser.error('--run requires --out NEW and --concurrency')
    arguments = args.official_args
    if arguments[:1] == ['--']:
        arguments = arguments[1:]
    verify_sources()
    sys.path.insert(0, str(ROOT))
    serve = importlib.import_module('monolith.serve')
    generate = importlib.import_module('monolith.generate')
    events = importlib.import_module('monolith.serving.events')
    for module, relative in ((serve, 'monolith/serve.py'), (generate, 'monolith/generate.py'), (events, 'monolith/serving/events.py')):
        need(Path(module.__file__).resolve() == ROOT/relative, 'unexpected installed module')
    parsed = serve.parse_args(arguments)
    expected = specification(args.out.parent, args.concurrency, parsed.port, 'not-a-secret', observed=False)['command'][4:]
    need(arguments == expected, 'qualification requires exact original Lithos-none arguments')
    args.out.mkdir(parents=True, exist_ok=False)
    lock = threading.Lock()
    with (args.out/'observer.jsonl').open('x', encoding='utf-8') as stream:
        def emit(row):
            with lock:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n'); stream.flush()
        emit(dict(event='open', schema=SCHEMA, concurrency=args.concurrency, source_root=str(ROOT), sources=PINS,
                  probe_sha256=sha(__file__), official_args=arguments, qualification_only=True))
        observer = Observer(generate.Session, events.WireResponse, emit,
                            expected_count=8 if args.concurrency == 1 else 16, source_check=verify_sources)
        try:
            with observer:
                serve.main(arguments)
            verify_sources()
            need(observer.sealed, 'server returned before finite observation completed')
        except BaseException as exc:
            emit(dict(event='error', type=type(exc).__name__, message=str(exc), **observer.counts()))
            raise


if __name__ == '__main__':
    main()
