"""Qualification-only observer around the pinned official oMLX CLI.

--run --out NEW --concurrency 1|8 -- serve <original official arguments>
No imports of oMLX/MLX until --run. Never use this observer for formal timings.
Only response/request identity and ordinary Python output metadata are copied;
the original arguments, model math, output objects and SSE chunks are unchanged.
The external owner supplies source/model/settings pins, resources and teardown.
"""
import argparse
from contextvars import ContextVar
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys
import threading

ROOT = Path('/Users/ryanli/code/minivllm/.artifacts/omlx-baseline/.venv/lib/python3.13/site-packages/omlx')
PINS = {
    'server.py': 'b05c1388278db419082220ede5aa2e3e13569857a256f3f3ec3d60b800d708c4',
    'cli.py': '721b4012d5bd9b8e3b67e8fc321020e4271a2396661ebd795c12ff007d2c5ce3',
    'scheduler.py': '88bf4d988934a60e268d4205f30a86aae3012eefaef4b1f9672a0b86b412b083',
    'engine/vlm.py': 'b24406d697cd549ed083a9cb83e21cec4a7efb3f11f9fbf3747ae6bccdbf5121',
    'engine_core.py': 'c9b37d06a462b0fbc253c2087b6bd1a74884a22ced16c55ac059344c76e39a1e',
    'patches/mlx_lm_mtp/batch_generator.py': 'f758e39cff16b3aff036f621775cbd540d5a2dee2e0de228c63c3c333f51dea9',
}
SAMPLING = ('max_tokens', 'temperature', 'top_p', 'top_k', 'min_p',
            'repetition_penalty', 'presence_penalty', 'frequency_penalty',
            'xtc_probability', 'stop', 'stop_token_ids', 'thinking_budget')
REPO = Path('/Users/ryanli/code/minivllm')
E020_PLAN = REPO / 'bench/m5_ultra_256gb/external_spec_http/prereg-v1/plan.json'
E020_SHA = 'e3f0c3e24d4ea841d5e6c0b0094652e65e463c02367836e765ef6d9e3c6e2bdc'


def specification(out, concurrency, port, key, *, observed=True):
    """Pure config/command builder. The external owner materializes and runs it."""
    need(type(concurrency) is int and concurrency in (1, 8), 'C1/C8 only')
    need(type(observed) is bool and type(port) is int and 1024 <= port <= 65535, 'explicit mode/port')
    need(type(key) is str and key, 'API key required')
    out = Path(out).resolve(); state, models = out / 'state', out / 'models'
    need(sha(E020_PLAN) == E020_SHA, 'unchanged official settings source')
    plan = json.loads(E020_PLAN.read_text()); model = 'Qwen3.8-27B-nvfp4'
    arguments = ['serve', '--model-dir', str(models), '--base-path', str(state),
        '--host', '127.0.0.1', '--port', str(port), '--api-key', key,
        '--headless', '--no-hf-cache', '--no-cache', '--max-concurrent-requests', str(concurrency),
        '--sse-keepalive-mode', 'off']
    python = str(REPO / '.artifacts/omlx-baseline/.venv/bin/python')
    command = ([python, '-B', str(Path(__file__).resolve()), '--run', '--out', str(out / 'probe'),
                '--concurrency', str(concurrency), '--', *arguments] if observed else
               [str(REPO / '.artifacts/omlx-baseline/.venv/bin/omlx'), *arguments])
    return dict(command=command, cwd=str(REPO),
        environment=dict(MLX_MAX_OPS_PER_BUFFER='50', MLX_MAX_MB_PER_BUFFER='50', MLX_BFS_MAX_WIDTH='20',
                         MLX_METAL_FAST_SYNCH='0', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false'),
        files={str(state/'settings.json'):dict(server={'burst_decode_mode':'balanced'},
            scheduler=dict(max_concurrent_requests=concurrency, chunked_prefill=False,
                           prefill_priority='speed', decode_fairness=True), cache={'enabled':False}),
            str(state/'model_settings.json'):dict(version=1, models={model:plan['qualified_model_settings']['omlx-none']})},
        symlinks={str(models/model):plan['workload']['model_dir']},
        observer_enabled=observed, original_settings_plan_sha256=E020_SHA)


def jobs_for(fixtures, concurrency):
    need(type(concurrency) is int and concurrency in (1, 8) and len(fixtures) == 8, 'fixed8 fixtures/C1/C8')
    jobs = []
    for i in range(8 if concurrency == 1 else 16):
        fixture = fixtures[i % 8]
        need(fixture['normal_eos'] is True and fixture['max_output_tokens'] == 512, 'D512 normal EOS')
        jobs.append(dict(request_id=f'qual-omlx-c{concurrency}-r{i:02d}', fixture_id=fixture['fixture_id'],
            path='/v1/completions', body=dict(model='Qwen3.8-27B-nvfp4', prompt=fixture['prompt_text'],
                temperature=0, top_p=1.0, repetition_penalty=1.0, max_tokens=512, n=1, stream=True,
                stream_options={'include_usage':True})))
    return jobs


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_sources():
    for name, digest in PINS.items():
        need(sha(ROOT / name) == digest, 'official source changed: ' + name)


def route(scheduler, mtp):
    """Read the actual loaded-model routing switches, not requested CLI flags."""
    return dict(max_num_seqs=scheduler.config.max_num_seqs,
        vlm_mtp_drafter=scheduler._vlm_mtp_drafter is not None,
        vlm_mtp_active=bool(scheduler._vlm_mtp_active),
        block_drafter=mtp._drafter_for(scheduler.model) is not None,
        embedded_mtp=bool(mtp._model_mtp_decode_enabled(scheduler.model)),
        specprefill=scheduler._specprefill_draft_model is not None,
        turboquant=scheduler._turboquant_kv_bits is not None)


class Observer:
    """Explicit process-local qualification scope; restores every patched seam."""
    def __init__(self, server, core_class, scheduler_class, route_read, emit, concurrency):
        self.server, self.core_class, self.scheduler_class = server, core_class, scheduler_class
        self.route_read, self.emit, self.C = route_read, emit, concurrency
        self.current = ContextVar('omlx_d512_response', default=None)
        self.originals = None

    def __enter__(self):
        need(self.originals is None, 'observer cannot be installed twice')
        old_stream, old_add, old_outputs = (self.server.stream_completion,
            self.core_class.add_request, self.scheduler_class._process_batch_responses)
        self.originals = old_stream, old_add, old_outputs
        stream_signature, add_signature = inspect.signature(old_stream), inspect.signature(old_add)

        async def stream(*args, **kwargs):
            bound = stream_signature.bind(*args, **kwargs); bound.apply_defaults()
            values = bound.arguments
            request = values['request']
            ctx = dict(response_id=values['response_id'], prompt=values['prompt'],
                       prompt_ids=list(values['prompt_token_ids']),
                       body=request.model_dump(mode='json', exclude_unset=True))
            self.emit(dict(event='response', **ctx))
            iterator = old_stream(*args, **kwargs)
            digest, size, complete, exhausted = hashlib.sha256(), 0, False, False
            try:
                while True:
                    token = self.current.set(ctx)
                    try:
                        chunk = await anext(iterator)
                    except StopAsyncIteration:
                        exhausted = True
                        break
                    finally:
                        self.current.reset(token)
                    need(type(chunk) is str, 'official stream must yield original str chunks')
                    raw = chunk.encode('utf-8'); digest.update(raw); size += len(raw)
                    # The client closes after DONE. Its disconnect may cancel
                    # this generator before the following anext observes EOF.
                    # Record the actual protocol terminal separately from EOF.
                    complete |= chunk == 'data: [DONE]\n\n'
                    yield chunk
            finally:
                token = self.current.set(ctx)
                try:
                    await iterator.aclose()
                finally:
                    self.current.reset(token)
                    self.emit(dict(event='wire', response_id=ctx['response_id'],
                        complete=complete, iterator_exhausted=exhausted,
                        bytes=size, sha256=digest.hexdigest()))

        async def add(core, *args, **kwargs):
            bound = add_signature.bind(core, *args, **kwargs); bound.apply_defaults()
            ctx = self.current.get()
            need(ctx is not None, 'unbound engine request in qualification server')
            params = bound.arguments['sampling_params']
            # Await the exact original call before observing its returned UUID.
            rid = await old_add(core, *args, **kwargs)
            self.emit(dict(event='binding', response_id=ctx['response_id'], request_id=rid,
                scheduler_id=id(core.engine.scheduler),
                sampling={name: getattr(params, name) for name in SAMPLING}))
            return rid

        def outputs(scheduler, responses):
            before = []
            for response in responses:
                rid = scheduler.uid_to_request_id.get(response.uid)
                need(rid in scheduler.running, 'unknown response UID/request')
                request = scheduler.running[rid]
                before.append(dict(request_id=rid, token=response.token,
                    finish_reason=response.finish_reason,
                    prompt_ids=list(request.prompt_token_ids),
                    output_parser=scheduler._output_parser_factory is not None))
            actual_route = self.route_read(scheduler)
            result = old_outputs(scheduler, responses)
            out = []
            for item in result[0]:
                out.append(dict(request_id=item.request_id, finished=item.finished,
                    finish_reason=item.finish_reason, error=item.error,
                    prompt_tokens=item.prompt_tokens, completion_tokens=item.completion_tokens,
                    cached_tokens=item.cached_tokens,
                    committed_ids=list(item.output_token_ids) if item.finished else None))
            self.emit(dict(event='batch', scheduler_id=id(scheduler), route=actual_route,
                           tokens=before, outputs=out))
            return result

        self.server.stream_completion = stream
        self.core_class.add_request = add
        self.scheduler_class._process_batch_responses = outputs
        return self

    def __exit__(self, *_):
        if self.originals is not None:
            self.server.stream_completion, self.core_class.add_request, self.scheduler_class._process_batch_responses = self.originals
            self.originals = None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--concurrency', type=int, choices=(1, 8), required=True)
    parser.add_argument('official_args', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    need(args.run, '--run required; this entrypoint starts the official server')
    command = args.official_args
    if command[:1] == ['--']:
        command = command[1:]
    need(command[:1] == ['serve'], 'original official serve arguments required')
    verify_sources()
    args.out.mkdir(parents=True, exist_ok=False)
    modules = {n: importlib.import_module('omlx.' + n) for n in
               ('server', 'cli', 'scheduler', 'engine_core', 'engine.vlm', 'patches.mlx_lm_mtp.batch_generator')}
    for name, module in modules.items():
        need(Path(module.__file__).resolve() == ROOT / (name.replace('.', '/') + '.py'), 'loaded official path mismatch')
    lock = threading.Lock()
    with (args.out / 'observer.jsonl').open('x') as ledger:
        def emit(value):
            raw = json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n'
            with lock:
                ledger.write(raw); ledger.flush()
        emit(dict(event='open', schema='E023-omlx-D512-observer-v1', concurrency=args.concurrency,
                  source_root=str(ROOT), sources=PINS, probe_sha256=sha(__file__), performance_qualified=False))
        original_argv = sys.argv
        ok = False
        try:
            sys.argv = ['omlx', *command]
            with Observer(modules['server'], modules['engine_core'].AsyncEngineCore,
                          modules['scheduler'].Scheduler,
                          lambda scheduler: route(scheduler, modules['patches.mlx_lm_mtp.batch_generator']),
                          emit, args.concurrency):
                modules['cli'].main()
            ok = True
        finally:
            sys.argv = original_argv
            verify_sources()
            emit(dict(event='close', restored=True, official_returned_normally=ok))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
