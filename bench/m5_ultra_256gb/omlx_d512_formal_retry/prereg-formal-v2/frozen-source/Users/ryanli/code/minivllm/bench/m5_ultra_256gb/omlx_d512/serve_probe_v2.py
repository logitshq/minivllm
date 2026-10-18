"""Qualification-only observer around the pinned official oMLX CLI.

--run --out NEW --concurrency 1|8 -- serve <original official arguments>
No imports of oMLX/MLX until --run. Never use this observer for formal timings.
The durable request seal is neither process close nor generator EOF proof.
Actual loaded VLM/model/runtime identity and ordinary Python metadata are copied;
the original arguments, model math, output objects and SSE chunks are unchanged.
The external owner supplies source/model/settings pins, resources and teardown.
"""
import argparse
import asyncio
from collections import Counter
import importlib.metadata
import os
import platform
import re
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
    'api/openai_models.py': '4095ed248c862c32f195075146f7b7ad66e0e1836089701da977e62c07572b09',
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
FIXTURES = REPO / 'bench/m5_ultra_256gb/pareto_realistic/fixtures/prepared-D-v1/fixtures.json'
FIXTURES_SHA = '6a4f7ce1578a0c6fead596e8b3d646033d90a2e2b843263747b89a3c6b3c88db'
VERSIONS = {'omlx':'0.7.1.dev1', 'mlx':'0.32.3', 'mlx-vlm':'0.7.7',
            'mlx-lm':'0.31.4.dev132+g94cdcae13'}
ENV_ROOT = REPO / '.artifacts/omlx-baseline/.venv'


def expected_model():
    need(sha(E020_PLAN) == E020_SHA, 'unchanged official settings source')
    return str(Path(json.loads(E020_PLAN.read_text())['workload']['model_dir']).resolve())


def runtime_identity():
    versions = {key: importlib.metadata.version(key) for key in VERSIONS}
    value = dict(python_executable=sys.executable, python_resolved=str(Path(sys.executable).resolve()),
                 prefix=sys.prefix, python_version=platform.python_version(), versions=versions,
                 pid=os.getpid(), ppid=os.getppid())
    validate_runtime(value)
    return value


def validate_runtime(value):
    need(value['versions'] == VERSIONS and Path(value['prefix']).resolve() == ENV_ROOT.resolve()
         and Path(value['python_executable']).absolute() == ENV_ROOT / 'bin/python'
         and value['python_resolved'] == str((ENV_ROOT / 'bin/python').resolve())
         and value['python_version'].startswith('3.13.')
         and all(type(value[k]) is int and value[k] > 1 for k in ('pid','ppid')),
         'actual dedicated oMLX Python/runtime metadata; outer binary pins still required')


def loaded_identity(engine):
    scheduler = engine._engine.engine.scheduler
    kind = type(engine); module = sys.modules[kind.__module__]
    value = dict(engine_id=id(engine), class_module=kind.__module__, class_name=kind.__qualname__,
        class_file=str(Path(module.__file__).resolve()), class_sha256=sha(module.__file__),
        model_name=str(engine.model_name), model_path=str(Path(engine.model_name).resolve()),
        core_class_module=type(engine._engine).__module__, core_class_name=type(engine._engine).__qualname__,
        scheduler_class_module=type(scheduler).__module__, scheduler_class_name=type(scheduler).__qualname__,
        scheduler_id=id(scheduler), prefix_cache_enabled=engine.prefix_cache_enabled)
    validate_loaded(value, expected_model())
    return value


def validate_loaded(value, model_path):
    need(value['class_module'] == 'omlx.engine.vlm' and value['class_name'] == 'VLMBatchedEngine'
         and value['class_file'] == str(ROOT / 'engine/vlm.py')
         and value['class_sha256'] == PINS['engine/vlm.py']
         and value['core_class_module'] == 'omlx.engine_core' and value['core_class_name'] == 'AsyncEngineCore'
         and value['scheduler_class_module'] == 'omlx.scheduler' and value['scheduler_class_name'] == 'Scheduler'
         and value['prefix_cache_enabled'] is False
         and value['model_path'] == model_path and str(Path(value['model_name']).resolve()) == model_path
         and all(type(value[k]) is int and value[k] > 0 for k in ('engine_id','scheduler_id')),
         'actual loaded VLM engine/model/source identity; fallback is outside this qualification')



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
                temperature=0.0, top_p=1.0, repetition_penalty=1.0, max_tokens=512, stream=True,
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
        turboquant=scheduler._turboquant_kv_bits is not None,
        block_aware_cache_present=scheduler.block_aware_cache is not None,
        paged_cache_manager_present=scheduler.paged_cache_manager is not None,
        paged_ssd_cache_manager_present=scheduler.paged_ssd_cache_manager is not None,
        paged_ssd_cache_dir=scheduler.config.paged_ssd_cache_dir,
        paged_ssd_cache_max_size=scheduler.config.paged_ssd_cache_max_size,
        hot_cache_max_size=scheduler.config.hot_cache_max_size, hot_cache_only=scheduler.config.hot_cache_only,
        hot_cache_budget_present=scheduler.config.hot_cache_budget is not None)


def expected_route(concurrency, ssd_max_bytes=0):
    return dict(max_num_seqs=concurrency, vlm_mtp_drafter=False, vlm_mtp_active=False,
        block_drafter=False, embedded_mtp=False, specprefill=False, turboquant=False,
        block_aware_cache_present=False, paged_cache_manager_present=False, paged_ssd_cache_manager_present=False,
        paged_ssd_cache_dir=None, paged_ssd_cache_max_size=ssd_max_bytes,
        hot_cache_max_size=0, hot_cache_only=False, hot_cache_budget_present=False)


def valid_route(value, concurrency):
    # The unused SSD capacity may remain positive when the cache has no directory/managers.
    unused = value['paged_ssd_cache_max_size']
    expected = expected_route(concurrency, unused)
    return type(unused) is int and unused >= 0 and value == expected and all(type(value[k]) is type(v) for k,v in expected.items())


def wire_errors_allowed(event):
    if event.get('complete') is not True: return False
    for key in ('stream_error','close_error'):
        value = event[key]
        if value is not None and (set(value) != {'type','message'} or
            value['type'] not in ('GeneratorExit','CancelledError')): return False
    return True


class Observer:
    """Explicit process-local qualification scope; restores every patched seam."""
    def __init__(self, server, core_class, scheduler_class, route_read, emit, concurrency, identity_read=loaded_identity):
        self.server, self.core_class, self.scheduler_class = server, core_class, scheduler_class
        self.route_read, self.emit, self.C = route_read, emit, concurrency
        self.identity_read = identity_read
        self.identities = {}
        self.current = ContextVar('omlx_d512_response', default=None)
        self.originals = None

    def __enter__(self):
        need(self.originals is None, 'observer cannot be installed twice')
        old_stream, old_add, old_outputs = (self.server.stream_completion,
            self.core_class.add_request, self.scheduler_class._process_batch_responses)
        self.originals = old_stream, old_add, old_outputs
        stream_signature, add_signature = inspect.signature(old_stream), inspect.signature(old_add)

        async def stream(*args, **kwargs):
            try:
                bound = stream_signature.bind(*args, **kwargs); bound.apply_defaults()
                values = bound.arguments
                request = values['request']
                engine = values['engine']
                identity = self.identities.setdefault(id(engine), self.identity_read(engine))
                need(identity == self.identity_read(engine), 'loaded engine identity changed')
                ctx = dict(response_id=values['response_id'], prompt=values['prompt'],
                           prompt_ids=list(values['prompt_token_ids']),
                           body=request.model_dump(mode='json', exclude_unset=True), loaded_identity=identity)
            except BaseException as exc:
                # An unexpected request after seal must leave evidence even if
                # its engine/body cannot be inspected. Preserve the original error.
                try:self.emit(dict(event='observer_error', phase='request-entry', type=type(exc).__name__, message=str(exc)))
                except BaseException:pass
                raise
            self.emit(dict(event='response', **ctx))
            iterator = old_stream(*args, **kwargs)
            digest, size, complete, exhausted = hashlib.sha256(), 0, False, False
            stream_error = close_error = None
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
            except BaseException as exc:
                stream_error = dict(type=type(exc).__name__, message=str(exc))
                raise
            finally:
                token = self.current.set(ctx)
                try:
                    await iterator.aclose()
                except BaseException as exc:
                    close_error = dict(type=type(exc).__name__, message=str(exc))
                    raise
                finally:
                    self.current.reset(token)
                    self.emit(dict(event='wire', response_id=ctx['response_id'],
                        complete=complete, iterator_exhausted=exhausted, stream_error=stream_error, close_error=close_error,
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
        self.identities.clear()



class Closure:
    """Server-side request closure; independent HTTP join/reap are still required.

    No CLI signal handlers are installed. SIGTERM may prevent Python close;
    a durable seal is a bounded request-set receipt, never process/EOF proof.
    """
    def __init__(self, fixtures, concurrency, model_path):
        self.C = concurrency; self.N = 8 if concurrency == 1 else 16
        jobs = jobs_for(fixtures, concurrency)
        self.allowed = Counter(json.dumps(j['body'], sort_keys=True) for j in jobs)
        self.by_prompt = {f['prompt_text']: f for f in fixtures}
        need(len(self.by_prompt) == 8, 'unique frozen prompts')
        self.model_path = model_path
        self.responses = {}; self.bindings = {}; self.inverse = {}; self.states = {}; self.wires = {}
        self.sealed = False

    def observe(self, event):
        kind = event['event']
        need(not self.sealed or kind == 'close', 'new observation after sealed request set')
        if kind == 'open' or kind == 'close': return
        if kind == 'response':
            rid = event['response_id']; key = json.dumps(event['body'], sort_keys=True)
            need(re.fullmatch(r'cmpl-[0-9a-f]{8}', rid) and rid not in self.responses
                 and self.allowed[key] > 0, 'unexpected/duplicate response or body')
            f = self.by_prompt[event['prompt']]
            need(event['prompt_ids'] == f['prompt_ids'], 'actual full prompt differs from frozen fixture')
            validate_loaded(event['loaded_identity'], self.model_path)
            self.allowed[key] -= 1; self.responses[rid] = event
        elif kind == 'binding':
            rid = event['request_id']; response = event['response_id']
            need(response in self.responses and response not in self.inverse and rid not in self.bindings,
                 'unique actual response/engine binding')
            expected = dict(max_tokens=512, temperature=0, top_p=1, top_k=0, min_p=0,
                repetition_penalty=1, presence_penalty=0, frequency_penalty=0, xtc_probability=0,
                stop=[], stop_token_ids=[], thinking_budget=None)
            need(event['sampling'] == expected and event['scheduler_id'] ==
                 self.responses[response]['loaded_identity']['scheduler_id'], 'effective sampling/actual scheduler')
            self.bindings[rid] = event; self.inverse[response] = rid
        elif kind == 'batch':
            tokens, outputs = event['tokens'], event['outputs']; rids = [x['request_id'] for x in tokens]
            need(valid_route(event['route'], self.C) and len(rids) == len(set(rids)) <= self.C
                 and [x['request_id'] for x in outputs] == rids, 'actual batch route/identity')
            for token, output in zip(tokens, outputs):
                rid = token['request_id']; raw = self.states.setdefault(rid, dict(raw_ids=[], finished=False,
                    prompt_ids=token['prompt_ids'], scheduler_id=event['scheduler_id']))
                need(not raw['finished'] and token['prompt_ids'] == raw['prompt_ids'] and
                     event['scheduler_id'] == raw['scheduler_id'] and token['output_parser'] is False,
                     'actual token route/prompt/terminal')
                need(type(token['token']) is int and 0 <= token['token'] < 248320, 'actual token ID')
                raw['raw_ids'].append(token['token']); finish = token['finish_reason']
                committed = raw['raw_ids'][:-1] if finish == 'stop' else raw['raw_ids']
                need(len(raw['raw_ids']) <= 512 and output['error'] is None
                     and type(output['cached_tokens']) is int and output['cached_tokens'] == 0
                     and type(output['prompt_tokens']) is int and output['prompt_tokens'] == len(raw['prompt_ids'])
                     and type(output['completion_tokens']) is int and output['completion_tokens'] == len(committed)
                     and type(output['finished']) is bool and output['finish_reason'] == finish
                     and output['finished'] == (finish is not None), 'raw/committed/cache/error counts')
                if output['finished']:
                    need(finish in ('stop','length') and output['committed_ids'] == committed, 'terminal committed IDs')
                    raw.update(finished=True, finish_reason=finish, committed_ids=list(committed))
                else: need(output['committed_ids'] is None, 'early committed sequence')
        elif kind == 'wire':
            response = event['response_id']
            need(response in self.responses and response not in self.wires and wire_errors_allowed(event)
                 and type(event['bytes']) is int and event['bytes'] > 0
                 and re.fullmatch('[0-9a-f]{64}', event['sha256']), 'actual complete DONE stream')
            self.wires[response] = event
        else: raise ValueError('unknown observation for seal')

    def completed(self):
        if len(self.responses) != self.N or len(self.bindings) != self.N or len(self.wires) != self.N:
            return False
        need(set(self.responses) == set(self.wires) == set(self.inverse) and
             set(self.states) == set(self.bindings) and not any(self.allowed.values()), 'complete unique request set')
        for rid, binding in self.bindings.items():
            state = self.states[rid]; response = self.responses[binding['response_id']]
            f = self.by_prompt[response['prompt']]; raw = state['raw_ids']; eos = f['eos_token_ids']
            need(state['finished'] and raw and state['prompt_ids'] == f['prompt_ids'] and
                 state['scheduler_id'] == binding['scheduler_id'] and not any(t in eos for t in raw[:-1]),
                 'actual finished normal-EOS request/prompt')
            stop = raw[-1] in eos
            need((stop or len(raw) == 512) and state['finish_reason'] == ('stop' if stop else 'length')
                 and state['committed_ids'] == (raw[:-1] if stop else raw), 'actual EOS/cap sequence')
        return True

    def receipt(self, prefix_sha256, prefix_lines):
        need(not self.sealed and self.completed(), 'cannot seal an incomplete request set')
        self.sealed = True
        return dict(event='seal', schema='E023-omlx-D512-request-seal-v1', server_side_qualified=True,
            request_count=self.N, response_ids=sorted(self.responses), engine_request_ids=sorted(self.bindings),
            scheduler_ids=sorted({x['scheduler_id'] for x in self.bindings.values()}),
            raw_output_count=sum(len(x['raw_ids']) for x in self.states.values()),
            committed_output_count=sum(len(x['committed_ids']) for x in self.states.values()),
            prefix_sha256=prefix_sha256, prefix_lines=prefix_lines,
            process_close_proven=False, generator_eof_required=False, http_client_verified=False)


class Ledger:
    def __init__(self, stream, closure):
        self.stream, self.closure = stream, closure
        self.lock = threading.Lock(); self.digest = hashlib.sha256(); self.lines = 0

    def _write(self, event, durable=False):
        raw = (json.dumps(event, ensure_ascii=False, allow_nan=False)+'\n').encode()
        self.stream.write(raw); self.stream.flush(); self.digest.update(raw); self.lines += 1
        if durable: os.fsync(self.stream.fileno())

    def __call__(self, event):
        with self.lock:
            self._write(event)
            try:
                self.closure.observe(event)
                if not self.closure.sealed and self.closure.completed():
                    self._write(self.closure.receipt(self.digest.hexdigest(), self.lines), durable=True)
            except BaseException as exc:
                self._write(dict(event='violation', type=type(exc).__name__, message=str(exc)), durable=True)
                raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--concurrency', type=int, choices=(1, 8), required=True)
    parser.add_argument('official_args', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    need(args.run, '--run required; this entrypoint starts the official server')
    command = args.official_args
    if command[:1] == ['--']: command = command[1:]
    need(command[:1] == ['serve'], 'original official serve arguments required')
    verify_sources(); need(sha(FIXTURES) == FIXTURES_SHA, 'fixed full D fixtures')
    fixtures = json.loads(FIXTURES.read_text())['fixtures']
    args.out.mkdir(parents=True, exist_ok=False)
    modules = {n: importlib.import_module('omlx.' + n) for n in
               ('server', 'cli', 'scheduler', 'engine_core', 'engine.vlm', 'patches.mlx_lm_mtp.batch_generator')}
    for name, module in modules.items():
        need(Path(module.__file__).resolve() == ROOT / (name.replace('.', '/') + '.py'), 'loaded official path mismatch')
    runtime = runtime_identity()
    with (args.out / 'observer.jsonl').open('xb') as stream:
        emit = Ledger(stream, Closure(fixtures, args.concurrency, expected_model()))
        emit(dict(event='open', schema='E023-omlx-D512-observer-v2', concurrency=args.concurrency,
            source_root=str(ROOT), sources=PINS, probe_sha256=sha(__file__), fixture_sha256=FIXTURES_SHA,
            model_path=expected_model(), runtime=runtime, official_args=command,
            performance_qualified=False))
        original_argv = sys.argv; ok = False
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
