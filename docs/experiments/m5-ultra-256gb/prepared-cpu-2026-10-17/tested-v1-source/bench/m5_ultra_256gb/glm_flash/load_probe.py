"""Official GLM-5.3-Flash EnginePool load / four short plain completions.

CPU-safe import/help; only --run imports oMLX/MLX. This is NOT an HTTP benchmark,
chat-quality test, reference comparison, or permission to fit 190 GiB of weights.
An external cold/resource owner must admit this exact plan, own this direct child,
monitor physical memory, and hard TERM/KILL/wait it if cooperative deadlines fail.

Run: <pinned-python> -I -B load_probe.py --run --plan ABS --plan-sha256 SHA
     --admission ABS --admission-sha256 SHA --out NEW

Plan (all paths absolute): schema=GLM-official-load-plan-v1, snapshot,
prompt={path,sha256}, download_receipt={path,sha256},
header_inventory={path,sha256}, source_pins={absolute_small_source:sha256},
environment=ENVIRONMENT, deadlines={work_seconds,request_seconds,shutdown_seconds}.
source_pins must include every path returned by contract(). Full model/runtime
verification belongs to the owner's pinned receipts, not this small-source check.

Admission: schema=GLM-official-load-admission-v1, owner_pid, plan_sha256,
probe_sha256, output_directory, snapshot, issued_at, expires_at,
cold_service_absent=true, exclusive_resources=true, hard_reap_owner=true,
model_runtime_verified=true, evidence={cold_release,resources,model_runtime},
where each evidence entry is a {path,sha256} ref. Those receipts are retained and
bound here; their safety predicates must be independently checked by the owner.
No service mutation or resource-policy relaxation is performed by this script.
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import functools
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
import traceback
from unittest.mock import patch

REPO = Path('/Users/ryanli/code/minivllm')
HERE = REPO / 'bench/m5_ultra_256gb/glm_flash'
VENV = REPO / '.artifacts/omlx-baseline/.venv'
SITE = VENV / 'lib/python3.13/site-packages'
VENDOR = SITE / 'omlx/patches/mlx_vlm_glm5_next_compat/vendor/mlx_vlm/models/glm5_next'
REVISION = 'fa04cbc3622b16aa7475404060b05e506d81bd4e'
SNAPSHOT = REPO / '.artifacts/models/huggingface/models--mlx-community--GLM-5.3-Flash-4bit/snapshots' / REVISION
MODEL_ID = 'GLM-5.3-Flash-4bit'
EOS = [154820, 154827, 154829]
CAPS = [8, 32, 32, 32]
EXPECTED = {'omlx': '0.7.1.dev1', 'mlx': '0.32.3', 'mlx-metal': '0.32.3',
            'mlx-vlm': '0.7.7', 'mlx-lm': '0.31.4.dev132+g94cdcae13'}
ENVIRONMENT = dict(MLX_MAX_OPS_PER_BUFFER='50', MLX_MAX_MB_PER_BUFFER='50',
    MLX_BFS_MAX_WIDTH='20', MLX_METAL_FAST_SYNCH='0', HF_HUB_OFFLINE='1',
    TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false')
FIXED_RECEIPTS = {
    'download_receipt': ('complete.json', 'f42666fd0b18adb07a25c3e07c4e32f5bb374529059cf38520e52618311f8363'),
    'header_inventory': ('header-inventory.json', '6bcfc482687d61746d7e6fac5768c4bba27b8d006631dace17992950637497f3')}
REQUIRED_SOURCES = [Path(__file__).resolve(), HERE / 'metadata-v1/manifest.json',
    HERE / 'metadata-v1/config.json', *[SITE / p for p in (
        'omlx/engine_pool.py', 'omlx/engine/vlm.py', 'omlx/engine_core.py',
        'omlx/models/vlm.py', 'omlx/model_settings.py', 'omlx/settings.py',
        'omlx/request.py', 'omlx/scheduler.py', 'omlx/api/utils.py',
        'omlx/utils/model_loading.py', 'omlx/patches/moe_gate_up_fusion.py',
        'omlx/patches/mlx_lm_mtp/batch_generator.py',
        'omlx/patches/deepseek_v4/switch_layers.py',
        'omlx/patches/mlx_vlm_glm5_next_compat/__init__.py', 'mlx_vlm/utils.py')],
    *[VENDOR / p for p in ('__init__.py', 'config.py', 'glm5_next.py', 'language.py', 'vision.py')]]
SAMPLING = dict(temperature=0.0, top_p=1.0, top_k=0, min_p=0.0,
    repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0,
    xtc_probability=0.0, stop=[], stop_token_ids=[], thinking_budget=None)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def small_bytes(path, limit=64 * 1024**2):
    path = Path(path)
    need(path.is_absolute() and path.is_file() and path.stat().st_size <= limit,
         'missing/nonabsolute/oversized small evidence: ' + str(path))
    return path.read_bytes()


def sha(path):
    return hashlib.sha256(small_bytes(path)).hexdigest()


def read_ref(ref):
    need(set(ref) == {'path', 'sha256'}, 'expected exact path/SHA reference')
    raw = small_bytes(ref['path'])
    need(hashlib.sha256(raw).hexdigest() == ref['sha256'], 'reference SHA changed: ' + ref['path'])
    return raw


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def contract():
    return dict(schema='GLM-official-load-contract-v1', snapshot=str(SNAPSHOT),
        python=str(VENV / 'bin/python'), versions=EXPECTED, model_id=MODEL_ID,
        concurrency=1, context_request_bound=2048, prompt_token_bound=128,
        completion_caps=CAPS, normal_eos=EOS, prompt_mode='plain-completion-no-chat-template',
        required_source_paths=[str(p) for p in REQUIRED_SOURCES], environment=ENVIRONMENT,
        fixed_evidence={key: dict(path=str(REPO / '.artifacts/m5-ultra-256gb/glm-flash-download-v1' / filename),
                                 sha256=digest) for key, (filename, digest) in FIXED_RECEIPTS.items()},
        source_checks='small listed files only; complete runtime/model pins owned externally',
        allocator_policy='official EnginePool defaults; no allocator-limit override',
        reset_scope='distinct requests, drained queues/UID maps, prefix cache disabled; no tensor-state equivalence',
        admission_scope='bound external assertion, not independently established safety',
        performance_qualified=False, full_model_correctness_qualified=False)


def source_check(plan):
    pins = plan['source_pins']
    need(type(pins) is dict and set(map(str, REQUIRED_SOURCES)) <= set(pins), 'missing required source pin')
    for path, digest in pins.items():
        need(sha(path) == digest, 'source changed: ' + path)
    return pins


def prepare(args):
    plan_raw = read_ref(dict(path=str(args.plan), sha256=args.plan_sha256))
    plan = json.loads(plan_raw)
    need(plan['schema'] == 'GLM-official-load-plan-v1' and plan['snapshot'] == str(SNAPSHOT), 'fixed GLM plan')
    need(plan['environment'] == ENVIRONMENT, 'fixed offline/runtime environment')
    deadlines = plan['deadlines']
    need(set(deadlines) == {'work_seconds', 'request_seconds', 'shutdown_seconds'}, 'deadline fields')
    for key, ceiling in (('work_seconds', 1800), ('request_seconds', 180), ('shutdown_seconds', 60)):
        need(type(deadlines[key]) in (int, float) and 0 < deadlines[key] <= ceiling, 'invalid ' + key)
    need(deadlines['request_seconds'] <= deadlines['work_seconds'], 'request exceeds work budget')
    source_check(plan)
    receipts = {}
    for key, (filename, digest) in FIXED_RECEIPTS.items():
        expected_path = REPO / '.artifacts/m5-ultra-256gb/glm-flash-download-v1' / filename
        need(plan[key] == dict(path=str(expected_path), sha256=digest), 'fixed ' + key)
        receipts[key] = json.loads(read_ref(plan[key]))
        need(receipts[key]['snapshot'] == str(SNAPSHOT), 'receipt snapshot mismatch')
    need(receipts['download_receipt']['status'] == 'PASS_PINNED_DOWNLOAD_AND_SHA_ONLY', 'incomplete download')
    need(receipts['header_inventory']['status'] == 'PASS_HEADER_INVENTORY_ONLY', 'incomplete inventory')
    prompt_raw = read_ref(plan['prompt'])
    need(0 < len(prompt_raw) <= 4096 and b'\x00' not in prompt_raw, 'bounded nonempty UTF8 prompt')
    prompt = prompt_raw.decode('utf-8', errors='strict')
    admission_raw = read_ref(dict(path=str(args.admission), sha256=args.admission_sha256))
    admission = json.loads(admission_raw)
    need(admission['schema'] == 'GLM-official-load-admission-v1', 'external admission schema')
    need(type(admission['owner_pid']) is int and admission['owner_pid'] == os.getppid() > 1, 'direct owner PPID binding')
    need(admission['plan_sha256'] == args.plan_sha256 and admission['probe_sha256'] == sha(__file__), 'admission source/plan binding')
    need(admission['output_directory'] == str(args.out) and admission['snapshot'] == str(SNAPSHOT), 'admission output/model binding')
    now = time.time()
    need(all(type(admission[k]) in (int, float) for k in ('issued_at', 'expires_at')) and
         admission['issued_at'] <= now < admission['expires_at'] <= now + 3600, 'stale/unbounded admission')
    for key in ('cold_service_absent', 'exclusive_resources', 'hard_reap_owner', 'model_runtime_verified'):
        need(admission[key] is True, 'required external admission assertion: ' + key)
    need(set(admission['evidence']) == {'cold_release', 'resources', 'model_runtime'}, 'external evidence refs')
    evidence = {key: read_ref(ref) for key, ref in admission['evidence'].items()}
    need(Path(sys.prefix).absolute() == VENV and sys.flags.isolated and sys.dont_write_bytecode,
         'use fixed venv Python with -I -B; runtime switching forbidden')
    versions = {name: importlib.metadata.version(name) for name in EXPECTED}
    need(versions == EXPECTED, 'installed package version mismatch')
    need(not any(n in ('mlx', 'omlx') or n.startswith(('mlx.', 'omlx.')) for n in sys.modules), 'fresh runtime import required')
    need(not any(k.startswith('OMLX_') for k in os.environ), 'unexpected inherited OMLX environment (values not read)')
    for key, value in ENVIRONMENT.items():
        need(os.environ.get(key) == value, 'owner must supply fixed ' + key)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'plan.json').write_bytes(plan_raw)
    (args.out / 'admission.json').write_bytes(admission_raw)
    (args.out / 'prompt.utf8').write_bytes(prompt_raw)
    for key, raw in evidence.items():
        (args.out / ('admission-' + key + '.json')).write_bytes(raw)
    write(args.out / 'contract.json', contract())
    write(args.out / 'source-pins-before.json', source_check(plan))
    return plan, admission, prompt


def model_metadata():
    """Small metadata content and shard stat only; never hash weight payload here."""
    manifest = json.loads((HERE / 'metadata-v1/manifest.json').read_text())
    need(manifest['revision'] == REVISION and len(manifest['shards']) == 62, 'fixed model manifest')
    result = {'metadata': {}, 'shard_stats': {}, 'weight_payload_SHA_rechecked': False}
    for name, entry in manifest['metadata'].items():
        raw = small_bytes(SNAPSHOT / name)
        need(len(raw) == entry['bytes'] and hashlib.sha256(raw).hexdigest() == entry['sha256'], 'model metadata changed: ' + name)
        result['metadata'][name] = entry
    for item in manifest['shards']:
        st = (SNAPSHOT / item['name']).stat()
        need(st.st_size == item['bytes'], 'model shard size changed')
        result['shard_stats'][item['name']] = dict(bytes=st.st_size, mtime_ns=st.st_mtime_ns,
                                                 inode=st.st_ino, device=st.st_dev)
    need(not (SNAPSHOT / 'offload_index.json').exists(), 'offload snapshot forbidden')
    return result


class Ledger:
    def __init__(self, out, mx):
        self.out, self.mx = out, mx
        self.lock = threading.Lock()
        self.events = []

    def emit(self, kind, **values):
        item = dict(kind=kind, time=time.time(), monotonic=time.monotonic(),
                    pid=os.getpid(), thread=threading.get_ident(), **values)
        with self.lock:
            self.events.append(item)
            with (self.out / 'events.jsonl').open('a') as f:
                f.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n')
        return item

    def memory(self, phase, **values):
        return self.emit('memory', phase=phase, active_bytes=self.mx.get_active_memory(),
            cache_bytes=self.mx.get_cache_memory(), peak_bytes=self.mx.get_peak_memory(),
            memory_limit_bytes=self.mx.get_memory_limit(), **values)


def actual_route(scheduler, mtp):
    return dict(max_num_seqs=scheduler.config.max_num_seqs,
        vlm_mtp_drafter=scheduler._vlm_mtp_drafter is not None,
        vlm_mtp_active=bool(scheduler._vlm_mtp_active),
        block_drafter=mtp._drafter_for(scheduler.model) is not None,
        embedded_mtp=bool(mtp._model_mtp_decode_enabled(scheduler.model)),
        specprefill=scheduler._specprefill_draft_model is not None,
        turboquant=scheduler._turboquant_kv_bits is not None,
        moe_offload=bool(scheduler.config.moe_offload_active),
        prefix_cache=scheduler.block_aware_cache is not None,
        output_parser=scheduler._output_parser_factory is not None)


def require_target_route(route):
    need(route == dict(max_num_seqs=1, **{k: False for k in (
        'vlm_mtp_drafter', 'vlm_mtp_active', 'block_drafter', 'embedded_mtp',
        'specprefill', 'turboquant', 'moe_offload', 'prefix_cache', 'output_parser')}),
        'unsupported non-ordinary target route; preserve evidence, do not fall back')


def state_counts(scheduler):
    names = ('running', 'waiting', 'prefilling', 'uid_to_request_id', 'request_id_to_uid',
             '_pending_async_removes', '_inflight_store_futures')
    return {name: len(getattr(scheduler, name)) for name in names}


def source_of(obj, plan):
    path = Path(inspect.getsourcefile(obj)).resolve()
    need(plan['source_pins'].get(str(path)) == sha(path), 'loaded unpinned source: ' + str(path))
    return dict(file=str(path), sha256=sha(path), module=obj.__module__, name=obj.__qualname__)


class Observer:
    """Scoped call-through hooks: ordinary Python metadata only, no tensor eval."""
    def __init__(self, ledger, plan, mtp):
        self.log, self.plan, self.mtp = ledger, plan, mtp
        self.patches = []
        self.case = None
        self.lock = threading.Lock()
        self.active = {}

    def active_calls(self):
        with self.lock:
            return {name: count for name, count in self.active.items() if count}

    def hook(self, module, name, factory):
        original = getattr(module, name)
        descriptor = inspect.getattr_static(module, name)
        # All four pinned seams are plain functions (one is an instance method
        # accessed on its class).
        # Reject staticmethod/classmethod/property descriptors instead of losing
        # their binding behavior through a generic setattr wrapper.
        need(inspect.isfunction(descriptor) and original is descriptor, 'unsupported seam descriptor')
        self.log.emit('seam', name=name, source=source_of(original, self.plan))
        observed = factory(original)
        @functools.wraps(original)
        def replacement(*args, **kwargs):
            with self.lock:
                self.active[name] = self.active.get(name, 0) + 1
            try:
                return observed(*args, **kwargs)
            finally:
                with self.lock:
                    self.active[name] -= 1
        item = patch.object(module, name, replacement)
        item.start(); self.patches.append((module, name, descriptor, item))

    def install(self, utils, loading, fusion, scheduler_class):
        def boundary(label):
            def factory(original):
                def call(*args, **kwargs):
                    extra = (dict(path=str(Path(args[0]).resolve()), lazy=kwargs.get('lazy', False),
                                  trust_remote_code=kwargs.get('trust_remote_code', False))
                             if label == 'official_vlm_load' else {})
                    self.log.memory(label + ':before', **extra)
                    result = original(*args, **kwargs)
                    self.log.memory(label + ':after', **extra,
                                    **({'fused_layers': result} if label == 'gate_up_fusion' else {}))
                    return result
                return call
            return factory
        self.hook(utils, 'load', boundary('official_vlm_load'))
        self.hook(loading, 'materialize_lazy_state', boundary('materialize_lazy_state'))
        self.hook(fusion, 'apply_moe_gate_up_fusion', boundary('gate_up_fusion'))

        def outputs(original):
            def call(scheduler, responses):
                case = self.case
                route = actual_route(scheduler, self.mtp)
                before = []
                for response in responses:
                    rid = scheduler.uid_to_request_id.get(response.uid)
                    need(rid in scheduler.running and case is not None, 'unbound response')
                    need(type(response.token) is int, 'observer requires already-materialized Python token IDs')
                    request = scheduler.running[rid]
                    before.append(dict(request_id=rid, token=response.token,
                        finish_reason=response.finish_reason, prompt_ids=list(request.prompt_token_ids),
                        sampling={k: getattr(request.sampling_params, k) for k in (*SAMPLING, 'max_tokens')}))
                self.log.emit('batch_before', case=case, route=route, responses=before)
                require_target_route(route)
                result = original(scheduler, responses)
                final = [dict(request_id=o.request_id, committed_ids=list(o.output_token_ids),
                    text=o.output_text, finish_reason=o.finish_reason, prompt_tokens=o.prompt_tokens,
                    completion_tokens=o.completion_tokens, cached_tokens=o.cached_tokens, error=o.error)
                    for o in result[0] if o.finished]
                self.log.emit('batch_after', case=case, finals=final)
                return result
            return call
        self.hook(scheduler_class, '_process_batch_responses', outputs)

    def close(self):
        active = self.active_calls()
        restored, errors = [], []
        for module, name, original, item in reversed(self.patches):
            try:
                item.stop()
                need(inspect.getattr_static(module, name) is original, 'seam restore mismatch: ' + name)
                restored.append(name)
            except BaseException as exc:
                errors.append(dict(name=name, type=type(exc).__name__, message=str(exc)))
        self.patches.clear()
        return dict(complete=not errors, restored=restored, errors=errors,
                    active_at_restore=active,
                    scope='attribute identities restored; active old wrapper calls may still require outer hard reap')


def shutdown_is_complete(*, returned, retained, engine_cleared, load_attempted, load_returned,
                         active_calls, hooks_restored):
    # Cancelled get_engine can clear entry.is_loading in its finally while its
    # run_in_executor loader still runs. Empty entries alone cannot close it.
    return bool(returned and not retained and engine_cleared and not active_calls and
                hooks_restored and (not load_attempted or load_returned))


def request_receipt(log, index, cap, prompt_ids, result, tokenizer, clean):
    before = [row for e in log.events if e['kind'] == 'batch_before' and e['case'] == index for row in e['responses']]
    final = [row for e in log.events if e['kind'] == 'batch_after' and e['case'] == index for row in e['finals']]
    need(len(final) == 1 and before, 'one nonempty final request required')
    output = final[0]
    need(all(r['request_id'] == output['request_id'] and r['prompt_ids'] == prompt_ids and
             r['sampling'] == dict(SAMPLING, max_tokens=cap) for r in before), 'request binding/prompt/sampling changed')
    raw = [r['token'] for r in before]
    need(all(r['finish_reason'] is None for r in before[:-1]), 'early/duplicate finish')
    need(output['finish_reason'] == before[-1]['finish_reason'] in ('stop', 'length'), 'unsupported finish')
    committed = output['committed_ids']
    if output['finish_reason'] == 'stop':
        need(raw[-1] in EOS and not any(t in EOS for t in raw[:-1]) and committed == raw[:-1], 'EOS exclusion mismatch')
        need(len(raw) <= cap, 'EOS request exceeded cap')
    else:
        need(len(raw) == cap and committed == raw and not any(t in EOS for t in raw), 'length/count mismatch')
    need(output['error'] is None and output['cached_tokens'] == result.cached_tokens == 0, 'error/cache reuse')
    need(output['prompt_tokens'] == result.prompt_tokens == len(prompt_ids) and
         output['completion_tokens'] == result.completion_tokens == len(committed), 'usage mismatch')
    need(result.finish_reason == output['finish_reason'], 'engine/scheduler finish differs')
    decoded = tokenizer.decode(committed)
    need(output['text'] == decoded and result.text == clean(decoded), 'ordinary decode/official cleanup mismatch')
    return dict(index=index, cap=cap, prompt_ids=prompt_ids, raw_ids=raw, committed_ids=committed,
        request_id=output['request_id'], scheduler_output=output, tokenizer_decoded_text=decoded,
        engine_text=result.text, official_text_transform='omlx.api.utils.clean_special_tokens',
        usage_rule='normal EOS excluded from committed output/completion_tokens',
        performance_qualified=False)


async def probe(args, plan, admission, prompt):
    # This function is reached only after explicit --run and all external bindings.
    start = time.monotonic()
    state = args.out / 'state'; state.mkdir()
    models = args.out / 'models'; models.mkdir()
    (models / MODEL_ID).symlink_to(SNAPSHOT)
    os.environ.update(OMLX_BASE_PATH=str(state), HF_HOME=str(state / 'hf'),
                      HUGGINGFACE_HUB_CACHE=str(state / 'hf/hub'))
    write(state / 'settings.json', dict(version='1.0', cache={'enabled': False},
          server={'burst_decode_mode': 'balanced'}))
    import mlx.core as mx
    from mlx_vlm import utils
    from omlx.engine_pool import EnginePool
    from omlx.model_settings import ModelSettings
    from omlx.scheduler import Scheduler, SchedulerConfig
    from omlx.settings import init_settings, burst_decode_env
    from omlx.utils import model_loading
    from omlx.patches import moe_gate_up_fusion
    from omlx.patches.mlx_lm_mtp import batch_generator as mtp
    from omlx.api.utils import clean_special_tokens

    settings = init_settings(base_path=state)
    need(settings.base_path == state and settings.cache.enabled is False, 'isolated cache settings not effective')
    os.environ.update(burst_decode_env('balanced'))
    log = Ledger(args.out, mx)
    observer = Observer(log, plan, mtp)
    settings = ModelSettings(max_context_window=2048, vlm_mtp_enabled=False, mtp_enabled=False,
        dflash_enabled=False, specprefill_enabled=False, turboquant_kv_enabled=False,
        moe_expert_offload_enabled=False, trust_remote_code=False)
    config = SchedulerConfig(max_num_seqs=1, completion_batch_size=1, max_num_batched_tokens=2048,
        prefill_step_size=2048, chunked_prefill=False, prefill_speed_priority=True,
        decode_fairness=True, paged_ssd_cache_dir=None, hot_cache_only=False, hot_cache_max_size=0)
    config_record = dataclasses.asdict(config); config_record['policy'] = config.policy.value
    write(args.out / 'settings.json', dict(model=settings.to_dict(), scheduler=config_record,
        context_enforcement='probe checks prompt+cap<=2048; direct API has no HTTP context admission',
        environment={**ENVIRONMENT, **burst_decode_env('balanced')},
        memory_enforcer='external owner only; EnginePool server callbacks are not installed'))
    pool, engine, scheduler, tokenizer, engine_executor = None, None, None, None, None
    model = layers = switches = None
    load_attempted = load_returned = False
    completed, error, shutdown = [], None, dict(attempted=False, complete=False)
    model_before = model_metadata()
    write(args.out / 'model-metadata-before.json', model_before)
    try:
        remaining = plan['deadlines']['work_seconds'] - (time.monotonic() - start)
        need(remaining > 0, 'work deadline exhausted before pool creation')
        async with asyncio.timeout(remaining):
            observer.install(utils, model_loading, moe_gate_up_fusion, Scheduler)
            log.memory('before_pool_load')
            pool = EnginePool(scheduler_config=config)
            pool.discover_models(str(models))
            discovered = {k: {'path': str(e.model_path), 'engine_type': e.engine_type} for k, e in pool._entries.items()}
            write(args.out / 'discovery.json', discovered)
            need(set(discovered) == {MODEL_ID} and Path(discovered[MODEL_ID]['path']).resolve() == SNAPSHOT, 'unexpected discovered model')
            load_attempted = True
            engine = await pool.get_engine(MODEL_ID, runtime_settings=settings)
            load_returned = True
            log.memory('after_pool_load')
            need(type(engine).__name__ == 'VLMBatchedEngine' and engine.model_type == 'glm5_next', 'wrong actual engine/model')
            scheduler = engine._engine.engine.scheduler
            engine_executor = engine._engine.engine._mlx_executor
            tokenizer = engine.tokenizer
            require_target_route(actual_route(scheduler, mtp))
            need(engine.prefix_cache_enabled is False and sorted(scheduler._get_stop_tokens()) == EOS, 'cache/EOS contract')
            need(engine._moe_offload_wrapped == 0 and engine._model_settings.to_dict() == settings.to_dict(), 'runtime setting fallback')
            model = engine._vlm_model
            layers = model.language_model.layers
            switches = [layer.mlp.switch_mlp for layer in layers if hasattr(layer.mlp, 'switch_mlp')]
            loaded = dict(engine=source_of(type(engine), plan), wrapper=source_of(type(model), plan),
                language=source_of(type(model.language_model), plan), adapter=source_of(type(scheduler.model), plan),
                switch=source_of(type(switches[0]), plan) if switches else None,
                model_type=engine.model_type, layers=len(layers), kda_layers=sum(bool(l.is_linear) for l in layers),
                moe_layers=len(switches), fused_gate_up_layers=sum(hasattr(s, 'gate_up_proj') for s in switches),
                stop_token_ids=sorted(scheduler._get_stop_tokens()), route=actual_route(scheduler, mtp),
                quantized_gate_up=[dict(bits=s.gate_up_proj.bits, group_size=s.gate_up_proj.group_size,
                    mode=s.gate_up_proj.mode, weight_shape=list(s.gate_up_proj.weight.shape),
                    scales_shape=list(s.gate_up_proj.scales.shape)) for s in switches if hasattr(s, 'gate_up_proj')])
            write(args.out / 'loaded-engine.json', loaded)
            need(loaded['wrapper']['file'] == str(VENDOR / 'glm5_next.py') and
                 loaded['language']['file'] == str(VENDOR / 'language.py'), 'upstream/vendor mismatch')
            need((loaded['layers'], loaded['kda_layers'], loaded['moe_layers'], loaded['fused_gate_up_layers']) == (45, 34, 42, 42), 'unexpected GLM structure/fusion fallback')
            load_events = [e for e in log.events if e['kind'] == 'memory' and e['phase'] == 'official_vlm_load:after']
            need(len(load_events) == 1 and load_events[0]['path'] == str(SNAPSHOT) and
                 load_events[0]['lazy'] is False and load_events[0]['trust_remote_code'] is False, 'ordinary official loader not witnessed')
            need(any(e['kind'] == 'memory' and e['phase'] == 'gate_up_fusion:after' and e['fused_layers'] == 42 for e in log.events), 'missing official fusion witness')
            # No model/array references retained by evidence or across shutdown.
            model = layers = switches = None
            prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
            need(0 < len(prompt_ids) <= 128 and all(type(t) is int and t >= 0 for t in prompt_ids), 'prompt token bound/type')
            write(args.out / 'input.json', dict(prompt=prompt, prompt_sha256=plan['prompt']['sha256'],
                prompt_ids=prompt_ids, tokenization='encode(add_special_tokens=False); no chat template', normal_eos=EOS))
            loop = asyncio.get_running_loop()
            for index, cap in enumerate(CAPS):
                need(len(prompt_ids) + cap <= 2048, 'context request bound')
                empty = await loop.run_in_executor(engine_executor, state_counts, scheduler)
                log.emit('reset_before', case=index, counts=empty)
                need(not any(empty.values()), 'previous request not fully drained')
                observer.case = index
                log.memory('request_before', case=index, cap=cap)
                result = await asyncio.wait_for(engine.generate(prompt=list(prompt_ids), max_tokens=cap,
                    temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, repetition_penalty=1.0,
                    presence_penalty=0.0, frequency_penalty=0.0, xtc_probability=0.0,
                    stop=[], thinking_budget=None), timeout=plan['deadlines']['request_seconds'])
                row = request_receipt(log, index, cap, prompt_ids, result, tokenizer, clean_special_tokens)
                need(row['request_id'] not in {r['request_id'] for r in completed}, 'request ID reused')
                drained = await loop.run_in_executor(engine_executor, state_counts, scheduler)
                row['reset_after'] = drained
                write(args.out / f'request-{index}.json', row)
                need(not any(drained.values()), 'finished request still owns scheduler slots')
                require_target_route(actual_route(scheduler, mtp))
                completed.append(row)
                log.memory('request_after', case=index, cap=cap)
                observer.case = None
            need(completed[1]['raw_ids'] == completed[2]['raw_ids'] == completed[3]['raw_ids'], 'repeated greedy requests differ')
    except BaseException as exc:
        error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        write(args.out / 'primary-error.json', error)
    finally:
        # async cancellation cannot preempt blocked C++/driver/executor teardown;
        # only the external owner's hard deadline and wait/reap close that gap.
        # Keep the request label until hooks are removed: a cancelled generation
        # may still deliver a batch on the engine executor while stop() drains it.
        scheduler = tokenizer = model = layers = switches = engine_executor = None
        shutdown.update(load_attempted=load_attempted, load_await_returned=load_returned,
                        returned=False, retained_entries=[], engine_references_cleared=engine is None)
        try:
            if pool is not None:
                shutdown['attempted'] = True
                log.memory('before_pool_shutdown')
                await asyncio.wait_for(pool.shutdown(), plan['deadlines']['shutdown_seconds'])
                shutdown['returned'] = True
                shutdown['retained_entries'] = [k for k, e in pool._entries.items() if e.engine is not None or e.is_loading]
                shutdown['engine_references_cleared'] = engine is None or (
                    not engine._loaded and engine._engine is None and engine._vlm_model is None and engine._vlm_mtp_drafter is None)
                need(not shutdown['retained_entries'] and shutdown['engine_references_cleared'], 'pool teardown retained engine')
                log.memory('after_pool_shutdown')
        except BaseException as exc:
            shutdown['error'] = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        finally:
            shutdown['hook_restoration'] = observer.close()
            observer.case = None
            shutdown['active_observed_calls'] = observer.active_calls()
            shutdown['complete'] = shutdown_is_complete(returned=shutdown['returned'] and 'error' not in shutdown,
                retained=shutdown['retained_entries'], engine_cleared=shutdown['engine_references_cleared'],
                load_attempted=load_attempted, load_returned=load_returned,
                active_calls=shutdown['active_observed_calls'] or shutdown['hook_restoration']['active_at_restore'],
                hooks_restored=shutdown['hook_restoration']['complete'])
            shutdown['runtime_work_unsettled'] = not shutdown['complete']
            shutdown['outer_wait_reap_required'] = True
            engine = pool = None
            write(args.out / 'shutdown.json', shutdown)
    post_error = None
    try:
        need(shutdown['complete'], 'runtime work unsettled: external owner must hard-reap before restore')
        need(os.getppid() == admission['owner_pid'], 'external owner identity lost')
        source_check(plan)
        write(args.out / 'source-pins-after.json', plan['source_pins'])
        after = model_metadata()
        write(args.out / 'model-metadata-after.json', after)
        need(after == model_before, 'metadata/shard stat changed')
    except BaseException as exc:
        post_error = dict(type=type(exc).__name__, message=str(exc))
    passed = error is None and post_error is None and shutdown['complete'] and len(completed) == 4
    final = dict(schema='GLM-official-load-result-v1', status='PASS' if passed else 'FAIL',
        pid=os.getpid(), owner_pid=admission['owner_pid'], plan_sha256=args.plan_sha256,
        admission_sha256=args.admission_sha256, probe_sha256=sha(__file__),
        versions=EXPECTED, snapshot=str(SNAPSHOT), completed_requests=completed,
        short_lifecycle_diagnostic_complete=passed, performance_qualified=False,
        full_model_correctness_qualified=False, chat_quality_qualified=False,
        physical_resource_safety_qualified=False, admission_scope='external receipt assertions only',
        reset_scope=contract()['reset_scope'], shutdown=shutdown, error=error, postcheck_error=post_error,
        elapsed_wall_seconds=time.monotonic() - start, timing_scope='instrumented diagnostic, not performance',
        limits=['No independent logits/cache/reference comparison.',
                'Scalar MLX boundaries are observations, not physical peak or hard memory caps.',
                'No whole-weight content rehash here; external owner must verify its model/runtime pins.',
                'Source-pinned ordinary load proves a route, not every intermediate sanitize operation.',
                'Cooperative async deadlines require an external hard child deadline and reap.'])
    write(args.out / 'complete.json', final)
    return 0 if passed else 1


async def run_cancellable(args, plan, admission, prompt):
    loop = asyncio.get_running_loop()
    task = asyncio.create_task(probe(args, plan, admission, prompt))
    cancelled = False
    def cancel_once():
        nonlocal cancelled
        if not cancelled:
            cancelled = True
            task.cancel()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, cancel_once)
    try:
        return await task
    finally:
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.remove_signal_handler(sig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--plan-sha256')
    parser.add_argument('--admission', type=Path)
    parser.add_argument('--admission-sha256')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    if not args.run:
        print(json.dumps(contract(), indent=2))
        return 0
    need(all(getattr(args, k) is not None for k in ('plan', 'plan_sha256', 'admission', 'admission_sha256', 'out')), 'explicit run bindings required')
    need(all(getattr(args, k).is_absolute() for k in ('plan', 'admission', 'out')), 'absolute paths required')
    args.out = args.out.resolve()
    plan, admission, prompt = prepare(args)
    try:
        return asyncio.run(run_cancellable(args, plan, admission, prompt))
    except BaseException as exc:
        # Imports/initialization can fail before a pool exists. Preserve this
        # separately; absence of complete.json never constitutes success.
        write(args.out / 'entry-error.json', dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
