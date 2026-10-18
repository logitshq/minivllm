"""C2/C4 domain and single-pair construction contracts, default inert.

This module does not import a model backend, launch a server, or grant any
qualification. An explicit future numerical runner must own the original
1800s deadline, full pre/post source/model verification and finally release.
The immutable C1/C8 numeric plan remains provenance, not a C2/C4 qualification.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
JOBS = HERE.parent / 'pareto_realistic/http_qualification.py'
JOBS_SHA = '471bebd8a114d3a7f5a190c20909f6896f17fd39f282656546c1366c3fa2c6e2'
BASE_PLAN_SHA = '21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb'
CONCURRENCIES = (2, 4)
CANDIDATE = dict(compile_decode=True, fuse_decode=True, gates_compile=True, norm_reuse=False)
REFERENCE = dict(compile_decode=False, fuse_decode=False, gates_compile=False, norm_reuse=False)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def concurrency(value):
    need(type(value) is int and value in CONCURRENCIES, 'new domain is exactly C2/C4')
    return value


def request_count(value):
    return max(8, 2 * concurrency(value))


def jobs_for(fixtures, value):
    """Reuse the frozen eight request bodies, changing only client request IDs.

    The original C1 jobs helper supplies body templates, not recorded C1 tokens,
    schedules or latency. Both new domains use each of the same D8 fixtures once.
    """
    C = concurrency(value)
    need(hashlib.sha256(JOBS.read_bytes()).hexdigest() == JOBS_SHA, 'frozen job/body source changed')
    spec = importlib.util.spec_from_file_location('_c24_frozen_body_templates', JOBS)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    original = helper.jobs_for(fixtures, 1)
    need(len(original) == request_count(C), 'exact D8 request count')
    return [dict(job, request_id=f'qual-c{C}-r{i:02d}') for i, job in enumerate(original)]


def configuration(numeric, value):
    """Return explicit new capacities without rewriting the registered old plan."""
    C = concurrency(value)
    need(numeric['schema'] == 'E023-native-qualification-plan-v1', 'original numeric plan schema')
    need(numeric['configuration'] == dict(max_len=2048,
         max_seqs_by_case={'D-C1': 1, 'D-C8': 8, 'lifecycle': 8},
         max_batched_tokens=2048, cache_limit_gib=2), 'unchanged original base capacity settings')
    need(numeric['candidate'] == CANDIDATE and numeric['deadline_s'] == 1800,
         'unchanged candidate flags and numeric deadline')
    return dict(max_len=2048, max_seqs=C, max_batched_tokens=2048, cache_limit_gib=2)


def load_pair_into(engines, observers, *, numeric, fixtures, value, engine_type,
                   mx, tree_flatten, observer_type, comparator, deadline, phase):
    """Construct exactly one independent pair at the actual new capacity.

    Constructor/flag/weight/EOS/observer order matches frozen qualify_native's
    load_pair. Dependencies are explicit so CPU fakes need no backend import.
    The caller must finally restore observers and release both lists even after
    failure; every returned engine is appended before subsequent checks. There
    is no output sampling, schedule substitution or candidate-token input here.
    """
    cfg = configuration(numeric, value)
    need(type(engines) is list and type(observers) is list and not engines and not observers,
         'one fresh engine/observer pair; no reuse from another domain')
    phase('load-pair-start', max_seqs=cfg['max_seqs'], model_instances=2)
    for flags in (REFERENCE, CANDIDATE):
        deadline()
        engine = engine_type(numeric['model'], max_len=cfg['max_len'], max_seqs=cfg['max_seqs'],
            max_batched_tokens=cfg['max_batched_tokens'], cache_limit_gb=cfg['cache_limit_gib'],
            dtype=mx.bfloat16, compile_decode=flags['compile_decode'], fuse_decode=flags['fuse_decode'])
        engines.append(engine)
        need(engine.scheduler.max_seqs == cfg['max_seqs'] and engine.runner.max_len == 2048,
             'actual engine capacity differs from new domain')
        for layer in engine.model.layers:
            if not layer.is_attn:
                layer.linear_attn.set_gates_compile(flags['gates_compile'])
                layer.linear_attn.set_norm_reuse(False)
        need(engine.runner.spec is None and len(engine.model.layers) == 64,
             'target-only model with original64-layer layout')
    weights = [dict(tree_flatten(engine.model.parameters())) for engine in engines]
    comparator.meta('independent weights have same names', sorted(weights[0]), sorted(weights[1]))
    need(not any(weights[0][key] is weights[1][key] for key in weights[0]),
         'independent engine weights must not share array handles')
    comparator.meta('EOS set', sorted(engines[0].eos_ids), sorted(engines[1].eos_ids))
    for fixture in fixtures:
        comparator.meta('fixture EOS set', sorted(fixture['eos_token_ids']), sorted(engines[0].eos_ids))
    for engine, arm in zip(engines, ('reference', 'candidate')):
        # The supplied candidate Observer installs the frozen combined candidate
        # before wrapping its true model/decode calls, as in the existing replay.
        observers.append(observer_type(engine, comparator, arm))
    mx.synchronize()
    phase('load-pair-complete', max_seqs=cfg['max_seqs'], independent_weight_arrays=len(weights[0]))


def description():
    return dict(schema='GDN-graph-combined-c24-domain-description-v1', status='SOURCE_ONLY',
        concurrency=list(CONCURRENCIES), requests_per_C=8, normal_eos=True, max_output_tokens=512,
        configuration='context2048 / gatesON / normOFF / E028OFF',
        route_graph_batch_guard='unchanged B1..8', gdn_prework_batch_guard='unchanged B1 only',
        actual_http_qualified=False, numerical_qualified=False, performance_qualified=False,
        missing='new server/capture/replay/auditor binding and externally owned lifecycle')


if __name__ == '__main__':
    print(json.dumps(description(), sort_keys=True))
