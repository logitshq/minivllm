"""Join captured native HTTP evidence, then replay one C with the frozen qualifier.

--prepare-only joins evidence on CPU, without importing MLX or loading weights.
Numerical execution must be invoked by a resource owner AFTER the captured HTTP
server has been stopped/reaped. This module never owns a server or changes the
resident service. Neither preparation nor a replay PASS grants performance or
future-arrival qualification. Original 37-request qualification remains separate.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import traceback

HERE = Path(__file__).resolve().parent
BASE = HERE / 'qualify_native.py'
BASE_SHA = '26964fcbed9a2e5dcefaee4888244987897f3138d4a2b7add59bf63a7461b7bf'
BASE_PLAN_SHA = '21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb'
FIXED = {BASE: BASE_SHA,
    HERE / 'client.py': 'ef64a114f4f7d90ec5044d6bdcf058ca05f98dea445d6639db706a4fc75217dc',
    HERE / 'native_server.py': '9bcb37eff8bdecd79800afdedc82deacf9f7a0a9a7a1e0d62dc8ba49e70fb68f'}
ADAPTERS = (HERE / 'http_qualification.py', HERE / 'http_join.py', HERE / 'http_replay.py')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def read(path):
    with Path(path).open() as f:
        return json.load(f)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def pinned(item):
    need(set(item) == {'path', 'sha256'} and Path(item['path']).is_absolute(), 'absolute pinned file required')
    need(sha(item['path']) == item['sha256'], 'input SHA changed: ' + item['path'])
    return Path(item['path'])


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate(path, expected_sha):
    need(sha(path) == expected_sha, 'HTTP replay plan SHA mismatch')
    plan = read(path)
    need(set(plan) == {'schema', 'qualification', 'concurrency', 'transport', 'inputs', 'sources'}
         and plan['schema'] == 'E023-recorded-http-plan-v1', 'explicit HTTP replay plan required')
    need(type(plan['concurrency']) is int and plan['concurrency'] in (1, 8), 'only independently prepared C1/C8')
    q = plan['qualification']
    need(q['sha256'] == BASE_PLAN_SHA, 'unchanged numerical base plan required')
    numeric = read(pinned(q))
    transport = plan['transport']
    need(set(transport) == {'tcp_nodelay', 'io_timeout_seconds'}
         and type(transport['tcp_nodelay']) is bool
         and type(transport['io_timeout_seconds']) in (int, float)
         and math.isfinite(transport['io_timeout_seconds']) and transport['io_timeout_seconds'] > 0,
         'explicit socket and finite positive IO timeout required')
    sources = plan['sources']
    need(set(sources) == {str(p) for p in (*FIXED, *ADAPTERS)}, 'exact six source pins required')
    for source, digest in sources.items():
        need(sha(source) == digest, 'adapter/source changed: ' + source)
    for source, digest in FIXED.items():
        need(sources[str(source)] == digest, 'frozen source changed: ' + str(source))
    need(set(plan['inputs']) == {'group', 'trace'}, 'captured group and trace required')
    for item in plan['inputs'].values():
        pinned(item)
    return plan, numeric


def jobs_for(fixtures, concurrency):
    need(type(concurrency) is int and concurrency in (1, 8), 'C1/C8 only')
    need(len(fixtures) == 8 and len({f['fixture_id'] for f in fixtures}) == 8, 'eight distinct D fixtures required')
    jobs = []
    for i in range(8 if concurrency == 1 else 16):
        f = fixtures[i % 8]
        need(f['normal_eos'] is True and f['max_output_tokens'] == 512, 'fixed normal-EOS D512 required')
        jobs.append({'request_id': f'qual-c{concurrency}-r{i:02d}', 'fixture_id': f['fixture_id'],
            'path': '/v1/completions', 'body': {'model': 'native-target-only', 'prompt': f['prompt_text'],
                'temperature': 0, 'top_p': 1.0, 'repetition_penalty': 1.0, 'max_tokens': 512,
                'stream': True, 'stream_options': {'include_usage': True}, 'n': 1}})
    return jobs


def expected_identity(numeric, fixtures, source, model, concurrency, transport):
    cfg = numeric['configuration']; flags = numeric['candidate']
    eos = sorted(fixtures[0]['eos_token_ids'])
    need(all(sorted(f['eos_token_ids']) == eos for f in fixtures), 'consistent fixture EOS sets required')
    configuration = {k: cfg[k] for k in ('max_len', 'max_batched_tokens', 'cache_limit_gib')}
    configuration['max_seqs'] = cfg['max_seqs_by_case'][f'D-C{concurrency}']
    configuration.update({k: 'on' if v else 'off' for k, v in flags.items()})
    configuration.update(tcp_nodelay='on' if transport['tcp_nodelay'] else 'off',
                         io_timeout_seconds=float(transport['io_timeout_seconds']))
    path = Path(numeric['model']).resolve()
    return {'configuration': configuration, 'source_files': source['files'],
        'source_manifest_sha256': numeric['source_manifest']['sha256'], 'manifest_all_sources_match': True,
        'package_root_loaded': str(Path(numeric['package_root']).resolve() / 'tutorial_code/miniengine'),
        'model_snapshot_path': str(path), 'model_snapshot_name': path.name,
        'model_config_sha256': model['files']['config.json'],
        'server_source_sha256': FIXED[HERE / 'native_server.py'],
        'step_observer': True, 'per_token_observer': False, 'qualification_trace_enabled': True,
        'drafter': None, 'normal_eos': True, 'eos_ids': eos, 'prefix_cache': False,
        'completion_usage': 'committed target IDs excluding terminal EOS; raw IDs include EOS',
        'text_decode': 'skip_special_tokens=False; terminal EOS excluded',
        'cancellation': 'pending skipped; active cancellation permanently fails this benchmark server',
        'weights_identity_scope': 'external controller must verify model weight pins; this server checks source/config identity'}


def prepare(plan, numeric):
    fixtures = read(pinned(numeric['fixtures']))['fixtures']
    source, model = [read(pinned(numeric[k])) for k in ('source_manifest', 'model_manifest')]
    tokenizer_path = Path(numeric['model']) / 'tokenizer.json'
    need(sha(tokenizer_path) == model['files']['tokenizer.json'], 'exact model tokenizer required')
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    for fixture in fixtures:
        need(tokenizer.encode(fixture['prompt_text'], add_special_tokens=False).ids == fixture['prompt_ids'],
             'full prepared prompt IDs changed: ' + fixture['fixture_id'])
    jobs = jobs_for(fixtures, plan['concurrency'])
    identity = expected_identity(numeric, fixtures, source, model, plan['concurrency'], plan['transport'])
    module = load(HERE / 'http_join.py', '_e023_private_http_join')
    group = read(pinned(plan['inputs']['group']))
    with pinned(plan['inputs']['trace']).open('rb') as trace:
        recording = module.join(group, trace, jobs, {f['fixture_id']: f for f in fixtures}, identity,
                                lambda ids: tokenizer.decode(ids, skip_special_tokens=False))
    return recording, fixtures


def run_adapter(base, replay, recording, fixtures, qualification, out):
    """Reuse the original single-process load/hash/deadline/release implementation."""
    name = f"D-C{recording['concurrency']}"
    case = {'name': name, 'concurrency': recording['concurrency']}
    old_workloads, old_run = base.workloads, base.run_case
    def workloads(actual):
        need(actual == fixtures, 'base loaded different fixtures')
        return [case]
    def run_case(actual, engines, observers, sampling, runner_module, cmp, ledger, deadline):
        need(actual == case, 'base changed the selected recorded case')
        return replay.run_recorded_case(recording, engines, observers, sampling, runner_module,
                                        cmp, ledger, deadline, base=base)
    base.workloads, base.run_case = workloads, run_case
    try:
        return base.main(['--plan', qualification['path'], '--plan-sha256', qualification['sha256'],
                          '--out', str(out)])
    finally:
        base.workloads, base.run_case = old_workloads, old_run


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--plan-sha256', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--prepare-only', action='store_true')
    args = ap.parse_args(argv)
    plan, numeric = validate(args.plan, args.plan_sha256)
    args.out.mkdir(parents=True, exist_ok=False)
    write(args.out / 'plan.json', plan)
    result = {'schema': 'E023-recorded-http-result-v1', 'status': 'INCOMPLETE',
        'plan_sha256': args.plan_sha256, 'sources': plan['sources'], 'inputs': plan['inputs'],
        'concurrency': plan['concurrency'], 'prepare_only': args.prepare_only,
        'numerical_qualified': False, 'performance_qualified': False,
        'scope': 'Only the captured HTTP schedule. No resource/reap, original 37-request, future-arrival or performance qualification.'}
    write(args.out / 'complete.json', result)
    try:
        recording, fixtures = prepare(plan, numeric)
        write(args.out / 'recording.json', recording)
        result['recording_sha256'] = sha(args.out / 'recording.json')
        if args.prepare_only:
            result['status'] = 'PREPARED'
        else:
            base = load(BASE, '_e023_private_recorded_qualifier')
            replay = load(HERE / 'http_replay.py', '_e023_private_recorded_loop')
            code = run_adapter(base, replay, recording, fixtures, plan['qualification'], args.out / 'base')
            inner = read(args.out / 'base/complete.json')
            result['base_result'] = {'path': 'base/complete.json', 'sha256': sha(args.out / 'base/complete.json'),
                                    'returncode': code, 'status': inner['status']}
            count = 8 if plan['concurrency'] == 1 else 16
            need(code == 0 and inner['status'] == 'PASS' and inner['source_model_pins_rechecked'] is True,
                 'recorded numerical replay did not pass')
            need(inner['total_requests'] == count and len(inner['cases']) == 1
                 and inner['cases'][0]['name'] == f"HTTP-C{plan['concurrency']}"
                 and len(inner['cases'][0]['completed']) == count, 'complete one-C captured workload required')
            result.update(status='PASS', numerical_qualified=True, total_requests=count)
        need(validate(args.plan, args.plan_sha256) == (plan, numeric), 'plan/input/source changed after work')
        result['input_source_pins_rechecked'] = True
    except BaseException as error:
        result.update(status='FAIL', numerical_qualified=False,
                      error={'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()})
    finally:
        write(args.out / 'complete.json', result)
    print(json.dumps({'status': result['status'], 'out': str(args.out)}), flush=True)
    return int(result['status'] not in ('PASS', 'PREPARED'))


if __name__ == '__main__':
    raise SystemExit(main())
