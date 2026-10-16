"""C1 serial NOL/OLN/LNO screen using the frozen official client definitions."""
import argparse
import ast
import concurrent.futures
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import traceback

P = Path('/private/tmp/minivllm-engine-optimization-20261014')
D = Path(__file__).resolve().parent
OLD = P / 'm4-final-native15-official-attempt2'
sys.path.insert(0, str(OLD))
import common as controls
spec = importlib.util.spec_from_file_location('v45_official_client', OLD / 'client.py')
official = importlib.util.module_from_spec(spec); spec.loader.exec_module(official)
import httpx
from tokenizers import Tokenizer


def write_new(path, value): controls.write_new(path, value)


def run_arm(item, output, config, previous_env):
    engine = item['engine']; label = item['label']; batch = 1
    run_dir = output / label; run_dir.mkdir()
    model = Path(config['model_dir'])
    w = json.loads((P / 'workload.json').read_text())
    assert str(model) == w['model_dir']
    tok = Tokenizer.from_file(str(model / 'tokenizer.json'))
    prompt = tok.decode(w['prompt_ids'], skip_special_tokens=False)
    assert tok.encode(prompt, add_special_tokens=False).ids == w['prompt_ids']
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]
    key = 'benchmark-' + secrets.token_urlsafe(24)
    env = controls.child_environment(os.environ)
    if previous_env:
        assert env == previous_env
    else:
        previous_env.update(env)
    if engine == 'native':
        env['DYLD_INSERT_LIBRARIES'] = config['native_library']
        command = [sys.executable, '-B', str(D / 'server.py'), '--config', str(D / 'runtime-config.json'),
                   '--model', str(model), '--port', str(port), '--trace', str(run_dir / 'engine-trace.json'),
                   '--run-dir', str(run_dir), '--owner-pid', str(os.getpid())]
    else:
        command = official.official_command(engine, batch, run_dir, port, key, model, env)
    samples = []; proc = None; birth = None; lf = None; error = None; cleanup = None
    result = None
    def sample(name):
        raw = controls.probe(samples, run_dir / 'resources.json', name, 'AC', proc.pid if proc else None)
        assert samples[-1]['processinfo_thermal_metadata']['code'] == 0, 'nominal thermal required'
        return raw
    try:
        sample('before-model-launch')
        write_new(run_dir / 'launch.json', {
            'engine': engine, 'command': ['<api-key>' if x == key else x for x in command],
            'environment': controls.boot.environment_record(env), 'source_config_sha256': controls.sha(D / 'runtime-config.json'),
            'selected_environment': {k: v for k, v in env.items() if k.startswith(('MLX_', 'DYLD_', 'MONOLITH_'))},
            'native_treatment': 'qualified native C1 adapter; native DYLD only; ordinary extra startup warm explicitly recorded'})
        lf = (run_dir / 'server.log').open('x')
        started = time.perf_counter()
        proc = subprocess.Popen(command, stdout=lf, stderr=subprocess.STDOUT, env=env)
        birth = controls.identity(proc.pid); assert birth is not None
        write_new(run_dir / 'owned-pid.json', {'pid': proc.pid, 'ppid': os.getpid()})
        write_new(run_dir / 'owned-birth.json', {'identity': birth})
        base = f'http://127.0.0.1:{port}'; headers = {'Authorization': 'Bearer ' + key}
        with httpx.Client(headers=headers, trust_env=False, timeout=180) as client:
            while True:
                if proc.poll() is not None: raise RuntimeError('owned server exited before ready')
                try:
                    response = client.get(base + '/v1/models', timeout=1)
                    if response.status_code == 200: break
                except httpx.HTTPError: pass
                if time.perf_counter() - started > 180: raise TimeoutError('server readiness')
                time.sleep(0.2)
        controls.exact_listener(proc.pid, port, run_dir / 'server-ready-exact-listener.json')
        birth = controls.lock_owned_identity(proc, birth, engine, run_dir, 'ready')
        runtime_identity = response.json().get('benchmark_identity')
        if engine == 'native':
            assert runtime_identity['manifest_all_sources_match'] and runtime_identity['measurement_hooks'] is False
            assert runtime_identity['startup_ordinary_warm_requests'] == 1
            assert runtime_identity['prior_fullqual_sha256'] == config['fullqual_receipt_sha256']
        name = response.json()['data'][0]['id']
        sample('server-ready-before-warmup')
        path = '/v1/completions'
        body = {'model': name, 'prompt': prompt, 'temperature': 0, 'repetition_penalty': 1.0, 'max_tokens': 32}
        if engine == 'lithos':
            path = '/v1/chat/completions'
            body = {'model': name, 'messages': [{'role': 'user', 'content': '用一句话解释什么是 KV cache。'}],
                    'temperature': 0, 'top_p': 1, 'max_completion_tokens': 32}
        a = argparse.Namespace(engine='tutorial' if engine == 'native' else engine)
        # Frozen SSE request and group functions are inserted with identical AST.

        def request(barrier,group_id,request_id):
         with httpx.Client(headers=headers,trust_env=False,timeout=180) as c:
          barrier.wait();start=time.perf_counter();first=None;pieces=[];events=[];usage=None;finish=None;done=False;done_time=None;output_ids=None
          with c.stream('POST',base+path,json={**body,'stream':True,'stream_options':{'include_usage':True}}) as resp:
           resp.raise_for_status()
           for line in resp.iter_lines():
            if not line.startswith('data:'):continue
            payload=line[5:].strip()
            if payload=='[DONE]':done_time=time.perf_counter();done=True;break
            d=json.loads(payload)
            if 'error' in d:raise RuntimeError(d['error'])
            usage=d.get('usage') or usage
            if 'benchmark_output_ids' in d:output_ids=d['benchmark_output_ids']
            for choice in d.get('choices',[]):
             finish=choice.get('finish_reason') or finish;piece=choice.get('text',choice.get('delta',{}).get('content',''))
             if piece:
              now=time.perf_counter();first=first or now;pieces.append(piece);events.append({'client_perf_counter_s':now,'text':piece,'engine_delivery':d.get('benchmark_delivery')})
          cleanup_complete=time.perf_counter();end=done_time;assert done and usage and finish=='length',(done,usage,finish);assert usage['prompt_tokens']==19 and usage['completion_tokens']==32,usage
          cached=usage.get('prompt_tokens_details',{}).get('cached_tokens',0);assert cached==0,cached
          text=''.join(pieces);visible_ids=tok.encode(text,add_special_tokens=False).ids;first_ids=tok.encode(pieces[0],add_special_tokens=False).ids
          prefix_ok=visible_ids[:len(first_ids)]==first_ids and text.startswith(pieces[0]);full_roundtrip_ok=len(visible_ids)==32
          assert len(visible_ids)==32,'visible canonical token count mismatch'
          if a.engine=='tutorial':assert output_ids is not None,'native raw token IDs missing'
          if output_ids is not None:assert isinstance(output_ids,list) and len(output_ids)==32 and output_ids==visible_ids,'actual engine token IDs mismatch/length/EOS'
          first_count=len(first_ids);remaining=32-first_count
          # HTTP APIs expose text chunks. Reconstructed counts are an explicitly labelled client estimate.
          interactivity=remaining/(end-first) if prefix_ok and full_roundtrip_ok and remaining>0 else None
          return {'group_id':group_id,'request_id':request_id,'cleanup_complete_perf_counter_s':cleanup_complete,'stream_cleanup_after_done_s':cleanup_complete-end,'start_perf_counter_s':start,'done_perf_counter_s':end,'first_text_perf_counter_s':first,'client_total_s':end-start,'client_ttft_text_s':first-start,'client_post_first_chunk_s':end-first,'usage':usage,'finish_reason':finish,'text':text,'output_ids_when_available':output_ids,'visible_roundtrip_ids':visible_ids,'first_chunk_text':pieces[0],'first_chunk_reconstructed_tokens':first_count,'first_chunk_prefix_verified':prefix_ok,'full_visible_roundtrip_count_matches_usage':full_roundtrip_ok,'x_client_estimated_interactivity_tok_s_user':interactivity,'events':events}

        def group(batch,group_id):
         barrier=threading.Barrier(batch)
         with concurrent.futures.ThreadPoolExecutor(max_workers=batch) as ex:runs=list(ex.map(lambda i:request(barrier,group_id,f"{group_id}-R{i}"),range(batch)))
         begin=min(r['start_perf_counter_s'] for r in runs);end=max(r['done_perf_counter_s'] for r in runs);window=end-begin;xs=[r['x_client_estimated_interactivity_tok_s_user'] for r in runs]
         return {'group_id':group_id,'actual_request_count':batch,'requests':runs,'arrival_spread_s':max(r['start_perf_counter_s'] for r in runs)-begin,'group_window_s':window,'total_completion_tokens':sum(r['usage']['completion_tokens'] for r in runs),'y_total_output_throughput_tok_s_mac':batch*32/window,'x_client_estimated_interactivity_tok_s_user':statistics.median(xs) if all(x is not None for x in xs) else None}
        groups = {}; warm_metadata = None
        for phase in ('warm', 'measure'):
            groups[phase] = []
            for i in range(3):
                gid = f'C1-{phase}-{i}'
                sample('before-' + gid)
                group_result = group(batch, gid)
                write_new(run_dir / (gid + '.json'), group_result)
                groups[phase].append(group_result)
                sample('after-' + gid)
            if phase == 'warm' and engine == 'native':
                deadline = time.monotonic() + 30
                while not (run_dir / 'warm-metadata.json').exists():
                    assert proc.poll() is None
                    if time.monotonic() > deadline: raise TimeoutError('warm metadata')
                    time.sleep(0.01)
                warm_metadata = json.loads((run_dir / 'warm-metadata.json').read_text())
                assert warm_metadata['native_steps'] == 93 and warm_metadata['completed_requests'] == 3
                assert warm_metadata['parameter_tree_unchanged'] and warm_metadata['terminal_cleanup']
                assert warm_metadata['measurement_hooks'] is False
            if phase == 'warm': sample('after-three-warm-before-measure')
        sample('after-measure-before-unload')
        requests = [q for phase in ('warm', 'measure') for g in groups[phase] for q in g['requests']]
        final_native = None; lithos_proof = None
        if engine == 'native':
            assert all(q['output_ids_when_available'] == w['continuation_ids'] for q in requests)
            with httpx.Client(headers=headers, trust_env=False, timeout=30) as client:
                response = client.get(base + '/benchmark-state'); response.raise_for_status()
                final_native = response.json()
            assert final_native['active'] is None and final_native['native_steps'] == 186
            assert final_native['request_count'] == 6
            f = final_native['native']['feedback']
            assert final_native['native']['status'] == 'OK' and f['total_steps'] == 186 and f['resets'] == 6
            assert f['icb']['execution_count'] == 186 and f['icb']['bank_executions'] == [96, 90]
            assert f['pingpong']['eligible_count'] == 48
            write_new(run_dir / 'native-final-metadata.json', final_native)
        if engine == 'lithos':
            text = (run_dir / 'server.log').read_text()
            assert 'Serving backend=m4_max,' in text and 'draft=None' in text and 'backend=m5' not in text
            native_metrics = [ast.literal_eval(line.split('Generation: ', 1)[1])
                              for line in text.splitlines() if 'Generation: ' in line]
            assert len(native_metrics) >= 6 and all(v['cached_prompt_tokens'] == 0 for v in native_metrics)
            lithos_proof = {'backend': 'm4_max', 'draft': None, 'metrics': native_metrics,
                            'precleanup_server_log_sha256': controls.sha(run_dir / 'server.log')}
        result = {'engine': engine, 'label': label, 'concurrency': 1,
                  'warmups': groups['warm'], 'runs': groups['measure'],
                  'runtime_identity': runtime_identity, 'warm_metadata': warm_metadata,
                  'final_native_metadata': final_native, 'lithos_proof': lithos_proof,
                  'x': statistics.median(g['x_client_estimated_interactivity_tok_s_user'] for g in groups['measure']),
                  'y': statistics.median(g['y_total_output_throughput_tok_s_mac'] for g in groups['measure']),
                  'ttft_s': statistics.median(q['client_ttft_text_s'] for g in groups['measure'] for q in g['requests']),
                  'E2E_s': statistics.median(q['client_total_s'] for g in groups['measure'] for q in g['requests']),
                  'source_config_sha256': controls.sha(D / 'runtime-config.json'),
                  'sampling': {'greedy': True, 'output_cap': 32, 'EOS': 'normal', 'draft': None, 'prefix_cache': 0},
                  'scope': 'fresh same-campaign C1 screen only; no C2/C4 or full-frontier conclusion'}
    except BaseException as exc:
        error = exc
        write_new(run_dir / 'first-failure.json', {'type': type(exc).__name__, 'message': str(exc),
                                                   'traceback': traceback.format_exc(), 'raw_preserved': True})
    finally:
        cleanup_errors = []
        if proc is not None:
            try:
                birth = controls.lock_owned_identity(proc, birth, engine, run_dir, 'pre-cleanup')
                cleanup = controls.cleanup_owned(proc, birth, run_dir / 'cleanup-signal-identities.json')
                write_new(run_dir / 'owned-process-exit.json', cleanup)
            except BaseException as exc:
                cleanup_errors.append({'phase': 'owned-process', 'type': type(exc).__name__, 'message': str(exc)})
        try: write_new(run_dir / 'port-absence.json', controls.wait_port_absence(port))
        except BaseException as exc:
            cleanup_errors.append({'phase': 'port', 'type': type(exc).__name__, 'message': str(exc)})
        try:
            controls.probe(samples, run_dir / 'resources.json', 'after-owned-unload', 'AC')
            assert samples[-1]['processinfo_thermal_metadata']['code'] == 0
        except BaseException as exc:
            cleanup_errors.append({'phase': 'resource', 'type': type(exc).__name__, 'message': str(exc)})
        if lf is not None: lf.close()
        write_new(run_dir / 'cleanup.json', {'error': None if error is None else str(error),
                                            'cleanup_errors': cleanup_errors, 'owned_exit': cleanup})
    if error is not None: raise error
    if cleanup_errors: raise RuntimeError('cleanup/postflight failed: ' + str(cleanup_errors))
    assert result is not None
    result.update(resource_snapshots=samples, owned_exit=cleanup)
    write_new(run_dir / 'result.json', result)
    return {k: result[k] for k in ('label', 'engine', 'x', 'y', 'ttft_s', 'E2E_s')}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--cpu-dry-run', action='store_true'); mode.add_argument('--run', action='store_true')
    ap.add_argument('--out', type=Path); ap.add_argument('--pins-sha256', required=True)
    args = ap.parse_args()
    assert controls.sha(D / 'pins.json') == args.pins_sha256
    pins = json.loads((D / 'pins.json').read_text())
    for path, digest in pins['files'].items(): assert controls.sha(path) == digest, path
    config = json.loads((D / 'runtime-config.json').read_text())
    for path, digest in config['source_pins'].items(): assert controls.sha(path) == digest, path
    q = json.loads(Path(config['fullqual_receipt']).read_text())
    assert q['status'] == config['fullqual_status'] and q['terminal_cleanup']
    assert controls.sha(config['fullqual_receipt']) == config['fullqual_receipt_sha256']
    assert not any(n == 'mlx' or n.startswith('mlx.') for n in sys.modules)
    assert not controls.selected(os.environ)
    assert sys.executable == controls.PYTHON
    plan = [{'block': b, 'label': f'C1-B{b}-{name}', 'engine': name}
            for b, order in enumerate((('native', 'omlx', 'lithos'), ('omlx', 'lithos', 'native'),
                                       ('lithos', 'native', 'omlx')), 1) for name in order]
    if args.cpu_dry_run:
        print(json.dumps({'status': 'PASS_CPU_DRY_RUN', 'plan': plan, 'fresh_loads': 9,
                          'HTTP_requests': 54, 'HTTP_completion_tokens': 1728,
                          'native_startup_ordinary_requests': 3, 'native_startup_ordinary_tokens': 96,
                          'total_executed_requests': 57, 'total_executed_tokens': 1824, 'GPU_used': False}))
        return 0
    assert args.out is not None and args.out.is_absolute(); args.out.mkdir()
    write_new(D / 'attempt.consumed.json', {'pid': os.getpid(), 'pins_sha256': args.pins_sha256})
    completed = []; previous_env = {}; samples = []; error = None; completion = None
    def stop_owned_driver(sig, frame):
        raise KeyboardInterrupt('owned driver signal ' + str(sig))
    previous_term = signal.signal(signal.SIGTERM, stop_owned_driver)
    try:
        controls.probe(samples, args.out / 'owner-resources.json', 'before-campaign', 'AC')
        assert samples[-1]['processinfo_thermal_metadata']['code'] == 0
        for item in plan:
            print(json.dumps({'event': 'ARM_STARTED', **item}), flush=True)
            completed.append(run_arm(item, args.out, config, previous_env))
            print(json.dumps({'event': 'ARM_COMPLETED', 'completed': len(completed), 'total': 9,
                              **completed[-1]}), flush=True)
        for path, digest in pins['files'].items(): assert controls.sha(path) == digest, path
        assert len(completed) == 9
        medians = {engine: {key: statistics.median(r[key] for r in completed if r['engine'] == engine)
                            for key in ('x', 'y', 'ttft_s', 'E2E_s')}
                   for engine in ('native', 'omlx', 'lithos')}
        completion = {'status': 'PASS_C1_SCREEN', 'arms': completed,
            'medians': medians, 'all_negative_pairs_retained': True, 'full_frontier_win': False,
            'HTTP_requests': 54, 'HTTP_completion_tokens': 1728,
            'native_startup_ordinary_requests': 3, 'native_startup_ordinary_tokens': 96,
            'total_executed_requests': 57, 'total_executed_tokens': 1824,
            'measured_requests': 27, 'measured_tokens': 864}
    except BaseException as exc:
        error = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
        write_new(args.out / 'suite-failure.json', {'error': error, 'completed': completed, 'raw_preserved': True})
        traceback.print_exc(); return 1
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        controls.probe(samples, args.out / 'owner-resources.json', 'after-campaign', 'AC')
        assert samples[-1]['processinfo_thermal_metadata']['code'] == 0
        write_new(args.out / 'owner-cleanup.json', {'pid': os.getpid(), 'completed_arms': len(completed),
                                                 'error': error, 'GPU_released': False})
    assert completion is not None and error is None
    write_new(args.out / 'suite-complete.json', completion)
    print(json.dumps({'event': 'SUITE_COMPLETE', 'medians': completion['medians'],
                      'full_frontier_win': False}), flush=True)
    return 0


if __name__ == '__main__': raise SystemExit(main())
