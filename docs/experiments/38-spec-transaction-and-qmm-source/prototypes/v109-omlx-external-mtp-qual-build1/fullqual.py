"""One-use external-MTP correctness/resources qualification, never performance."""
import argparse
import ast
import concurrent.futures
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

D = Path(__file__).resolve().parent
P = D.parent
OLD = P / 'm4-final-native15-official-attempt2'
sys.path.insert(0, str(OLD))
import common as controls
from support import verify, sha
from checks import inspect_observations, equal_streams


def write_new(path, value):
    controls.write_new(path, value)


def run_arm(mode, output, config, pins_sha, previous_env):
    # Imports occur only with --run; CPU inert never loads an inference library.
    import httpx
    from tokenizers import Tokenizer
    spec = importlib.util.spec_from_file_location('v109_old_official_client', OLD / 'client.py')
    official = importlib.util.module_from_spec(spec); spec.loader.exec_module(official)
    run_dir = output / mode; run_dir.mkdir()
    model = Path(config['target_model_dir'])
    tok = Tokenizer.from_file(str(model / 'tokenizer.json'))
    prompt = tok.decode(config['prompt_ids'], skip_special_tokens=False)
    assert tok.encode(prompt, add_special_tokens=False).ids == config['prompt_ids']
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    key = 'benchmark-' + secrets.token_urlsafe(24)
    env = controls.child_environment(os.environ)
    if previous_env:
        assert env == previous_env
    else:
        previous_env.update(env)
    old_command = official.official_command('omlx', 1, run_dir, port, key, model, env)
    vendor_command = old_command[old_command.index('--') + 1:]
    command = [old_command[0], '-B', str(D / 'entry.py'), '--mode', mode,
               '--run-dir', str(run_dir), '--pins-sha256', pins_sha, '--', *vendor_command]
    settings = {'mtp_enabled': False, 'dflash_enabled': False, 'specprefill_enabled': False,
                'vlm_mtp_enabled': mode == 'mtp',
                'vlm_mtp_draft_model': config['draft_model_dir'] if mode == 'mtp' else None,
                'vlm_mtp_draft_block_size': None}
    write_new(run_dir / 'state/model_settings.json', {'version': 1, 'models': {model.name: settings}})
    write_new(run_dir / 'launch.json', {'mode': mode, 'command': ['<api-key>' if x == key else x for x in command],
              'environment': controls.boot.environment_record(env), 'model_settings': settings,
              'target_revision': model.name, 'external_draft_revision': Path(config['draft_model_dir']).name,
              'qualification_observers': True, 'performance_eligible': False,
              'official_cli_unchanged': True, 'official_command_AST_unchanged': True})
    samples = []; proc = None; birth = None; log = None; error = None; cleanup = None; result = None
    cleanup_errors = []
    def sample(label):
        controls.probe(samples, run_dir / 'resources.json', label, 'AC', proc.pid if proc else None)
        assert samples[-1]['processinfo_thermal_metadata']['code'] == 0, 'nominal thermal required'
    try:
        sample('before-model-launch')
        log = (run_dir / 'server.log').open('x')
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        birth = controls.identity(proc.pid); assert birth is not None
        write_new(run_dir / 'owned-pid.json', {'pid': proc.pid, 'ppid': os.getpid()})
        write_new(run_dir / 'owned-birth.json', {'identity': birth})
        base = f'http://127.0.0.1:{port}'; headers = {'Authorization': 'Bearer ' + key}
        deadline = time.monotonic() + 180
        with httpx.Client(headers=headers, trust_env=False, timeout=180) as client:
            while True:
                assert proc.poll() is None, 'owned server exited before ready'
                try:
                    response = client.get(base + '/v1/models', timeout=1)
                    if response.status_code == 200: break
                except httpx.HTTPError: pass
                if time.monotonic() > deadline: raise TimeoutError('readiness')
                time.sleep(0.2)
        controls.exact_listener(proc.pid, port, run_dir / 'server-ready-exact-listener.json')
        birth = controls.lock_owned_identity(proc, birth, 'omlx', run_dir, 'ready')
        write_new(run_dir / 'models.json', response.json())
        name = response.json()['data'][0]['id']; assert name == model.name, name
        sample('server-ready')
        path = '/v1/completions'
        body = {'model': name, 'prompt': prompt, 'temperature': 0, 'repetition_penalty': 1.0, 'max_tokens': 32}
        a = argparse.Namespace(engine='omlx')
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
        requests = []
        for index in range(3):
            gid = f'C1-qualification-{index}'
            sample('before-' + gid)
            current = group(1, gid)
            write_new(run_dir / (gid + '.json'), current)
            requests.extend(current['requests'])
            sample('after-' + gid)
        rows = [json.loads(line) for line in (run_dir / 'observer.jsonl').read_text().splitlines()]
        observed = inspect_observations(rows, mode, requests, config)
        result = {'mode': mode, 'observations': observed, 'requests': requests,
                  'observer_sha256': sha(run_dir / 'observer.jsonl'), 'source_pins_sha256': pins_sha,
                  'target_revision': model.name, 'drafter_revision': Path(config['draft_model_dir']).name if mode == 'mtp' else None,
                  'correctness_scope': 'three serial normal-EOS greedy uncached raw32; compared to fresh actual vendor target control',
                  'timings_are_instrumented_and_not_performance_evidence': True}
    except BaseException as exc:
        error = exc
        write_new(run_dir / 'first-failure.json', {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()})
    finally:
        if proc is not None:
            try:
                birth = controls.lock_owned_identity(proc, birth, 'omlx', run_dir, 'pre-cleanup')
                cleanup = controls.cleanup_owned(proc, birth, run_dir / 'cleanup-signal-identities.json')
                write_new(run_dir / 'owned-process-exit.json', cleanup)
            except BaseException as exc:
                cleanup_errors.append({'phase': 'owned-process', 'message': str(exc)})
        try: write_new(run_dir / 'port-absence.json', controls.wait_port_absence(port))
        except BaseException as exc: cleanup_errors.append({'phase': 'port', 'message': str(exc)})
        try:
            controls.probe(samples, run_dir / 'resources.json', 'after-owned-unload', 'AC')
            assert samples[-1]['processinfo_thermal_metadata']['code'] == 0
        except BaseException as exc: cleanup_errors.append({'phase': 'resource', 'message': str(exc)})
        if log is not None: log.close()
        write_new(run_dir / 'cleanup.json', {'first_error': None if error is None else str(error),
                  'cleanup_errors': cleanup_errors, 'owned_exit': cleanup})
    if error is not None: raise error
    assert not cleanup_errors, cleanup_errors
    assert result is not None
    result['resource_snapshots'] = samples
    result['owned_exit'] = cleanup
    write_new(run_dir / 'result.json', result)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--inert', action='store_true'); mode.add_argument('--run', action='store_true')
    ap.add_argument('--out', type=Path); ap.add_argument('--pins-sha256', required=True)
    args = ap.parse_args()
    config = verify(args.pins_sha256)
    assert not any(name == 'mlx' or name.startswith('mlx.') for name in sys.modules)
    assert not controls.selected(os.environ)
    assert sys.executable == controls.PYTHON
    if args.inert:
        print(json.dumps({'status': 'CPU_INERT_ONLY', 'GPU_used': False, 'fresh_loads': 2,
                          'requests': 6, 'target': config['target_model_dir'], 'draft': config['draft_model_dir']}))
        return 0
    assert args.out and args.out.is_absolute(); args.out.mkdir()
    write_new(D / 'attempt.consumed.json', {'pid': os.getpid(), 'pins_sha256': args.pins_sha256})
    def terminate(sig, frame): raise KeyboardInterrupt('owned controller signal ' + str(sig))
    previous = signal.signal(signal.SIGTERM, terminate)
    results = []; previous_env = {}; failure = None
    try:
        for mode in ('none', 'mtp'):
            print(json.dumps({'event': 'ARM_STARTED', 'mode': mode}), flush=True)
            results.append(run_arm(mode, args.out, config, args.pins_sha256, previous_env))
            print(json.dumps({'event': 'ARM_COMPLETED', 'mode': mode}), flush=True)
        canonical = equal_streams(results[0]['observations'], results[1]['observations'])
        verify(args.pins_sha256)
        write_new(args.out / 'receipt.json', {'status': 'PASS_EXTERNAL_MTP_GREEDY32_RAW_TARGET_CONTROL_RESOURCES',
                  'source_pins_sha256': args.pins_sha256, 'actual_fresh_servers': 2, 'actual_requests': 6,
                  'ordinary_control_raw32': canonical, 'MTP_per_request': results[1]['observations'],
                  'target_revision': Path(config['target_model_dir']).name,
                  'external_draft_revision': Path(config['draft_model_dir']).name,
                  'terminal_cleanup': True, 'performance_qualified': False, 'global_best_claim': False,
                  'scope': 'fixed 19-prompt C1 greedy32 normal-EOS uncached; external MTP only; not DFlash2 or broad quality',
                  'controller_pid': os.getpid()})
    except BaseException as exc:
        failure = exc
        write_new(args.out / 'suite-failure.json', {'status': 'FAIL_STOP', 'type': type(exc).__name__,
                  'message': str(exc), 'traceback': traceback.format_exc(), 'completed_arms': len(results),
                  'performance_qualified': False, 'raw_preserved': True})
    finally:
        signal.signal(signal.SIGTERM, previous)
    if failure is not None: raise failure
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
