"""V120 one-arm client draft; frozen105 request/group AST preserved. No model import."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import statistics
import sys
import threading
import time
from adapter import D, P, qualification


def measure_arm(args):
    qualification(args.qualification_binding)
    import httpx
    from tokenizers import Tokenizer
    sys.path.insert(0, str(P / 'm4-final-native15-official-attempt2'))
    import common as controls
    w = json.loads((P / 'workload.json').read_text())
    tok = Tokenizer.from_file(str(Path(w['model_dir']) / 'tokenizer.json'))
    prompt = tok.decode(w['prompt_ids'], skip_special_tokens=False)
    assert tok.encode(prompt, add_special_tokens=False).ids == w['prompt_ids']
    base = args.base_url.rstrip('/')
    headers = {}
    with httpx.Client(trust_env=False, timeout=1) as c:
        response = c.get(base + '/v1/models'); response.raise_for_status()
        metadata = response.json()
    assert metadata['benchmark_identity']['mode'] == args.mode
    assert metadata['benchmark_identity']['qualification_observer_callbacks'] is False
    body = {'model': metadata['data'][0]['id'], 'prompt': prompt, 'temperature': 0,
            'repetition_penalty': 1.0, 'max_tokens': 32}
    path = '/v1/completions'
    a = argparse.Namespace(engine='tutorial')
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

    args.out.mkdir(exist_ok=False)
    samples = []
    def sample(label):
        controls.probe(samples, args.out / 'resources.json', label,
                       'AC' if args.power_domain == 'AC Power' else 'Battery', args.server_pid)
        assert samples[-1]['processinfo_thermal_metadata']['code'] == 0, 'nominal thermal required'
    groups = {}
    for phase in ('warm', 'measure'):
        groups[phase] = []
        for i in range(3):
            label = f'C1-{phase}-{i}'
            sample('before-' + label)
            current = group(1, label)
            (args.out / (label + '.json')).write_text(json.dumps(current, indent=2) + '\n')
            assert current['requests'][0]['output_ids_when_available'] == w['continuation_ids']
            groups[phase].append(current)
            sample('after-' + label)
    rows = [r for g in groups['measure'] for r in g['requests']]
    result = {'status': 'C1_SINGLE_ARM_COMPLETED_NO_ADOPTION', 'mode': args.mode,
        'groups': groups, 'runtime': metadata['benchmark_identity'], 'all_raw32_match_old_fixture': True,
        'latency_p50_ms': 1000 * statistics.median(r['done_perf_counter_s'] - r['start_perf_counter_s'] for r in rows),
        'total_output_tok_s': statistics.median(g['y_total_output_throughput_tok_s_mac'] for g in groups['measure']),
        'ttft_p50_ms': 1000 * statistics.median(r['client_ttft_text_s'] for r in rows),
        'scope': 'One fresh C1 arm, not paired proof or HTTP/full-frontier win', 'power_domain': args.power_domain}
    (args.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'mode', 'latency_p50_ms', 'total_output_tok_s')}))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inert', action='store_true')
    ap.add_argument('--execute-http', action='store_true')
    ap.add_argument('--mode', choices=('ordinary', 'speculative'), default='ordinary')
    ap.add_argument('--qualification-binding', type=Path, default=D / 'qualification-binding.json')
    ap.add_argument('--base-url')
    ap.add_argument('--server-pid', type=int)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--power-domain', choices=('AC Power', 'Battery Power'), default='AC Power')
    args = ap.parse_args(argv)
    if not args.execute_http:
        assert not any(n == 'mlx' or n.startswith(('mlx.', 'numpy')) for n in sys.modules)
        print(json.dumps({'status': 'CPU_READY_SHORT_C1_ACTUAL117_BOUND_GPU_UNRUN', 'HTTP_requests': 0, 'GPU_used': False}))
        return 0
    if not (args.base_url and args.base_url.startswith('http://127.0.0.1:') and args.server_pid and args.out and args.out.is_absolute()):
        ap.error('root-owned loopback server PID and absolute new output required')
    measure_arm(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
