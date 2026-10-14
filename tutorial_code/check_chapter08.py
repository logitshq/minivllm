"""Validate token budget, late admission, different prompt lengths, and batch greedy parity."""
import argparse
import json
from pathlib import Path


def scheduler_checks():
    # No MLX import: this test runs on ordinary Python, even without a GPU runtime.
    from tutorial_code.miniengine.engine.request import Request
    from tutorial_code.miniengine.engine.scheduler import Scheduler
    from tutorial_code.miniengine.sampling_params import SamplingParams
    s = Scheduler(2, 32, 3, {99})
    p = SamplingParams(max_tokens=3, ignore_eos=True)
    a, b, c = [Request(name, list(range(n)), p) for name, n in [('a',5),('b',2),('c',1)]]
    for r in [a,b,c]:
        s.add(r)
    events = []
    while s.has_work():
        out = s.schedule()
        assert out.num_tokens <= 3
        assert len(s.running) <= 2
        sampled = {r.req_id: 7 for r in out.decode}
        sampled.update({r.req_id: 7 for r,n in out.prefill if r.num_computed_tokens+n == r.num_prompt_tokens})
        events.append({'decode':[r.req_id for r in out.decode], 'prefill':[(r.req_id,n) for r,n in out.prefill]})
        s.update(out, sampled)
    assert all(r.output_token_ids == [7,7,7] for r in [a,b,c])
    assert len(s.free_slots) == 2
    assert events[0]['prefill'] == [('a',3)]
    assert any(e['decode'] and e['prefill'] for e in events)
    stop = Scheduler(1, 8, 1, {99})
    r = Request('eos',[1],SamplingParams(max_tokens=3))
    stop.add(r);out=stop.schedule();stop.update(out,{'eos':99})
    assert r.is_finished and r.output_token_ids == [99] and not stop.has_work()
    try:
        SamplingParams(max_tokens=0)
    except ValueError:
        pass
    else:
        raise AssertionError('max_tokens=0 must be rejected')
    return {'scheduler_budget_and_admission':True, 'eos_stop':True,'max_tokens_zero_rejected':True,'events':events}


def gpu_checks(model_dir, count):
    import mlx.core as mx
    from tutorial_code.miniengine.engine.llm_engine import LLMEngine
    from tutorial_code.miniengine.sampling_params import SamplingParams
    trace=json.loads((Path(__file__).parent/'examples/chapter01-trace.json').read_text())
    p=trace['prompt_ids']
    prompts=[p,p[:7],p[:12]]
    engine=LLMEngine(str(model_dir),max_seqs=3,max_len=256,max_batched_tokens=9,dtype=mx.float32)
    embedded = engine.model.embed_tokens(mx.array(p))
    probe_logits = engine.model(mx.array(p))
    assert embedded.dtype == mx.float32 and probe_logits.dtype == mx.float32, (embedded.dtype, probe_logits.dtype)
    params=SamplingParams(max_tokens=count,ignore_eos=True)
    # Compute the single-request references with exactly this model and weight format.
    single=[engine.generate([ids],params)[0] for ids in prompts]
    batch=engine.generate(prompts,params)
    assert batch == single, (batch,single)
    assert not engine.requests and not engine.runner.staging and not engine.runner.batch.size
    # A new request arrives while the first is decoding; prompt lengths differ.
    r0=engine.add_request(prompts[0],params,'late-0')
    outputs={r0:[]}
    while not outputs[r0]:
        for rid,t,_ in engine.step():outputs[rid].append(t)
    r1=engine.add_request(prompts[1],params,'late-1');outputs[r1]=[]
    while engine.has_work():
        for rid,t,_ in engine.step():outputs[rid].append(t)
    assert [outputs[r0],outputs[r1]] == single[:2]
    assert not engine.runner.staging and not engine.runner.batch.size
    return {'same_format_batch_greedy_equal':True,'late_arrival_equal':True,'finished_cache_released':True,
            'activation_dtype':str(embedded.dtype),'logits_dtype':str(probe_logits.dtype),'max_new_tokens':count,'prompt_lengths':list(map(len,prompts)),'single_outputs':single,'batch_outputs':batch}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scheduler-only',action='store_true')
    ap.add_argument('--model-dir',type=Path)
    ap.add_argument('--max-new-tokens',type=int,default=12)
    ap.add_argument('--out',type=Path)
    args=ap.parse_args()
    report=scheduler_checks()
    if not args.scheduler_only:
        if not args.model_dir:ap.error('--model-dir required for GPU checks')
        report.update(gpu_checks(args.model_dir,args.max_new_tokens))
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
