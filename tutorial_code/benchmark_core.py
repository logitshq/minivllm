"""Repeated warmed token-ID engine benchmark. HTTP-client timings belong in a separate column."""
import argparse
import hashlib
import json
import platform
import statistics
import time
from importlib.metadata import version
from pathlib import Path
import mlx.core as mx
from tutorial_code.miniengine.engine.llm_engine import LLMEngine
from tutorial_code.miniengine.sampling_params import SamplingParams


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-dir',type=Path,required=True)
    ap.add_argument('--drafter-dir',type=Path)
    ap.add_argument('--spec-block',type=int,default=5)
    ap.add_argument('--prompt-json',type=Path,default=Path(__file__).parent/'examples/chapter01-trace.json')
    ap.add_argument('--max-new-tokens',type=int,default=128)
    ap.add_argument('--repeat',type=int,default=3);ap.add_argument('--warmup-runs',type=int,default=1)
    ap.add_argument('--ignore-eos',action='store_true',help='Fixed output-step experiment; report separately from normal EOS-stopping oMLX requests')
    ap.add_argument('--fuse-decode',action='store_true',help='Opt-in B1 projection packing and GDN preparation fusion; requires --compile-decode')
    ap.add_argument('--compile-decode',action='store_true',help='Opt-in bounded M5 Ultra decode graph reuse')
    ap.add_argument('--cache-limit-gb',type=int,default=2,help='MLX allocator pool, not prompt/prefix caching')
    ap.add_argument('--dtype',choices=['float32','bfloat16'],default='bfloat16');ap.add_argument('--out',type=Path)
    args=ap.parse_args()
    if args.repeat<1 or args.warmup_runs<1 or args.max_new_tokens<2:ap.error('need repeat>=1, warmup-runs>=1, max-new-tokens>=2')
    prompt=json.loads(args.prompt_json.read_text())['prompt_ids']
    model_dir=args.model_dir.resolve()
    dtype=getattr(mx,args.dtype)
    mx.random.seed(42)
    engine=LLMEngine(str(model_dir),max_seqs=1,max_len=len(prompt)+args.max_new_tokens+16,
                     max_batched_tokens=max(512,len(prompt)),dtype=dtype,
                     drafter_dir=str(args.drafter_dir) if args.drafter_dir else None,spec_block=args.spec_block,
                     compile_decode=args.compile_decode,fuse_decode=args.fuse_decode,cache_limit_gb=args.cache_limit_gb)
    params=SamplingParams(max_tokens=args.max_new_tokens,temperature=0,ignore_eos=args.ignore_eos)
    def run():
        if hasattr(mx,'reset_peak_memory'):mx.reset_peak_memory()
        mx.synchronize()  # Exclude previous pending GPU work before the clock starts.
        start=time.perf_counter();rid=engine.add_request(prompt,params)
        ids=[];first=None;first_delivery_tokens=0;round_times=[]
        while engine.has_work():
            output=engine.step()  # ModelRunner calls mx.eval before handing sampled IDs to Python.
            finished=time.perf_counter()
            if output:
                produced=[tok for req,tok,_ in output if req==rid]
                if first is None:
                    first=finished;first_delivery_tokens=len(produced)
                ids.extend(produced);round_times.append({'elapsed_s':finished-start,'tokens':len(produced)})
        mx.synchronize();end=time.perf_counter()
        total=end-start;decode=end-first if first else None
        terminal_eos = bool(ids and not args.ignore_eos and ids[-1] in engine.eos_ids)
        assert not engine.runner.staging and not engine.runner.batch.size
        stats=engine.runner.spec.finished_stats.pop(rid,None) if engine.runner.spec else None
        return {'output_ids':ids,'output_tokens':len(ids),'engine_elapsed_s':total,
                'terminal_eos_in_output_ids':terminal_eos,
                'logical_completion_tokens':len(ids)-int(terminal_eos),
                'engine_ttft_s':first-start if first else None,'post_first_delivery_s':decode,'first_delivery_tokens':first_delivery_tokens,
                'decode_tokens_per_s':(len(ids)-first_delivery_tokens)/decode if decode and len(ids)>first_delivery_tokens else None,
                'output_tokens_per_engine_s':len(ids)/total,'step_delivery_times':round_times,
                'accepted_drafts_and_rounds':stats,
                'mlx_peak_memory_bytes':mx.get_peak_memory() if hasattr(mx,'get_peak_memory') else None,
                'mlx_active_memory_after_finish_bytes':mx.get_active_memory() if hasattr(mx,'get_active_memory') else None}
    for _ in range(args.warmup_runs):run()
    runs=[run() for _ in range(args.repeat)]
    assert all(r['output_ids']==runs[0]['output_ids'] for r in runs), 'greedy IDs changed between repetitions'
    config_bytes=(model_dir/'config.json').read_bytes();config=json.loads(config_bytes)
    report={'baseline':'tutorial miniengine','measurement_scope':'token-ID synchronous engine; includes Python scheduling and GPU evaluation; excludes loading/tokenization/HTTP',
            'platform':platform.platform(),'machine':platform.machine(),'processor':platform.processor(),'random_seed':42,'mlx_version':version('mlx'),'model_dir':str(model_dir),
            'config_sha256':hashlib.sha256(config_bytes).hexdigest(),'quantization':config.get('quantization'),
            'model_shape':{name:getattr(engine.cfg,name,None) for name in ['hidden_size','num_hidden_layers','vocab_size','num_attention_heads','num_key_value_heads','head_dim','linear_num_key_heads','linear_num_value_heads','linear_key_head_dim','linear_value_head_dim']},
            'memory_scope':'MLX allocator active and peak bytes; not process RSS or total unified-memory use',
            'output_count_definition':'output_tokens counts every emitted ID, including a terminal EOS; logical_completion_tokens excludes a terminal EOS when EOS stopping is enabled. Rates use emitted-ID counts.',
            'decode_rate_definition':'(output tokens minus first delivery tokens)/(end minus first delivery time); token ID delivery can be a speculative burst',
            'dtype':args.dtype,'compile_decode':args.compile_decode,'fuse_decode':args.fuse_decode,'allocator_cache_limit_gb':args.cache_limit_gb,
            'drafter_dir':str(args.drafter_dir) if args.drafter_dir else None,
            'request':{'prompt_ids':prompt,'prompt_tokens':len(prompt),'max_new_tokens':args.max_new_tokens,
                       'ignore_eos':args.ignore_eos,'eos_token_ids':sorted(engine.eos_ids),'temperature':0,'concurrency':1},
            'warmup_runs':args.warmup_runs,'repeat':args.repeat,'runs':runs,
            'median_engine_ttft_s':statistics.median(r['engine_ttft_s'] for r in runs),
            'median_output_tokens_per_engine_s':statistics.median(r['output_tokens_per_engine_s'] for r in runs),
            'median_decode_tokens_per_s':statistics.median(v for r in runs if (v:=r['decode_tokens_per_s']) is not None) if any(r['decode_tokens_per_s'] is not None for r in runs) else None,
            'comparison_rule':'Keep oMLX HTTP and Lithos timing scopes separate; do not derive a speedup from different timing boundaries or published projections.'}
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
