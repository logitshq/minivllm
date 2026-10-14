"""Greedy acceptance, attention rewind, GDN replay; optional real target + DFlash2 parity."""
import argparse
import json
from pathlib import Path
import mlx.core as mx
from tutorial_code.miniengine.cache import AttnCache,GDNCache
from tutorial_code.miniengine.kernels.gdn import gated_delta_rule_recurrent
from tutorial_code.miniengine.engine.spec_runner import accept_greedy_prefix


def max_error(a,b):return float(mx.max(mx.abs(a-b)).item())


def rollback_checks():
    assert accept_greedy_prefix([10,11,12],[10,11,90,91]) == ([10,11,90],2)
    assert accept_greedy_prefix([10,11],[99,11,12]) == ([99],0)
    assert accept_greedy_prefix([10,11],[10,11,12]) == ([10,11,12],2)
    kv=AttnCache();initial=mx.arange(6,dtype=mx.float32).reshape(1,1,3,2)
    kv.update(initial,initial);kv.snapshot()
    kv.update(mx.ones((1,1,4,2)),mx.ones((1,1,4,2)));kv.rollback(2)
    assert kv.offset==5
    got,_=kv.update(mx.full((1,1,1,2),7.),mx.full((1,1,1,2),7.))
    assert got.shape[2]==6 and max_error(got[:,:,:3],initial)==0 and float(got[0,0,-1,0].item())==7
    mx.random.seed(9)
    T,H,D=5,2,128
    q,k,v=[mx.random.normal((T,H,D)) for _ in range(3)]
    g=-mx.random.uniform(shape=(T,H));beta=mx.sigmoid(mx.random.normal((T,H)))
    S0=mx.random.normal((H,D,D))*.1
    raw=mx.random.normal((T,6));conv0=mx.random.normal((3,6))
    errors=[]
    for n in [0,1,3,5]:
        c=GDNCache();c.S=S0;c.conv_state=conv0;c.offset=7;c.snapshot()
        _,c.S=gated_delta_rule_recurrent(q,k,v,g,beta,S0)
        c.conv_state=mx.concatenate([conv0,raw])[-3:];c.offset+=T;c.tape=(raw,q,k,v,g,beta)
        c.rollback(n)
        expected=S0 if n==0 else gated_delta_rule_recurrent(q[:n],k[:n],v[:n],g[:n],beta[:n],S0)[1]
        expected_conv=mx.concatenate([conv0,raw[:n]])[-3:]
        mx.eval(c.S,expected,c.conv_state)
        err=max_error(c.S,expected);assert err<2e-5,err
        assert max_error(c.conv_state,expected_conv)==0 and c.offset==7+n
        assert c._snap is None and c.tape is None and not c.record
        errors.append({'kept_verify_tokens':n,'state_max_abs':err})
    return {'greedy_accept_cases_passed':3,'attention_offset_and_overwrite_passed':True,'gdn_replay_checks':errors}


def spec_checks(model_dir,drafter_dir,count,block):
    from tutorial_code.miniengine.engine.llm_engine import LLMEngine
    from tutorial_code.miniengine.engine.model_runner import ModelRunner
    from tutorial_code.miniengine.spec.dflash2 import load_drafter
    from tutorial_code.miniengine.sampling_params import SamplingParams
    trace=json.loads((Path(__file__).parent/'examples/chapter01-trace.json').read_text())
    prompt=trace['prompt_ids'];params=SamplingParams(max_tokens=count,ignore_eos=True)
    engine=LLMEngine(str(model_dir),max_seqs=1,max_len=len(prompt)+count+16,max_batched_tokens=512)
    plain=engine.generate([prompt],params)[0]
    drafter=load_drafter(str(drafter_dir))
    engine.runner=ModelRunner(engine.model,len(prompt)+count+16,drafter,block)
    speculative=engine.generate([prompt],params)[0]
    assert speculative==plain,(plain,speculative)
    assert not engine.runner.staging and not engine.runner.batch.size and not engine.runner.feats
    try:engine.add_request(prompt,SamplingParams(max_tokens=2,temperature=.7))
    except ValueError:pass
    else:raise AssertionError('spec non-greedy must be rejected')
    stats=engine.runner.spec.finished_stats
    assert stats and sum(rounds for accepted,rounds in stats.values())>0
    return {'real_model_greedy_equal':True,'temperature_guard':True,'finished_cache_released':True,
            'model_dir':str(model_dir),'drafter_dir':str(drafter_dir),'max_new_tokens':count,
            'plain_ids':plain,'speculative_ids':speculative,'finished_stats':stats,
            'scope':'same target weights greedy parity on this prompt; no throughput claim'}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-dir',type=Path);ap.add_argument('--drafter-dir',type=Path)
    ap.add_argument('--max-new-tokens',type=int,default=16);ap.add_argument('--block',type=int,default=5)
    ap.add_argument('--out',type=Path);args=ap.parse_args()
    report=rollback_checks()
    if args.model_dir or args.drafter_dir:
        if not(args.model_dir and args.drafter_dir):ap.error('real spec check requires both model and drafter directories')
        report.update(spec_checks(args.model_dir,args.drafter_dir,args.max_new_tokens,args.block))
    else:report['real_model_spec_test']='not requested; algorithm and cache replay only'
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
