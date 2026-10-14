"""Evaluate both custom Metal kernels against the previous chapter's MLX reference."""
import argparse
import json
import statistics
import time
from pathlib import Path
import mlx.core as mx
from tutorial_code.miniengine.kernels.gdn import gated_delta_step_batch, _solve_unit_lower, gated_delta_rule_recurrent, gated_delta_rule_chunk
from tutorial_code.miniengine.kernels.gdn_metal import (gated_delta_step_metal,solve_unit_lower_metal,DISPATCH_COUNTS,reset_dispatch_counts)


def error(a,b):return float(mx.max(mx.abs(a-b)).item())


def timed(fn, warmups=2, repeats=5):
    def evaluate():
        result=fn()
        mx.eval(*(result if isinstance(result, tuple) else [result]))
    for _ in range(warmups):evaluate()
    samples=[]
    for _ in range(repeats):
        mx.synchronize();start=time.perf_counter()
        evaluate();mx.synchronize()
        samples.append(time.perf_counter()-start)
    return {'warmups':warmups,'repeats':repeats,'samples_s':samples,'median_s':statistics.median(samples),
            'timing_scope':'one complete evaluated operation; includes graph building and dispatch'}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path);args=ap.parse_args()
    mx.random.seed(42);reset_dispatch_counts();checks=[]
    for b in [1,3]:
        shape=(b,4,128)
        q,k,v=[mx.random.normal(shape) for _ in range(3)]
        g=-mx.random.uniform(shape=(b,4));beta=mx.sigmoid(mx.random.normal((b,4)))
        S=mx.random.normal((b,4,128,128))*.1
        ref,rs=gated_delta_step_batch(q,k,v,g,beta,S)
        actual,ms=gated_delta_step_metal(q,k,v,g,beta,S)
        mx.eval(ref,rs,actual,ms)
        errors={'output_max_abs':error(ref,actual),'state_max_abs':error(rs,ms)}
        assert max(errors.values()) < 2e-5, errors
        checks.append({'kernel':'decode_step','batch':b,'shape':list(S.shape),'dtype':'float32',**errors,
                       'mlx_reference_timing':timed(lambda:gated_delta_step_batch(q,k,v,g,beta,S)),
                       'metal_timing':timed(lambda:gated_delta_step_metal(q,k,v,g,beta,S))})
    for c in [8,16,64]:
        A=mx.tril(mx.random.normal((2,c,c))*.02,-1)
        V=mx.random.normal((2,c,256))
        ref=_solve_unit_lower(A,V);actual=solve_unit_lower_metal(A,V)
        mx.eval(ref,actual)
        diff=error(ref,actual)
        assert diff < 1e-5,diff
        checks.append({'kernel':'unit_lower_solve','chunk':c,'shape_A':list(A.shape),'shape_V':list(V.shape),'dtype':'float32','max_abs':diff,
                       'mlx_reference_timing':timed(lambda:_solve_unit_lower(A,V)),
                       'metal_timing':timed(lambda:solve_unit_lower_metal(A,V))})
    T,H,D=73,2,128
    q,k,v=[mx.random.normal((T,H,D)) for _ in range(3)]
    g=-mx.random.uniform(shape=(T,H));beta=mx.sigmoid(mx.random.normal((T,H)))
    state=mx.random.normal((H,D,D))*.1
    ro,rs=gated_delta_rule_recurrent(q,k,v,g,beta,state)
    co,cs=gated_delta_rule_chunk(q,k,v,g,beta,state,chunk=16)
    mx.eval(ro,rs,co,cs)
    tail_errors={'output_max_abs':error(ro,co),'state_max_abs':error(rs,cs)}
    assert max(tail_errors.values()) < 2e-5,tail_errors
    checks.append({'kernel':'chunk_vs_recurrent','tokens':T,'chunk':16,'tail_tokens':T%16,'nonzero_initial_state':True,**tail_errors})
    assert DISPATCH_COUNTS['decode_step'] == 2*(1+2+5)
    assert DISPATCH_COUNTS['unit_lower_solve'] == 3*(1+2+5)+1
    report={'checks':checks,'evaluated_dispatch_counts':dict(DISPATCH_COUNTS),
            'scope':'correctness, evaluated Metal execution, and warmed operation-level timing; no full-engine speedup claim'}
    print(json.dumps(report,indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
