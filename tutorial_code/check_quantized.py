"""Chapter 07: compare the same quantized checkpoint with its mlx-lm reference."""
import argparse
import numpy as np
import mlx.core as mx
from tutorial_code.reference_helpers import raw_model_dir, write_report, HF_REFERENCE
from tutorial_code.reference_helpers import NV_REFERENCE
from tutorial_code.miniengine.model.loader import load_model

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", default="/Users/ryanli/code/minivllm/models/Qwen3.5-0.8B-nvfp4")
    p.add_argument("--tokens", type=int, default=12)
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    p.add_argument("--out", default="tutorial_code/examples/chapter07-checks.json")
    p.add_argument("--reference", default=str(NV_REFERENCE))
    args = p.parse_args()
    ref = np.load(args.reference)
    if not 8 <= args.tokens <= len(ref["greedy"]):
        p.error("--tokens must be between 8 and reference length")
    dtype = getattr(mx, args.dtype)
    model,cfg = load_model(args.model_dir, dtype=dtype)
    assert cfg.quantization["mode"] == "nvfp4"
    ids = ref["prompt"].tolist()
    embedding = model.embed_tokens(mx.array(ids))
    actual_logits = model(mx.array(ids))
    assert embedding.dtype == dtype and actual_logits.dtype == dtype, (embedding.dtype, actual_logits.dtype, dtype)
    logits = np.array(actual_logits.astype(mx.float32))
    error = float(np.max(np.abs(logits-ref["logits"])))
    agreement = float((logits.argmax(-1) == ref["logits"].argmax(-1)).mean())
    assert error < 2.0 and agreement >= .8, (error,agreement)
    out=[]
    for _ in range(args.tokens):
        nxt=int(mx.argmax(model(mx.array(ids),last_only=True)[-1]).item())
        ids.append(nxt);out.append(nxt)
    target=ref["greedy"][:args.tokens].tolist()
    prefix = next((i for i,(a,b) in enumerate(zip(out,target)) if a != b),args.tokens)
    assert prefix >= 8, prefix
    write_report(args.out,{"chapter":"07","baseline":"mlx-lm same nvfp4 checkpoint reference","quantization":cfg.quantization,"requested_dtype":args.dtype,"embedding_dtype":str(embedding.dtype),"logits_dtype":str(actual_logits.dtype),"max_abs_error":error,"prompt_argmax_agreement":agreement,"greedy_prefix_match":prefix,"greedy_tokens_checked":args.tokens,"generated_ids":out,"note":"quantized outputs are not required to equal raw HF fp32"})

if __name__ == "__main__":
    main()
