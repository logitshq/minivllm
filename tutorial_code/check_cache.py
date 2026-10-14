"""Chapter 06: cached/chunked logits equal an uncached full-prefix forward."""
import argparse
import numpy as np
import mlx.core as mx
from tutorial_code.reference_helpers import raw_model_dir, write_report, HF_REFERENCE
from tutorial_code.miniengine.model.loader import load_model
from tutorial_code.miniengine.cache import make_cache
from tutorial_code.miniengine.context import ForwardContext

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="tutorial_code/examples/chapter06-checks.json")
    p.add_argument("--reference", default=str(HF_REFERENCE))
    args = p.parse_args()
    ref = np.load(args.reference)
    model, cfg = load_model(raw_model_dir(), dtype=mx.float32)
    prompt = ref["prompt"].tolist()
    full_ids = prompt + ref["fp32_greedy"][:12].tolist()
    full = np.array(model(mx.array(full_ids)))
    ctx = ForwardContext("prefill", make_cache(model))
    parts = [np.array(model(mx.array(prompt), ctx))]
    for token in full_ids[len(prompt):]:
        parts.append(np.array(model(mx.array([token]), ctx)))
    cached = np.concatenate(parts)
    error = float(np.max(np.abs(cached-full)))
    assert error < 1e-2, error
    assert (cached.argmax(-1) == full.argmax(-1)).all()
    assert all(c.offset == len(full_ids) for c in ctx.caches)
    chunks = {}
    one = np.array(model(mx.array(prompt)))
    for size in (7,13):
        ctx = ForwardContext("prefill", make_cache(model))
        out = np.concatenate([np.array(model(mx.array(prompt[i:i+size]),ctx)) for i in range(0,len(prompt),size)])
        diff = float(np.max(np.abs(out-one)))
        assert diff < 1e-2 and (out.argmax(-1) == one.argmax(-1)).all()
        assert all(c.offset == len(prompt) for c in ctx.caches)
        chunks[str(size)] = diff
    write_report(args.out, {"chapter":"06", "dtype":"float32", "prompt_tokens":len(prompt),"decode_tokens_checked":12,"cached_max_abs_error":error,"chunk_sizes_max_abs_error":chunks,"integration":"generation uses single-request cache; no scheduler/batch yet"})

if __name__ == "__main__":
    main()
