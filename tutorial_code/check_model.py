"""Chapter 05: independent full-model logits and greedy against fixed HF reference."""
import argparse
import numpy as np
import mlx.core as mx
from tutorial_code.reference_helpers import raw_model_dir, write_report, HF_REFERENCE
from tutorial_code.miniengine.model.loader import load_model

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens", type=int, default=12)
    p.add_argument("--out", default="tutorial_code/examples/chapter05-checks.json")
    p.add_argument("--reference", default=str(HF_REFERENCE))
    args = p.parse_args()
    ref = np.load(args.reference)
    if not 1 <= args.tokens <= len(ref["fp32_greedy"]):
        p.error("--tokens must be between 1 and reference length")
    model, cfg = load_model(raw_model_dir(), dtype=mx.float32)
    ids = ref["prompt"].tolist()
    logits = np.array(model(mx.array(ids)).astype(mx.float32))
    error = float(np.max(np.abs(logits-ref["fp32_logits"])))
    assert error < 1e-3, error
    generated = []
    for _ in range(args.tokens):
        nxt = int(mx.argmax(model(mx.array(ids), last_only=True)[-1]).item())
        generated.append(nxt)
        ids.append(nxt)
    assert generated == ref["fp32_greedy"][:args.tokens].tolist(), generated
    write_report(args.out, {"chapter":"05", "baseline":"fixed HF fp32 reference", "max_abs_error":error, "greedy_tokens_checked":args.tokens, "generated_ids":generated, "integration":"complete model.forward and greedy loop now use miniengine"})

if __name__ == "__main__":
    main()
