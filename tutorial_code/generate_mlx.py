"""Chapter 06: generate with the independent MLX model. Tokenizer stays at the boundary."""
import argparse
import json
from pathlib import Path
import mlx.core as mx
from tutorial_code.reference_helpers import raw_model_dir, write_report
from tutorial_code.miniengine.model.loader import load_model
from tutorial_code.miniengine.eos import read_eos_ids
from tutorial_code.miniengine.cache import make_cache
from tutorial_code.miniengine.context import ForwardContext

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt", default="用一句话解释什么是 KV cache。")
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--model-dir", type=Path)
    p.add_argument("--dtype", choices=("float32","bfloat16"), default="float32")
    p.add_argument("--trace-json", type=Path)
    args = p.parse_args()
    if args.max_new_tokens < 1:
        p.error("--max-new-tokens must be positive")
    from transformers import AutoTokenizer
    model_dir = args.model_dir or raw_model_dir()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    prompt_ids = tokenizer.apply_chat_template([{"role":"user", "content":args.prompt}], tokenize=True, add_generation_prompt=True, enable_thinking=False)
    if hasattr(prompt_ids, "keys"):
        prompt_ids = prompt_ids["input_ids"]
    prompt_ids = list(prompt_ids)
    model, cfg = load_model(model_dir, dtype=getattr(mx, args.dtype))
    eos = read_eos_ids(str(model_dir))
    ids, out, rounds = list(prompt_ids), [], []
    ctx = ForwardContext("prefill", make_cache(model))
    current = list(prompt_ids)
    for step in range(args.max_new_tokens):
        logits = model(mx.array(current), ctx, last_only=True)
        nxt = int(mx.argmax(logits[-1]).item())
        rounds.append({"round":step+1,"input_tokens":len(current),"logits_shape":list(logits.shape),"next_token_id":nxt})
        ids.append(nxt); out.append(nxt)
        current = [nxt]
        if nxt in eos:
            break
    print("generated IDs:", out)
    print("generated text:", tokenizer.decode(out, skip_special_tokens=True))
    if args.trace_json:
        write_report(args.trace_json, {"chapter":"06", "model_snapshot":str(model_dir),"dtype":args.dtype,"use_cache":True,"prompt_ids":prompt_ids,"generated_ids":out,"rounds":rounds})

if __name__ == "__main__":
    main()
