"""Chapter 01: observe next-token generation with a reference model.

This version deliberately recomputes the full prefix with use_cache=False.
It does not import the finished minivllm engine. Later chapter commits replace
the reference implementation and then add cache, batching, and scheduling.
"""
import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="用一句话解释什么是 KV cache。")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--trace-json", type=Path)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable; run with --device cpu")

    model_dir = args.model_dir or Path(snapshot_download(
        "Qwen/Qwen3.5-0.8B",
        revision="2fc06364715b967f1860aea9cf38778875588b17",
        local_files_only=True,
    ))
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, dtype=torch.float32,
    ).to(args.device).eval()
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False,
        return_dict=True, return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(args.device)
    prompt_ids = input_ids[0].tolist()
    eos_ids = model.generation_config.eos_token_id
    eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
    generated = []
    rounds = []
    print(f"model snapshot: {model_dir}")
    print(f"prompt tokens: {len(prompt_ids)}")

    with torch.inference_mode():
        for step in range(args.max_new_tokens):
            # Every round processes the entire known prefix. No KV/GDN cache.
            logits = model(input_ids=input_ids, use_cache=False).logits
            next_id = int(logits[0, -1].argmax().item())
            rounds.append({
                "round": step + 1,
                "input_tokens": input_ids.shape[1],
                "logits_shape": list(logits.shape),
                "next_token_id": next_id,
            })
            print(f"round {step + 1}: input={input_ids.shape[1]}, "
                  f"logits={tuple(logits.shape)}, next_id={next_id}")
            generated.append(next_id)
            input_ids = torch.cat([
                input_ids,
                torch.tensor([[next_id]], device=args.device),
            ], dim=1)
            if next_id in eos_ids:
                break

    print("generated IDs:", generated)
    print("generated text:", tokenizer.decode(generated, skip_special_tokens=True))
    if args.trace_json:
        trace = {
            "chapter": "01", "model_snapshot": str(model_dir),
            "device": args.device, "dtype": "float32", "use_cache": False,
            "prompt": args.prompt, "prompt_ids": prompt_ids,
            "generated_ids": generated, "rounds": rounds,
        }
        args.trace_json.parent.mkdir(parents=True, exist_ok=True)
        args.trace_json.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
