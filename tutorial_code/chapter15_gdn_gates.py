"""Chapter 15: explicit GDN-gates opt-in; token-ID smoke, not an HTTP benchmark."""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--gates", choices=("off", "on"), default="off")
    ap.add_argument("--prompt-json", type=Path,
                    default=Path(__file__).parent / "examples/chapter01-trace.json")
    args = ap.parse_args()
    prompt = json.loads(args.prompt_json.read_text())["prompt_ids"]
    if len(prompt) != 19 or not all(type(v) is int for v in prompt):
        ap.error("this example uses the existing 19-token chapter01 prompt")

    import mlx.core as mx
    from tutorial_code.miniengine.engine.llm_engine import LLMEngine
    from tutorial_code.miniengine.model.gdn import GatedDeltaNet
    from tutorial_code.miniengine.sampling_params import SamplingParams

    model_dir = str(args.model_dir.resolve())
    engine = LLMEngine(model_dir, max_seqs=4, max_len=2048,
                       max_batched_tokens=512, dtype=mx.bfloat16)
    gdns = [layer.linear_attn for layer in engine.model.layers if not layer.is_attn]
    if len(gdns) != 48 or not all(type(g) is GatedDeltaNet for g in gdns):
        raise ValueError("this example targets the qualified 48-GDN 27B model")
    enabled = args.gates == "on"
    for gdn in gdns:
        gdn.set_gates_compile(enabled)  # Explicitly opt in after model construction.
    params = SamplingParams(max_tokens=32, temperature=0, ignore_eos=False)
    output_ids = engine.generate([prompt], params)[0]
    assert not engine.requests and not engine.runner.staging and not engine.runner.batch.size
    callable_count = sum(callable(object.__getattribute__(g, "_compiled_decode_gates"))
                         for g in gdns)
    print(json.dumps({
        "scope": "token-ID single-request smoke; no performance claim",
        "model_dir": model_dir, "prompt_ids": prompt,
        "gates_requested": args.gates, "gdn_layers": len(gdns),
        "compiled_wrappers_created": callable_count,
        "output_ids": output_ids, "actual_output_tokens": len(output_ids),
        "temperature": 0, "ignore_eos": False,
        "note": "unsupported signatures retain original gates even when on",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
