"""Chapter 16: unadopted experiment-branch GDN fusion; token-ID smoke only."""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--fusion", choices=("off", "on"), default="off")
    args = ap.parse_args()
    cfg_path = Path(__file__).parent / "examples/chapter16-gdn-fusion.json"
    cfg = json.loads(cfg_path.read_text())
    prompt, expected = cfg["prompt_ids"], cfg["expected_output_ids"]
    if len(prompt) != 19 or not all(type(v) is int for v in prompt):
        ap.error("expected the qualified 19-token prompt")
    if args.model_dir.resolve() != Path(cfg["checkpoint"]["actual_path"]).resolve():
        ap.error("use the pinned Qwen3.8-27B-nvfp4 snapshot in this example")

    import mlx.core as mx
    from tutorial_code.miniengine.engine.llm_engine import LLMEngine
    from tutorial_code.miniengine.model.gdn import GatedDeltaNet
    from tutorial_code.miniengine.sampling_params import SamplingParams

    engine = LLMEngine(str(args.model_dir.resolve()), dtype=mx.bfloat16, **cfg["engine"])
    gdns = [layer.linear_attn for layer in engine.model.layers if not layer.is_attn]
    if len(gdns) != 48 or not all(type(g) is GatedDeltaNet for g in gdns):
        raise ValueError("this example targets the qualified 48-GDN 27B model")
    enabled = args.fusion == "on"
    for gdn in gdns:
        gdn.set_gates_compile(True)  # Keep the accepted chapter15 path in both arms.
        gdn.set_postconv_fusion(enabled)  # Independent switch, after model construction.
    output_ids = engine.generate([prompt], SamplingParams(**cfg["generation"]))[0]
    assert not engine.requests and not engine.runner.staging and not engine.runner.batch.size
    if output_ids != expected:
        raise AssertionError("actual output IDs differ from qualified fullmodel IDs")
    print(json.dumps({
        "scope": "token-ID smoke; no performance or dispatch-count claim",
        "model_dir": str(args.model_dir.resolve()), "prompt_ids": prompt,
        "gates_compile_requested": True, "fusion_requested": args.fusion,
        "gdn_layers": len(gdns), "expected_ids_exact": True,
        "output_ids": output_ids, "actual_output_tokens": len(output_ids),
        "cache_limit_gb": 0, "drafter_dir": None, **cfg["generation"],
        "note": "unsupported signatures retain the original path even when on",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
