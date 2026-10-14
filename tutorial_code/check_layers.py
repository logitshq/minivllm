"""Chapter 02 v2: check basic blocks with hand examples and real HF inputs.

HF supplies Attention/GDN and the full reference forward. Our independent MLX
blocks get the exact captured inputs and weights; this isolates each component.
"""
import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM

from layers import Embedding, FeedForwardBlock, Linear, MLP, RMSNorm, silu


def to_numpy(x):
    return x.detach().float().cpu().numpy()


def linear_from_hf(layer):
    return Linear(mx.array(to_numpy(layer.weight)))


def norm_from_hf(layer):
    return RMSNorm(mx.array(to_numpy(layer.weight)), eps=layer.eps)


def check(name, actual, expected, errors, tolerance=1e-3):
    mx.eval(actual)
    error = float(np.abs(np.array(actual) - expected).max())
    assert error < tolerance, f"{name} mismatch: {error}"
    errors[name] = error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--report-json", type=Path)
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS unavailable; use --device cpu")

    errors = {}
    toy_weight = mx.array([[1., 0., 1.], [0., 1., 1.], [1., 1., 0.],
                           [-1., 0., 1.], [0., -1., 1.]])
    check("toy_embedding", Embedding(toy_weight)(mx.array([2, 0])),
          np.array([[1., 1., 0.], [1., 0., 1.]]), errors, 1e-6)
    check("toy_linear", Linear(toy_weight)(mx.array([[1., 2., 0.]])),
          np.array([[1., 2., 3., -1., -2.]]), errors, 1e-6)
    toy_x = np.array([[3., 4.]], dtype=np.float32)
    check("toy_rms_norm", RMSNorm(mx.zeros(2), eps=0.)(mx.array(toy_x)),
          toy_x / np.sqrt(np.mean(toy_x**2, axis=-1, keepdims=True)), errors, 1e-6)
    toy_act_x = np.array([-2., 0., 2.], dtype=np.float32)
    check("toy_silu", silu(mx.array(toy_act_x)),
          toy_act_x / (1. + np.exp(-toy_act_x)), errors, 1e-6)
    identity = Linear(mx.eye(2))
    toy_mlp = MLP(identity, identity, identity)
    toy_ff_x = np.array([[1., 2.]], dtype=np.float32)
    check("toy_mlp", toy_mlp(mx.array(toy_ff_x)),
          (toy_ff_x / (1. + np.exp(-toy_ff_x))) * toy_ff_x, errors, 1e-6)
    zero = Linear(mx.zeros((2, 2)))
    zero_branch = FeedForwardBlock(RMSNorm(mx.zeros(2)), MLP(identity, identity, zero))
    check("zero_branch_preserves_residual", zero_branch(mx.array(toy_ff_x)), toy_ff_x, errors, 1e-6)

    model_dir = snapshot_download(
        "Qwen/Qwen3.5-0.8B",
        revision="2fc06364715b967f1860aea9cf38778875588b17",
        local_files_only=True,
    )
    first_trace = json.loads((Path(__file__).parent / "examples/chapter01-trace.json").read_text())
    prompt_ids = first_trace["prompt_ids"]
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, dtype=torch.float32,
    ).to(args.device).eval()
    backbone = model.model
    if hasattr(backbone, "language_model"):
        backbone = backbone.language_model
    layer = backbone.layers[0]
    hf_embedding = model.get_input_embeddings()
    hf_head = model.get_output_embeddings()
    weights_shared = hf_embedding.weight.data_ptr() == hf_head.weight.data_ptr()
    embedding_weight = to_numpy(hf_embedding.weight)
    embedding = Embedding(mx.array(embedding_weight))
    head = Linear(embedding.weight) if weights_shared else linear_from_hf(hf_head)
    input_norm = norm_from_hf(layer.input_layernorm)
    post_norm = norm_from_hf(layer.post_attention_layernorm)
    final_norm = norm_from_hf(backbone.norm)
    mlp = MLP(linear_from_hf(layer.mlp.gate_proj),
              linear_from_hf(layer.mlp.up_proj), linear_from_hf(layer.mlp.down_proj))
    feed_forward = FeedForwardBlock(post_norm, mlp)

    captured = {}
    def capture(name):
        def hook(module, inputs, output):
            captured[name + "_in"] = to_numpy(inputs[0])
            captured[name + "_out"] = to_numpy(output)
        return hook
    hooks = [
        layer.input_layernorm.register_forward_hook(capture("input_norm")),
        layer.post_attention_layernorm.register_forward_hook(capture("post_norm")),
        layer.mlp.register_forward_hook(capture("mlp")),
        layer.register_forward_hook(capture("layer")),
        backbone.norm.register_forward_hook(capture("final_norm")),
        hf_head.register_forward_hook(capture("head")),
    ]
    ids = torch.tensor([prompt_ids], device=args.device)
    try:
        with torch.inference_mode():
            ref_embedding = to_numpy(hf_embedding(ids))
            outputs = model(input_ids=ids, use_cache=False)
            ref_logits = to_numpy(outputs.logits)
    finally:
        for hook in hooks:
            hook.remove()

    ours_embedding = embedding(mx.array([prompt_ids]))
    check("embedding", ours_embedding, ref_embedding, errors, 1e-6)
    for name, block in [("input_norm", input_norm), ("post_norm", post_norm),
                        ("final_norm", final_norm), ("mlp", mlp)]:
        check(name, block(mx.array(captured[name + "_in"])), captured[name + "_out"], errors)
    # post_norm input already includes HF's first mixer residual. Rebuild the
    # complete MLP sub-block with our norm, MLP and residual addition.
    ours_ff = feed_forward(mx.array(captured["post_norm_in"]))
    check("feed_forward_with_residual", ours_ff, captured["layer_out"], errors)
    ours_logits = head(mx.array(captured["head_in"]))
    check("output_head", ours_logits, ref_logits, errors)
    ours_norm_head = head(final_norm(mx.array(captured["final_norm_in"])))
    check("final_norm_and_output_head", ours_norm_head, ref_logits, errors)
    agree = bool(np.array_equal(np.array(ours_norm_head).argmax(-1), ref_logits.argmax(-1)))
    next_id = int(mx.argmax(ours_norm_head[0, -1]).item())
    assert agree, "Final norm + output head argmax mismatch"
    assert next_id == first_trace["generated_ids"][0]

    report = {
        "chapter": "02-v2", "model_snapshot": model_dir,
        "hf_device": args.device, "dtype": "float32", "checked_layer": 0,
        "embedding_weight_shape": list(embedding.weight.shape),
        "ids_shape": list(ids.shape), "embedding_output_shape": list(ours_embedding.shape),
        "mlp_gate_weight_shape": list(mlp.gate_proj.weight.shape),
        "mlp_down_weight_shape": list(mlp.down_proj.weight.shape),
        "feed_forward_shape": list(ours_ff.shape),
        "logits_shape": list(ours_norm_head.shape), "weights_shared": weights_shared,
        "max_errors": errors, "argmax_all_positions_agree": agree, "next_token_id": next_id,
        "scope": "isolated blocks, one MLP residual sub-block and final norm+head; full decoder still HF",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
