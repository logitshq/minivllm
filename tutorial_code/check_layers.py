"""Chapter 02: compare our Embedding/Linear with HF using real weights.

HF still supplies the decoder layers and final norm. We capture the tensor
entering its output head so our Linear gets exactly the same input.
"""
import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM

from layers import Embedding, Linear


def to_numpy(x):
    return x.detach().float().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--report-json", type=Path)
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS unavailable; use --device cpu")

    # Hand-checkable example: select rows, then take a dot product per row.
    toy_weight = mx.array([[1., 0., 1.], [0., 1., 1.], [1., 1., 0.],
                           [-1., 0., 1.], [0., -1., 1.]])
    toy_vectors = Embedding(toy_weight)(mx.array([2, 0]))
    toy_logits = Linear(toy_weight)(mx.array([[1., 2., 0.]]))
    assert toy_vectors.tolist() == [[1., 1., 0.], [1., 0., 1.]]
    assert toy_logits.tolist() == [[1., 2., 3., -1., -2.]]
    print("toy embedding:", toy_vectors.tolist())
    print("toy logits:", toy_logits.tolist())

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
    hf_embedding = model.get_input_embeddings()
    hf_head = model.get_output_embeddings()
    embedding_weight = to_numpy(hf_embedding.weight)
    weights_shared = hf_embedding.weight.data_ptr() == hf_head.weight.data_ptr()
    head_weight = embedding_weight if weights_shared else to_numpy(hf_head.weight)
    embedding = Embedding(mx.array(embedding_weight))
    head = Linear(embedding.weight if weights_shared else mx.array(head_weight))

    # Capture the contextual, normalized hidden vectors entering lm_head.
    captured = {}
    def capture_head_input(module, inputs):
        captured["hidden"] = inputs[0].detach()
    hook = hf_head.register_forward_pre_hook(capture_head_input)
    ids = torch.tensor([prompt_ids], device=args.device)
    try:
        with torch.inference_mode():
            ref_embedding = to_numpy(hf_embedding(ids))
            outputs = model(input_ids=ids, use_cache=False)
            ref_logits = to_numpy(outputs.logits)
            hidden = to_numpy(captured["hidden"])
    finally:
        hook.remove()

    ours_embedding = embedding(mx.array([prompt_ids]))
    ours_logits = head(mx.array(hidden))
    mx.eval(ours_embedding, ours_logits)
    emb_error = float(np.abs(np.array(ours_embedding) - ref_embedding).max())
    head_error = float(np.abs(np.array(ours_logits) - ref_logits).max())
    agree = bool(np.array_equal(np.array(ours_logits).argmax(-1), ref_logits.argmax(-1)))
    next_id = int(mx.argmax(ours_logits[0, -1]).item())
    assert emb_error == 0., f"Embedding mismatch: {emb_error}"
    assert head_error < 1e-3, f"Linear mismatch: {head_error}"
    assert agree, "Output-head argmax mismatch"
    assert next_id == first_trace["generated_ids"][0]

    report = {
        "chapter": "02", "model_snapshot": model_dir,
        "hf_device": args.device, "dtype": "float32",
        "embedding_weight_shape": list(embedding_weight.shape),
        "ids_shape": list(ids.shape),
        "embedding_output_shape": list(ours_embedding.shape),
        "output_head_input_shape": list(hidden.shape),
        "logits_shape": list(ours_logits.shape),
        "weights_shared": weights_shared,
        "embedding_max_error": emb_error, "output_head_max_error": head_error,
        "argmax_all_positions_agree": agree, "next_token_id": next_id,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
