"""HF fixture adapted from tutorial-02-v2:tutorial_code/check_layers.py.

Fixed baseline commit: f3b203ba44b135693bf0c6d6dbc82dd8072e9a63.
HF supplies the full model and real intermediate inputs. Notebook-defined blocks
are checked separately; this helper is not a handwritten full-model forward.
Importing this module does not import MLX, Torch, Transformers, or a model.
"""

import gc
import json
from pathlib import Path


MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"


def load_hf_fixture(device="mps") -> dict:
    """Run one HF reference forward and return independently owned NumPy data.

    The fixed snapshot must already be in the local HF cache. The full HF model
    is released before returning; tied embedding/head weights share one NumPy
    object so the notebook need not keep two copies of the same large array.
    """
    if device not in ("mps", "cpu"):
        raise ValueError("device must be 'mps' or 'cpu'")
    import torch

    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; call load_hf_fixture(device='cpu')")
    try:
        return _capture_hf_fixture(device, torch)
    finally:
        # _capture_hf_fixture's frame has exited on the successful return path.
        # NumPy copies retain no Torch/model storage owners.
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()


def _capture_hf_fixture(device, torch):
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM

    def to_numpy(value):
        # copy() is intentional: .numpy() alone can retain a Torch storage.
        return value.detach().float().cpu().numpy().copy()

    trace_path = (
        Path(__file__).resolve().parents[1]
        / "ch01" / "reference" / "chapter01-trace.json"
    )
    trace = json.loads(trace_path.read_text())
    prompt_ids = list(trace["prompt_ids"])
    expected_first_id = int(trace["generated_ids"][0])
    model_dir = snapshot_download(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True,
    )

    model = backbone = layer = hf_embedding = hf_head = ids = outputs = None
    hooks = []
    captured = {}

    def capture(name):
        def hook(module, inputs, output):
            captured[name + "_in"] = to_numpy(inputs[0])
            captured[name + "_out"] = to_numpy(output)
        return hook

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, dtype=torch.float32,
        )
        model.to(device)
        model.eval()
        backbone = model.model
        if hasattr(backbone, "language_model"):
            backbone = backbone.language_model
        layer = backbone.layers[0]
        hf_embedding = model.get_input_embeddings()
        hf_head = model.get_output_embeddings()
        weights_shared = (
            hf_embedding.weight.data_ptr() == hf_head.weight.data_ptr()
        )
        embedding_weight = to_numpy(hf_embedding.weight)
        head_weight = (
            embedding_weight if weights_shared else to_numpy(hf_head.weight)
        )
        norms = {
            "input_norm": {
                "weight": to_numpy(layer.input_layernorm.weight),
                "eps": float(layer.input_layernorm.eps),
            },
            "post_norm": {
                "weight": to_numpy(layer.post_attention_layernorm.weight),
                "eps": float(layer.post_attention_layernorm.eps),
            },
            "final_norm": {
                "weight": to_numpy(backbone.norm.weight),
                "eps": float(backbone.norm.eps),
            },
        }
        mlp_weights = {
            "gate": to_numpy(layer.mlp.gate_proj.weight),
            "up": to_numpy(layer.mlp.up_proj.weight),
            "down": to_numpy(layer.mlp.down_proj.weight),
        }

        # Register inside the try so even partial registration is cleaned up.
        hooks.append(layer.input_layernorm.register_forward_hook(capture("input_norm")))
        hooks.append(layer.post_attention_layernorm.register_forward_hook(capture("post_norm")))
        hooks.append(layer.mlp.register_forward_hook(capture("mlp")))
        hooks.append(layer.register_forward_hook(capture("layer")))
        hooks.append(backbone.norm.register_forward_hook(capture("final_norm")))
        hooks.append(hf_head.register_forward_hook(capture("head")))
        ids = torch.tensor([prompt_ids], device=device)
        with torch.inference_mode():
            ref_embedding = to_numpy(hf_embedding(ids))
            outputs = model(input_ids=ids, use_cache=False)
            ref_logits = to_numpy(outputs.logits)

        return {
            "prompt_ids": prompt_ids,
            "expected_first_id": expected_first_id,
            "weights_shared": weights_shared,
            "embedding_weight": embedding_weight,
            "head_weight": head_weight,
            "norms": norms,
            "mlp_weights": mlp_weights,
            "captured": captured,
            "ref_embedding": ref_embedding,
            "ref_logits": ref_logits,
            "device": device,
        }
    finally:
        for hook in hooks:
            hook.remove()
        hooks.clear()
        # Explicitly release module/tensor references on both success and error.
        outputs = ids = hf_head = hf_embedding = layer = backbone = model = None
