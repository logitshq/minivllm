"""Shared baseline setup; used only by verification, never engine execution."""
import json
from pathlib import Path

HF_REPO = "Qwen/Qwen3.5-0.8B"
HF_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
ROOT = Path(__file__).resolve().parents[1]
HF_REFERENCE = ROOT / "tests/data/hf_0.8b.npz"
NV_REFERENCE = ROOT / "tests/data/mlxlm_0.8b_nvfp4.npz"

def raw_model_dir():
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(HF_REPO, revision=HF_REVISION, local_files_only=True))

def write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))

def hf_mixer_fixture(layer_index):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM
    if not torch.backends.mps.is_available():
        raise RuntimeError("This checker requires MPS")
    model = AutoModelForCausalLM.from_pretrained(raw_model_dir(), local_files_only=True, dtype=torch.float32).to("mps").eval()
    backbone = model.model
    if hasattr(backbone, "language_model"):
        backbone = backbone.language_model
    layer = backbone.layers[layer_index]
    mixer = layer.self_attn if hasattr(layer, "self_attn") else layer.linear_attn
    captured = {}
    def before(module, args, kwargs):
        x = args[0] if args else kwargs["hidden_states"]
        captured["input"] = x.detach().float().cpu().numpy()[0]
    def after(module, args, output):
        y = output[0] if isinstance(output, tuple) else output
        captured["output"] = y.detach().float().cpu().numpy()[0]
    hooks = [mixer.register_forward_pre_hook(before, with_kwargs=True), mixer.register_forward_hook(after)]
    ids = np.load(HF_REFERENCE)["prompt"].tolist()
    try:
        with torch.inference_mode():
            model(input_ids=torch.tensor([ids], device="mps"), use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()
    weights = {k: v.detach().float().cpu().numpy() for k, v in mixer.state_dict().items()}
    del model
    return captured["input"], captured["output"], weights
