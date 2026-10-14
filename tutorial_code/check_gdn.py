"""Chapter 04: compare the complete recurrent GDN component with real HF tensors."""
import argparse
import numpy as np
import mlx.core as mx
from tutorial_code.reference_helpers import raw_model_dir, write_report, HF_REFERENCE
from tutorial_code.reference_helpers import hf_mixer_fixture
from tutorial_code.miniengine.config import ModelConfig
from tutorial_code.miniengine.model.gdn import GatedDeltaNet, causal_conv1d

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="tutorial_code/examples/chapter04-checks.json")
    args = p.parse_args()
    cfg = ModelConfig.from_dir(raw_model_dir())
    index = cfg.layer_types.index("linear_attention")
    x, expected, weights = hf_mixer_fixture(index)
    layer = GatedDeltaNet(cfg)
    mapped = []
    for name, value in weights.items():
        if name == "conv1d.weight":
            name, value = "conv1d_weight", value.reshape(value.shape[0], -1)
        mapped.append((name, mx.array(value)))
    layer.load_weights(mapped, strict=True)
    actual = np.array(layer(mx.array(x)))
    error = float(np.max(np.abs(actual - expected)))
    assert actual.shape == expected.shape and error < 1e-3, error
    # Independent Torch Conv1d, plus a split inside the history window.
    import torch
    rng = np.random.default_rng(4)
    raw, weight = rng.normal(size=(11, 5)).astype(np.float32), rng.normal(size=(5, 4)).astype(np.float32)
    torch_out = torch.nn.functional.conv1d(torch.from_numpy(raw.T)[None], torch.from_numpy(weight[:, None]), padding=3, groups=5)[0, :, :11].T.numpy()
    conv, state = causal_conv1d(mx.array(raw), mx.array(weight))
    a, tail = causal_conv1d(mx.array(raw[:2]), mx.array(weight))
    b, tail = causal_conv1d(mx.array(raw[2:]), mx.array(weight), tail)
    conv_error = float(np.max(np.abs(np.array(conv)-torch_out)))
    split_error = float(np.max(np.abs(np.array(mx.concatenate([a,b]))-torch_out)))
    assert conv_error < 1e-5 and split_error < 1e-5
    write_report(args.out, {"chapter": "04", "baseline": "HF float32 MPS + torch grouped conv1d", "layer": index, "max_abs_error": error, "conv_error": conv_error, "conv_split_error": split_error, "integration": "component check; generation remains HF"})

if __name__ == "__main__":
    main()
