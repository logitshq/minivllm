"""Chapter 03: compare Attention with the same real HF layer input and weights."""
import argparse
import numpy as np
import mlx.core as mx
from tutorial_code.reference_helpers import raw_model_dir, write_report, HF_REFERENCE
from tutorial_code.reference_helpers import hf_mixer_fixture
from tutorial_code.miniengine.config import ModelConfig
from tutorial_code.miniengine.model.attention import Attention

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="tutorial_code/examples/chapter03-checks.json")
    args = p.parse_args()
    cfg = ModelConfig.from_dir(raw_model_dir())
    index = cfg.layer_types.index("full_attention")
    x, expected, weights = hf_mixer_fixture(index)
    layer = Attention(cfg)
    layer.load_weights([(k, mx.array(v)) for k, v in weights.items()], strict=True)
    actual = np.array(layer(mx.array(x)))
    error = float(np.max(np.abs(actual - expected)))
    assert actual.shape == expected.shape and error < 1e-3, error
    changed = x.copy()
    changed[-1] += 0.5
    changed_out = np.array(layer(mx.array(changed)))
    causal_error = float(np.max(np.abs(changed_out[:-1] - actual[:-1])))
    assert causal_error < 1e-5, causal_error
    write_report(args.out, {"chapter": "03", "baseline": "HF float32 MPS", "layer": index, "input_shape": list(x.shape), "output_shape": list(actual.shape), "max_abs_error": error, "causal_future_change_error": causal_error, "integration": "component check; generation remains HF"})

if __name__ == "__main__":
    main()
