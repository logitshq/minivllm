"""生成参考数据（只跑一次，结果存 tests/data/）：
  hf_0.8b.npz       HF transformers：fp32 和 bf16 两种精度下 prompt 的 logits [T, V] 和 greedy 48 个 token
用 .venv（torch + transformers）跑：.venv/bin/python tests/make_refs.py
"""
import glob, os, numpy as np, torch
from transformers import AutoModelForCausalLM

PROMPT = [248045, 846, 198, 95826, 110827, 98682, 108829, 81726, 6297, 1710, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271]
HF_DIR = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/*"))[0]

out = {"prompt": np.array(PROMPT)}
for name, dtype in [("fp32", torch.float32), ("bf16", torch.bfloat16)]:
    model = AutoModelForCausalLM.from_pretrained(HF_DIR, dtype=dtype).to("mps").eval()
    ids = torch.tensor([PROMPT], device="mps")
    with torch.no_grad():
        logits = model(ids).logits[0].float().cpu().numpy()
        gen = model.generate(ids, max_new_tokens=48, do_sample=False)[0, len(PROMPT):].cpu().numpy()
    out[f"{name}_logits"] = logits; out[f"{name}_greedy"] = gen
    print(name, "greedy:", gen.tolist()[:16], "...")
    del model
np.savez_compressed("tests/data/hf_0.8b.npz", **out)
print("saved tests/data/hf_0.8b.npz")
