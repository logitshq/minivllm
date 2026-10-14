"""逐层对比我们的模型和 mlx-lm（同一个 checkpoint、同一个 prompt），找第一个分叉的层。用 .venv-mlx 跑。"""
import sys; sys.path.insert(0, ".")
import numpy as np, mlx.core as mx
from mlx_lm import load
from minivllm.model.loader import load_model

path = sys.argv[1]
PROMPT = [248045, 846, 198, 95826, 110827, 98682, 108829, 81726, 6297, 1710, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271]
ref, _ = load(path); ours, cfg = load_model(path)
lm = ref.language_model.model
x = mx.array(PROMPT)

def d(a, b):
    a, b = np.array(a.astype(mx.float32)).reshape(-1, a.shape[-1]), np.array(b.astype(mx.float32)).reshape(-1, b.shape[-1])
    return np.abs(a - b).max(), np.abs(b).max()

h_o = ours.embed_tokens(x); h_r = lm.embed_tokens(x[None])
print("embed diff %.5f" % d(h_o, h_r[0])[0])
mask = "causal"
for i, (lo, lr) in enumerate(zip(ours.layers, lm.layers)):
    # 同一个输入分别过两边的层（隔离每层自己的误差）
    inp_o = h_r[0]
    out_o = lo(inp_o); out_r = lr(h_r, mask=(mask if lo.is_attn else None), cache=None)
    md, sc = d(out_o, out_r[0])
    kind = "attn" if lo.is_attn else "gdn "
    flag = "  <-- 分叉" if md > 0.05 * sc else ""
    print(f"layer {i:2d} {kind} isolated diff {md:.4f} (scale {sc:.2f}){flag}")
    if md > 0.05 * sc and i < 8:
        # 拆开看：mixer 和 mlp
        n1 = lo.input_layernorm(inp_o); n1r = lr.input_layernorm(h_r)
        print("    norm1 diff %.5f" % d(n1, n1r[0])[0])
        mo = lo.self_attn(n1) if lo.is_attn else lo.linear_attn(n1)
        mr = (lr.self_attn(n1r, mask=mask, cache=None) if lo.is_attn else lr.linear_attn(n1r, mask=None, cache=None))
        print("    mixer diff %.5f (scale %.3f)" % d(mo, mr[0]))
    h_r = out_r
logits_o = ours(x); logits_r = ref(x[None])[0]
md, sc = d(logits_o, logits_r)
print(f"final logits diff {md:.4f} (scale {sc:.1f}); argmax agree {np.mean(np.array(logits_o.astype(mx.float32)).argmax(-1) == np.array(logits_r.astype(mx.float32)).argmax(-1))*100:.0f}%")
