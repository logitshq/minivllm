"""AST-extracted accepted15 conv expressions; no MLX import at module import."""
def conv_reference(raw, old_tail, weights, mx, nn):
    qkv = raw
    w = weights.astype(qkv.dtype)
    xp = mx.concatenate([old_tail.astype(qkv.dtype), qkv[:, None]], axis=1)
    product = xp * w.T[None]
    qkv = product.sum(axis=1)
    act = nn.silu(qkv)
    return (qkv, act, xp[:, 1:], product)
