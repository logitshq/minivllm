"""E012: same DFlash2 path scores, one final device-to-host path transfer."""
from contextlib import contextmanager
import mlx.core as mx
from tutorial_code.miniengine.spec.dflash2 import DFlash2Drafter

ORIGINAL = DFlash2Drafter._select


def select(drafter, h, logits, anchor, *, device=True, trace=False):
    k = drafter.cfg.selector_top_k
    cand = mx.argpartition(-logits, k - 1, axis=-1)[:, :k]
    unary = mx.take_along_axis(logits, cand, axis=-1)
    proj = drafter.selector_proj(h).astype(mx.float32)
    succ = drafter.successor_codebook[cand].astype(mx.float32)
    path, scores = [], []
    prev = mx.array(anchor, dtype=mx.int32) if device else anchor
    if not device:
        mx.eval(cand, unary, proj, succ)
    for p in range(h.shape[0]):
        pred = mx.take(drafter.predecessor_codebook, prev, axis=0).astype(mx.float32) if device else drafter.predecessor_codebook[prev].astype(mx.float32)
        score = unary[p] + (succ[p] @ (pred * proj[p]))
        j = mx.argmax(score)
        if device:
            prev = mx.take(cand[p], j, axis=0)
        else:
            prev = int(cand[p, int(j.item())].item())
        path.append(prev)
        if trace:
            scores.append(score)
    result = mx.stack(path).tolist() if device and path else path
    return (result, scores) if trace else result


@contextmanager
def enabled(device, observe=None):
    def route(self, h, logits, anchor):
        if observe is not None:
            observe(h, logits, anchor)
        return select(self, h, logits, anchor) if device else ORIGINAL(self, h, logits, anchor)
    DFlash2Drafter._select = route
    try:
        yield
    finally:
        DFlash2Drafter._select = ORIGINAL
