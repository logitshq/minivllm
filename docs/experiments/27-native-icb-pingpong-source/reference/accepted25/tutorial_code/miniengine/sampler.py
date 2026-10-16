"""采样：logits [B, V] → token [B]。temperature 0 = greedy；否则 temperature → top-k → top-p → 按概率抽。"""
import mlx.core as mx

from .sampling_params import SamplingParams


def sample(logits: mx.array, params: list[SamplingParams]) -> mx.array:
    if all(p.temperature == 0 for p in params):
        return mx.argmax(logits, axis=-1)
    logits = logits.astype(mx.float32)
    temps = mx.array([max(p.temperature, 1e-5) for p in params])[:, None]
    logits = logits / temps
    V = logits.shape[-1]
    # top-k：保留每行最大的 k 个
    ks = [min(p.top_k, V) if p.top_k > 0 else V for p in params]
    if any(k < V for k in ks):
        sorted_desc = -mx.sort(-logits, axis=-1)
        kth = mx.take_along_axis(sorted_desc, mx.array(ks)[:, None] - 1, axis=-1)
        logits = mx.where(logits < kth, -mx.inf, logits)
    # top-p：按概率从大到小累加，超过 p 之后的全部丢掉
    if any(p.top_p < 1.0 for p in params):
        order = mx.argsort(-logits, axis=-1)
        sorted_logits = mx.take_along_axis(logits, order, axis=-1)
        probs = mx.softmax(sorted_logits, axis=-1)
        cum = mx.cumsum(probs, axis=-1)
        tp = mx.array([p.top_p for p in params])[:, None]
        keep = (cum - probs) < tp                     # 自己之前的累计还没超过 p 的保留
        sorted_logits = mx.where(keep, sorted_logits, -mx.inf)
        logits = mx.put_along_axis(mx.full(logits.shape, -mx.inf), order, sorted_logits, axis=-1)
    greedy_rows = mx.array([p.temperature == 0 for p in params])
    sampled = mx.random.categorical(logits, axis=-1)
    return mx.where(greedy_rows, mx.argmax(logits, axis=-1), sampled)
