"""ModelRunner：把调度器的一步变成模型的前向。

每步两类工作：
  decode  —— 所有 prompt 已算完的请求排成一个 batch，一次前向出 B 个 token。它们的 cache 在 DecodeBatch 里。
  prefill —— 每个 (请求, 块长) 单独一次前向，用请求自己的单序列 cache（staging）；算完 prompt 的那一块产出第一个 token。
请求第一次进 decode 时，它的 staging cache 并进 DecodeBatch（重排一次）；结束的请求在下一次重排时被丢掉。
"""
import mlx.core as mx

from ..cache import DecodeBatch, make_cache
from ..forward_context import ForwardContext
from ..sampler import sample
from .request import Request
from .scheduler import SchedulerOutput


class ModelRunner:
    def __init__(self, model, max_len: int):
        self.model = model
        self.max_len = max_len
        self.batch = DecodeBatch(model)
        self.staging: dict[str, list] = {}          # req_id → 单序列 cache（prefill 阶段）

    def run(self, out: SchedulerOutput) -> dict[str, int]:
        sampled: dict[str, int] = {}
        to_eval = []
        # ---- decode ----
        if out.decode:
            self._sync_batch(out.decode)
            positions = [r.num_computed_tokens for r in out.decode]
            tokens = mx.array([r.output_token_ids[-1] for r in out.decode])
            ctx = ForwardContext("decode", self.batch.layers, positions=mx.array(positions),
                                 lens=mx.array([p + 1 for p in positions]), max_len=max(positions) + 1)
            logits = self.model(tokens, ctx)
            toks = sample(logits, [r.sampling for r in out.decode])
            to_eval.append(toks)
            decode_reqs = out.decode
        elif self.batch.size:
            self.batch.clear()                                  # 没人在 decode 了：释放批量 cache
        # ---- prefill ----
        prefill_toks = []
        for req, n in out.prefill:
            caches = self.staging.setdefault(req.req_id, make_cache(self.model))
            start = req.num_computed_tokens
            ids = mx.array(req.all_token_ids[start:start + n])
            logits = self.model(ids, ForwardContext("prefill", caches), last_only=True)
            if start + n >= req.num_prompt_tokens:          # 这一块算完了 prompt → 第一个 token
                t = sample(logits, [req.sampling])
                prefill_toks.append((req, t)); to_eval.append(t)
            else:
                to_eval.append(logits)                      # 让这一块的计算现在就执行
        mx.eval(*to_eval, *self.batch.arrays())
        if out.decode:
            for r, t in zip(decode_reqs, toks.tolist()):
                sampled[r.req_id] = t
        for r, t in prefill_toks:
            sampled[r.req_id] = int(t.item())
        return sampled

    def _sync_batch(self, decode: list[Request]) -> None:
        """让 DecodeBatch 的行顺序 == decode 列表：保留还在的，追加新来的（从 staging 搬进来），丢掉结束的。"""
        if self.batch.req_ids == [r.req_id for r in decode]:
            return
        pos = {rid: i for i, rid in enumerate(self.batch.req_ids)}
        keep = [pos[r.req_id] for r in decode if r.req_id in pos]
        new = []
        for r in decode:
            if r.req_id not in pos:
                caches = self.staging.pop(r.req_id)
                new.append((r.req_id, caches, r.num_prompt_tokens + r.sampling.max_tokens))
        self.batch.rebuild(keep, new)
        # 丢掉的行（结束的请求）不需要做任何事：它们的 staging 早已搬走，batch 里的行被 rebuild 过滤掉了

    def drop(self, req_id: str) -> None:
        self.staging.pop(req_id, None)
