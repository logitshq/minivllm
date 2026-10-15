"""ModelRunner：把调度器的一步变成模型的前向。

每步两类工作：
  decode  —— 所有 prompt 已算完的请求排成一个 batch，一次前向出 B 个 token。它们的 cache 在 DecodeBatch 里。
  prefill —— 每个 (请求, 块长) 单独一次前向，用请求自己的单序列 cache（staging）；算完 prompt 的那一块产出第一个 token。
请求第一次进 decode 时，它的 staging cache 并进 DecodeBatch（重排一次）；结束的请求在 drop() 中立即释放或重排丢掉。
"""
import mlx.core as mx

from ..cache import BatchKV, BatchState, DecodeBatch, make_cache
from ..context import ForwardContext
from ..sampler import sample
from .request import Request
from .scheduler import SchedulerOutput


class ModelRunner:
    def __init__(self, model, max_len: int, drafter=None, spec_block: int | None = None):
        self.model = model
        self.max_len = max_len
        self.batch = DecodeBatch(model)
        self.staging: dict[str, list] = {}          # req_id → 单序列 cache（prefill 阶段，以及投机解码的 solo 模式）
        self.spec = None
        self.feats: dict[str, list] = {}            # req_id → 全部上下文特征（按块存；drafter 按需补吃）
        # One target-only decode may be queued on the device while its predecessor
        # is delivered. Keep its cache objects separate from committed batch state.
        self._ahead = None
        if drafter is not None:
            from .spec_runner import SpecRunner
            self.spec = SpecRunner(model, drafter, spec_block)

    def run(self, out: SchedulerOutput) -> dict[str, int | list[int]]:
        sampled: dict[str, int | list[int]] = {}
        to_eval = []
        decode = out.decode                                      # 不改 out：scheduler.update 还要用它
        # ---- solo 投机解码：只有一个请求在 decode，且装了 drafter ----
        if self.spec is not None and len(out.decode) == 1 and out.decode[0].req_id in self.staging:
            req = out.decode[0]
            hist = self.feats.setdefault(req.req_id, [])
            produced, new_feats = self.spec.step(req, self.staging[req.req_id], hist)
            if len(hist) > 64:                                   # 合并，免得列表越来越长
                hist[:] = [mx.concatenate(hist)]
            hist.append(new_feats)
            sampled[req.req_id] = produced
            decode = []
        elif self.spec is not None and len(out.decode) == 1 and self.batch.size == 1 and self.batch.req_ids == [out.decode[0].req_id]:
            # batch 缩到只剩一个：抽回 solo 模式，下一步走投机解码
            self.staging[out.decode[0].req_id] = self.batch.extract(0)
            self.batch.clear()
            return self.run(out)
        # ---- decode ----
        if decode:
            self._sync_batch(decode)
            positions = [r.num_computed_tokens for r in decode]
            sampling_key = tuple((r.sampling.temperature, r.sampling.top_p,
                                  r.sampling.top_k, r.sampling.max_tokens,
                                  r.sampling.ignore_eos, tuple(r.sampling.stop_token_ids))
                                 for r in decode)
            key = ([r.req_id for r in decode], positions, sampling_key,
                   tuple(r.output_token_ids[-1] for r in decode))
            if self._ahead is not None and self._ahead[0] == key:
                _, toks, layers = self._ahead
                self.batch.layers = layers
            else:
                tokens = mx.array([r.output_token_ids[-1] for r in decode])
                ctx = ForwardContext("decode", self.batch.layers, positions=mx.array(positions),
                                     lens=mx.array([p + 1 for p in positions]), max_len=max(positions) + 1)
                if self.spec is not None:
                    logits, f = self.model(tokens, ctx, capture=self.spec.capture)
                    for i, r in enumerate(decode):
                        self.feats.setdefault(r.req_id, []).append(f[i:i + 1])
                    to_eval.append(f)
                else:
                    logits = self.model(tokens, ctx)
                toks = sample(logits, [r.sampling for r in decode])
            self._ahead = None
            if (self.spec is None and not out.prefill
                    and all(r.sampling.temperature == 0 for r in decode)
                    and all(len(r.output_token_ids) + 1 < r.sampling.max_tokens
                            and r.num_tokens + 1 < self.max_len for r in decode)):
                # Submit current arithmetic before constructing the successor.
                # The successor consumes the device token, never a host roundtrip.
                mx.async_eval(toks, *self.batch.arrays())
                # __setitem__ replaces an mx.array object's graph in place. New
                # view objects prevent a future KV write from changing the
                # committed object's handle, while sharing its evaluated buffer.
                layers = [BatchKV(l.K[:], l.V[:]) if isinstance(l, BatchKV)
                          else BatchState(l.conv, l.S) for l in self.batch.layers]
                next_pos = [p + 1 for p in positions]
                next_ctx = ForwardContext("decode", layers, positions=mx.array(next_pos),
                                          lens=mx.array([p + 1 for p in next_pos]), max_len=max(next_pos) + 1)
                next_logits = self.model(toks.astype(mx.int32), next_ctx)
                next_toks = sample(next_logits, [r.sampling for r in decode])
                arrays = [a for l in layers for a in
                          ((l.K, l.V) if isinstance(l, BatchKV) else (l.conv, l.S))]
                mx.async_eval(next_toks, *arrays)
                self._ahead = ((key[0], next_pos, sampling_key, None), next_toks, layers)
            to_eval.append(toks)
            decode_reqs = decode
            self.batch.lens = [p + 1 for p in positions]
        elif self.batch.size:
            self._ahead = None
            self.batch.clear()                                  # 没人在 decode 了：释放批量 cache
        # ---- prefill ----
        prefill_toks = []
        for req, n in out.prefill:
            caches = self.staging.setdefault(req.req_id, make_cache(self.model))
            start = req.num_computed_tokens
            ids = mx.array(req.all_token_ids[start:start + n])
            if self.spec is not None:
                logits, f = self.model(ids, ForwardContext("prefill", caches), last_only=True, capture=self.spec.capture)
                self.feats.setdefault(req.req_id, []).append(f); to_eval.append(f)
            else:
                logits = self.model(ids, ForwardContext("prefill", caches), last_only=True)
            if start + n >= req.num_prompt_tokens:          # 这一块算完了 prompt → 第一个 token
                t = sample(logits, [req.sampling])
                prefill_toks.append((req, t)); to_eval.append(t)
            else:
                to_eval.append(logits)                      # 让这一块的计算现在就执行
        mx.eval(*to_eval, *self.batch.arrays())
        if decode:
            delivered = toks.tolist()
            if self._ahead is not None:
                ahead_key, ahead_toks, ahead_layers = self._ahead
                self._ahead = (ahead_key[:3] + (tuple(delivered),), ahead_toks, ahead_layers)
            for r, t in zip(decode_reqs, delivered):
                sampled[r.req_id] = t
        for r, t in prefill_toks:
            sampled[r.req_id] = int(t.item())
        return sampled

    def _sync_batch(self, decode: list[Request]) -> None:
        """让 DecodeBatch 的行顺序 == decode 列表：保留还在的，追加新来的（从 staging 搬进来），丢掉结束的。"""
        if self.batch.req_ids == [r.req_id for r in decode]:
            return
        self._ahead = None
        pos = {rid: i for i, rid in enumerate(self.batch.req_ids)}
        keep = [pos[r.req_id] for r in decode if r.req_id in pos]
        new = []
        for r in decode:
            if r.req_id not in pos:
                caches = self.staging.pop(r.req_id)
                new.append((r.req_id, caches, r.num_prompt_tokens + r.sampling.max_tokens))
        self.batch.rebuild(keep, new)
        if self.spec is not None:
            self.spec.drafter.reset()                       # 有人进了 batch：drafter 上下文作废，pending 特征保留着
        # 丢掉的行（结束的请求）不需要做任何事：它们的 staging 早已搬走，batch 里的行被 rebuild 过滤掉了

    def drop(self, req_id: str) -> None:
        self._ahead = None
        self.staging.pop(req_id, None)
        self.feats.pop(req_id, None)
        if req_id in self.batch.req_ids:
            keep = [i for i, rid in enumerate(self.batch.req_ids) if rid != req_id]
            if keep:
                self.batch.rebuild(keep, [])
            else:
                self.batch.clear()
        if self.spec is not None:
            self.spec.drop(req_id)
            self.spec.drafter.reset()                       # drafter 的上下文 cache 只服务一个 solo 请求
