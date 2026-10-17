"""投机解码（DFlash2）的 solo 路径：一个请求、一步最多出 8 个 token。

ModelRunner 在只有一个请求在 decode 时走这里（batch ≥ 2 时投机解码在 Mac 上不划算，见 tutorial 00 的 verify 成本表）。
每步：
  draft   drafter 吃新的上下文特征，吐 7 个草稿 token
  verify  目标模型一次前向 [锚点, 7 个草稿]（prefill 路径，顺便捕获上下文特征），posterior = argmax
  accept  最长匹配前缀 n + 1 个 bonus token；cache 回滚到只保留锚点 + 前 n 个草稿；下一步的锚点 = bonus
"""
import mlx.core as mx

from ..context import ForwardContext


class SpecState:
    def __init__(self):
        self.stats = [0, 0]                    # [接受的草稿数, 轮数]
        self.q = 0.8                           # 最近的逐位置接受率（接受数 / 草稿数）的滑动平均，自适应块长用


class SpecRunner:
    def __init__(self, model, drafter, block_size: int | None = None):
        if block_size is not None and not 2 <= block_size <= drafter.cfg.block_size:
            raise ValueError("spec block must be between 2 and the trained block size")
        self.model, self.drafter = model, drafter
        self.capture = tuple(drafter.cfg.target_layer_ids)
        if not self.capture or min(self.capture) < 0 or max(self.capture) >= len(model.layers):
            raise ValueError("drafter target layer IDs do not fit this target model")
        if drafter.cfg.hidden_size != model.cfg.hidden_size:
            raise ValueError("drafter and target hidden dimensions differ")
        self.set_verify_transaction(False)
        self.block = block_size                                # None = 自适应：按逐位置接受率在 5 和 3 之间切
        self.state: dict[str, SpecState] = {}
        self.finished_stats: dict[str, tuple[int, int]] = {}   # req_id → (接受数, 轮数)，请求结束后留着给跑分读

    def step(self, req, caches: list, history: list[mx.array]) -> tuple[list[int], mx.array]:
        """caches: 请求的单序列 cache；history: 这个请求到目前为止全部的上下文特征（位置 0..start-1，按块存）。
        drafter 自己的上下文 cache 长度就是它已经吃到哪了；差的部分从 history 补上（刚从 batch 抽回来时是全部）。
        返回产出的 token 和要追加进 history 的新特征。"""
        if self._verify_transaction_enabled:
            return self._step_transaction108(req, caches, history)
        if req.sampling.temperature != 0:
            raise ValueError("DFlash2 tutorial verification supports greedy (temperature=0) only")
        st = self.state.setdefault(req.req_id, SpecState())
        start = req.num_computed_tokens                      # 锚点的位置
        anchor = req.all_token_ids[-1]
        fed = self.drafter.layers[0].self_attn.ctx_len
        feats = None
        if fed < start:
            full = history[0] if len(history) == 1 else mx.concatenate(history)
            feats = full[fed:start]
        bs = self.block or (5 if st.q >= 0.45 else 3)          # 块长。Mac 的 verify 成本在 5 之后陡增：默认 5，接受率太低时缩到 3
        drafts = self.drafter.draft(feats, anchor, start, self.model.embed_tokens, self.model.lm_head_fn, bs)
        block = [anchor] + drafts
        for c in caches:
            c.snapshot()
        logits, new_feats = self.model(mx.array(block), ForwardContext("prefill", caches), capture=self.capture)
        posterior = mx.argmax(logits, axis=-1)
        mx.eval(posterior, new_feats)
        post = posterior.tolist()
        produced, n = accept_greedy_prefix(drafts, post)                    # 接受 n 个草稿 + bonus
        for c in caches:
            c.rollback(n + 1)                                # cache 保留锚点 + n 个草稿
        st.stats[0] += n; st.stats[1] += 1
        st.q = 0.8 * st.q + 0.2 * (n / max(1, bs - 1))
        return produced, new_feats[: n + 1]                 # 锚点 + 接受的草稿的特征，追加进 history

    def set_verify_transaction(self, enabled, *, eos_ids=(), capacity=64, observer=None):
        if type(enabled) is not bool or type(capacity) is not int or capacity < 1:
            raise TypeError("strict bool and positive integer capacity required")
        self._verify_transaction_enabled = enabled
        self._verify_eos_ids = frozenset(eos_ids)
        self._verify_capacity = capacity
        self._transaction_observer = observer
        self.transaction_rounds = []


    def set_verify_broadcast_m1(self, enabled=False, *, logits_observer=None):
        """Explicit V114 opt-in, only inside the active V108 target verify call."""
        from ..spec.verify_broadcast_m1 import prepare_verify_m1
        if type(enabled) is not bool or (logits_observer is not None and not callable(logits_observer)):
            raise TypeError("strict bool and optional callable observer required")
        self._verify_m1_ops114 = prepare_verify_m1(self.model, mx) if enabled else ()
        self._verify_m1_enabled114 = enabled
        self._verify_m1_logits_observer114 = logits_observer
        self.verify_m1_rounds114 = []

    def _step_transaction108(self, req, caches, history):
        from ..spec.transaction import VerifyTransaction, select_delivery
        from ..spec.verify_broadcast_m1 import verify_m1_context
        if req.sampling.temperature != 0:
            raise ValueError("V108 greedy only")
        remaining = req.sampling.max_tokens - len(req.output_token_ids)
        start = req.num_computed_tokens
        if remaining <= 0 or start >= self._verify_capacity:
            raise ValueError("no remaining output/context budget")
        if any(cache.offset != start for cache in caches):
            raise RuntimeError("request/cache frontiers differ")
        st = self.state.setdefault(req.req_id, SpecState())
        anchor = req.all_token_ids[-1]
        bs = min(self.block or (5 if st.q >= 0.45 else 3), remaining, self._verify_capacity - start)
        fed = self.drafter.layers[0].self_attn.ctx_len
        feats = None
        if fed < start:
            full = history[0] if len(history) == 1 else mx.concatenate(history)
            feats = full[fed:start]
        drafts = [] if bs == 1 else self.drafter.draft(feats, anchor, start, self.model.embed_tokens, self.model.lm_head_fn, bs)
        if len(drafts) != bs - 1:
            raise RuntimeError("drafter returned wrong bounded length")
        with VerifyTransaction(caches, bs) as transaction:
            observer = self._transaction_observer
            if observer is not None:
                observer("begin", transaction, {"request": req.req_id, "start": start})
            with verify_m1_context(self, caches) as m1_scope114:
                logits, new_feats = self.model(mx.array([anchor] + drafts), ForwardContext("prefill", caches), capture=self.capture)
            posterior = mx.argmax(logits, axis=-1)
            mx.eval(posterior, new_feats, *transaction.eval_arrays())
            post = posterior.tolist()
            produced, accepted, keep = select_delivery(drafts, post, remaining, self._verify_eos_ids, req.sampling.stop_token_ids, req.sampling.ignore_eos)
            info = {"transaction": transaction.id, "request": req.req_id, "start": start, "verify_inputs": bs,
                    "output_count_before": len(req.output_token_ids), "remaining_before": remaining, "drafts": list(drafts), "posterior": post,
                    "draft_calls": int(bs > 1), "drafted_tokens": len(drafts), "accepted_before_clip": accepted,
                    "delivered": len(produced), "committed_inputs": keep, "next_anchor": produced[-1],
                    "next_frontier": start + keep, "GDN_helper_calls": sum(t.updates for t in transaction.traces if t.kind == "gdn")}
            if m1_scope114 is not None:
                metadata114 = m1_scope114.metadata()
                self.verify_m1_rounds114.append(metadata114)
                info["verify_broadcast_m1_114"] = metadata114
                callback114 = getattr(self, "_verify_m1_logits_observer114", None)
                if callback114 is not None:
                    callback114(logits, transaction, info)
            if observer is not None:
                observer("before_commit", transaction, info)
            transaction.commit(keep)
            if observer is not None:
                observer("after_commit", transaction, info)
            self.transaction_rounds.append(info)
        st.stats[0] += min(accepted, max(0, len(produced) - 1)); st.stats[1] += 1
        st.q = 0.8 * st.q + 0.2 * (accepted / max(1, bs - 1))
        return produced, new_feats[:keep]

    def drop(self, req_id: str):
        st = self.state.pop(req_id, None)
        if st is not None:
            self.finished_stats[req_id] = tuple(st.stats)


def accept_greedy_prefix(drafts: list[int], posterior: list[int]) -> tuple[list[int], int]:
    """draft i must match the target logit at anchor+i. Keep the first mismatch's target token as bonus."""
    if len(posterior) != len(drafts) + 1:
        raise ValueError("posterior must have one prediction per anchor + draft token")
    n = 0
    while n < len(drafts) and drafts[n] == posterior[n]:
        n += 1
    return drafts[:n] + [posterior[n]], n
