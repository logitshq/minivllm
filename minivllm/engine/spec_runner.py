"""投机解码（DFlash2）的 solo 路径：一个请求、一步最多出 8 个 token。

ModelRunner 在只有一个请求在 decode 时走这里（batch ≥ 2 时投机解码在 Mac 上不划算，见 tutorial 00 的 verify 成本表）。
每步：
  draft   drafter 吃新的上下文特征，吐 7 个草稿 token
  verify  目标模型一次前向 [锚点, 7 个草稿]（prefill 路径，顺便捕获上下文特征），posterior = argmax
  accept  最长匹配前缀 n + 1 个 bonus token；cache 回滚到只保留锚点 + 前 n 个草稿；下一步的锚点 = bonus
"""
import mlx.core as mx

from ..cache import AttnCache, GDNCache
from ..forward_context import ForwardContext


class SpecState:
    def __init__(self):
        self.stats = [0, 0]                    # [接受的草稿数, 轮数]


class SpecRunner:
    def __init__(self, model, drafter):
        self.model, self.drafter = model, drafter
        self.capture = tuple(drafter.cfg.target_layer_ids)
        self.block = drafter.cfg.block_size
        self.state: dict[str, SpecState] = {}

    def step(self, req, caches: list, history: list[mx.array]) -> tuple[list[int], mx.array]:
        """caches: 请求的单序列 cache；history: 这个请求到目前为止全部的上下文特征（位置 0..start-1，按块存）。
        drafter 自己的上下文 cache 长度就是它已经吃到哪了；差的部分从 history 补上（刚从 batch 抽回来时是全部）。
        返回产出的 token 和要追加进 history 的新特征。"""
        st = self.state.setdefault(req.req_id, SpecState())
        start = req.num_computed_tokens                      # 锚点的位置
        anchor = req.all_token_ids[-1]
        fed = self.drafter.layers[0].self_attn.ctx_len
        feats = None
        if fed < start:
            full = history[0] if len(history) == 1 else mx.concatenate(history)
            feats = full[fed:start]
        drafts = self.drafter.draft(feats, anchor, start, self.model.embed_tokens, self.model.lm_head_fn)
        block = [anchor] + drafts
        for c in caches:
            c.snapshot()
        logits, new_feats = self.model(mx.array(block), ForwardContext("prefill", caches), capture=self.capture)
        posterior = mx.argmax(logits, axis=-1)
        mx.eval(posterior, new_feats)
        post = posterior.tolist()
        n = 0
        while n < len(drafts) and drafts[n] == post[n]:
            n += 1
        produced = drafts[:n] + [post[n]]                    # 接受 n 个草稿 + bonus
        for c in caches:
            c.rollback(n + 1)                                # cache 保留锚点 + n 个草稿
        st.stats[0] += n; st.stats[1] += 1
        return produced, new_feats[: n + 1]                 # 锚点 + 接受的草稿的特征，追加进 history

    def drop(self, req_id: str):
        self.state.pop(req_id, None)
