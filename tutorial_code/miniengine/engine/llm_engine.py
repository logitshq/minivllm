"""LLMEngine：同步引擎。add_request / step / generate。只认 token id，分词在外面。"""

import mlx.core as mx

from ..model.loader import load_model
from ..eos import read_eos_ids
from ..sampling_params import SamplingParams
from .model_runner import ModelRunner
from .request import Request
from .scheduler import Scheduler


class LLMEngine:
    def __init__(self, model_dir: str, max_seqs: int = 8, max_len: int = 4096, max_batched_tokens: int = 512,
                 dtype=mx.bfloat16, cache_limit_gb: int = 2, drafter_dir: str | None = None, spec_block: int | None = None):
        # MLX 的 allocator 会把释放的 buffer 留着复用；decode 每步都在生成新的 K/V 拷贝，尺寸各不相同，
        # 不限制的话几分钟就攒出 20 GB 的"缓存"，逼近 37 GB 工作集后系统开始压缩内存，整体变慢。
        mx.set_cache_limit(cache_limit_gb << 30)
        self.model, self.cfg = load_model(model_dir, dtype=dtype)
        self.eos_ids = read_eos_ids(model_dir)
        self.scheduler = Scheduler(max_seqs, max_len, max_batched_tokens, self.eos_ids)
        drafter = None
        if drafter_dir:
            from ..spec.dflash2 import load_drafter
            drafter = load_drafter(drafter_dir)
        self.runner = ModelRunner(self.model, max_len, drafter, spec_block)
        self.requests: dict[str, Request] = {}
        self._counter = 0

    def add_request(self, prompt_ids: list[int], sampling: SamplingParams, req_id: str | None = None) -> str:
        if req_id is None:
            self._counter += 1
            req_id = f"req-{self._counter}"
        if req_id in self.requests:
            raise ValueError(f"duplicate active request id: {req_id}")
        if self.runner.spec is not None and sampling.temperature != 0:
            raise ValueError("speculative decoding supports greedy (temperature=0) only")
        req = Request(req_id, list(prompt_ids), sampling)
        self.scheduler.add(req)
        self.requests[req_id] = req
        return req_id

    def has_work(self) -> bool:
        return self.scheduler.has_work()

    def step(self) -> list[tuple[str, int, bool]]:
        """跑一步。返回这一步产出 token 的 (req_id, token, 是否结束)。"""
        out = self.scheduler.schedule()
        for rid in out.finished_ids:
            self.runner.drop(rid)
        if out.is_empty:
            return []
        sampled = self.runner.run(out)
        before = {rid: len(self.requests[rid].output_token_ids) for rid in sampled}
        self.scheduler.update(out, sampled)          # 可能在 EOS / max_tokens 处截断投机解码多出来的 token
        results = []
        for rid in sampled:
            req = self.requests[rid]
            toks = req.output_token_ids[before[rid]:]
            for i, tok in enumerate(toks):
                results.append((rid, tok, req.is_finished and i == len(toks) - 1))
            if req.is_finished:
                self.runner.drop(rid)  # Free the final request even if no next step runs.
                self.requests.pop(rid)
        return results

    def generate(self, prompts: list[list[int]], sampling: SamplingParams) -> list[list[int]]:
        ids = [self.add_request(p, sampling) for p in prompts]
        outs = {rid: [] for rid in ids}
        while self.has_work():
            for rid, tok, _ in self.step():
                outs[rid].append(tok)
        return [outs[rid] for rid in ids]
