"""调度器（有主见的 vLLM V1 简化版）。

资源模型：固定槽位。每个请求整个生命周期占一个槽，槽里的 KV / GDN 状态按 max_len 预分配。
进来时就检查 prompt + max_tokens ≤ max_len，所以运行中永远不会缺内存 —— **不需要抢占**，也没有 block 表。
代价是短请求也占满一个槽的内存；换来的是调度器只剩一个 token 预算。

每一步：
  1. 所有 prompt 已算完的在跑请求各 decode 1 个 token（decode 优先，永远不被 prefill 饿死）。
  2. 剩余预算给 prompt 还没算完的请求做分块 prefill（chunked prefill），先在跑的、再从等待队列按 FCFS 接新的（要有空槽）。
  3. 输出 SchedulerOutput：decode 列表 + prefill 块列表 [(请求, 这一步算几个 token)]。
一个 prefill 块如果正好把 prompt 算完，这一步就要对它采样（它的最后一个位置产出第一个 token）。

step 之后 update()：写入采样结果，推进 num_computed_tokens，判结束，释放槽。
"""
from collections import deque
from dataclasses import dataclass, field

from .request import Request, Status


@dataclass
class SchedulerOutput:
    decode: list[Request] = field(default_factory=list)                 # 每个算 1 个 token
    prefill: list[tuple[Request, int]] = field(default_factory=list)    # (请求, 这一步的 token 数)
    finished_ids: list[str] = field(default_factory=list)               # 上一步 update 里结束的请求

    @property
    def is_empty(self) -> bool:
        return not self.decode and not self.prefill

    @property
    def num_tokens(self) -> int:
        return len(self.decode) + sum(n for _, n in self.prefill)


class Scheduler:
    def __init__(self, max_seqs: int, max_len: int, max_batched_tokens: int, eos_token_ids: set[int]):
        assert max_batched_tokens >= max_seqs, "预算至少要够每个槽 decode 一个 token"
        self.max_seqs, self.max_len, self.max_batched_tokens = max_seqs, max_len, max_batched_tokens
        self.eos_token_ids = set(eos_token_ids)
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.free_slots = list(range(max_seqs - 1, -1, -1))
        self._finished_since_last_step: list[str] = []

    # ---- 对外 ----
    def add(self, req: Request) -> None:
        if req.num_prompt_tokens + req.sampling.max_tokens > self.max_len:
            raise ValueError(f"{req.req_id}: prompt {req.num_prompt_tokens} + max_tokens {req.sampling.max_tokens} > max_len {self.max_len}")
        if req.num_prompt_tokens == 0:
            raise ValueError(f"{req.req_id}: empty prompt")
        self.waiting.append(req)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def schedule(self) -> SchedulerOutput:
        out = SchedulerOutput(finished_ids=self._finished_since_last_step)
        self._finished_since_last_step = []
        budget = self.max_batched_tokens
        # 1. decode
        for req in self.running:
            if req.prompt_done:
                out.decode.append(req)
                budget -= 1
        # 2. 在跑但 prompt 没算完的（上一步被分块了）
        for req in self.running:
            if not req.prompt_done and budget > 0:
                n = min(req.num_prompt_tokens - req.num_computed_tokens, budget)
                out.prefill.append((req, n))
                budget -= n
        # 3. 接新请求
        while self.waiting and self.free_slots and budget > 0:
            req = self.waiting.popleft()
            req.slot = self.free_slots.pop()
            req.status = Status.RUNNING
            self.running.append(req)
            n = min(req.num_prompt_tokens, budget)
            out.prefill.append((req, n))
            budget -= n
        return out

    def update(self, out: SchedulerOutput, sampled: dict[str, int | list[int]]) -> None:
        """sampled: 这一步产出了 token 的请求 → token id（或投机解码时的多个 token id）。
        不变量：cache 里的 token 数 = 已有 token 数 − 1（最新采样的那个还没进 cache，下一步作为输入）。
        所以 decode 请求产出 m 个 token 时 num_computed_tokens += m。"""
        for req, n in out.prefill:
            req.num_computed_tokens += n
        for req in out.decode + [r for r, _ in out.prefill if r.prompt_done]:
            toks = sampled[req.req_id]
            toks = [toks] if isinstance(toks, int) else list(toks)
            if req in out.decode:
                req.num_computed_tokens += len(toks)
            for tok in toks:
                req.output_token_ids.append(tok)
                self._check_finish(req, tok)
                if req.is_finished:
                    break
        still = []
        for req in self.running:
            if req.is_finished:
                self.free_slots.append(req.slot)
                self._finished_since_last_step.append(req.req_id)
            else:
                still.append(req)
        self.running = still

    # ---- 内部 ----
    def _check_finish(self, req: Request, tok: int) -> None:
        s = req.sampling
        if not s.ignore_eos and (tok in self.eos_token_ids or tok in s.stop_token_ids):
            req.status = Status.FINISHED_STOP
        elif len(req.output_token_ids) >= s.max_tokens or req.num_tokens >= self.max_len:
            req.status = Status.FINISHED_LENGTH
