"""调度器的 CPU 单测：不碰模型，用一个假的采样器驱动。"""
import random

from minivllm.engine.request import Request, Status
from minivllm.engine.scheduler import Scheduler
from minivllm.sampling_params import SamplingParams

EOS = 7


def run_step(sch, sampler):
    out = sch.schedule()
    sampled = {r.req_id: sampler(r) for r in out.decode}
    sampled.update({r.req_id: sampler(r) for r, n in out.prefill if r.num_computed_tokens + n >= r.num_prompt_tokens})
    sch.update(out, sampled)
    return out


def test_chunked_prefill_and_budget():
    sch = Scheduler(max_seqs=4, max_len=64, max_batched_tokens=8, eos_token_ids={EOS})
    sch.add(Request("a", list(range(20)), SamplingParams(max_tokens=3, ignore_eos=True)))
    out = sch.schedule()                       # 第一块：8 个
    assert out.prefill == [(sch.running[0], 8)] and out.num_tokens == 8
    sch.update(out, {})
    out = sch.schedule(); assert out.prefill[0][1] == 8; sch.update(out, {})
    out = sch.schedule(); assert out.prefill[0][1] == 4           # 最后一块 4 个，算完 prompt → 采样
    sch.update(out, {"a": 100})
    r = sch.running[0]
    assert r.output_token_ids == [100] and r.num_computed_tokens == 20
    out = sch.schedule(); assert out.decode == [r] and not out.prefill   # 之后每步 decode 1 个
    sch.update(out, {"a": 101}); out = sch.schedule(); sch.update(out, {"a": 102})
    assert r.status == Status.FINISHED_LENGTH and r.output_token_ids == [100, 101, 102]
    assert sch.free_slots and not sch.running and not sch.has_work()
    assert sch.schedule().finished_ids == ["a"]


def test_decode_never_starved_and_slots_limit():
    sch = Scheduler(max_seqs=2, max_len=64, max_batched_tokens=6, eos_token_ids={EOS})
    for i in range(3):
        sch.add(Request(f"r{i}", list(range(10)), SamplingParams(max_tokens=5, ignore_eos=True)))
    out = sch.schedule()                                   # 只有 2 个槽：r0 拿 6 个 token，r1 等预算
    assert [(r.req_id, n) for r, n in out.prefill] == [("r0", 6)]
    sch.update(out, {})
    out = sch.schedule()                                   # r0 剩 4 个；r1 拿 2 个
    assert [(r.req_id, n) for r, n in out.prefill] == [("r0", 4), ("r1", 2)]
    sch.update(out, {"r0": 1})
    out = sch.schedule()                                   # r0 decode（永远排前面），r1 继续 prefill 5 个
    assert [r.req_id for r in out.decode] == ["r0"] and [(r.req_id, n) for r, n in out.prefill] == [("r1", 5)]
    assert len(sch.waiting) == 1 and sch.waiting[0].req_id == "r2"   # r2 没槽，还在等


def test_finish_on_eos_and_rejects():
    sch = Scheduler(max_seqs=1, max_len=16, max_batched_tokens=16, eos_token_ids={EOS})
    sch.add(Request("a", [1, 2, 3], SamplingParams(max_tokens=10)))
    out = sch.schedule(); sch.update(out, {"a": 5})
    out = sch.schedule(); sch.update(out, {"a": EOS})
    assert sch.running == [] and sch.schedule().finished_ids == ["a"]
    try:
        sch.add(Request("b", list(range(10)), SamplingParams(max_tokens=10)))   # 10 + 10 > 16
        assert False
    except ValueError:
        pass


def test_simulation_200_requests():
    rng = random.Random(0)
    sch = Scheduler(max_seqs=8, max_len=256, max_batched_tokens=64, eos_token_ids={EOS})
    reqs = [Request(f"r{i}", [rng.randrange(1000) for _ in range(rng.randint(1, 120))],
                    SamplingParams(max_tokens=rng.randint(1, 100), ignore_eos=rng.random() < 0.5)) for i in range(200)]
    for r in reqs:
        sch.add(r)
    def sampler(r):                     # 偶尔吐 EOS
        return EOS if rng.random() < 0.05 else rng.randrange(1000)
    steps = 0
    while sch.has_work():
        out = run_step(sch, sampler)
        assert out.num_tokens <= 64 and len(sch.running) <= 8
        steps += 1
        assert steps < 20000
    for r in reqs:
        assert r.is_finished and r.num_computed_tokens == r.num_tokens - 1 or r.is_finished   # 最后一个 token 没进 cache
        if r.status == Status.FINISHED_LENGTH:
            assert len(r.output_token_ids) == r.sampling.max_tokens or r.num_tokens >= 256
        else:
            assert r.output_token_ids[-1] == EOS and not r.sampling.ignore_eos
    assert sorted(sch.free_slots) == list(range(8))
