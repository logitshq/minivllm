from dataclasses import dataclass, field


@dataclass
class SamplingParams:
    max_tokens: int = 256
    temperature: float = 0.0        # 0 = greedy
    top_p: float = 1.0
    top_k: int = 0                  # 0 = 不限
    ignore_eos: bool = False        # 跑分用：不在 EOS 停
    stop_token_ids: tuple[int, ...] = ()
