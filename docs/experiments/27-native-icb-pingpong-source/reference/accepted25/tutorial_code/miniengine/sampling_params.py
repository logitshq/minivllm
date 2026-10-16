from dataclasses import dataclass


@dataclass
class SamplingParams:
    max_tokens: int = 256
    temperature: float = 0.0        # 0 = greedy
    top_p: float = 1.0
    top_k: int = 0                  # 0 = 不限
    ignore_eos: bool = False        # 跑分用：不在 EOS 停
    stop_token_ids: tuple[int, ...] = ()

    def __post_init__(self):
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if self.temperature < 0 or not 0 < self.top_p <= 1 or self.top_k < 0:
            raise ValueError("invalid temperature / top_p / top_k")
