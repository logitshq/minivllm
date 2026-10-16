"""一个请求的全部状态。调度器只读写这里的字段。"""
from enum import Enum, auto

from ..sampling_params import SamplingParams


class Status(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED_STOP = auto()      # EOS 或 stop token
    FINISHED_LENGTH = auto()    # max_tokens 或 max_len


class Request:
    def __init__(self, req_id: str, prompt_token_ids: list[int], sampling: SamplingParams):
        self.req_id = req_id
        self.prompt_token_ids = prompt_token_ids
        self.output_token_ids: list[int] = []
        self.sampling = sampling
        self.num_computed_tokens = 0     # 已经进了 cache 的 token 数（prompt + 已生成）
        self.slot: int | None = None     # 占的槽
        self.status = Status.WAITING

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def all_token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids

    @property
    def prompt_done(self) -> bool:
        return self.num_computed_tokens >= self.num_prompt_tokens

    @property
    def is_finished(self) -> bool:
        return self.status in (Status.FINISHED_STOP, Status.FINISHED_LENGTH)
