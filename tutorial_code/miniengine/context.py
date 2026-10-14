"""一步前向的元数据。模型层通过它知道自己在哪种模式、该读写哪份 cache。"""
from dataclasses import dataclass

import mlx.core as mx


@dataclass
class ForwardContext:
    mode: str                         # "prefill"（单序列，x [T, hidden]）或 "decode"（批量，x [B, hidden]）
    caches: list                      # prefill: 单序列 cache 列表；decode: DecodeBatch.layers
    positions: mx.array | None = None # decode: [B] 每行新 token 的位置（= 已有长度）
    lens: mx.array | None = None      # decode: [B] 写入后的长度（positions + 1）
    max_len: int = 0                  # decode: 这一步要读的 K/V 列数（= max(lens)）
