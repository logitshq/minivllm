"""Pure-host greedy speculative selection with EOS / output-cap handling.

One verify round feeds the target model a block of ``[anchor] + drafts``
(width 4 or 8).  ``posterior[j]`` is the target argmax sampled after
``input[j]``.  Greedy acceptance takes the longest matching prefix
``drafts[j] == posterior[j]`` (``a`` tokens), then appends the
correction/bonus token ``posterior[a]``.  The emitted stream is trimmed at
the earliest EOS (inclusive) or at the remaining output budget.

The anchor itself was already delivered before the round, so it is never
re-emitted.  ``computed_count`` is the number of block positions whose KV
state is retained: the anchor plus every emitted token except the last
(pending) one, i.e. ``computed_count == len(produced)`` and the retained
prefix of the block is ``[anchor] + produced[:-1]``.  ``next_anchor`` is
the last emitted token (still pending, not yet committed) unless the
request finished.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

DRAFT_WIDTHS = (3, 7)  # verify-block widths 4 and 8, minus the anchor


@dataclass(frozen=True)
class Selection:
    """Result of one speculative greedy selection round."""

    produced: Tuple[int, ...]
    accepted_drafts: int
    computed_count: int
    next_anchor: Optional[int]
    finished: bool


def _is_token_id(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _check_tokens(name: str, values: object) -> Tuple[int, ...]:
    if isinstance(values, (list, tuple)):
        items = tuple(values)
    else:
        raise TypeError(f"{name} must be a list or tuple of token ids, got {type(values).__name__}")
    for item in items:
        if not _is_token_id(item):
            raise ValueError(f"{name} entries must be nonnegative strict ints, got {item!r}")
    return items


def _check_eos_ids(eos_ids: object) -> frozenset:
    if isinstance(eos_ids, (tuple, list, set, frozenset)):
        items = tuple(eos_ids)
    else:
        raise TypeError(f"eos_ids must be a tuple/list/set/frozenset of token ids, got {type(eos_ids).__name__}")
    for item in items:
        if not _is_token_id(item):
            raise ValueError(f"eos_ids entries must be nonnegative strict ints, got {item!r}")
    return frozenset(items)


def select(drafts: object, posterior: object, remaining: object, eos_ids: object) -> Selection:
    """Run one greedy speculative selection round.

    Parameters
    ----------
    drafts:
        Draft tokens, width 3 or 7 (verify block width 4/8 minus anchor).
    posterior:
        Target argmax after each block input; ``len(posterior) == len(drafts) + 1``.
    remaining:
        Positive strict int output budget left for this request.
    eos_ids:
        Iterable (tuple/list/set/frozenset) of nonnegative strict int EOS ids.

    Returns
    -------
    Selection
        ``produced`` tuple of emitted tokens, ``accepted_drafts`` count of
        accepted draft tokens actually emitted (``min(a, len(produced))``),
        ``computed_count == len(produced)`` retained block positions,
        ``next_anchor`` the last emitted token unless finished, and
        ``finished`` when EOS was emitted or the output budget is exhausted.
    """
    draft_toks = _check_tokens("drafts", drafts)
    post_toks = _check_tokens("posterior", posterior)
    if len(draft_toks) not in DRAFT_WIDTHS:
        raise ValueError(f"drafts width must be 3 or 7 (block width 4/8), got {len(draft_toks)}")
    if len(post_toks) != len(draft_toks) + 1:
        raise ValueError(
            f"posterior must have exactly one entry per block input "
            f"({len(draft_toks) + 1}), got {len(post_toks)}"
        )
    if isinstance(remaining, bool) or not isinstance(remaining, int):
        raise TypeError(f"remaining must be a strict int, got {type(remaining).__name__}")
    if remaining <= 0:
        raise ValueError(f"remaining must be positive; no output is possible with remaining={remaining}")
    eos = _check_eos_ids(eos_ids)

    accepted = 0
    while accepted < len(draft_toks) and draft_toks[accepted] == post_toks[accepted]:
        accepted += 1

    out = list(draft_toks[:accepted])
    out.append(post_toks[accepted])

    eos_index = None
    for index, token in enumerate(out):
        if token in eos:
            eos_index = index
            break
    keep = len(out) if eos_index is None else eos_index + 1
    if keep > remaining:
        keep = remaining

    produced = tuple(out[:keep])
    eos_hit = eos_index is not None and eos_index < keep
    # ``keep == remaining`` also ends the request: the output budget is spent.
    finished = eos_hit or len(produced) >= remaining

    return Selection(
        produced=produced,
        accepted_drafts=min(accepted, len(produced)),
        computed_count=len(produced),
        next_anchor=None if finished else produced[-1],
        finished=finished,
    )
