"""Pure greedy selection for private verifier prefixes; no cache mutation."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Selection:
    accepted_drafts: int
    produced: tuple
    computed_count: int
    next_anchor: int
    layers: object
    features: object


def select_greedy(drafts, posterior, output):
    """One anchor plus T-1 drafts; retain exactly the accepted causal prefix.

    Caller owns EOS/output-cap handling and committing the returned state.
    This selects a complete round only; it never mutates canonical caches.
    """
    if type(drafts) not in (list, tuple) or type(posterior) not in (list, tuple):
        raise TypeError("host token lists required")
    width = len(drafts) + 1
    if width not in (2, 4, 8) or len(posterior) != width:
        raise ValueError("one posterior per anchor/draft input at T2/T4/T8")
    if any(type(t) is not int or t < 0 for t in (*drafts, *posterior)):
        raise ValueError("nonnegative integer token IDs required")
    if len(output.prefixes) != width or output.features is None:
        raise ValueError("complete private prefixes and captured features required")
    if output.features.shape != (width, 25600):
        raise ValueError("five fixed target-layer features per input required")
    accepted = 0
    while accepted < len(drafts) and drafts[accepted] == posterior[accepted]:
        accepted += 1
    count = accepted + 1
    # The correction/bonus has been emitted, but has not yet been computed.
    # Prefix[count-1] contains anchor + accepted drafts, never that new token.
    return Selection(accepted, tuple(drafts[:accepted]) + (posterior[accepted],),
                     count, posterior[accepted], output.prefixes[count-1],
                     output.features[:count])
