"""E019 isolated history suffix helper; no MLX import or core monkeypatch.

Callers provide the array backend's concatenate function and expected metadata.
The existing history list remains the full committed prefix, read-only here.
"""


def pending_features(history, fed, start, *, width, dtype, concatenate):
    """Return rows [fed:start], concatenating only intersecting tail pieces.

    Contract: history is a list of rank-2 arrays with the declared width/dtype;
    its total row count is exactly start; 0 <= fed <= start. Empty pieces are
    allowed. Inputs are not mutated or cast. The return is None iff fed==start.
    ``concatenate`` has the MLX/NumPy signature (pieces, axis=0).

    For the current target the caller must supply width=25600 and BF16 dtype.
    This helper is generic only in metadata; it does not infer/promote dtypes.
    """
    if type(history) is not list:
        raise TypeError("history must be the existing list of complete prefix pieces")
    if type(fed) is not int or type(start) is not int or type(width) is not int:
        raise TypeError("fed, start, and width must be integers")
    if not 0 <= fed <= start or width <= 0:
        raise ValueError("requires 0 <= fed <= start and a positive feature width")
    total = 0
    for piece in history:
        if piece.ndim != 2 or piece.shape[1] != width or piece.dtype != dtype:
            raise ValueError("history rank, width, or dtype differs from the declared contract")
        total += piece.shape[0]
    if total != start:
        raise ValueError("history must contain exactly the committed prefix [0:start]")
    if fed == start:
        return None

    remaining = start - fed
    pieces = []
    for piece in reversed(history):
        take = min(remaining, piece.shape[0])
        if take:
            pieces.append(piece if take == piece.shape[0] else piece[-take:])
            remaining -= take
        if remaining == 0:
            break
    pieces.reverse()
    return pieces[0] if len(pieces) == 1 else concatenate(pieces, axis=0)


def full_history_reference(history, fed, start, *, concatenate):
    """Literal existing SpecRunner.step extraction for valid-domain comparison."""
    if fed < start:
        full = history[0] if len(history) == 1 else concatenate(history)
        return full[fed:start]
    return None
