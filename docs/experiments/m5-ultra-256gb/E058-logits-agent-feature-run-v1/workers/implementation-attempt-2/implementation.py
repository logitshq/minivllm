"""Token-major feature reference and E058 candidate wrapper.

The reference mirrors the original native one-row traversal but is independent
of the verifier candidates. Imports stay inert; ``runtime`` supplies every
array/cache operation. Root owns GPU admission and execution.
"""
from types import SimpleNamespace
import sys

REPO = '/Users/ryanli/code/minivllm'
if REPO not in sys.path:
    sys.path.insert(0, REPO)

CAPTURE = (5, 19, 33, 47, 61)
FEATURE_SIZE = 5 * 5120


def _candidate_module():
    from bench.m5_ultra_256gb.rowwise_mlp_shared_features import candidate
    return candidate


def _snapshot(caches, runtime):
    """Copy every backing so a prefix never aliases canonical or sibling state."""
    return [runtime.BatchKV(cache.K[:], cache.V[:]) if type(cache) is runtime.BatchKV
            else runtime.BatchState(cache.conv, cache.S) for cache in caches]


def _rows(rows, runtime):
    return rows[0] if len(rows) == 1 else runtime.mx.concatenate(rows, axis=0)


def reference_with_features(model, tokens, context, runtime):
    """Independent sequential native forward for T=1..8 fixed token rows.

    Features are complete post-layer hidden rows from CAPTURE, concatenated on
    the hidden axis and then on the token axis. ``prefixes[t]`` is the cache
    state after input token ``t``. Canonical context and backings are untouched.
    """
    mx = runtime.mx
    if tokens.ndim != 1 or tokens.shape[0] not in (1, 2, 4, 8):
        raise ValueError('tokens must be a one-dimensional T1/T2/T4/T8 row')
    T = tokens.shape[0]
    state = _snapshot(context.caches, runtime)
    captured = [[] for _ in CAPTURE]
    outputs, prefixes = [], []
    for step in range(T):
        ctx = runtime.ForwardContext(
            'decode', state,
            positions=context.positions + step,
            lens=context.lens + step,
            max_len=context.max_len + step)
        x = model.embed_tokens(tokens[step:step+1])
        for index, layer in enumerate(model.layers):
            x = layer(x, ctx, index, _norm_reuse_normal=True)
            if index in CAPTURE:
                captured[CAPTURE.index(index)].append(x)
        normalized = model.norm(x)
        outputs.append(model.embed_tokens.as_linear(normalized)
                       if model.cfg.tie_word_embeddings else model.lm_head(normalized))
        prefixes.append(_snapshot(state, runtime))
    features = _rows([runtime.mx.concatenate(
        captured[i], axis=0) for i in range(len(CAPTURE))], runtime)
    return SimpleNamespace(logits=_rows(outputs, runtime), features=features,
                           prefixes=prefixes)


def candidate_with_features(model, tokens, context, runtime, project):
    """Run the current E058 candidate with the fixed five-layer capture."""
    result = _candidate_module().verify(
        model, tokens, context, runtime, enabled=True, head_project=project,
        mlp_input_project=project, capture=CAPTURE)
    return SimpleNamespace(logits=result.logits, features=result.features,
                           prefixes=result.prefixes)
