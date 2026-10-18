"""Experimental DFlash2 path selection with a single final host transfer."""
import mlx.core as mx


def select_path(predecessor_codebook, candidates, unary, projected, successors, anchor):
    path = []
    previous = mx.array(anchor, dtype=mx.int32)
    for position in range(candidates.shape[0]):
        predecessor = mx.take(predecessor_codebook, previous, axis=0).astype(mx.float32)
        score = unary[position] + (successors[position] @ (predecessor * projected[position]))
        previous = mx.take(candidates[position], mx.argmax(score), axis=0)
        path.append(previous)
    return mx.stack(path).tolist() if path else []
