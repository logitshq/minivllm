"""Tiny byte-backed CPU fixtures; no real MLX, NumPy, model, or GPU imports."""
import builtins
import importlib.util
from pathlib import Path
import struct
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent


class Piece:
    """Rank-2 byte storage: slicing copies rows but preserves every bit."""
    def __init__(self, payload, *, width=4, dtype="bfloat16"):
        self.width, self.dtype = width, dtype
        self.itemsize = 4 if dtype == "float32" else 2
        self.rowbytes = width * self.itemsize
        assert len(payload) % self.rowbytes == 0
        self.payload = payload
        self.shape, self.ndim = (len(payload) // self.rowbytes, width), 2

    def __getitem__(self, selection):
        assert isinstance(selection, slice)
        lo, hi, step = selection.indices(self.shape[0])
        assert step == 1
        return Piece(self.payload[lo * self.rowbytes:hi * self.rowbytes], width=self.width, dtype=self.dtype)


def make_piece(first, rows, *, width=4, dtype="bfloat16"):
    itemsize = 4 if dtype == "float32" else 2
    # Includes arbitrary bit patterns, not floating arithmetic or recasting.
    payload = b"".join(struct.pack("<I", (first + row) * width + col)[:itemsize]
                       for row in range(rows) for col in range(width))
    return Piece(payload, width=width, dtype=dtype)


class ConcatSpy:
    def __init__(self):
        self.calls = []

    def __call__(self, pieces, axis=0):
        assert axis == 0 and pieces
        self.calls.append(tuple(pieces))
        return Piece(b"".join(piece.payload for piece in pieces), width=pieces[0].width, dtype=pieces[0].dtype)


class ViewPiece:
    """Retains a base buffer and row index mapping, including non-unit stride."""
    def __init__(self, base, indices):
        self.base, self.indices = base, tuple(indices)
        self.width, self.dtype = base.width, base.dtype
        self.shape, self.ndim = (len(self.indices), self.width), 2

    @property
    def payload(self):
        return b"".join(self.base.payload[i * self.base.rowbytes:(i + 1) * self.base.rowbytes]
                        for i in self.indices)

    def __getitem__(self, selection):
        return ViewPiece(self.base, self.indices[selection])


class TailTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        original_import = builtins.__import__
        def no_runtime(name, *args, **kwargs):
            if name.split('.')[0] in {"mlx", "numpy", "torch"}:
                raise AssertionError("forbidden runtime import: " + name)
            return original_import(name, *args, **kwargs)
        spec = importlib.util.spec_from_file_location("e019_candidate", HERE / "candidate.py")
        cls.candidate = importlib.util.module_from_spec(spec)
        with patch.object(builtins, "__import__", no_runtime):
            spec.loader.exec_module(cls.candidate)

    def compare(self, history, fed, start, *, width=4, dtype="bfloat16"):
        ids = tuple(map(id, history))
        before = tuple(piece.payload for piece in history)
        reference_concat, candidate_concat = ConcatSpy(), ConcatSpy()
        expected = self.candidate.full_history_reference(history, fed, start, concatenate=reference_concat)
        actual = self.candidate.pending_features(history, fed, start, width=width, dtype=dtype,
                                                concatenate=candidate_concat)
        if expected is None:
            self.assertIsNone(actual)
        else:
            self.assertEqual(actual.payload, expected.payload)
            self.assertEqual(actual.shape, (start - fed, width))
            self.assertEqual(actual.dtype, dtype)
        self.assertEqual(tuple(map(id, history)), ids)
        self.assertEqual(tuple(piece.payload for piece in history), before)
        return actual, candidate_concat, reference_concat

    def test_single_whole_piece_returns_identity_without_concat(self):
        history = [make_piece(0, 19), make_piece(19, 3)]
        actual, candidate_concat, reference_concat = self.compare(history, 19, 22)
        self.assertIs(actual, history[-1])
        self.assertEqual(candidate_concat.calls, [])
        self.assertEqual(sum(p.shape[0] for p in reference_concat.calls[0]), 22)

    def test_partial_last_piece_without_concat(self):
        _, candidate_concat, _ = self.compare([make_piece(0, 19), make_piece(19, 5)], 21, 24)
        self.assertEqual(candidate_concat.calls, [])

    def test_cross_chunk_only_concatenates_required_rows(self):
        history = [make_piece(0, 19), make_piece(19, 3), make_piece(22, 5)]
        _, candidate_concat, reference_concat = self.compare(history, 21, 27)
        self.assertEqual([[p.shape[0] for p in call] for call in candidate_concat.calls], [[1, 5]])
        self.assertEqual(sum(p.shape[0] for p in reference_concat.calls[0]), 27)

    def test_reset_fed_zero_delivers_complete_prefix(self):
        history = [make_piece(0, 4), make_piece(4, 7), make_piece(11, 3)]
        actual, candidate_concat, _ = self.compare(history, 0, 14)
        self.assertEqual(actual.payload, b"".join(p.payload for p in history))
        self.assertEqual(candidate_concat.calls, [tuple(history)])
        single = [make_piece(0, 14)]
        actual, candidate_concat, _ = self.compare(single, 0, 14)
        self.assertIs(actual, single[0])
        self.assertEqual(candidate_concat.calls, [])

    def test_caught_up_and_empty_prefix(self):
        _, candidate_concat, _ = self.compare([make_piece(0, 19)], 19, 19)
        self.assertEqual(candidate_concat.calls, [])
        self.compare([], 0, 0)

    def test_empty_pieces_preserve_boundaries(self):
        history = [make_piece(0, 0), make_piece(0, 3), make_piece(3, 0), make_piece(3, 2), make_piece(5, 0)]
        _, candidate_concat, _ = self.compare(history, 2, 5)
        self.assertEqual([[p.shape[0] for p in call] for call in candidate_concat.calls], [[1, 2]])

    def test_offset_and_strided_views_are_not_materialized_by_helper(self):
        base = make_piece(0, 18)
        history = [ViewPiece(base, range(2, 8)), ViewPiece(base, range(9, 18, 2))]
        actual, candidate_concat, _ = self.compare(history, 6, 11)
        self.assertIs(actual, history[-1])
        self.assertEqual(candidate_concat.calls, [])
        actual, candidate_concat, _ = self.compare(history, 8, 11)
        self.assertIs(actual.base, base)
        self.assertEqual(actual.indices, (13, 15, 17))
        self.assertEqual(candidate_concat.calls, [])
        self.compare(history, 4, 11)

    def test_all_small_split_and_fed_boundaries(self):
        for a in range(1, 5):
            for b in range(1, 5):
                for c in range(1, 5):
                    history = [make_piece(0, a), make_piece(a, b), make_piece(a + b, c)]
                    for fed in range(a + b + c + 1):
                        self.compare(history, fed, a + b + c)

    def test_compression_exactly_matches_existing_greater_than_64_rule(self):
        for count in (64, 65):
            history = [make_piece(i, 1) for i in range(count)]
            self.compare(history, count - 2, count)
            # ModelRunner compresses after step, before appending new features.
            compression_concat = ConcatSpy()
            if len(history) > 64:
                history[:] = [compression_concat(history)]
            history.append(make_piece(count, 2))
            self.assertEqual(len(compression_concat.calls), int(count > 64))
            self.assertEqual(len(history), 65 if count == 64 else 2)
            actual, candidate_concat, _ = self.compare(history, count, count + 2)
            self.assertIs(actual, history[-1])
            self.assertEqual(candidate_concat.calls, [])
            self.compare(history, count - 1, count + 2)
            self.compare(history, 0, count + 2)

    def test_dtype_and_current_width_contract(self):
        for dtype in ("bfloat16", "float16", "float32"):
            self.compare([make_piece(0, 2, dtype=dtype), make_piece(2, 2, dtype=dtype)], 1, 4, dtype=dtype)
        self.compare([make_piece(0, 1, width=25600)], 0, 1, width=25600)
        # Signed-zero and NaN payload bits are preserved, with no float conversion.
        raw = bytes.fromhex("00000080007fc17f")
        piece = Piece(raw)
        actual, _, _ = self.compare([piece], 0, 1)
        self.assertEqual(actual.payload, raw)

    def test_invalid_contract_rejected_before_concat(self):
        good = [make_piece(0, 3)]
        cases = [(good, -1, 3, 4, "bfloat16"), (good, 4, 3, 4, "bfloat16"),
                 (good, 0, 2, 4, "bfloat16"), (good, 0, 4, 4, "bfloat16"),
                 (good, 0, 3, 5, "bfloat16"), (good, 0, 3, 4, "float16"),
                 (good, 3, 3, 5, "bfloat16"), ([], 0, 1, 4, "bfloat16"),
                 ([make_piece(0, 1), make_piece(1, 2, dtype="float32")], 1, 3, 4, "bfloat16")]
        for history, fed, start, width, dtype in cases:
            spy = ConcatSpy()
            with self.assertRaises(ValueError):
                self.candidate.pending_features(history, fed, start, width=width, dtype=dtype, concatenate=spy)
            self.assertEqual(spy.calls, [])
        for fed in (True, 0.0):
            with self.assertRaises(TypeError):
                self.candidate.pending_features(good, fed, 3, width=4, dtype="bfloat16", concatenate=ConcatSpy())
        with self.assertRaises(TypeError):
            self.candidate.pending_features(tuple(good), 0, 3, width=4, dtype="bfloat16", concatenate=ConcatSpy())
        wrong_rank = make_piece(0, 3)
        wrong_rank.ndim = 3
        with self.assertRaises(ValueError):
            self.candidate.pending_features([wrong_rank], 0, 3, width=4, dtype="bfloat16", concatenate=ConcatSpy())


if __name__ == "__main__":
    unittest.main()
