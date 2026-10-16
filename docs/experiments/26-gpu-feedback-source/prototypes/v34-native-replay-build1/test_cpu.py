"""Pure CPU range/staging invariants, no native library or MLX import."""
import unittest


def overlaps(a, n, b, m): return a < b+m and b < a+n


def staged_feedback(arena, outputs, links):
    staged = {key: bytes(arena[offset:offset+n]) for key, offset, n in outputs}
    for destination, output, n in links:
        arena[destination:destination+n] = staged[output]
    return staged


class AliasTests(unittest.TestCase):
    def test_cross_range_cycle_preserves_all_graph_outputs(self):
        arena = bytearray(b"AAAABBBBCCCC")
        outputs = [(1, 4, 4), (2, 0, 4), (3, 2, 8)]
        observed = staged_feedback(arena, outputs, [(0, 1, 4), (4, 2, 4)])
        self.assertEqual(arena, b"BBBBAAAACCCC")
        self.assertEqual(observed, {1: b"BBBB", 2: b"AAAA", 3: b"AABBBBCC"})

    def test_input_disjointness_is_byte_range_based(self):
        self.assertFalse(overlaps(0, 4, 4, 4))
        self.assertTrue(overlaps(0, 4, 2, 4))
        self.assertTrue(overlaps(4, 4, 4, 4))
        self.assertFalse(overlaps(1024, 61440, 65536, 4))


if __name__ == "__main__": unittest.main()
