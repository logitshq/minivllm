"""Pure CPU bank/reset invariants; no MLX or dylib loading."""
import unittest


def run(seed, steps):
    banks = [seed, None]
    next_input_bank = 0
    outputs = []
    for _ in range(steps):
        written = next_input_bank ^ 1
        banks[written] = banks[next_input_bank] * 3 + 1
        next_input_bank ^= 1
        outputs.append((next_input_bank, banks[next_input_bank]))
    return outputs


class BankTests(unittest.TestCase):
    def test_odd_even_banks_and_original_recurrence(self):
        self.assertEqual(run(2, 4), [(1, 7), (0, 22), (1, 67), (0, 202)])

    def test_reset_and_altered_seed(self):
        self.assertEqual(run(2, 3), run(2, 3))
        self.assertEqual(run(4, 2), [(1, 13), (0, 40)])



if __name__ == "__main__": unittest.main()
