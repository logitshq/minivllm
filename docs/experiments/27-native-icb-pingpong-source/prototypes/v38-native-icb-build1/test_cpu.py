"""CPU-only reference checks for before-command barriers and byte banks."""
import unittest


def fold_barriers(events):
    before, commands = False, []
    for event in events:
        if event == "barrier": before = True
        else:
            commands.append((event, before))
            before = False
    return commands, before


class EncodingTests(unittest.TestCase):
    def test_before_command_barriers_and_trailing_fence(self):
        events = ["A", "barrier", "barrier", "B", "C", "barrier", "D", "barrier"]
        commands, trailing = fold_barriers(events)
        self.assertEqual(commands, [("A", False), ("B", True), ("C", False), ("D", True)])
        self.assertTrue(trailing)  # graphDone fence orders final graph → feedback

    def test_payload_bank_preserves_every_byte_at_aligned_offsets(self):
        payloads = [bytes(range(255)), b"\x00\xff\x80\x7f\x01", bytes(range(64))]
        bank, offsets = bytearray(), []
        for payload in payloads:
            offset = (len(bank)+255) & ~255
            bank.extend(b"\x00"*(offset-len(bank)))
            offsets.append(offset)
            bank.extend(payload)
        self.assertEqual(offsets, [0, 256, 512])
        self.assertEqual([bytes(bank[o:o+len(p)]) for o,p in zip(offsets,payloads)], payloads)


if __name__ == "__main__": unittest.main()
