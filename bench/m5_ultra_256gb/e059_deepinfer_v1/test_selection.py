"""CPU-only tests for pure-host speculative greedy selection (selection.py).

Two independent angles:

* Exhaustive enumeration over block width (4/8), mismatch length, remaining
  budget and EOS position, checked against an independent reimplementation
  written straight from the spec.
* A multi-round causal simulation driven by a literal scripted target model:
  the concatenation of committed (KV-retained) inputs plus the pending token
  must always equal the emitted token history, the emitted history must be a
  prefix of the target's greedy stream, and every round must consume exactly
  ``computed_count`` fresh block positions.

Plus input-validation and no-mutation tests.  Standard library only.
"""

import unittest

from selection import Selection, select

EOS = 999
OTHER_EOS = 998


def make_case(width, accepted, eos_at=None, eos_ids=(EOS,)):
    """Build (drafts, posterior) whose longest matching prefix is ``accepted``.

    Drafts are distinct small ids; posterior matches drafts for the first
    ``accepted`` slots, then differs.  Optionally one EOS id is stamped at
    absolute output position ``eos_at`` of the untrimmed stream
    ``drafts[:accepted] + [posterior[accepted]]`` (positions beyond that
    stream are ignored by callers that only test trimming).
    """
    drafts = list(range(10, 10 + width - 1))
    posterior = [100 + i for i in range(width)]
    for j in range(accepted):
        posterior[j] = drafts[j]
    # ensure a real mismatch right after the accepted prefix (ids disjoint)
    out = drafts[:accepted] + [posterior[accepted]]
    if eos_at is not None and 0 <= eos_at < len(out):
        # stamp EOS id into whichever slot produces position eos_at
        if eos_at < accepted:
            drafts[eos_at] = EOS if EOS in eos_ids else eos_ids[0]
            posterior[eos_at] = drafts[eos_at]
        else:
            posterior[accepted] = EOS if EOS in eos_ids else eos_ids[0]
    return drafts, posterior


def expected_select(drafts, posterior, remaining, eos_ids):
    """Independent reference: literal transcription of the spec."""
    a = 0
    while a < len(drafts) and drafts[a] == posterior[a]:
        a += 1
    out = drafts[:a] + [posterior[a]]
    produced = []
    for token in out:
        if len(produced) >= remaining:
            break
        produced.append(token)
        if token in eos_ids:
            break
    finished = bool(produced) and (produced[-1] in eos_ids or len(produced) >= remaining)
    return Selection(
        produced=tuple(produced),
        accepted_drafts=min(a, len(produced)),
        computed_count=len(produced),
        next_anchor=None if finished else produced[-1],
        finished=finished,
    )


class ExhaustiveEnumerationTest(unittest.TestCase):
    """width 4/8 x mismatch length x remaining x EOS position."""

    def test_all_combinations(self):
        eos_forms = (frozenset([EOS]), {EOS}, [EOS], (EOS,))
        checked = 0
        for width in (4, 8):
            draft_len = width - 1
            for accepted in range(draft_len + 1):
                out_len = accepted + 1
                # EOS positions: none, every slot of the untrimmed output,
                # and past-the-end (must be irrelevant).
                eos_positions = [None] + list(range(out_len + 2))
                for eos_at in eos_positions:
                    drafts, posterior = make_case(width, accepted, eos_at)
                    for remaining in range(1, width + 3):
                        for eos_ids in eos_forms:
                            d_copy, p_copy = list(drafts), list(posterior)
                            got = select(drafts, posterior, remaining, eos_ids)
                            want = expected_select(drafts, posterior, remaining, set(eos_ids))
                            self.assertEqual(
                                got, want,
                                msg=f"width={width} accepted={accepted} eos_at={eos_at} "
                                    f"remaining={remaining} form={type(eos_ids).__name__} "
                                    f"drafts={drafts} posterior={posterior}",
                            )
                            self.assertIsInstance(got.produced, tuple)
                            self.assertEqual(drafts, d_copy, "select mutated drafts")
                            self.assertEqual(posterior, p_copy, "select mutated posterior")
                            # structural invariants
                            self.assertEqual(got.computed_count, len(got.produced))
                            self.assertLessEqual(got.accepted_drafts, accepted)
                            self.assertLessEqual(len(got.produced), min(out_len, remaining))
                            if got.finished:
                                self.assertIsNone(got.next_anchor)
                            else:
                                self.assertEqual(got.next_anchor, got.produced[-1])
                            checked += 1
        # exact sweep size:
        # width 4: sum(accepted+4 for accepted in 0..3) * 6 remaining * 4 eos forms = 528
        # width 8: sum(accepted+4 for accepted in 0..7) * 10 remaining * 4 eos forms = 2400
        self.assertEqual(checked, 528 + 2400)

    def test_multiple_eos_earliest_wins(self):
        drafts = [11, 12, 13]
        posterior = [11, 12, 13, 14]
        got = select(drafts, posterior, 10, {13, 14})
        self.assertEqual(got.produced, (11, 12, 13))
        self.assertTrue(got.finished)
        self.assertEqual(got.accepted_drafts, 3)
        self.assertEqual(got.computed_count, 3)

    def test_eos_beyond_cap_ignored(self):
        drafts = [11, 12, 13]
        posterior = [11, 12, 13, EOS]
        got = select(drafts, posterior, 2, [EOS])
        self.assertEqual(got.produced, (11, 12))  # cap hits before EOS slot
        self.assertTrue(got.finished)  # finished by budget, not EOS
        self.assertEqual(got.accepted_drafts, 2)

    def test_first_draft_mismatch(self):
        got = select([5, 6, 7], [42, 6, 7, 8], 10, [])
        self.assertEqual(got.produced, (42,))
        self.assertEqual(got.accepted_drafts, 0)
        self.assertEqual(got.computed_count, 1)
        self.assertEqual(got.next_anchor, 42)
        self.assertFalse(got.finished)


class CausalHistoryTest(unittest.TestCase):
    """Multi-round simulation against a literal scripted target model.

    The target is a literal token stream T: after consuming the token at
    absolute position p the target's argmax is T[p + 1].  Each round feeds
    ``[next_anchor] + drafts``; drafts follow T for a scripted number of
    positions then go wrong, so every mismatch length is exercised.  After
    each round the cache keeps ``computed_count`` fresh block positions:
    ``[anchor] + produced[:-1]``.
    """

    def simulate(self, width, target, accept_pattern, remaining, eos_ids):
        # Committed block inputs drawn from the emitted stream; target[0] is the
        # pre-round prompt token and never re-enters the emitted history.
        committed = []
        pending = target[1]              # last emitted token, not yet committed
        emitted = [target[1]]
        rounds = []
        step = 0
        while True:
            accepted = accept_pattern[step % len(accept_pattern)]
            accepted = min(accepted, width - 1)
            base = len(committed)        # number of committed fresh block inputs
            self.assertEqual(pending, target[base + 1])  # anchor is last emitted token
            drafts = []
            for j in range(width - 1):
                if j < accepted:
                    drafts.append(target[base + 2 + j])
                else:
                    drafts.append(-(1000 + j))  # guaranteed mismatch id
            drafts = [d if d >= 0 else 5000 + j for j, d in enumerate(drafts)]
            posterior = [target[base + 2 + j] for j in range(width)]
            d_copy, p_copy = list(drafts), list(posterior)

            budget = remaining - len(emitted)
            self.assertGreater(budget, 0)  # previous round must have finished
            result = select(drafts, posterior, budget, eos_ids)
            rounds.append(result)
            self.assertEqual(drafts, d_copy)
            self.assertEqual(posterior, p_copy)

            # retain computed_count fresh block positions: anchor + emitted[:-1]
            block = [pending] + drafts
            retained = block[: result.computed_count]
            self.assertEqual(retained, [pending] + list(result.produced[:-1]))
            committed.extend(retained)
            emitted.extend(result.produced)

            # causal invariant: committed inputs + pending == emitted history
            if result.finished:
                self.assertIsNone(result.next_anchor)
                self.assertEqual(committed + [result.produced[-1]], emitted)
            else:
                self.assertEqual(result.next_anchor, result.produced[-1])
                pending = result.next_anchor
                self.assertEqual(committed + [pending], emitted)
            # greedy verification never diverges from the target stream
            self.assertEqual(emitted, target[1 : 1 + len(emitted)])

            if result.finished:
                break
            step += 1
            self.assertLess(step, 1000, "simulation did not terminate")
        return rounds, emitted

    def test_runs_until_eos(self):
        for width in (4, 8):
            # literal target stream with EOS near the end; padding keeps block
            # reads in range even for the widest verify block
            target = [7, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                      30, 31, 32, 33, 34, 35, 36, 37, EOS] + [100 + i for i in range(32)]
            for accept_pattern in ([0], [width - 1], [1, width - 2, 0, 3 if width == 8 else 1]):
                rounds, emitted = self.simulate(width, target, accept_pattern, 10 ** 6, {EOS})
                self.assertEqual(emitted, target[1 : target.index(EOS) + 1])
                self.assertTrue(rounds[-1].finished)
                self.assertIn(EOS, rounds[-1].produced)

    def test_runs_until_cap(self):
        for width in (4, 8):
            target = [7] + [100 + i for i in range(64)]
            remaining = 7
            rounds, emitted = self.simulate(width, target, [2, width - 1, 0], remaining, {EOS})
            self.assertEqual(len(emitted), remaining)
            self.assertEqual(emitted, target[1 : 1 + remaining])
            self.assertTrue(rounds[-1].finished)
            self.assertNotIn(EOS, emitted)

    def test_zero_accept_every_round(self):
        # target always contradicts every draft: pure correction-token crawl
        for width in (4, 8):
            target = [3, 11, 12, 13, 14] + [EOS] + [100 + i for i in range(16)]
            rounds, emitted = self.simulate(width, target, [0], 10 ** 6, [EOS])
            self.assertTrue(all(r.accepted_drafts == 0 for r in rounds))
            self.assertEqual(emitted, [11, 12, 13, 14, EOS])


class ValidationTest(unittest.TestCase):
    def test_bad_token_ids(self):
        bad = [-1, True, False, 1.5, "3", None, 2 ** 63 * -1]
        for value in bad:
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    select([value, 2, 3], [1, 2, 3, 4], 5, [])
                with self.assertRaises((TypeError, ValueError)):
                    select([1, 2, 3], [value, 2, 3, 4], 5, [])
                with self.assertRaises((TypeError, ValueError)):
                    select([1, 2, 3], [1, 2, 3, 4], 5, [value])

    def test_width_and_length(self):
        with self.assertRaises(ValueError):
            select([1, 2], [1, 2, 3], 5, [])          # width 3 block unsupported
        with self.assertRaises(ValueError):
            select([1, 2, 3, 4], [1, 2, 3, 4, 5], 5, [])  # width 5 block unsupported
        with self.assertRaises(ValueError):
            select([1, 2, 3], [1, 2, 3], 5, [])       # posterior too short
        with self.assertRaises(ValueError):
            select([1, 2, 3], [1, 2, 3, 4, 5], 5, [])  # posterior too long
        with self.assertRaises(TypeError):
            select("123", [1, 2, 3, 4], 5, [])
        with self.assertRaises(TypeError):
            select({1, 2, 3}, [1, 2, 3, 4], 5, [])    # unordered container rejected

    def test_remaining(self):
        for bad in (0, -3):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    select([1, 2, 3], [1, 2, 3, 4], bad, [])
        for bad in (True, 1.0, "2", None):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    select([1, 2, 3], [1, 2, 3, 4], bad, [])

    def test_eos_container_forms(self):
        drafts, posterior = [1, 2, 3], [1, 2, 3, EOS]
        for form in ([EOS], (EOS,), {EOS}, frozenset([EOS])):
            with self.subTest(form=type(form).__name__):
                got = select(drafts, posterior, 10, form)
                self.assertTrue(got.finished)
        for bad in (EOS, "999", {EOS: 1}, 3.5):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    select(drafts, posterior, 10, bad)

    def test_empty_eos(self):
        got = select([1, 2, 3], [1, 2, 3, 4], 10, [])
        self.assertFalse(got.finished)
        self.assertEqual(got.produced, (1, 2, 3, 4))

    def test_no_input_mutation_and_result_immutability(self):
        drafts, posterior = [1, 2, 3], [1, 2, 3, 9]
        eos = [9]
        snapshot = (list(drafts), list(posterior), list(eos))
        result = select(drafts, posterior, 10, eos)
        self.assertEqual((drafts, posterior, eos), snapshot)
        self.assertIsInstance(result.produced, tuple)
        with self.assertRaises(AttributeError):
            result.finished = False  # frozen dataclass


if __name__ == "__main__":
    unittest.main()
