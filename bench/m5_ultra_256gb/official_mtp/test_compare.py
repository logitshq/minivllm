"""CPU-only corruption checks for the independent E015 receipt comparator."""
import json
from pathlib import Path
import tempfile
import unittest

from compare import compare


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.paths = [Path(self.temp.name) / name for name in ("none", "mtp")]
        self.results = []
        for mode, path in zip(("none", "mtp"), self.paths):
            path.mkdir()
            plan = {key: "same" for key in ("target", "drafter", "workload_sha256", "distribution_identity", "harness", "environment")}
            plan["pid"] = 100 if mode == "none" else 200
            (path / "plan.json").write_text(json.dumps(plan))
            (path / "settings.json").write_text(json.dumps({"request": {"prompt_ids": list(range(19))}}))
            (path / "pool-shutdown.json").write_text('{"completed": true}')
            requests = []
            for index in range(3):
                rid = f"{mode}-{index}"
                row = {"raw": {"request_id": rid, "raw_ids": list(range(32)), "prompt_ids": list(range(19)),
                               "prompt_tokens": 19, "completion_tokens": 32, "cached_tokens": 0,
                               "finish_reason": "length", "error": None},
                       "routes": [], "verifies": [], "rounds": []}
                if mode == "mtp":
                    draft = [2] * 17
                    accept = [1] * 15 + [0, 0]
                    row.update(
                        routes=[{"request_id": rid, "uid": -1-index, "registered": True}],
                        verifies=[{"request_id": rid, "input_shape": [1, 3]} for _ in draft],
                        rounds=[{"request_id": rid, "proposed": d, "accepted": a} for d, a in zip(draft, accept)],
                        draft_lens=draft, accept_lens=accept,
                        counters_before={"rounds": 0, "accepted": 0, "drafted": 0},
                        counters_after={"rounds": 17, "accepted": 15, "drafted": 34},
                        counter_delta={"rounds": 17, "accepted": 15, "drafted": 34},
                        counter_owner_id=1000 + index, underlying_model_id=999,
                        underlying_counters_before={"rounds": 0, "accepted": 0, "drafted": 0},
                        underlying_counters_after={"rounds": 0, "accepted": 0, "drafted": 0})
                    running = {"rounds": 0, "accepted": 0, "drafted": 0}
                    for item in row["rounds"]:
                        before = dict(running)
                        running = {"rounds": running["rounds"] + 1,
                                   "accepted": running["accepted"] + item["accepted"],
                                   "drafted": running["drafted"] + item["proposed"]}
                        item.update(counter_owner_id=row["counter_owner_id"], underlying_model_id=999,
                                    counter_owner_class="omlx.speculative.vlm_mtp._MTPResetBindingProxy",
                                    counters_before=before, counters_after=dict(running))
                requests.append(row)
            result = {"mode": mode, "status": "PASS_ARM_ONLY_REQUIRES_FRESH_CONTROL_COMPARISON", "requests": requests,
                      "loads": [{"loaded": True, "kind_resolved": "mtp"}] if mode == "mtp" else [],
                      "attaches": [{"scheduler_same_drafter": True, "model_block_size": 3}] if mode == "mtp" else []}
            self.results.append(result)
        self.save()

    def save(self):
        for path, result in zip(self.paths, self.results):
            (path / "result.json").write_text(json.dumps(result))

    def test_valid_independent_arms(self):
        self.assertEqual(compare(*self.paths)["raw_ids_compared"], 192)

    def test_reject_changed_raw_token(self):
        self.results[1]["requests"][1]["raw"]["raw_ids"][5] += 1
        self.save()
        with self.assertRaises(AssertionError):
            compare(*self.paths)

    def test_reject_changed_prompt_token(self):
        self.results[1]["requests"][1]["raw"]["prompt_ids"][5] += 1
        self.save()
        with self.assertRaises(AssertionError):
            compare(*self.paths)

    def test_reject_soft_fallback(self):
        self.results[1]["requests"][0]["routes"][0]["uid"] = None
        self.save()
        with self.assertRaises((AssertionError, TypeError)):
            compare(*self.paths)

    def test_reject_counter_disagreement(self):
        self.results[1]["requests"][2]["counters_after"]["accepted"] += 1
        self.save()
        with self.assertRaises(AssertionError):
            compare(*self.paths)

    def test_reject_verify_width_disagreement(self):
        self.results[1]["requests"][0]["verifies"][0]["input_shape"] = [1, 1]
        self.save()
        with self.assertRaises(AssertionError):
            compare(*self.paths)

    def test_reject_proxy_replacement(self):
        self.results[1]["requests"][0]["rounds"][4]["counter_owner_id"] += 10
        self.save()
        with self.assertRaises(AssertionError):
            compare(*self.paths)

    def test_reject_counter_chain_gap(self):
        item = self.results[1]["requests"][0]["rounds"][4]
        item["counters_before"]["rounds"] += 1
        item["counters_after"]["rounds"] += 1
        self.save()
        with self.assertRaises(AssertionError):
            compare(*self.paths)


if __name__ == "__main__":
    unittest.main()
