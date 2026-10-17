"""Compare separate official none/MTP qualification receipts; never rank timings."""
import argparse
import hashlib
import json
from pathlib import Path


def compare(control, candidate):
    plans = [json.loads((p / "plan.json").read_text()) for p in (control, candidate)]
    results = [json.loads((p / "result.json").read_text()) for p in (control, candidate)]
    assert [r["mode"] for r in results] == ["none", "mtp"]
    assert all(r["status"] == "PASS_ARM_ONLY_REQUIRES_FRESH_CONTROL_COMPARISON" for r in results)
    assert plans[0]["pid"] != plans[1]["pid"], "requires fresh processes"
    for key in ("target", "drafter", "workload_sha256", "distribution_identity", "harness", "environment"):
        assert plans[0][key] == plans[1][key], "identity mismatch: " + key
    ids = results[0]["requests"][0]["raw"]["raw_ids"]
    assert len(ids) == 32
    assert all(r["raw"]["raw_ids"] == ids for result in results for r in result["requests"])
    assert all(len(result["requests"]) == 3 for result in results)
    settings = [json.loads((p / "settings.json").read_text()) for p in (control, candidate)]
    expected_prompt = settings[0]["request"]["prompt_ids"]
    assert len(expected_prompt) == 19 and settings[1]["request"]["prompt_ids"] == expected_prompt
    for result in results:
        for row in result["requests"]:
            raw = row["raw"]
            assert raw["completion_tokens"] == 32 and raw["prompt_tokens"] == 19
            assert raw["prompt_ids"] == expected_prompt and raw["cached_tokens"] == 0
            assert raw["finish_reason"] == "length" and raw["error"] is None
    assert not results[0]["loads"] and not results[0]["attaches"]
    assert all(not r["routes"] and not r["verifies"] and not r["rounds"] for r in results[0]["requests"])
    loads, attaches = results[1]["loads"], results[1]["attaches"]
    assert len(loads) == len(attaches) == 1
    assert loads[0]["loaded"] and loads[0]["kind_resolved"] == "mtp"
    assert attaches[0]["scheduler_same_drafter"] and attaches[0]["model_block_size"] == 3
    for row in results[1]["requests"]:
        routes, verifies, rounds = row["routes"], row["verifies"], row["rounds"]
        assert len(routes) == 1 and isinstance(routes[0]["uid"], int) and routes[0]["uid"] < 0 and routes[0]["registered"]
        assert all(e["request_id"] == row["raw"]["request_id"] for e in routes + verifies + rounds)
        draft, accept = row["draft_lens"], row["accept_lens"]
        assert len(verifies) == len(rounds) == len(draft) == len(accept) > 0
        assert draft == [r["proposed"] for r in rounds] and accept == [r["accepted"] for r in rounds]
        assert [v["input_shape"] for v in verifies] == [[1, d + 1] for d in draft]
        assert all(0 <= a <= d for a, d in zip(accept, draft))
        assert sum(draft) > 0 and sum(accept) > 0
        assert {r["counter_owner_id"] for r in rounds} == {row["counter_owner_id"]}
        assert {r["underlying_model_id"] for r in rounds} == {row["underlying_model_id"]}
        assert all(r["counter_owner_class"] == "omlx.speculative.vlm_mtp._MTPResetBindingProxy" for r in rounds)
        assert row["counters_before"] == rounds[0]["counters_before"]
        assert row["counters_after"] == rounds[-1]["counters_after"]
        for i, item in enumerate(rounds):
            before, after = item["counters_before"], item["counters_after"]
            assert {key: after[key] - before[key] for key in ("rounds", "accepted", "drafted")} == {
                "rounds": 1, "accepted": item["accepted"], "drafted": item["proposed"]}
            if i:
                assert before == rounds[i - 1]["counters_after"]
        delta = {key: row["counters_after"][key] - row["counters_before"][key]
                 for key in ("rounds", "accepted", "drafted")}
        assert delta == row["counter_delta"] == {"rounds": len(rounds), "accepted": sum(accept), "drafted": sum(draft)}
        assert 1 + sum(1 + n for n in accept) >= 32
    assert all(json.loads((p / "pool-shutdown.json").read_text())["completed"] for p in (control, candidate))
    return {"status": "PASS_OFFICIAL_MTP_ENGINE_IDS_AND_ACTUAL_ROUTE", "requests": 6,
            "raw_ids_compared": 192, "reference": "fresh official ordinary engine control",
            "raw_ids": ids, "no_performance_claim": True,
            "receipts": {str(p / "result.json"): hashlib.sha256((p / "result.json").read_bytes()).hexdigest()
                         for p in (control, candidate)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.control, args.candidate)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(result["status"])


if __name__ == "__main__":
    main()
