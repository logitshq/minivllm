"""CPU-only independent evidence/source audit and immutable E015 archive builder."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

from compare import compare


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    base = args.base.resolve()
    control, candidate = (base / f"official-mtp-{mode}-v1" for mode in ("none", "mtp"))
    independently_compared = compare(control, candidate)
    saved_comparison = read(base / "official-mtp-comparison-v1.json")
    for key in ("status", "requests", "raw_ids_compared", "raw_ids", "no_performance_claim"):
        assert independently_compared[key] == saved_comparison[key]
    plans = [read(path / "plan.json") for path in (control, candidate)]
    source_hashes = plans[0]["distribution_identity"]["files"]
    assert source_hashes == plans[1]["distribution_identity"]["files"]
    for path, expected in source_hashes.items():
        assert sha(path) == expected, "current installed file changed: " + path

    source_files = {}
    resource_rows = []
    request_rows = []
    package_root = None
    for mode, directory, plan in zip(("none", "mtp"), (control, candidate), plans):
        for path, expected in plan["harness"].items():
            assert sha(path) == expected
            source_files["harness/" + Path(path).name] = Path(path)
        loaded = read(directory / "loaded-sources.json")
        for module, identity in loaded.items():
            assert source_hashes[identity["file"]] == identity["sha256"], module
            path = Path(identity["file"])
            if path.suffix == ".py":
                site_index = path.parts.index("site-packages")
                relative = Path(*path.parts[site_index + 1:])
                source_files[str(Path("official-source") / relative)] = path
                package_root = Path(*path.parts[:site_index + 1])
        for seam, identity in read(directory / "observer-seams.json").items():
            assert source_hashes[identity["file"]] == identity["sha256"], seam
        events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
        result = read(directory / "result.json")
        assert [e for e in events if e["kind"] == "load"] == result["loads"]
        assert [e for e in events if e["kind"] == "attach"] == result["attaches"]
        ids = {row["raw"]["request_id"] for row in result["requests"]}
        assert len(ids) == 3
        assert all(e.get("request_id") in ids for e in events if e["kind"] not in ("load", "attach"))
        for index, row in enumerate(result["requests"]):
            assert read(directory / f"request-{index}.json") == row
            rid = row["raw"]["request_id"]
            actual = [e for e in events if e.get("request_id") == rid]
            assert [e for e in actual if e["kind"] == "output"] == [row["raw"]]
            for kind, field in (("route", "routes"), ("verify", "verifies"), ("round", "rounds")):
                assert [e for e in actual if e["kind"] == kind] == row[field]
            request_rows.append({"mode": mode, "request_index": index, "raw_output_tokens": 32,
                                 "verify_calls": len(row["verifies"]),
                                 "actual_proxy_counter_delta": row.get("counter_delta"),
                                 "draft_lens": row.get("draft_lens"), "accept_lens": row.get("accept_lens"),
                                 "underlying_counters_after": row["underlying_counters_after"]})
        assert read(directory / "pool-shutdown.json")["completed"]
        owner = base / f"official-mtp-{mode}-v1-owner"
        release, complete = read(owner / "release.json"), read(owner / "complete.json")
        assert complete["status"] == "PASS"
        assert release["pid"] == read(owner / "child.json")["pid"] == plan["pid"]
        assert release["returncode"] == 0 and release["waited_reaped"] and release["strict_absence"]
        probe = subprocess.run(["/bin/ps", "-p", str(release["pid"]), "-o", "pid="], capture_output=True, text=True)
        assert probe.returncode == 1 and not probe.stdout and not probe.stderr
        rows = read(owner / "resources.json")
        for index, row in enumerate(rows):
            raw = read(owner / f"resource-{index:03d}.json")
            for key, value in raw.items():
                assert row[key] == value
            assert row["pressure_level_value"] == 1
            assert row["power_metadata"]["source"] == "AC Power"
            assert row["processinfo_thermal_metadata"]["code"] == 0
            assert row["only_owned_known_inference_processes"] and row["no_active_foreign_native_processes"]
            assert row["native_process_inventory_ok"] and row["process_inventory_ok"]
            assert row["authorized_suspended_pids"] == [74384]
            if row["label"] == "child-running":
                assert row["owned_model_pid"] == release["pid"]
                assert row["exact_owned_pid_metadata"]["status"] == "known"
            resource_rows.append(row)

    lease_dir = base / "official-mtp-service-lease-v1"
    lease, resume = read(lease_dir / "lease.json"), read(lease_dir / "resume.json")
    assert lease["closed"] and lease["service_pid"] == 74384
    assert resume["identity_match"] and resume["state_before"] == "T" and resume["state_after"] == "S"
    assert lease["jobs"] == read(base / "official-mtp-jobs-v1.json")
    for index, mode in enumerate(("none", "mtp")):
        job = read(lease_dir / f"job-{index}.json")
        assert job["returncode"] == 0 and job["waited"] and job["command"] == lease["jobs"][index]
        assert job["pid"] == read(base / f"official-mtp-{mode}-v1-owner/complete.json")["owner_pid"]
    assert len({r["swap_used_mb"] for r in resource_rows}) == 1
    assert len({r["vm_counters"]["Swapouts"] for r in resource_rows}) == 1
    checks = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", str(Path(__file__).parent),
                             "-p", "test_*.py", "-v"], capture_output=True, text=True)
    assert checks.returncode == 0, checks.stderr
    args.out.mkdir(parents=True, exist_ok=False)
    write(args.out / "cpu-checks.json", {"command": checks.args, "returncode": checks.returncode,
                                       "stdout": checks.stdout, "stderr": checks.stderr})
    summary = {"status": independently_compared["status"], "requests": request_rows,
               "compared_raw_token_ids": 192, "distinct_prompts": 1,
               "resource_samples": len(resource_rows), "thermal_codes": [0], "pressure_levels": [1],
               "power": "AC Power", "swap_used_mb": resource_rows[0]["swap_used_mb"],
               "swapout_delta": 0, "both_children_reaped_and_absent": True, "external_service_restored": True,
               "installed_files_rehashed": len(source_hashes),
               "loaded_source_snapshot_counts": [len(read(p / "loaded-sources.json")) for p in (control, candidate)],
               "versions": plans[0]["distribution_identity"]["versions"],
               "target_revision": plans[0]["target"]["revision"], "drafter_revision": plans[0]["drafter"]["revision"],
               "no_performance_claim": True,
               "scope": "M5 instrumented official EnginePool API, 19->32 greedy normal EOS, one prompt, 3 requests/arm; not HTTP or all-workload losslessness"}
    write(args.out / "summary.json", summary)
    write(args.out / "audit.json", {"status": "PASS", "independent_compare": independently_compared,
          "raw_events_equal_receipts": True, "source_identity_pass": True,
          "model_weight_hash_scope": "full SHA recorded by each child and equal across arms; not reread during this CPU post-audit",
          "resource_process_scope": "known executable inventory and exact child ownership; unrecognized runtimes/GUI GPU use unknown",
          "source_snapshot_scope": "loaded official Python files at post-load snapshot plus engine_pool; native binaries hashed but not copied"})
    files = dict(source_files)
    assert package_root is not None
    engine_pool = package_root / "omlx/engine_pool.py"
    assert sha(engine_pool) == source_hashes[str(engine_pool)]
    files["official-source/omlx/engine_pool.py"] = engine_pool
    files["harness/audit.py"] = Path(__file__).resolve()
    for parent in (control, candidate, base / "official-mtp-none-v1-owner", base / "official-mtp-mtp-v1-owner", lease_dir):
        for path in parent.rglob("*"):
            if path.is_file() and not path.is_symlink():
                files[str(Path("raw") / path.relative_to(base))] = path
    for name in ("official-mtp-jobs-v1.json", "official-mtp-comparison-v1.json"):
        files["raw/" + name] = base / name
    for name in ("summary.json", "audit.json", "cpu-checks.json"):
        files["audit/" + name] = args.out / name
    manifests = {name: {"source": str(path), "size": path.stat().st_size, "sha256": sha(path)}
                 for name, path in sorted(files.items())}
    archive = args.out / "evidence.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, path in sorted(files.items()):
            tar.add(path, arcname=name, recursive=False)
    with tarfile.open(archive, "r:gz") as tar:
        assert set(tar.getnames()) == set(manifests)
        for entry in tar.getmembers():
            assert entry.isfile() and entry.size == manifests[entry.name]["size"]
            assert hashlib.sha256(tar.extractfile(entry).read()).hexdigest() == manifests[entry.name]["sha256"]
    write(args.out / "manifest.json", {"archive_sha256": sha(archive), "archive_bytes": archive.stat().st_size,
          "member_count": len(manifests), "members": manifests, "all_members_hash_verified": True})
    print(json.dumps({"status": "PASS", "archive_files": len(manifests), "archive_bytes": archive.stat().st_size,
                      "resource_samples": len(resource_rows), "source_files_checked": len(source_hashes)}))


if __name__ == "__main__":
    main()
