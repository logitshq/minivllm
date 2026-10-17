"""Independent stdlib E019 evidence audit/archive; never imports MLX or screen.

Outside history_tail/ to preserve the measured source set. Source bytes always
come from preregistration commit be12f82, including the subsequently updated doc.
"""
import array
import copy
import hashlib
import io
import json
import math
import mmap
from pathlib import Path
import statistics
import struct
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / ".artifacts/m5-ultra-256gb"
RUN = BASE / "history-tail-v1"
OUT = ROOT / "docs/experiments/m5-ultra-256gb/E019-history-tail-v1"
COMMIT = "be12f82"
WIDTH = 25600
LENGTHS = (19, 128, 512, 2048)
TUTORIAL = "docs/optimization-guide/tutorials/019-drafter-history-tail.md"


def read(path):
    return json.loads(Path(path).read_text())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def cases():
    result = []
    for n in LENGTHS:
        names = ["single", "short_tail", "cross_chunks", "offset_single", "row_stride_cross", "column_stride_cross"]
        if n != 19:
            names += ["old64_then_append", "old65_compress_then_append"]
        for layout in names:
            for role, missing in [("tail1", 1), ("tail3", 3), ("tail7", 7), ("reset", n), ("caught_up", 0)]:
                result.append({"id": f"N{n}-{layout}-{role}", "N": n, "layout": layout,
                               "role": role, "fed": n - missing, "start": n, "rows": missing,
                               "timed": missing > 0, "gate_group": "none" if not missing else
                               "long_tail" if n >= 512 and role.startswith("tail") else "other"})
    return result


def piece_lengths(n, layout):
    if layout in ("single", "offset_single"):
        return [n]
    if layout in ("short_tail", "old65_compress_then_append"):
        return [n - 7, 7]
    if layout == "old64_then_append":
        return [((i + 1) * (n - 7)) // 64 - (i * (n - 7)) // 64 for i in range(64)] + [7]
    return [n - 3, 2, 1]


def audit_tensor(path, metadata, n):
    raw = path.read_bytes()
    assert len(raw) == metadata["bytes"] and sha(raw) == metadata["sha256"]
    header_length, = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8:8 + header_length])
    header.pop("__metadata__", None)
    assert set(header) == {"flat"}
    flat = header["flat"]
    count = (2 * n + 4) * WIDTH
    assert flat == {"dtype": "BF16", "shape": [count], "data_offsets": [0, 2 * count]}
    assert metadata["flat_shape"] == [count] and metadata["dtype"] == "mlx.core.bfloat16"
    offset = 8 + header_length
    assert offset + 2 * count == len(raw)
    # Every generated value is an exact power-of-two fraction representable in
    # BF16, so taking F32's high half gives its exact two stored little-endian bytes.
    period = b"".join(struct.pack("<f", (((i + 19019) % 257) - 128) / 128)[2:] for i in range(257))
    chunk = period * 4096
    body = memoryview(raw)[offset:]
    for start in range(0, len(body), len(chunk)):
        selected = body[start:start + len(chunk)]
        assert selected == chunk[:len(selected)], "frozen input differs from fixed preregistered pattern"
    return {"file_sha256": sha(raw), "payload_sha256": sha(body), "bytes": len(raw),
            "elements": count, "payload_offset": offset, "dtype": "BF16", "full_pattern_checked": True}


def logical_sha(mm, payload_offset, layout, first, rows):
    base = WIDTH + 1 if layout == "offset_single" else 0
    row_stride = 2 * WIDTH if layout in ("row_stride_cross", "column_stride_cross") else WIDTH
    col_stride = 2 if layout == "column_stride_cross" else 1
    digest = hashlib.sha256()
    for row in range(first, first + rows):
        start = payload_offset + 2 * (base + row * row_stride)
        if col_stride == 1:
            digest.update(mm[start:start + 2 * WIDTH])
        else:
            words = array.array("H")
            words.frombytes(mm[start:start + 4 * WIDTH])
            digest.update(words[::2].tobytes())
    return digest.hexdigest()


def audit_check(row, case, expected_inputs, expected_output):
    assert {key: row[key] for key in case} == case
    for key in ("exact", "inputs_unchanged", "input_objects_unchanged", "concat_call_contract", "passed",
                "python_calls_do_not_prove_physical_metal_traffic"):
        assert row[key] is True, key
    assert row["inputs_before"] == row["inputs_after"] == expected_inputs
    lengths = [piece["shape"][0] for piece in expected_inputs]
    selected, cursor = [], 0
    for length in lengths:
        overlap = max(0, min(cursor + length, case["start"]) - max(cursor, case["fed"]))
        if overlap:
            selected.append(overlap)
        cursor += length
    expected_candidate = [selected] if len(selected) > 1 else []
    expected_reference = [lengths] if case["timed"] and len(lengths) > 1 else []
    assert row["expected_candidate_concat_rows"] == expected_candidate
    assert row["expected_reference_concat_rows"] == expected_reference
    assert row["candidate_python_concat_calls"] == [{"axis": 0, "rows": values} for values in expected_candidate]
    assert row["reference_python_concat_calls"] == [{"axis": 0, "rows": values} for values in expected_reference]
    output = row["output"]
    if case["timed"]:
        assert output["reference_sha256"] == output["candidate_sha256"] == expected_output
        assert output["shape"] == [case["rows"], WIDTH] and output["dtype"] == "mlx.core.bfloat16"
        full_indices, position = [], 0
        for index, length in enumerate(lengths):
            if position == case["fed"] and position + length == case["start"]:
                full_indices.append(index)
            position += length
        assert output["candidate_same_piece_indices"] == full_indices
    else:
        assert output == {"reference_none": True, "candidate_none": True}


def audit_timing(row, case):
    assert {key: row[key] for key in case} == case and case["timed"]
    arms = ("reference", "candidate")
    assert set(row["first_post_qualification_ms"]) == set(row["warmups_ms"]) == set(arms)
    assert all(len(row["warmups_ms"][arm]) == 2 for arm in arms)
    assert [(r["block"], r["arm"]) for r in row["rounds"]] == [
        (block, arm) for block in range(3) for arm in ("reference", "candidate", "candidate", "reference")]
    samples = list(row["first_post_qualification_ms"].values())
    samples += [sample for arm in arms for sample in row["warmups_ms"][arm]] + row["rounds"]
    assert len(samples) == 18
    for sample in samples:
        a, b, total = [sample[key] for key in ("build_and_eval_ms", "release_ms", "build_eval_and_release_ms")]
        assert all(math.isfinite(x) and x >= 0 for x in (a, b, total)) and total > 0
        assert math.isclose(a + b, total, rel_tol=1e-12, abs_tol=1e-12)
    medians = {arm: statistics.median(r["build_eval_and_release_ms"] for r in row["rounds"] if r["arm"] == arm) for arm in arms}
    assert row["median_build_eval_and_release_ms"] == medians
    ratio = medians["reference"] / medians["candidate"]
    assert math.isclose(row["speedup"], ratio, rel_tol=1e-12)
    return {"id": case["id"], "N": case["N"], "layout": case["layout"], "role": case["role"],
            "gate_group": case["gate_group"], "speedup": ratio, "median_ms": medians}


def main():
    assert not OUT.exists(), "Immutable evidence archive already exists"
    plan, recorded = read(RUN / "plan.json"), read(RUN / "summary.json")
    expected_cases = cases()
    assert plan["cases"] == expected_cases and len(expected_cases) == 150
    assert plan["width"] == WIDTH and plan["dtype"] == "bfloat16" and plan["seed"] == 19019
    assert plan["warmups_per_arm"] == 2 and plan["abba_blocks"] == 3 and plan["pool_limit_bytes"] == 2 * 1024 ** 3
    assert plan["timing_metric"] == "build_eval_and_release_ms"
    assert plan["gate"] == {"long_N512_N2048_all_tails_geomean_min": 1.25, "all_other_nonempty_points_min": .97}
    identity = plan["identity"]
    assert identity["mlx"] == identity["mlx_metal"] == "0.32.3"
    assert identity["device"]["device_name"] == "Apple M5 Ultra" and identity["device"]["memory_size"] == 256 * 1024 ** 3
    assert identity["candidate_loaded_source"] == str(ROOT / "bench/m5_ultra_256gb/history_tail/candidate.py")
    files = {}
    def collect(path, name=None):
        path = Path(path)
        key = name or str(path.relative_to(ROOT))
        data = path.read_bytes()
        assert key not in files or files[key] == data
        files[key] = data
        return data
    for path, expected in plan["source_sha256"].items():
        relative = str(Path(path).relative_to(ROOT))
        data = subprocess.check_output(["git", "-C", str(ROOT), "show", COMMIT + ":" + relative])
        assert sha(data) == expected, "preregistered source differs from actual run pin: " + relative
        files[relative] = data
    assert len(plan["source_sha256"]) == 7
    files["preregistration/tutorial019-be12f82.md"] = files[TUTORIAL]
    tensor_audit = {}
    expected_histories, expected_outputs = {}, {}
    for n in LENGTHS:
        metadata = plan["inputs"][str(n)]
        path = RUN / metadata["file"]
        tensor_audit[str(n)] = audit_tensor(path, metadata, n)
        with path.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            for case in [c for c in expected_cases if c["N"] == n]:
                layout = case["layout"]
                key = f"N{n}-{layout}"
                if key not in expected_histories:
                    cursor, pieces = 0, []
                    for length in piece_lengths(n, layout):
                        pieces.append({"shape": [length, WIDTH], "dtype": "mlx.core.bfloat16",
                                       "sha256": logical_sha(mm, tensor_audit[str(n)]["payload_offset"], layout, cursor, length)})
                        cursor += length
                    assert cursor == n
                    expected_histories[key] = pieces
                expected_outputs[case["id"]] = logical_sha(mm, tensor_audit[str(n)]["payload_offset"], layout, case["fed"], case["rows"])
    checks, timing = read(RUN / "qualification.json"), read(RUN / "timing.json")
    assert len(checks) == 150 and len(timing) == 120
    for case, row in zip(expected_cases, checks):
        assert row == read(RUN / ("qualification-" + case["id"] + ".json"))
        audit_check(row, case, expected_histories[f"N{case['N']}-{case['layout']}"], expected_outputs[case["id"]])
    assert read(RUN / "inputs-after-timing.json") == expected_histories
    rows = []
    for case, row in zip([c for c in expected_cases if c["timed"]], timing):
        assert row == read(RUN / ("timing-" + case["id"] + ".json"))
        rows.append(audit_timing(row, case))
    assert len(list(RUN.glob("qualification-*.json"))) == 150 and len(list(RUN.glob("timing-*.json"))) == 120
    assert not list(RUN.glob("failure*"))
    long_rows = [r for r in rows if r["gate_group"] == "long_tail"]
    other_rows = [r for r in rows if r["gate_group"] == "other"]
    gm = statistics.geometric_mean(r["speedup"] for r in long_rows)
    misses = [{"id": r["id"], "speedup": r["speedup"]} for r in other_rows if r["speedup"] < .97]
    advance = gm >= 1.25 and not misses
    assert len(long_rows) == 48 and len(other_rows) == 72
    assert math.isclose(gm, recorded["gate"]["long_tail_geomean"], rel_tol=1e-12)
    assert recorded["gate"]["other_below_097"] == misses and recorded["gate"]["advance"] is advance
    assert recorded["advance"] is advance and recorded["status"] == ("PASS_MODULE_SCREEN" if advance else "REJECT_PERFORMANCE_GATE")
    assert recorded["qualification_passed"] and recorded["qualification_cases"] == 150 and recorded["timed_cases"] == 120 and recorded["untimed_none_cases"] == 30
    assert recorded["all_timed_histories_unchanged"] and recorded["source_and_frozen_inputs_unchanged"] and not recorded["cold_compilation_claim"]
    assert plan["cold_compilation_claim"] is False and plan["no_model_or_full_draft_claim"] is True

    # Corruption probes target the independent byte replay and statistic checks.
    corruption = []
    for kind in ("output_hash", "input_hash", "concat_rows", "median", "abba_order", "sample_count"):
        try:
            if kind in ("output_hash", "input_hash", "concat_rows"):
                bad = copy.deepcopy(checks[0])
                if kind == "output_hash": bad["output"]["candidate_sha256"] = "0" * 64
                if kind == "input_hash": bad["inputs_after"][0]["sha256"] = "0" * 64
                if kind == "concat_rows": bad["candidate_python_concat_calls"] = [{"axis": 0, "rows": [1]}]
                audit_check(bad, expected_cases[0], expected_histories["N19-single"], expected_outputs[expected_cases[0]["id"]])
            else:
                bad = copy.deepcopy(timing[0])
                if kind == "median": bad["median_build_eval_and_release_ms"]["reference"] *= 2
                if kind == "abba_order": bad["rounds"][0]["arm"] = "candidate"
                if kind == "sample_count": bad["rounds"].pop()
                audit_timing(bad, expected_cases[0])
        except AssertionError:
            corruption.append({"mutation": kind, "rejected": True})
        else:
            raise AssertionError("audit accepted corruption: " + kind)

    owner = BASE / "history-tail-v1-owner"
    child, release, complete, attempt = [read(owner / name) for name in ("child.json", "release.json", "complete.json", "attempt.json")]
    assert child["pid"] == release["pid"] and release["returncode"] == 0 and release["waited_reaped"] and release["strict_absence"]
    assert complete["status"] == "PASS" and attempt["owner_pid"] == complete["owner_pid"]
    resources = [read(p) for p in sorted(owner.glob("resource-*.json"))]
    # Individual snapshots precede the guard's in-memory adjacent-delta fields.
    # Independently reconstruct those derived fields before checking the total.
    aggregate_resources = copy.deepcopy(resources)
    for previous, current in zip(aggregate_resources, aggregate_resources[1:]):
        current["swap_used_mb_delta"] = current["swap_used_mb"] - previous["swap_used_mb"]
        current["vm_counter_delta"] = {key: value - previous["vm_counters"][key]
                                       for key, value in current["vm_counters"].items()
                                       if key in previous["vm_counters"]}
    assert len(resources) >= 3 and aggregate_resources == read(owner / "resources.json")
    for resource in resources:
        assert resource["pressure_level_value"] == 1 and resource["processinfo_thermal_metadata"]["code"] == 0
        assert resource["power_metadata"]["source"] == "AC Power" and resource["no_active_foreign_native_processes"]
        paused = [p for p in resource["native_inference_processes"] if p["pid"] == 74384]
        assert len(paused) == 1 and paused[0]["authorized_suspended"] and "T" in paused[0]["state"]
        if resource["owned_model_pid"] is not None:
            assert resource["owned_model_pid"] == child["pid"]
            assert resource["exact_owned_pid_metadata"]["pid"] == child["pid"]
            assert resource["exact_owned_pid_metadata"]["ppid"] == complete["owner_pid"]
    assert len({r["swap_used_mb"] for r in resources}) == len({r["vm_counters"]["Swapouts"] for r in resources}) == 1
    lease_dir = BASE / "history-tail-v1-service-lease"
    lease, resume, job = [read(lease_dir / name) for name in ("lease.json", "resume.json", "job-0.json")]
    jobs_path = BASE / "history-tail-v1-jobs.json"
    jobs = read(jobs_path)
    assert lease["closed"] and lease["service_pid"] == 74384 and lease["jobs"] == jobs and len(jobs) == 1
    assert resume["identity_match"] and resume["state_before"] == "T" and resume["state_after"] == "S"
    assert job["command"] == jobs[0] and job["pid"] == complete["owner_pid"] and job["returncode"] == 0 and job["waited"]
    for folder in (RUN, owner, lease_dir):
        for path in sorted(folder.rglob("*")):
            if path.is_file(): collect(path)
    collect(jobs_path)
    collect(Path(__file__))
    review = ROOT / "bench/reviews/e019-source-review.md"
    if review.exists(): collect(review)
    for name in ("guarded_v2.py", "resources_v2.py", "resources.py", "resource_base.py", "suspended_service.py"):
        relative = "bench/m5_ultra_256gb/" + name
        files[relative] = subprocess.check_output(["git", "-C", str(ROOT), "show", COMMIT + ":" + relative])
    cpu = subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "unittest", "discover", "-s",
                          str(ROOT / "bench/m5_ultra_256gb/history_tail"), "-p", "test*.py", "-v"], capture_output=True, text=True)
    assert cpu.returncode == 0 and "Ran 17 tests" in cpu.stderr
    cpu_receipt = {"command": cpu.args, "returncode": cpu.returncode, "stdout": cpu.stdout, "stderr": cpu.stderr,
                   "runtime_scope": "stdlib tests and fake arrays only; real MLX/NumPy/Torch imports blocked by tests"}
    files["audit/cpu-tests.json"] = (json.dumps(cpu_receipt, indent=2) + "\n").encode()
    summary = {"status": "PASS_INDEPENDENT_AUDIT", "experiment_decision": recorded["status"], "advance": advance,
               "preregistration_commit": COMMIT, "qualification_cases": 150, "timed_cases": 120, "untimed_none_cases": 30,
               "formal_samples": 1440, "warm_samples": 480, "first_post_qualification_samples": 240,
               "long_tail_geomean": gm, "long_tail_points": 48, "other_points": 72, "other_below_097": misses,
               "all_points": rows, "input_tensors": tensor_audit, "source_files_checked": 7,
               "independent_frozen_input_logical_byte_replay": True, "history_layouts_reconstructed": len(expected_histories),
               "corruption_probes": corruption, "release": release, "owner_pid": complete["owner_pid"],
               "service_restore": resume, "resource_samples": len(resources), "swap_used_mb": resources[0]["swap_used_mb"],
               "swapouts": resources[0]["vm_counters"]["Swapouts"], "runtime_memory": {key: recorded[key] for key in
                   ("input_preparation_peak_bytes", "micro_peak_bytes", "before_timing_active_bytes", "pool_limit_bytes")},
               "input_limitation": "pattern repeats every 257 elements; shifts of 257 entire rows may alias",
               "scope": "pure extraction micro; no model, full-draft, HTTP, or Pareto-frontier performance claim"}
    files["audit/summary.json"] = (json.dumps(summary, indent=2, ensure_ascii=False) + "\n").encode()
    OUT.mkdir(parents=True)
    archive = OUT / "evidence.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, data in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    manifest = {name: {"bytes": len(data), "sha256": sha(data)} for name, data in sorted(files.items())}
    with tarfile.open(archive) as tar:
        assert len(tar.getnames()) == len(manifest) and set(tar.getnames()) == set(manifest)
        for name, expected in manifest.items():
            data = tar.extractfile(name).read()
            assert len(data) == expected["bytes"] and sha(data) == expected["sha256"]
    archive_sha = sha(archive.read_bytes())
    write(OUT / "manifest.json", {"archive_sha256": archive_sha, "files": manifest})
    write(OUT / "tar.verify.json", {"status": "PASS", "members": len(manifest), "archive_sha256": archive_sha,
                                     "all_member_hashes_and_sizes_verified": True})
    write(OUT / "summary.json", summary)
    write(OUT / "cpu-checks.json", cpu_receipt)
    write(OUT / "audit.json", {"status": "PASS", "independent_reconstructed_inputs_and_outputs": True,
                                "timing_gate_recomputed": True, "corruption_probes": corruption,
                                "source_commit": COMMIT, "source_files_checked": 7})
    bad_table = "\n".join(f"| {row['id']} | {row['speedup']:.6f}× |" for row in misses)
    (OUT / "README.md").write_text(f'''# E019：hidden history 尾部提取

独立审计 PASS；实验结论 **{recorded['status']}**，不采用当前覆盖全部点的候选。
150 个资格点全部通过；120 个非空计时点中，48 个 N512/N2048 tail 点几何平均 {gm:.6f}×，但其余 72 点有 **{len(misses)} 点低于 0.97×**。完整预注册门失败，未剔除 single/reset 或坏点，未重跑。

CPU 审计直接解析四份原始 BF16 safetensors，逐字节检查固定输入 pattern，按独立的 offset、行 stride、列 stride 和压缩布局重建全部 history 与输出 SHA，再核对 150 份独立记录、总表和计时后输入记录。它不导入 MLX，也不复跑 GPU。输入 pattern 有 257 元素周期，不能排除任意长距离错行；仍需真实 features/full-draft 资格。

全部 1440 formal、480 warm、240 first-post-qualification 样本保留。逐点重算 build+eval+release、ABBA 顺序、中位数、speedup 与完整 gate。首个计时调用发生在资格后，不称冷编译。6 项破坏性记录检查均被独立审计拒绝，17 项 CPU 边界/CLI 检查通过。

所有 {len(resources)} 个资源样本为 AC、pressure=1、thermal=0，swap 268.81 MiB 与 Swapouts 保持不变。child PID {child['pid']} exit 0、wait/reap/strict-absence 记录通过；owner PID {complete['owner_pid']} 完成，授权 oMLX 服务 PID74384 恢复 T→S。进程审计仅覆盖已知命令/精确 owner PID，未声称检测所有 GUI GPU 使用。

archive 保存全部四份 tensor 输入（原样约 265 MiB，未省略）、全部 raw JSON/timing、owner/lease/jobs、源码及审查。7 份测量 pin 源码均从 **{COMMIT}** 读取并对照 plan SHA，包含当时 tutorial；后续文档追加不会改变归档 pin。{len(manifest)} 个 tar 成员逐 size/SHA 复核通过，完整 manifest 随附。

这是单进程纯提取 micro。没有目标模型、完整 draft、token/accept 或 HTTP 结果，**不进入主 throughput–latency Pareto 图**。原 history/64 块压缩/核心路径未改。

## 所有未过 0.97× 的 other 点

| Point | Reference / candidate |
|---|---:|
{bad_table}
''')
    print(json.dumps({"status": summary["status"], "experiment_decision": recorded["status"], "members": len(manifest),
                      "archive_bytes": archive.stat().st_size, "archive_sha256": archive_sha, "regressions": len(misses)}))


if __name__ == "__main__":
    main()
