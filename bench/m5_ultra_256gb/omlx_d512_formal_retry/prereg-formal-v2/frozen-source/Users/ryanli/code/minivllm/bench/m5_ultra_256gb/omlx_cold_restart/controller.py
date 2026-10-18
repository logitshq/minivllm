"""Explicitly approved, journaled cold restart of one fixed oMLX launchd job.

Import/--help are stdlib-only. Never starts an experiment. See README before use.
"""
import argparse
import contextlib
import fcntl
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import time
import uuid

PLAN_SHA = "e96a4a4d5be6f6a69d597387bbc566eadedae81065bfa879445352d986515cb8"
EVIDENCE_SHA = "9fcb46cb0d1b1824355b62bb0601c5a9722046f4118adcb828659b928ed0e5d6"
CONTRACT = "cold-restart-persisted-visible-v1"
ROOT = Path(__file__).resolve().parents[3]
PLAN = ROOT / "bench/reviews/omlx-cold-restart-plan-v1/plan.json"
EVIDENCE = PLAN.with_name("evidence.json")
JOB = "gui/501/local.omlx.server"
PLIST = "/Users/ryanli/Library/LaunchAgents/local.omlx.server.plist"
TARGET = "Qwen3.8-Flash-Next-oQ6e-mtp"
MODEL = "/Users/ryanli/code/models/mlx/" + TARGET
SETTINGS = Path("/Users/ryanli/.omlx/model_settings.json")
GLOBAL_SETTINGS = Path("/Users/ryanli/.omlx/settings.json")
LOG = Path("/Users/ryanli/.omlx/logs/service.stderr.log")
GIB = 1 << 30


class Refuse(RuntimeError):
    pass


class NotReady(Refuse):
    pass


def need(test, message):
    if not test:
        raise Refuse(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def file_stat(path):
    s = Path(path).stat()
    return {"dev": s.st_dev, "ino": s.st_ino, "size": s.st_size, "mtime_ns": s.st_mtime_ns}


def pins():
    need(sha(PLAN) == PLAN_SHA and sha(EVIDENCE) == EVIDENCE_SHA, "fixed proposal/evidence changed")
    p, e = read_json(PLAN), read_json(EVIDENCE)
    need(p["original"]["job"] == JOB and p["original"]["plist"] == PLIST, "wrong fixed target")
    need(sha(PLIST) == p["original"]["plist_sha256"], "original plist changed")
    for x in e["small_source_pins"]:
        need(sha(x["path"]) == x["sha256"], "reviewed source changed: " + x["path"])
    need(sha(SETTINGS) == "b474a171fe3022027842af02ee021206289a7cd9306720841da375029cf42acb", "model settings changed")
    return p


def atomic_write(path, obj):
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix=".journal-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        dfd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class Journal:
    def __init__(self, directory, create=False):
        self.directory = Path(directory).resolve()
        if create:
            self.directory.mkdir(mode=0o700, parents=False, exist_ok=False)
        need(self.directory.is_dir(), "journal directory missing")
        self.path = self.directory / "journal.json"
        self.lock = (self.directory / ".lock").open("a+")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lock.close()
            raise Refuse("another controller holds this journal") from exc
        self.data = {} if create else read_json(self.path)

    def close(self):
        self.lock.close()

    def save(self, event, **fields):
        self.data.update(fields)
        self.data.setdefault("events", []).append({"event": event, "time": time.time(), "monotonic": time.monotonic()})
        atomic_write(self.path, self.data)


class System:
    """Only this backend performs I/O; CPU tests substitute a fake."""
    def run(self, argv, timeout=10):
        try:
            r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
            return r.returncode, r.stdout
        except subprocess.TimeoutExpired:
            return None, ""  # outcome unknown; no stderr/argv environment retained

    def process(self, pid):
        rc, out = self.run(["/bin/ps", "-p", str(pid), "-o", "pid=,ppid=,lstart=,comm="])
        if rc == 1 and not out.strip():
            return None
        need(rc == 0, "cannot establish process identity")
        m = re.fullmatch(r"\s*(\d+)\s+(\d+)\s+([A-Za-z]{3} [A-Za-z]{3}\s+\d+ \d\d:\d\d:\d\d \d{4})\s+(.+?)\s*", out)
        need(m is not None, "unexpected ps identity format")
        return {"pid": int(m[1]), "ppid": int(m[2]), "start": " ".join(m[3].split()), "comm": m[4]}

    def job(self):
        rc, out = self.run(["/bin/launchctl", "print", JOB])
        if rc == 113:
            return None
        need(rc == 0, "launchd absence/running state unresolved")
        fields = dict(re.findall(r"^\t(pid|state|path|program|working directory) = (.*)$", out, re.M))
        need(fields.get("path") == PLIST, "unexpected launchd definition")
        need(fields.get("pid", "").isdigit(), "launchd has no stable PID yet")
        return {"pid": int(fields["pid"]), "state": fields.get("state"), "path": fields["path"]}

    def listener(self):
        rc, out = self.run(["/usr/sbin/lsof", "-nP", "-a", "-iTCP:8001", "-sTCP:LISTEN", "-Fpn"])
        if rc == 1 and not out.strip():
            return None
        need(rc == 0, "listener status unresolved")
        pids = set(re.findall(r"^p(\d+)$", out, re.M))
        names = set(re.findall(r"^n(.+)$", out, re.M))
        need(len(pids) == 1 and names == {"127.0.0.1:8001"}, "unexpected port owner/address")
        return int(next(iter(pids)))

    def http(self, method, path, timeout=2):
        need(method in ("GET", "POST"), "unsupported HTTP method")
        need(method != "POST" or path == "/v1/models/" + TARGET + "/load", "HTTP mutation outside scope")
        c = http.client.HTTPConnection("127.0.0.1", 8001, timeout=timeout)
        try:
            c.request(method, path, body=None, headers={"Connection": "close"})
            r = c.getresponse()
            need(r.status == 200, "management HTTP status " + str(r.status))
            raw = r.read(2 << 20)
            need(len(raw) < 2 << 20, "management response exceeds bound")
            return json.loads(raw)
        finally:
            c.close()

    def mutate(self, which):
        argv = {"stop": ["/bin/launchctl", "bootout", JOB],
                "bootstrap": ["/bin/launchctl", "bootstrap", "gui/501", PLIST]}[which]
        rc, _ = self.run(argv, timeout=30)
        return rc

    def sleep(self, seconds):
        time.sleep(seconds)

    def resources(self):
        with tempfile.TemporaryDirectory(prefix="omlx-reclaim-") as td:
            dest = Path(td) / "footprint.json"
            rc, _ = self.run(["/usr/bin/footprint", "-p", str(os.getpid()), "--noCategories", "--sysFootprint", "-f", "bytes", "-j", str(dest)], timeout=20)
            need(rc == 0 and dest.exists(), "physical footprint unavailable")
            f = read_json(dest)
        need(not f.get("errors") and not f.get("warnings"), "physical footprint probe had errors/warnings")
        rc, pressure = self.run(["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"])
        need(rc == 0, "pressure unavailable")
        rc, swap = self.run(["/usr/sbin/sysctl", "vm.swapusage"])
        need(rc == 0, "swap unavailable")
        m = re.search(r"used = ([\d.]+)M", swap)
        need(m is not None, "swap format unknown")
        rc, memory = self.run(["/usr/sbin/sysctl", "-n", "hw.memsize"])
        need(rc == 0, "physical RAM unavailable")
        return {"time": time.time(), "system_footprint": f["auxiliary"]["sys_footprint"],
                "pressure": int(pressure), "swap_bytes": round(float(m[1]) * (1 << 20)), "physical_ram": int(memory)}

    def log_identity(self):
        return file_stat(LOG)

    def new_log_lines(self, before):
        now = file_stat(LOG)
        need((now["dev"], now["ino"]) == (before["dev"], before["ino"]), "log rotated: require independent recovery review")
        need(before["size"] <= now["size"] <= before["size"] + (4 << 20), "new log window truncated/too large")
        with LOG.open("rb") as f:
            f.seek(before["size"])
            return f.read().decode("utf-8", errors="strict").splitlines()


def original_identity(s):
    p = s.process(74384)
    need(p == {"pid": 74384, "ppid": 1, "start": "Fri Oct 16 08:01:58 2026", "comm": "omlx-server"}, "original service identity changed")
    j = s.job()
    need(j == {"pid": 74384, "state": "running", "path": PLIST} and s.listener() == 74384, "original supervisor/listener mismatch")
    return p


def service_status(s, expected_set=None):
    a, b = s.http("GET", "/api/status"), s.http("GET", "/v1/models/status")
    need(a["version"] == "0.7.0" and a["default_model"] == TARGET, "version/default mismatch")
    need(all(type(a[k]) is int and a[k] >= 0 for k in ("active_requests", "waiting_requests", "models_loading", "models_loaded")), "invalid management counters")
    need(a["active_requests"] == 0 and a["waiting_requests"] == 0, "service has active/waiting requests")
    ids = a["loaded_models"]
    need(type(ids) is list and len(ids) == len(set(ids)) and set(ids) <= {TARGET}, "unexpected physical loaded set")
    need(a["models_loaded"] == len(ids), "loaded count inconsistent")
    if expected_set is not None:
        need(set(ids) == set(expected_set) and a["models_loading"] == 0, "residency/loading mismatch")
    rows = [m for m in b["models"] if m["id"] == TARGET]
    need(len(rows) == 1, "physical target missing/ambiguous")
    t = rows[0]
    need(t["model_path"] == MODEL and t["pinned"] is False and t["engine_type"] == "vlm" and t["config_model_type"] == "qwen4_exp", "target variant metadata changed")
    need(not any(m.get("pinned") is True for m in b["models"]), "startup may preload an unreviewed pinned model")
    need(type(t["loaded"]) is bool and type(t["is_loading"]) is bool and t["loaded"] == (TARGET in ids), "status endpoints disagree")
    return {"version": a["version"], "default_model": a["default_model"], "loaded": ids,
            "loading": a["models_loading"], "target_loading": t["is_loading"],
            "path": t["model_path"], "pinned": t["pinned"], "engine_type": t["engine_type"], "config_model_type": t["config_model_type"]}


def absent_service(s):
    need(s.process(74384) is None and s.job() is None and s.listener() is None, "service not fully removed")


def authorize(j, approved_contract, plan_sha):
    need(approved_contract == CONTRACT and plan_sha == PLAN_SHA, "explicit approved contract and fixed plan SHA required")
    need(j.data["plan_sha256"] == PLAN_SHA and j.data["controller_sha256"] == sha(__file__), "journal plan/controller identity mismatch")
    need(j.data["global_settings_stat"] == file_stat(GLOBAL_SETTINGS), "global settings metadata changed; do not read credentials")
    for path, digest in j.data["model_metadata"].items():
        need(sha(path) == digest, "model metadata changed")
    pins()


def prepare(j, s, owner_directory):
    need(not j.data, "journal already prepared")
    pins()
    need(os.getuid() == 501, "wrong launchd user")
    owner = Path(owner_directory).resolve()
    need(not owner.exists() and owner.parent.is_dir(), "predeclare a new owner directory before any experiment")
    identity, status = original_identity(s), service_status(s, [TARGET])
    metadata = {str(Path(MODEL) / name): sha(Path(MODEL) / name)
                for name in ("config.json", "model.safetensors.index.json") if (Path(MODEL) / name).exists()}
    need(str(Path(MODEL) / "config.json") in metadata, "model config missing")
    j.data = {"schema": "omlx-cold-restart-journal-v1", "id": str(uuid.uuid4()), "state": "PREPARED",
              "plan_sha256": PLAN_SHA, "controller_sha256": sha(__file__), "original": identity,
              "original_status": status, "global_settings_stat": file_stat(GLOBAL_SETTINGS),
              "model_metadata": metadata,
              "log_start": s.log_identity(), "owner_directory": str(owner), "events": []}
    j.save("prepared_read_only")


def stop(j, s, approved_contract, plan_sha):
    authorize(j, approved_contract, plan_sha)
    if j.data["state"] == "STOPPED":
        absent_service(s)
        return
    if j.data["state"] == "STOP_INTENT":
        absent_service(s)  # unknown outcome: reconcile, never blindly repeat bootout
        j.save("stop_reconciled", state="STOPPED", removed_at=time.time())
        return
    need(j.data["state"] == "PREPARED", "stop phase is not admissible")
    need(not Path(j.data["owner_directory"]).exists(), "experiment directory exists before service stop")
    original_identity(s)
    service_status(s, [TARGET])
    j.save("bootout_intent", state="STOP_INTENT", approved_contract=approved_contract)
    rc = s.mutate("stop")
    j.save("bootout_returned", bootout_returncode=rc)
    for _ in range(15):
        try:
            absent_service(s)
            j.save("service_absent", state="STOPPED", removed_at=time.time())
            return
        except Refuse:
            s.sleep(2)
    raise Refuse("bootout outcome unresolved; no experiment permitted")


def reaped(j, s, no_experiment):
    owner = Path(j.data["owner_directory"])
    if no_experiment:
        need(not owner.exists(), "cannot claim no experiment after owner directory appeared")
        return {"no_experiment": True}
    if not owner.exists():
        raise NotReady("owner has not run; explicit --no-experiment required to skip it")
    files = [owner / n for n in ("attempt.json", "child.json", "release.json")]
    if not all(p.exists() for p in files):
        raise NotReady("closed owner/reap receipts missing")
    need(all(p.stat().st_mtime >= j.data["removed_at"] for p in files), "stale owner receipts")
    attempt, child, release = map(read_json, files)
    pid, opid = child["pid"], attempt["owner_pid"]
    need(type(pid) is int and type(opid) is int and pid > 1 and opid > 1 and pid != opid, "invalid experiment identities")
    need(type(release["pid"]) is int and release["pid"] == pid and type(release["returncode"]) is int
         and release["waited_reaped"] is True and release["strict_absence"] is True, "child not proven reaped")
    if s.process(pid) is not None or s.process(opid) is not None:
        raise NotReady("experiment child or owner still exists (including PID reuse)")
    return {"no_experiment": False, "child_pid": pid, "owner_pid": opid,
            "receipts": [{"path": str(p), "sha256": sha(p)} for p in files]}


def resource_gate(j, s, loaded=False):
    rows = [s.resources()]
    s.sleep(5)
    rows.append(s.resources())
    j.save("resource_samples", latest_resources=rows)
    limit = rows[0]["physical_ram"] - 16 * GIB if loaded else 40 * GIB
    if not all(all(type(r[k]) is int for k in ("system_footprint", "physical_ram", "pressure", "swap_bytes"))
               and 0 < r["system_footprint"] <= limit and r["swap_bytes"] >= 0
               and r["physical_ram"] == 256 * GIB and r["pressure"] == 1 for r in rows):
        raise NotReady("physical admission/pressure not ready; no load")
    need(4 <= rows[1]["time"] - rows[0]["time"] <= 60, "resource sample interval invalid")
    if rows[1]["swap_bytes"] > rows[0]["swap_bytes"] or abs(rows[1]["system_footprint"] - rows[0]["system_footprint"]) > GIB:
        raise NotReady("physical footprint/swap not stable")


def replacement(s):
    job = s.job()
    need(job is not None and job["state"] == "running" and job["pid"] != 74384, "replacement job not ready")
    p = s.process(job["pid"])
    need(p is not None and p["ppid"] == 1 and p["comm"] == "omlx-server" and s.listener() == p["pid"], "replacement PID/listener mismatch")
    return p


def verify_logs(lines):
    starts = [i for i, x in enumerate(lines) if " - INFO - Loading model: " + TARGET in x]
    need(len(starts) == 1, "new target load transaction missing/ambiguous")
    lines = lines[starts[0]:]
    required = ["Qwen4-Exp PLE mode for " + MODEL + ": resident",
                "Qwen4-Exp Lightning MTP enabled for " + MODEL,
                "Fused 1 resident Qwen4-Exp PLE table into one packed device-side embedding",
                "VLMBatchedEngine loaded: " + MODEL, "Loaded model: " + TARGET + " ("]
    for marker in required:
        need(any(" - INFO - " in x and marker in x for x in lines), "new resident/MTP completion evidence missing")
    need(not any("PLE" in x and ("forced to SSD" in x or ": mmap" in x or ": ssd" in x) for x in lines), "SSD PLE observed; recovery incomplete")
    return [marker for marker in required]  # no arbitrary log text or requests retained


def recover(j, s, approved_contract, plan_sha, no_experiment=False):
    authorize(j, approved_contract, plan_sha)
    if j.data["state"] == "STOP_INTENT":
        absent_service(s)
        j.save("stop_reconciled_before_recovery", state="STOPPED", removed_at=time.time())
    need(j.data["state"] in ("STOPPED", "BOOTSTRAP_INTENT", "JOB_RESTORED", "LOAD_INTENT", "RESTORED"), "recovery phase inadmissible")
    reap = reaped(j, s, no_experiment)
    j.save("experiment_reaped_checked", reap=reap)
    if j.data["state"] == "STOPPED":
        absent_service(s)
        resource_gate(j, s)
        # Recheck ownership immediately before admission; still not a global GPU lock.
        reaped(j, s, no_experiment)
        absent_service(s)
        j.save("bootstrap_intent", state="BOOTSTRAP_INTENT", recovery_log_start=s.log_identity())
        rc = s.mutate("bootstrap")
        j.save("bootstrap_returned", bootstrap_returncode=rc)
    if j.data["state"] == "BOOTSTRAP_INTENT":
        for _ in range(30):
            try:
                identity = replacement(s)
                service_status(s)
                j.save("job_reconciled", state="JOB_RESTORED", replacement=identity)
                break
            except (Refuse, OSError, http.client.HTTPException):
                s.sleep(2)
        else:
            raise Refuse("bootstrap unknown/unready; no repeated bootstrap")
    identity = replacement(s)
    need(identity == j.data["replacement"], "replacement process changed during recovery")
    status = service_status(s)
    if j.data["state"] == "JOB_RESTORED":
        need(not status["loaded"] and status["loading"] == 0 and status["target_loading"] is False, "unexpected auto-load; inspect instead of issuing load")
        resource_gate(j, s)
        reaped(j, s, no_experiment)
        need(replacement(s) == identity, "replacement changed before load")
        j.save("load_intent", state="LOAD_INTENT")
        try:
            s.http("POST", "/v1/models/" + TARGET + "/load", timeout=600)
            j.save("load_call_returned")
        except (Refuse, OSError, http.client.HTTPException, TimeoutError) as exc:
            j.save("load_outcome_unknown", load_error_type=type(exc).__name__)
    for _ in range(120):
        need(replacement(s) == identity, "replacement changed while loading")
        status = service_status(s)
        if status["loaded"] == [TARGET] and status["loading"] == 0 and not status["target_loading"]:
            break
        need(status["loading"] > 0 or status["target_loading"], "load intent has no active/completed loader; manual reconciliation required, no duplicate POST")
        s.sleep(5)
    else:
        raise Refuse("load remains incomplete at bounded deadline")
    markers = verify_logs(s.new_log_lines(j.data["recovery_log_start"]))
    resource_gate(j, s, loaded=True)
    authorize(j, approved_contract, plan_sha)
    need(replacement(s) == identity, "replacement changed at completion")
    final = service_status(s, [TARGET])
    j.save("restored_persisted_visible_variant", state="RESTORED", final_status=final, log_markers=markers,
           exact_runtime_signature_restored=False, warm_kv_restored=False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("prepare", "stop", "recover"))
    p.add_argument("--journal", type=Path, required=True)
    p.add_argument("--plan-sha256", required=True)
    p.add_argument("--approved-contract")
    p.add_argument("--owner-directory", type=Path)
    p.add_argument("--no-experiment", action="store_true")
    p.add_argument("--watchdog-seconds", type=int, default=0)
    a = p.parse_args(argv)
    need(a.plan_sha256 == PLAN_SHA, "fixed plan SHA required")
    need(a.phase == "prepare" or a.approved_contract == CONTRACT, "explicit approval contract absent")
    need(0 <= a.watchdog_seconds <= 3600 and (not a.watchdog_seconds or a.phase == "recover"), "invalid watchdog bound")
    need(a.phase != "prepare" or a.owner_directory is not None, "prepare requires predeclared owner directory")
    need(a.phase == "recover" or not a.no_experiment, "no-experiment applies only to recovery")
    need(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), "existing alarm cannot be replaced")
    def timeout(*unused):
        raise TimeoutError("absolute controller deadline; inspect durable intent before recovery")
    with contextlib.closing(Journal(a.journal, create=a.phase == "prepare")) as j:
        s = System()
        previous_handler = signal.signal(signal.SIGALRM, timeout)
        signal.setitimer(signal.ITIMER_REAL, (1800 + a.watchdog_seconds) if a.phase == "recover" else 90)
        try:
            if a.phase == "prepare":
                prepare(j, s, a.owner_directory)
            elif a.phase == "stop":
                stop(j, s, a.approved_contract, a.plan_sha256)
            else:
                until = time.monotonic() + a.watchdog_seconds
                while True:
                    try:
                        recover(j, s, a.approved_contract, a.plan_sha256, a.no_experiment)
                        break
                    except NotReady:
                        if time.monotonic() >= until:
                            raise
                        j.save("watchdog_waiting_no_mutation")
                        s.sleep(5)
        except BaseException as exc:
            if j.data:
                j.save("controller_incomplete", last_error_type=type(exc).__name__, recovery_complete=j.data.get("state") == "RESTORED")
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
    print("controller phase complete; inspect journal for exact state")


if __name__ == "__main__":
    main()
