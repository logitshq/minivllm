"""No sockets, process commands, real service reads, MLX imports or mutations."""
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("cold_controller", Path(__file__).with_name("controller.py"))
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


class Fake:
    def __init__(self):
        self.mode = "old"
        self.loaded = True
        self.loading = False
        self.active = 0
        self.other_model = False
        self.extra_pinned = False
        self.mutations = []
        self.posts = 0
        self.pressure = 1
        self.footprint = 30 * c.GIB
        self.swap = 0
        self.clock = time.time()
        self.exists = {}
        self.boot_failure = False
        self.unknown_post = False
        self.bad_logs = False
        self.post_never_starts = False

    def process(self, pid):
        if pid in self.exists:
            return self.exists[pid]
        if (self.mode, pid) not in (("old", 74384), ("new", 90001)):
            return None
        return {"pid": pid, "ppid": 1, "start": "Fri Oct 16 08:01:58 2026" if pid == 74384 else "Sat Oct 17 17:00:00 2026", "comm": "omlx-server"}

    def job(self):
        if self.mode == "absent":
            return None
        return {"pid": 74384 if self.mode == "old" else 90001, "state": "running", "path": c.PLIST}

    def listener(self):
        return None if self.mode == "absent" else self.job()["pid"]

    def http(self, method, path, timeout=2):
        if method == "POST":
            self.posts += 1
            if self.post_never_starts:
                raise TimeoutError()
            self.loaded = True
            if self.unknown_post:
                raise TimeoutError()
            return {"status": "ok"}
        if path == "/api/status":
            ids = [c.TARGET] if self.loaded else []
            if self.other_model:
                ids.append("foreign")
            return {"version": "0.7.0", "default_model": c.TARGET, "active_requests": self.active,
                    "waiting_requests": 0, "models_loading": int(self.loading), "models_loaded": len(ids), "loaded_models": ids}
        return {"models": [{"id": c.TARGET, "model_path": c.MODEL, "pinned": False, "engine_type": "vlm",
                            "config_model_type": "qwen4_exp", "loaded": self.loaded, "is_loading": self.loading}]
                           + ([{"id": "other", "pinned": True}] if self.extra_pinned else [])}

    def mutate(self, which):
        self.mutations.append(which)
        if which == "stop":
            self.mode, self.loaded = "absent", False
        elif not self.boot_failure:
            self.mode = "new"
        return None if self.boot_failure else 0

    def sleep(self, seconds):
        self.clock += seconds

    def resources(self):
        return {"time": self.clock, "system_footprint": self.footprint, "pressure": self.pressure,
                "swap_bytes": self.swap, "physical_ram": 256 * c.GIB}

    def log_identity(self):
        return {"dev": 1, "ino": 2, "size": 123, "mtime_ns": 42}

    def new_log_lines(self, before):
        lines = ["Loading model: " + c.TARGET,
                 "Qwen4-Exp PLE mode for " + c.MODEL + (": mmap" if self.bad_logs else ": resident"),
                 "Qwen4-Exp Lightning MTP enabled for " + c.MODEL,
                 "Fused 1 resident Qwen4-Exp PLE table into one packed device-side embedding",
                 "VLMBatchedEngine loaded: " + c.MODEL,
                 "Loaded model: " + c.TARGET + " (actual: 140.48GB)"]
        return ["2026-10-17 - INFO - " + x for x in lines]


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.model = self.base / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text("{}")
        self.s = Fake()
        self.owner = self.base / "owner"
        self.j = c.Journal(self.base / "journal", create=True)
        self.addCleanup(self.j.close)
        for name, value in (("pins", lambda: {}), ("file_stat", lambda p: {"stable": True}),
                            ("MODEL", str(self.model))):
            p = patch.object(c, name, value)
            p.start()
            self.addCleanup(p.stop)
        p = patch.object(c.os, "getuid", return_value=501)
        p.start()
        self.addCleanup(p.stop)
        c.prepare(self.j, self.s, self.owner)

    def stop(self):
        c.stop(self.j, self.s, c.CONTRACT, c.PLAN_SHA)

    def recover(self, no=True):
        c.recover(self.j, self.s, c.CONTRACT, c.PLAN_SHA, no)

    def receipts(self):
        self.owner.mkdir()
        for name, body in {"attempt.json": {"owner_pid": 81000}, "child.json": {"pid": 81001},
                           "release.json": {"pid": 81001, "returncode": -15, "waited_reaped": True, "strict_absence": True}}.items():
            (self.owner / name).write_text(json.dumps(body))

    def test_prepare_does_not_mutate(self):
        self.assertEqual(self.j.data["state"], "PREPARED")
        self.assertEqual(self.s.mutations, [])
        self.assertEqual(self.s.posts, 0)

    def test_explicit_approval_and_plan_required(self):
        for contract, digest in ((None, c.PLAN_SHA), ("yes", c.PLAN_SHA), (c.CONTRACT, "0" * 64)):
            with self.assertRaises(c.Refuse):
                c.stop(self.j, self.s, contract, digest)
        self.assertEqual(self.s.mutations, [])

    def test_pid_listener_or_loaded_set_change_blocks_stop(self):
        self.s.exists[74384] = {"pid": 74384, "ppid": 1, "start": "new", "comm": "omlx-server"}
        with self.assertRaises(c.Refuse):
            self.stop()
        self.s.exists.clear()
        self.s.other_model = True
        with self.assertRaises(c.Refuse):
            self.stop()
        self.s.other_model = False
        with patch.object(self.s, "listener", return_value=123):
            with self.assertRaises(c.Refuse):
                self.stop()
        self.assertEqual(self.s.mutations, [])

    def test_active_or_other_pinned_model_rejected(self):
        self.s.active = 1
        with self.assertRaises(c.Refuse):
            self.stop()
        self.s.active = 0
        self.s.extra_pinned = True
        with self.assertRaises(c.Refuse):
            self.stop()

    def test_stop_journal_intent_precedes_command(self):
        original = self.s.mutate
        def mutate(which):
            self.assertEqual(c.read_json(self.j.path)["state"], "STOP_INTENT")
            return original(which)
        self.s.mutate = mutate
        self.stop()
        self.assertEqual(self.j.data["state"], "STOPPED")
        self.stop()
        self.assertEqual(self.s.mutations, ["stop"])

    def test_unknown_stop_does_not_repeat(self):
        self.j.save("simulate_crash_before_or_after_command", state="STOP_INTENT")
        with self.assertRaises(c.Refuse):
            self.stop()
        self.assertEqual(self.s.mutations, [])
        self.s.mode = "absent"
        self.stop()
        self.assertEqual(self.j.data["state"], "STOPPED")

    def test_cold_recovery_and_idempotent_verify(self):
        self.stop()
        self.recover()
        self.assertEqual(self.j.data["state"], "RESTORED")
        self.assertFalse(self.j.data["exact_runtime_signature_restored"])
        self.assertEqual(self.s.mutations, ["stop", "bootstrap"])
        self.assertEqual(self.s.posts, 1)
        self.recover()
        self.assertEqual(self.s.posts, 1)

    def test_live_child_or_owner_forbids_recovery(self):
        self.stop()
        self.receipts()
        for pid in (81001, 81000):
            self.s.exists[pid] = {"pid": pid}
            with self.assertRaises(c.NotReady):
                self.recover(no=False)
            del self.s.exists[pid]
        self.assertEqual(self.s.mutations, ["stop"])
        self.recover(no=False)
        self.assertEqual(self.j.data["state"], "RESTORED")

    def test_missing_reap_receipt_and_no_experiment_lie_rejected(self):
        self.stop()
        self.owner.mkdir()
        with self.assertRaises(c.NotReady):
            self.recover(no=False)
        with self.assertRaises(c.Refuse):
            self.recover(no=True)
        self.assertEqual(self.s.mutations, ["stop"])

    def test_pressure_and_footprint_block_bootstrap(self):
        self.stop()
        self.s.pressure = 2
        with self.assertRaises(c.NotReady):
            self.recover()
        self.s.pressure = 1
        self.s.footprint = 41 * c.GIB
        with self.assertRaises(c.NotReady):
            self.recover()
        self.assertEqual(self.s.mutations, ["stop"])

    def test_swap_growth_refuses(self):
        self.stop()
        original = self.s.sleep
        def sleep(n):
            original(n)
            self.s.swap += 1
        self.s.sleep = sleep
        with self.assertRaises(c.NotReady):
            self.recover()
        self.assertEqual(self.s.mutations, ["stop"])

    def test_bootstrap_unknown_outcome_never_repeats(self):
        self.stop()
        self.s.boot_failure = True
        with self.assertRaises(c.Refuse):
            self.recover()
        with self.assertRaises(c.Refuse):
            self.recover()
        self.assertEqual(self.s.mutations, ["stop", "bootstrap"])

    def test_unknown_bootstrap_with_live_replacement_reconciles(self):
        self.stop()
        self.j.save("simulated_crash", state="BOOTSTRAP_INTENT", recovery_log_start=self.s.log_identity())
        self.s.mode = "new"
        self.recover()
        self.assertEqual(self.s.mutations, ["stop"])
        self.assertEqual(self.s.posts, 1)

    def test_load_timeout_completed_reconciles_without_repeat(self):
        self.stop()
        self.s.unknown_post = True
        self.recover()
        self.assertEqual(self.j.data["state"], "RESTORED")
        self.assertEqual(self.s.posts, 1)

    def test_unknown_load_without_evidence_never_retries(self):
        self.stop()
        self.s.post_never_starts = True
        with self.assertRaises(c.Refuse):
            self.recover()
        with self.assertRaises(c.Refuse):
            self.recover()
        self.assertEqual(self.s.posts, 1)

    def test_ssd_or_stale_log_cannot_pass_restoration(self):
        self.stop()
        self.s.bad_logs = True
        with self.assertRaises(c.Refuse):
            self.recover()
        self.assertNotEqual(self.j.data["state"], "RESTORED")
        with self.assertRaises(c.Refuse):
            c.verify_logs([])
        with self.assertRaises(c.Refuse):
            c.verify_logs(self.s.new_log_lines({}) * 2)

    def test_exclusive_journal_lock(self):
        with self.assertRaises(c.Refuse):
            c.Journal(self.j.directory)

    def test_journal_controller_pin_and_settings_stat(self):
        self.j.data["controller_sha256"] = "0" * 64
        with self.assertRaises(c.Refuse):
            self.stop()
        self.assertEqual(self.s.mutations, [])

    def test_bool_pressure_not_integer_one(self):
        self.stop()
        self.s.pressure = True
        with self.assertRaises(c.NotReady):
            self.recover()

    def test_status_boolean_counter_is_rejected(self):
        original = self.s.http
        def request(method, path, timeout=2):
            out = original(method, path, timeout)
            if path == "/api/status":
                out["models_loaded"] = True
            return out
        self.s.http = request
        with self.assertRaises(c.Refuse):
            self.stop()

    def test_system_launchd_and_listener_parsers(self):
        s = c.System()
        with patch.object(s, "run", return_value=(0, "gui/501/local.omlx.server = {\n\tpath = " + c.PLIST + "\n\tstate = running\n\tpid = 74384\n\tnested = {\n\t\tpid = 999\n\t}\n}")):
            self.assertEqual(s.job()["pid"], 74384)
        with patch.object(s, "run", return_value=(113, "")):
            self.assertIsNone(s.job())
        with patch.object(s, "run", return_value=(1, "")):
            with self.assertRaises(c.Refuse):
                s.job()
        with patch.object(s, "run", return_value=(0, "p74384\nn127.0.0.1:8001\n")):
            self.assertEqual(s.listener(), 74384)
        with patch.object(s, "run", return_value=(0, "p74384\nn*:8001\n")):
            with self.assertRaises(c.Refuse):
                s.listener()

    def test_system_command_scope_and_process_parse(self):
        s = c.System()
        with patch.object(s, "run", return_value=(0, "")) as run:
            s.mutate("stop")
            run.assert_called_once_with(["/bin/launchctl", "bootout", c.JOB], timeout=30)
        with patch.object(s, "run", return_value=(0, "74384 1 Fri Oct 16 08:01:58 2026 omlx-server\n")):
            self.assertEqual(s.process(74384)["start"], "Fri Oct 16 08:01:58 2026")
        with patch.object(s, "run", return_value=(1, "")):
            self.assertIsNone(s.process(74384))
        with self.assertRaises(c.Refuse):
            s.http("POST", "/admin/api/login")

    def test_cli_missing_approval_has_no_backend_or_journal(self):
        with patch.object(c, "Journal", side_effect=AssertionError("must not open journal")):
            with self.assertRaises(c.Refuse):
                c.main(["stop", "--journal", str(self.base / "unused"), "--plan-sha256", c.PLAN_SHA])


if __name__ == "__main__":
    unittest.main()
