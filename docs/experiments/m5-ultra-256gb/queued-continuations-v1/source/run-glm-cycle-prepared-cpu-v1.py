"""All four GLM fake-only CPU modules, AFTER route formal v2 closure AND GDN CPU.

Default describes without opening evidence. --run creates one fresh result tree;
never starts a child/server, sends signals, imports a device backend, or opens
model payloads. A failed but independently RESTORED/archived experiment is a
valid prerequisite. The GDN CPU terminal PASS or FAIL is only a sequencing gate.
Neither prerequisite authorizes any device operation or claims GLM qualification.
"""
import argparse
import contextlib
import datetime
import hashlib
import importlib.abc
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
import unittest
from unittest.mock import patch

ROOT = Path('/Users/ryanli/code/minivllm')
ART = ROOT / '.artifacts/m5-ultra-256gb'
BENCH = ROOT / 'bench/m5_ultra_256gb'
PYTHON = ROOT / '.artifacts/omlx-baseline/.venv/bin/python'
SUITES = {
    'bench.m5_ultra_256gb.glm_flash_cycle.test_controller_cpu': 9,
    'bench.m5_ultra_256gb.glm_flash_cycle.test_preparation_cpu': 6,
    'bench.m5_ultra_256gb.glm_flash_cycle.test_audits_cpu': 8,
    'bench.m5_ultra_256gb.glm_flash_cycle.test_execute_cpu': 10,
}
EXPECTED_TESTS = sum(SUITES.values())
DEFAULT_OUT = ART / 'glm-cycle-prepared-cpu-v1'
GLM = BENCH / 'glm_flash_cycle'
PREPARED = GLM / 'prepared-source-v1.json'
PREPARED_SHA = 'fd86ae72adcae023b4595c1a4a416106a1761c611733d35dde98dc303d1e73ee'
SEALED = {
    "controller.py": "98bc920d97b529c07f9af0a64213e47e73728b63caee2e867e496f1ac30bde88",
    "audit_restore.py": "732c60870b9400d1d7b6bfb9c303bce19b9c92032bc06bc9db193b66ace3bf6d",
    "audit_closed.py": "0158be373f614c55ed03393320337b7cca4c1525360fd850f5081a5462694d48",
    "prepare_inputs.py": "da10a83b826150b88f44e1442423660b9e7edd990114e368fe2fea98032207fb",
    "preregister.py": "7f937f709b0c6806a0dfca7606122e1178b2c495b4574665e7df1eea2862e842",
    "execute.py": "cbc28848a124c74fabd9aede971b57b8fb9162051031e9a720dde5f87029635a",
    "test_controller_cpu.py": "30b4b4fa0d39d26299bdf9a9926f0bc201243093b10173be7e86922cfa1a428f",
    "test_preparation_cpu.py": "8798d44237059a4c5260cd5902826cdbb12211a59760cde8173289c30ed44452",
    "test_audits_cpu.py": "6717572506b87c9c502763445f24f05e83d3c12e1478df2791bbd9d3c086832f",
    "test_execute_cpu.py": "fc6a36bf99d48d17873c5ba58b57512e22030297900d5562d59c51dcbf5b2ad2",
    "README.md": "7e200208dbc83aa08102d136b87bc088f72740764e9424c066189a029b9e799f"
}
FROZEN_ORIGINALS = {
    'glm_flash_cycle/owner.py': '37f47300c14d8da721d6c30fcc71f52325132bad3dc3ba902c1cc08af4326b40',
    'glm_flash/load_probe.py': 'cd9df050384af83d92f0321d95f5bba70dca349defbedf0bb7d1d1e8b57385fc',
    'glm_flash/physical_guard.py': '908e0dc74812fd82b65c63a22e03ed8c5692e812bdbf3d6957e219aa52b14166',
}
GDN_RESULT = ART / 'gdn-http-prepared-cpu-v1/result.json'
GDN_QUEUE = ART / 'gdn-http-prepared-cpu-v1-queue/status.json'
GDN_QUEUE_RUNNER = Path('/tmp/continue-gdn-http-cpu-after-route-v2.py')
GDN_QUEUE_RUNNER_SHA = '698281417d9230a256120ed594119e97fbc619d71d26d4bfc49b28a06e142654'
GDN_RUNNER = Path('/tmp/run-gdn-http-prepared-cpu-v1.py')
GDN_RUNNER_SHA = '22e9ef7a46e1a2ea76940abc8e5152088e124d5f060f02c2ee63c1eeb51b35cc'
SOURCE_DIRS = ('glm_flash_cycle', 'glm_flash', 'route_http_formal', 'route_http_cycle',
    'route_decode_graph', 'identity_refill', 'pareto_realistic', 'long_decode_graph',
    'long_decode_isolation', 'gdn_prework_unbound', 'omlx_d512_formal_retry',
    'omlx_d512_formal', 'omlx_d512', 'omlx_cold_restart', 'cold_cycle',
    'cold_cycle_v2', 'cold_cycle_v3', 'lithos_d512_retry', 'lithos_d512_formal')
FORBIDDEN = {'mlx', 'omlx', 'mlx_lm', 'mlx_vlm', 'torch', 'numpy',
             'tutorial_code', 'minivllm', 'monolith', 'tokenizers'}
NAME = 'route-http-formal-v2'
ARCHIVE = ROOT / 'docs/experiments/m5-ultra-256gb/E024-route-http-formal-v2'
PLAN = BENCH / 'route_http_formal/prereg-formal-v2/owner-plan.json'
PLAN_SHA = '0920aceae4e8bb6ef5c68577d3ae905a2b5976287790a7bb7a4a6e4ea36470ce'
CONTROLLER_SHA = '98d48eb4ad608be5d8d65f6a28622126483ee96ff9a312b3cbd16ee4c63817f3'
RESTORE_SHA = '90a980d0dbe6a61fc1bab2ac6b2edaa9bf012bb60726cd0742801d11410eb0b1'
JOBS = [('audit', '/tmp/audit-closed-route-formal-v2.py',
         'e653955cace058183a6211c677898dbc3ef8cbb8b4d149b94199fdf14c4e2188'),
        ('archive', '/tmp/archive-route-formal-v2.py',
         '44b41d200881b3be6df29b1840da3673c60e43f87b2d10f4ef686352a2fe7477')]
MAX_SMALL_FILE = 32 * (1 << 20)


def need(ok, message):
    if not ok: raise AssertionError(message)


def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(raw): return hashlib.sha256(raw).hexdigest()
def write(path, value): path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
def finite(value): return type(value) in (int, float) and math.isfinite(value)


def small(path):
    path = Path(path)
    need(path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_SMALL_FILE,
         'regular bounded small evidence/source required: ' + str(path))
    before = path.stat(); raw = path.read_bytes(); after = path.stat()
    keys = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    need(keys(before) == keys(after) and len(raw) == before.st_size, 'changed while reading: ' + str(path))
    return raw


def parse(raw):
    def unique(items):
        result = {}
        for key, value in items:
            need(key not in result, 'duplicate JSON key: ' + key); result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def closure_gate():
    """Read the small closure status FIRST; never inspect a running journal."""
    paths = {'closure-status.json': ART / (NAME + '-closure/status.json'),
        'execution-final.json': ART / (NAME + '-execution/final.json'),
        'restored-journal.json': ART / (NAME + '-cold-journal/journal.json'),
        'restoration-audit.json': ART / (NAME + '-analysis/restoration-audit.json'),
        'audit-execution-summary.json': ART / (NAME + '-analysis/execution-summary.json'),
        'archive-verification.json': ARCHIVE / 'archive-verification.json',
        'archive-summary.json': ARCHIVE / 'summary.json',
        'archived-restored-journal.json': ARCHIVE / 'restored-journal.json',
        'archived-restoration-audit.json': ARCHIVE / 'restoration-audit.json',
        'registered-owner-plan.json': PLAN}
    raw = {'closure-status.json': small(paths['closure-status.json'])}
    closure = parse(raw['closure-status.json'])
    need(closure.get('status') in ('PASS', 'EXPERIMENT_FAILED_EVIDENCE_RETAINED')
         and finite(closure.get('ended_at')) and closure.get('service_actions') is False,
         'current formal closure must be terminal before any journal/source/test access')
    jobs = closure.get('jobs')
    need(type(jobs) is list and len(jobs) == 2, 'both actual closure jobs required')
    last = closure['started_at']; need(finite(last), 'closure start')
    for row, (label, script, digest) in zip(jobs, JOBS):
        need(row.get('label') == label and row.get('source_sha256') == digest
             and row.get('argv') == [str(ROOT / '.venv/bin/python'), '-I', '-B', script, '--run']
             and type(row.get('returncode')) is int and finite(row.get('started_at'))
             and finite(row.get('ended_at')) and last <= row['started_at'] <= row['ended_at'],
             'actual closed audit/archive invocation: ' + label)
        last = row['ended_at']
    need(last <= closure['ended_at'] and jobs[1]['returncode'] == 0, 'archive must have ended successfully')
    need(closure.get('audit_returncode') == jobs[0]['returncode'], 'closure audit return code binding')
    if closure['status'] == 'PASS':
        need(closure.get('audit_status') == 'PASS' and jobs[0]['returncode'] == 0, 'closure PASS consistency')
    else:
        need(closure.get('performance_qualified') is False, 'failed formal cannot gain performance qualification')
    raw.update({name: small(path) for name, path in paths.items() if name not in raw})
    final, journal, audit, execution, verified, summary = [parse(raw[n]) for n in (
        'execution-final.json', 'restored-journal.json', 'restoration-audit.json',
        'audit-execution-summary.json', 'archive-verification.json', 'archive-summary.json')]
    need(final.get('journal_state') == journal.get('state') == 'RESTORED'
         and type(final.get('recovery_returncode')) is int and final['recovery_returncode'] == 0,
         'actual current driver recovery0/RESTORED required')
    need(sha(raw['registered-owner-plan.json']) == PLAN_SHA
         and journal.get('schema') == 'E024-route-http-formal-cold-journal-v1'
         and journal.get('controller_sha256') == CONTROLLER_SHA
         and journal.get('owner_plan') == dict(path=str(PLAN), sha256=PLAN_SHA)
         and journal.get('owner_directory') == str(ART / (NAME + '-owner')), 'exact current cycle')
    need(closure.get('final_sha256') == sha(raw['execution-final.json'])
         and closure.get('journal_sha256') == sha(raw['restored-journal.json']), 'closure actual final/journal hashes')
    need(audit.get('schema') == 'cold-service-restoration-audit-v1' and audit.get('status') == 'PASS'
         and audit.get('qualified') is True and audit.get('audit_source_sha256') == RESTORE_SHA
         and audit.get('controller_sha256') == CONTROLLER_SHA and audit.get('owner_plan_sha256') == PLAN_SHA
         and audit.get('journal_sha256') == sha(raw['restored-journal.json'])
         and audit['service_restore']['replacement'] == journal['replacement']
         and audit['service_restore']['http_worker_returncode'] == 0, 'independent real restoration PASS')
    need(execution.get('schema') == 'E024-route-formal-closed-audits-v1'
         and execution.get('owner_plan_sha256') == PLAN_SHA and execution.get('restoration_qualified') is True
         and execution.get('status') == closure['audit_status'] and execution.get('ended_at')
         and execution['jobs']['restoration']['returncode'] == 0
         and execution['jobs']['restoration']['output']['sha256'] == sha(raw['restoration-audit.json']),
         'closed audit execution binds restoration bytes')
    need(verified.get('schema') == 'E024-route-formal-archive-verification-v1'
         and verified.get('status') == 'PASS' and all(verified.get(k) is True for k in (
             'all_safe_extracted_members_verified', 'all_original_files_rehashed', 'all_tree_members_rechecked'))
         and summary.get('schema') == 'E024-route-formal-archive-summary-v1'
         and summary.get('owner_plan_sha256') == PLAN_SHA and summary.get('restoration_qualified') is True,
         'closed verified actual archive required')
    need(raw['archived-restored-journal.json'] == raw['restored-journal.json']
         and raw['archived-restoration-audit.json'] == raw['restoration-audit.json'], 'archive sidecars must be actual closed bytes')
    return paths, raw, dict(status=closure['status'], experiment_audit_status=execution['status'],
        restoration_qualified=True, archive_verified=True, archive_stream=verified['archive'],
        scope='Closed historical evidence only; no live observation, device permission or new performance claim.')


def gdn_sequence_gate(closure_sha256):
    """Require the pinned parent's actual wait return, not the child's early result write."""
    queue_raw = small(GDN_QUEUE); queue = parse(queue_raw)
    command = [str(PYTHON), '-I', '-B', str(GDN_RUNNER), '--run', '--out', str(GDN_RESULT.parent)]
    need(queue.get('status') in ('PASS', 'FAIL') and queue.get('cpu_tests_started') is True
         and queue.get('cpu_tests_completed') is True and queue.get('service_actions') is False
         and queue.get('performance_qualified') is False and type(queue.get('returncode')) is int,
         'actual parent CPU wait-return terminal required before reading child result')
    times = [queue.get(k) for k in ('started_at','child_started_at','child_ended_at','ended_at')]
    need(all(finite(t) for t in times) and times == sorted(times), 'parent CPU wait-return timestamp order')
    need(queue.get('command') == command and queue.get('source_sha256') == GDN_RUNNER_SHA
         and queue.get('closure_sha256') == closure_sha256
         and sha(small(GDN_QUEUE_RUNNER)) == GDN_QUEUE_RUNNER_SHA
         and sha(small(GDN_QUEUE.parent/'runner.py')) == GDN_QUEUE_RUNNER_SHA
         and sha(small(GDN_QUEUE.parent/'cpu-runner.py')) == GDN_RUNNER_SHA,
         'actual pinned parent/child command, source and current closure binding')
    raw = small(GDN_RESULT); row = parse(raw)
    need(row.get('schema') == 'GDN-http-prepared-cpu-v1'
         and row.get('status') in ('PASS', 'FAIL')
         and row.get('module') == 'bench.m5_ultra_256gb.gdn_prework_http.test_http_cpu'
         and type(row.get('expected_tests')) is int and row['expected_tests'] == 16
         and row.get('device_executed') is False, 'GDN16 fake CPU terminal result required for serialization')
    start, end = row.get('started_at'), row.get('ended_at')
    need(type(start) is type(end) is str, 'GDN terminal timestamps required')
    a, b = (datetime.datetime.fromisoformat(t) for t in (start, end))
    need(a.tzinfo is not None and b.tzinfo is not None
         and times[1] <= a.timestamp() <= b.timestamp() <= times[2],
         'GDN child result must precede actual parent wait return')
    need(queue.get('result_sha256') == sha(raw) and queue.get('test_status') == row['status']
         and type(row.get('tests')) is int and type(queue.get('tests')) is int and queue['tests'] == row['tests']
         and (row['status'],queue['returncode']) in (('PASS',0),('FAIL',1))
         and (queue['status'] != 'PASS' or (row['status'] == 'PASS' and row['tests'] == 16)),
         'parent wait returncode and actual child result bytes must agree')
    need(row.get('argv') == [str(PYTHON), '-I', '-B', str(GDN_RUNNER.resolve()), '--run', '--out', str(GDN_RESULT.parent)]
         and sha(small(GDN_RUNNER)) == GDN_RUNNER_SHA
         and sha(small(GDN_RESULT.parent/'runner.py')) == GDN_RUNNER_SHA,
         'actual GDN runner invocation/source must match the queued predecessor')
    return raw, queue_raw, dict(path=str(GDN_RESULT), sha256=sha(raw), bytes=len(raw), status=row['status'],
        tests=row.get('tests'), expected_tests=16, runner_sha256=GDN_RUNNER_SHA,
        parent_queue=dict(path=str(GDN_QUEUE),sha256=sha(queue_raw),bytes=len(queue_raw),status=queue['status'],
            returncode=queue['returncode'],child_ended_at=queue['child_ended_at'],ended_at=queue['ended_at'],
            runner_sha256=GDN_QUEUE_RUNNER_SHA),
        scope='Actual parent wait-return serialization only; a child CPU FAIL is allowed and never qualifies GLM.')


def sources_and_metadata():
    paths = {Path(__file__).resolve(), PREPARED, *(GLM/name for name in SEALED),
             *(BENCH/name for name in FROZEN_ORIGINALS), GDN_RUNNER, GDN_QUEUE_RUNNER}
    for name in SOURCE_DIRS:
        folder = BENCH/name; need(folder.is_dir(), 'supplied source directory missing: '+name)
        paths.update(folder.glob('*.py'))
    paths.update(p for p in (ROOT/'bench/__init__.py', BENCH/'__init__.py') if p.is_file())
    for name in ('resources.py', 'resources_v2.py', 'resource_base.py', 'resource_reparse.py',
                 'cold_cycle/lifecycle.json', 'cold_cycle_v2/lifecycle.json', 'cold_cycle_v3/lifecycle.json'):
        paths.add(BENCH/name)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true'); parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not args.run:
        print('DESCRIPTION_ONLY: all four whole GLM fake CPU modules (33 tests), after actual route-formal-v2 restored/audited/archived closure and GDN16 terminal CPU result; no operations performed.')
        return 0
    need(Path(sys.executable) == PYTHON and Path(sys.prefix) == PYTHON.parent.parent
         and sys.flags.isolated == 1 and sys.dont_write_bytecode is True, 'actual vendor Python -I -B required')
    need(args.out.is_absolute() and args.out.parent == ART and not args.out.exists(), 'fresh artifact output required')
    args.out.mkdir(); (args.out / 'runner.py').write_bytes(Path(__file__).read_bytes())
    rec = dict(schema='GLM-cycle-prepared-cpu-v1', status='RUNNING', started_at=now(),
        argv=[sys.executable, '-I', '-B', str(Path(__file__).resolve()), *sys.argv[1:]],
        suites=[dict(module=m,expected_tests=n) for m,n in SUITES.items()],
        expected_tests=EXPECTED_TESTS, discovered_tests=None, discovered_by_module={}, tests=0,
        failures=0, errors=0, skipped=0, expected_failures=0, unexpected_successes=0,
        python_executable=sys.executable, python_resolved=str(Path(sys.executable).resolve()), python_version=sys.version,
        sealed_source_pins={**{str(GLM/n):h for n,h in SEALED.items()}, str(PREPARED):PREPARED_SHA,
            **{str(BENCH/n):h for n,h in FROZEN_ORIGINALS.items()}}, source_directories=list(SOURCE_DIRS),
        source_before={}, source_after={}, denied_operations=[], test_outcomes=[],
        device_executed=False, http_qualified=False, performance_qualified=False,
        scope='Four whole fake CPU modules (33 tests), no filtering. Not model/numerical/device/HTTP/performance qualification. Runtime/model inventory tests use synthetic files; no actual GLM preparation.',
        guard_scope='No subprocess/network/signal delivery or timers/backend imports/model payload reads. Local asyncio socketpair and SIGINT handler registration are allowed; no signal is sent.')
    def save(): write(args.out / 'result.json', rec)
    def deny(kind, detail):
        rec['denied_operations'].append(dict(kind=kind, detail=str(detail)))
        raise AssertionError('forbidden real operation: ' + kind)
    def forbidden(name): return name.split('.')[0] in FORBIDDEN
    def hook(event, values):
        if event in ('subprocess.Popen', 'os.system', 'os.fork', 'os.forkpty', 'os.posix_spawn',
                     'os.exec', 'os.spawn', 'os.kill', 'os.killpg', 'signal.pthread_kill',
                     'socket.connect', 'socket.bind', 'socket.getaddrinfo', 'socket.sendto'):
            deny(event, values)
        if event == 'socket.__new__' and values[1] in (socket.AF_INET, socket.AF_INET6): deny(event, values)
        if event == 'import' and forbidden(values[0]): deny('backend import', values[0])
        if event == 'exec':
            filename = getattr(values[0], 'co_filename', '')
            if filename.startswith((str(ROOT/'tutorial_code')+'/', str(ROOT/'minivllm')+'/')):
                deny('backend source execution', filename)
        if event == 'open' and isinstance(values[0], (str, bytes)):
            path = Path(os.fsdecode(values[0]))
            if path.suffix in ('.safetensors', '.bin', '.pack', '.metallib'):
                deny('model/device payload open', path)
    class Imports(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if forbidden(fullname): deny('backend import', fullname)
            return None
    class Result(unittest.TextTestResult):
        def startTest(self, test):
            rec['test_outcomes'].append(dict(id=test.id(), status='RUNNING', subtests=[])); super().startTest(test)
        def addSuccess(self, test):
            rec['test_outcomes'][-1]['status']='PASS'; super().addSuccess(test)
        def addFailure(self, test, error):
            rec['test_outcomes'][-1].update(status='FAIL', detail=self._exc_info_to_string(error,test)); super().addFailure(test,error)
        def addError(self, test, error):
            rec['test_outcomes'][-1].update(status='ERROR', detail=self._exc_info_to_string(error,test)); super().addError(test,error)
        def addSkip(self, test, reason):
            rec['test_outcomes'][-1].update(status='SKIP', reason=reason); super().addSkip(test,reason)
        def addExpectedFailure(self, test, error):
            rec['test_outcomes'][-1]['status']='EXPECTED_FAILURE'; super().addExpectedFailure(test,error)
        def addUnexpectedSuccess(self, test):
            rec['test_outcomes'][-1]['status']='UNEXPECTED_SUCCESS'; super().addUnexpectedSuccess(test)
        def addSubTest(self, test, subtest, error):
            row=dict(id=str(subtest), status='PASS' if error is None else 'FAIL')
            if error is not None:
                row['detail']=self._exc_info_to_string(error,test); rec['test_outcomes'][-1]['status']='FAIL'
            rec['test_outcomes'][-1]['subtests'].append(row); super().addSubTest(test,subtest,error)
    paths=set(); gate_raw={}; gate_paths={}; gdn_raw=None; gdn_queue_raw=None; guard=None; start=time.monotonic(); save()
    try:
        with (args.out/'stdout.txt').open('x') as out, (args.out/'stderr.txt').open('x') as err, \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            gate_paths,gate_raw,rec['closure_gate']=closure_gate()
            for name,raw in gate_raw.items():
                (args.out/name).write_bytes(raw)
                rec.setdefault('closure_inputs',{})[name]=dict(path=str(gate_paths[name]),sha256=sha(raw),bytes=len(raw))
            gdn_raw,gdn_queue_raw,rec['gdn_sequence_gate']=gdn_sequence_gate(sha(gate_raw['closure-status.json']))
            (args.out/'previous-gdn-cpu-result.json').write_bytes(gdn_raw)
            (args.out/'previous-gdn-cpu-queue.json').write_bytes(gdn_queue_raw)
            for name,digest in SEALED.items(): need(sha(small(GLM/name))==digest,'frozen GLM source changed: '+name)
            for name,digest in FROZEN_ORIGINALS.items(): need(sha(small(BENCH/name))==digest,'original frozen source changed: '+name)
            prepared_raw=small(PREPARED); need(sha(prepared_raw)==PREPARED_SHA,'root source-prepared manifest changed')
            manifest=parse(prepared_raw)
            need(manifest.get('schema')=='GLM-cycle-source-preparation-v1'
                 and manifest.get('status')=='SOURCE_PREPARED'
                 and manifest.get('cpu_test_methods')=={m.rsplit('.',1)[-1]+'.py':n for m,n in SUITES.items()}
                 and manifest.get('cpu_tests_executed') is False
                 and set(manifest['files'])==set(SEALED), 'frozen source-prepared manifest')
            for name,row in manifest['files'].items():
                raw=small(GLM/name); need(row==dict(sha256=sha(raw),bytes=len(raw)), 'manifest exact GLM file')
            paths=sources_and_metadata()
            for path in sorted(paths):
                raw=small(path);rec['source_before'][str(path)]=dict(sha256=sha(raw),bytes=len(raw))
                dest=args.out/'frozen-source'/str(path).lstrip('/');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
            need(not any(forbidden(name) for name in sys.modules),'no preloaded backend modules')
            os.chdir(ROOT);sys.path.insert(0,str(ROOT));guard=Imports();sys.meta_path.insert(0,guard);sys.addaudithook(hook);save()
            with contextlib.ExitStack() as stack:
                def block(label): return lambda *a,**k:deny(label,(a,k))
                for obj,name in ((subprocess,'Popen'),(subprocess,'run'),(socket.socket,'connect'),
                        (socket.socket,'connect_ex'),(socket.socket,'bind'),(socket.socket,'sendto'),
                        (socket,'getaddrinfo'),(os,'kill'),(os,'killpg'),(os,'system'),
                        (signal,'raise_signal'),(signal,'pthread_kill'),(signal,'alarm'),(signal,'setitimer')):
                    if hasattr(obj,name):stack.enter_context(patch.object(obj,name,side_effect=block(name)))
                suite=unittest.TestSuite()
                for module,expected in SUITES.items():
                    whole=unittest.defaultTestLoader.loadTestsFromName(module)
                    rec['discovered_by_module'][module]=whole.countTestCases();suite.addTests(whole)
                rec['discovered_tests']=suite.countTestCases();save()
                result=unittest.TextTestRunner(stream=err,verbosity=2,resultclass=Result).run(suite)
            rec.update(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                skipped=len(result.skipped),expected_failures=len(result.expectedFailures),
                unexpected_successes=len(result.unexpectedSuccesses),loader_errors=unittest.defaultTestLoader.errors,
                failures_detail=[dict(test=t.id(),traceback=s) for t,s in result.failures],
                errors_detail=[dict(test=t.id(),traceback=s) for t,s in result.errors])
            rec['status']='PASS' if (result.wasSuccessful() and result.testsRun==EXPECTED_TESTS
                and rec['discovered_tests']==EXPECTED_TESTS and rec['discovered_by_module']==SUITES and not result.skipped and not result.expectedFailures
                and not result.unexpectedSuccesses and not rec['loader_errors'] and not rec['denied_operations']) else 'FAIL'
    except BaseException as exc:
        rec.update(status='FAIL',error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
    finally:
        if guard in sys.meta_path:sys.meta_path.remove(guard)
        for path in sorted(paths):
            try:
                raw=small(path);rec['source_after'][str(path)]=dict(sha256=sha(raw),bytes=len(raw))
            except BaseException as exc:rec['source_after'][str(path)]=dict(error=str(exc))
        rec['source_unchanged']=bool(rec['source_before']) and rec['source_before']==rec['source_after']
        try:rec['closure_unchanged']=bool(gate_raw) and all(small(gate_paths[n])==raw for n,raw in gate_raw.items())
        except BaseException as exc:
            rec['closure_unchanged']=False;rec['closure_recheck_error']=dict(type=type(exc).__name__,message=str(exc))
        try:rec['gdn_sequence_unchanged']=(gdn_raw is not None and gdn_queue_raw is not None
            and small(GDN_RESULT)==gdn_raw and small(GDN_QUEUE)==gdn_queue_raw)
        except BaseException as exc:
            rec['gdn_sequence_unchanged']=False;rec['gdn_recheck_error']=dict(type=type(exc).__name__,message=str(exc))
        rec['forbidden_modules_loaded']=sorted(n for n in sys.modules if forbidden(n))
        if not rec['source_unchanged'] or not rec['closure_unchanged'] or not rec['gdn_sequence_unchanged'] or rec['forbidden_modules_loaded']:rec['status']='FAIL'
        rec.update(ended_at=now(),elapsed_seconds=time.monotonic()-start);save()
        inventory={str(p.relative_to(args.out)):dict(sha256=sha(small(p)),bytes=p.stat().st_size)
                   for p in sorted(args.out.rglob('*')) if p.is_file()}
        write(args.out/'artifact-manifest.json',dict(schema='cpu-artifact-inventory-v1',files=inventory))
    print(json.dumps({k:rec[k] for k in ('status','tests','failures','errors','skipped','source_unchanged','closure_unchanged','gdn_sequence_unchanged')}))
    return 0 if rec['status']=='PASS' else 1


if __name__=='__main__': raise SystemExit(main())
