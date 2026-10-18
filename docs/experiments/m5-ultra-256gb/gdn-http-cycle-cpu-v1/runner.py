"""Whole frozen GDN HTTP fake-only CPU module, AFTER route formal v2 closure.

Default describes without opening evidence. --run creates one fresh result tree;
never starts a child/server, sends signals, imports a device backend, or opens
model payloads. A failed but independently RESTORED/archived experiment is a
valid prerequisite. That prerequisite does not authorize any device operation.
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
MODULE = ['bench.m5_ultra_256gb.gdn_prework_http_cycle.test_' + n + '_cpu' for n in ('owner','controller','audit_restore','execute','prepare','audit_cycle')]
EXPECTED_TESTS = 83
DEFAULT_OUT = ART / 'gdn-http-cycle-cpu-v1'
GDN = BENCH / 'gdn_prework_http'
SEALED = {
    'http_server.py': 'e4408d9df0c26a2f7fc5db93312902e4797b72a8958f67f1912b4b9bd628e3d4',
    'http_capture.py': '0ecedc988d2b77a55f010bdd09e9f9581e4aa9618d7070a6eb4b81960e5849c8',
    'http_qualification.py': '3dddc8d00b42e0ebe34883805f13a018325f83b232cf007f170cd01a27c18209',
    'audit.py': '5ccb357e1f6a1296ab8673a604c48f2e183d01d83671e8d7857edc85cded7cd0',
    'test_http_cpu.py': '34b992a48cb5ca38e810c2bf71394a25d3182d779480289d839ec7c6529245f4',
    'README.md': 'a0dab9a0cb79c7340dd94a41be27c1865481cfd56295cabdccd33783405817ba',
    'source-manifest.json': '9d1ba2ab648f9ea10e0c0fea512257314358f838abb6705e4f27ca260fc98a1a',
}
SOURCE_DIRS = ('gdn_prework_http_cycle', 'route_http_cycle', 'route_http_formal', 'cold_cycle', 'omlx_cold_restart', 'glm_flash', 'gdn_prework_http', 'gdn_prework_unbound', 'pareto_realistic',
    'route_decode_graph', 'identity_refill', 'long_decode_graph',
    'long_decode_isolation', 'conv_tail_owned_copy', 'shared_w_m1')
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
BASE_PLAN = BENCH / 'pareto_realistic/prereg-qualification-v1/plan.json'
BASE_PLAN_SHA = '21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb'
SERVING_AUDIT = ROOT / 'docs/experiments/m5-ultra-256gb/E029-gdn-prework-serving-v1/independent-numerical.json'
SERVING_AUDIT_SHA = 'fee50c90c30c3224934489c66e5ecb4a5d7f0c4fb57ae0baa0576f6efe9a769b'
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


def sources_and_metadata():
    paths = {Path(__file__).resolve(), *(GDN / name for name in SEALED)}
    for name in SOURCE_DIRS:
        folder = BENCH / name; need(folder.is_dir(), 'supplied source directory missing: ' + name)
        paths.update(folder.glob('*.py'))
    # Snapshot actual core source read by metadata/source contracts, but forbid its import.
    paths.update((ROOT / 'tutorial_code/miniengine').rglob('*.py'))
    # These are namespace packages in this checkout; snapshot an initializer
    # only when one actually exists, never manufacture an empty source file.
    paths.update(p for p in (ROOT / 'bench/__init__.py', BENCH / '__init__.py') if p.is_file())
    for name in ('resources.py', 'resources_v2.py', 'resource_base.py', 'resource_reparse.py',
                 'long_decode_graph/model-source-pins.json'):
        paths.add(BENCH / name)
    numeric_raw = small(BASE_PLAN); need(sha(numeric_raw) == BASE_PLAN_SHA, 'original numeric plan changed')
    numeric = parse(numeric_raw); paths.add(BASE_PLAN)
    for key in ('fixtures', 'source_manifest', 'model_manifest'):
        ref = numeric[key]; p = Path(ref['path'])
        need(p.is_absolute() and p.suffix == '.json' and sha(small(p)) == ref['sha256'], 'original small metadata: ' + key)
        paths.add(p)
    audit_raw = small(SERVING_AUDIT); need(sha(audit_raw) == SERVING_AUDIT_SHA, 'actual GDN37 audit changed')
    audit = parse(audit_raw); child = Path(audit['child_directory'])
    need(child.is_absolute(), 'actual closed GDN37 child')
    paths.add(SERVING_AUDIT)
    for name in ('plan.json', 'complete.json'):
        p = child / name; need(sha(small(p)) == audit['terminal_sha256'][name], 'actual closed GDN37 ' + name); paths.add(p)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true'); parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not args.run:
        print('DESCRIPTION_ONLY: whole83 new cycle fake CPU tests after actual route-formal-v2 restored/audited/archived closure; no operations performed.')
        return 0
    need(Path(sys.executable) == PYTHON and Path(sys.prefix) == PYTHON.parent.parent
         and sys.flags.isolated == 1 and sys.dont_write_bytecode is True, 'actual vendor Python -I -B required')
    need(args.out.is_absolute() and args.out.parent == ART and not args.out.exists(), 'fresh artifact output required')
    args.out.mkdir(); (args.out / 'runner.py').write_bytes(Path(__file__).read_bytes())
    rec = dict(schema='GDN-http-cycle-cpu-v1', status='RUNNING', started_at=now(),
        argv=[sys.executable, '-I', '-B', str(Path(__file__).resolve()), *sys.argv[1:]],
        module=MODULE, expected_tests=EXPECTED_TESTS, discovered_tests=None, tests=0,
        failures=0, errors=0, skipped=0, expected_failures=0, unexpected_successes=0,
        python_executable=sys.executable, python_resolved=str(Path(sys.executable).resolve()), python_version=sys.version,
        sealed_source_pins={str(GDN/n): h for n,h in SEALED.items()}, source_directories=list(SOURCE_DIRS),
        source_before={}, source_after={}, denied_operations=[], test_outcomes=[],
        device_executed=False, http_qualified=False, performance_qualified=False,
        scope='Six whole fake CPU modules, no filtering. Not model/numerical/device/HTTP/performance qualification.',
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
    paths=set(); gate_raw={}; gate_paths={}; guard=None; start=time.monotonic(); save()
    try:
        with (args.out/'stdout.txt').open('x') as out, (args.out/'stderr.txt').open('x') as err, \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            gate_paths,gate_raw,rec['closure_gate']=closure_gate()
            for name,raw in gate_raw.items():
                (args.out/name).write_bytes(raw)
                rec.setdefault('closure_inputs',{})[name]=dict(path=str(gate_paths[name]),sha256=sha(raw),bytes=len(raw))
            for name,digest in SEALED.items(): need(sha(small(GDN/name))==digest,'frozen GDN source changed: '+name)
            manifest=parse(small(GDN/'source-manifest.json'))
            need(manifest['cpu_test_methods']==16 and manifest['cpu_tests_executed'] is False
                 and set(manifest['files'])==set(SEALED)-{'source-manifest.json'},'frozen source-prepared manifest')
            for name,row in manifest['files'].items():
                raw=small(GDN/name); need(row==dict(sha256=sha(raw),bytes=len(raw)),'manifest exact GDN file')
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
                suite=unittest.defaultTestLoader.loadTestsFromNames(MODULE)
                rec['discovered_tests']=suite.countTestCases();save()
                result=unittest.TextTestRunner(stream=err,verbosity=2,resultclass=Result).run(suite)
            rec.update(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                skipped=len(result.skipped),expected_failures=len(result.expectedFailures),
                unexpected_successes=len(result.unexpectedSuccesses),loader_errors=unittest.defaultTestLoader.errors,
                failures_detail=[dict(test=t.id(),traceback=s) for t,s in result.failures],
                errors_detail=[dict(test=t.id(),traceback=s) for t,s in result.errors])
            rec['status']='PASS' if (result.wasSuccessful() and result.testsRun==EXPECTED_TESTS
                and rec['discovered_tests']==EXPECTED_TESTS and not result.skipped and not result.expectedFailures
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
        rec['forbidden_modules_loaded']=sorted(n for n in sys.modules if forbidden(n))
        if not rec['source_unchanged'] or not rec['closure_unchanged'] or rec['forbidden_modules_loaded']:rec['status']='FAIL'
        rec.update(ended_at=now(),elapsed_seconds=time.monotonic()-start);save()
        inventory={str(p.relative_to(args.out)):dict(sha256=sha(small(p)),bytes=p.stat().st_size)
                   for p in sorted(args.out.rglob('*')) if p.is_file()}
        write(args.out/'artifact-manifest.json',dict(schema='cpu-artifact-inventory-v1',files=inventory))
    print(json.dumps({k:rec[k] for k in ('status','tests','failures','errors','skipped','source_unchanged','closure_unchanged')}))
    return 0 if rec['status']=='PASS' else 1


if __name__=='__main__': raise SystemExit(main())
