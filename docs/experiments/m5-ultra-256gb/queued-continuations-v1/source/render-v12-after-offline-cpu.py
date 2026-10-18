"""One inert-by-default v12 render continuation after actual offline CPU exit.

No service, model, experiment, commit, or current-link action. CPU PASS is not a
plot prerequisite: its actual terminal subprocess return only serializes work.
Complete route formal/resource/restoration/retention PASS remains mandatory in
the separately pinned v12 builder. Failed formal evidence leaves v11 current.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path('/Users/ryanli/code/minivllm')
ART = ROOT / '.artifacts/m5-ultra-256gb'
OUT = ART / 'progress-v12-offline-render-queue'
V12 = ROOT / 'docs/optimization-guide/progress/2026-10-17-v12'
PYTHON = ROOT / '.venv/bin/python'
VENDOR_PYTHON = ROOT / '.artifacts/omlx-baseline/.venv/bin/python'
CPU_QUEUE = ART / 'glm-cycle-prepared-cpu-v1-queue/status.json'
CPU_RUNNER = Path('/tmp/run-glm-cycle-prepared-cpu-v1.py')
CPU_OUTPUT = ART / 'glm-cycle-prepared-cpu-v1'
CLOSURE = ART / 'route-http-formal-v2-closure/status.json'
PLAN_SHA = '0920aceae4e8bb6ef5c68577d3ae905a2b5976287790a7bb7a4a6e4ea36470ce'
SOURCES = {
    'prepare': (V12 / 'prepare.py', '5bc8be7c8f6a5994eaff1895e254f9acfbaf483a309ccbf6af01a09c4ee0d966'),
    'render': (V12 / 'render.py', 'a3b3333bd4e89db07a29277e37cd2411037e06c54a4faf25b05fcefc02bde8cc'),
}
TARGETS = ('sources', 'chartdata.json', 'preparation.json', 'verification.json',
           'D512-throughput-latency.png', 'D512-throughput-latency.svg')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def save(value):
    pending = OUT / 'status.next.json'
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    pending.replace(OUT / 'status.json')


def retain(path, name):
    raw = Path(path).read_bytes()
    with (OUT / name).open('xb') as stream:
        stream.write(raw)
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def actual_cpu_exit():
    """Parent wait result, not the child's earlier result.json/ended_at write."""
    deadline = time.monotonic() + 43200
    while True:
        need(time.monotonic() < deadline, 'GLM CPU queue did not close within 12 hours')
        try:
            row = read(CPU_QUEUE)
        except (FileNotFoundError, json.JSONDecodeError):
            row = {}
        if row.get('ended_at') is not None:
            break
        time.sleep(30)
    need(row.get('status') in ('PASS', 'FAIL') and row.get('service_actions') is False
         and row.get('cpu_tests_started') is True and row.get('cpu_tests_completed') is True
         and type(row.get('returncode')) is int, 'actual terminal waited CPU attempt required')
    times = [row.get(k) for k in ('started_at', 'child_started_at', 'child_ended_at', 'ended_at')]
    need(all(finite(t) for t in times) and times == sorted(times), 'parent CPU wait timestamp order')
    need(row.get('command') == [str(VENDOR_PYTHON), '-I', '-B', str(CPU_RUNNER),
         '--run', '--out', str(CPU_OUTPUT)], 'actual fixed GLM CPU invocation')
    return row


def invoke(label, value):
    path, digest = SOURCES[label]
    need(sha(path) == digest, 'reviewed v12 source changed: ' + label)
    command = [str(PYTHON), '-I', '-B', str(path), '--run']
    row = dict(label=label, argv=command, cwd=str(ROOT), source_sha256=digest,
               started_at=time.time(), status='STARTED')
    value['jobs'].append(row)
    save(value)
    try:
        with (OUT / (label + '.stdout')).open('xb') as stdout, (OUT / (label + '.stderr')).open('xb') as stderr:
            child = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                   check=False, timeout=2400)
        need(type(child.returncode) is int, 'actual waited render subprocess result')
        row.update(status='RETURNED', returncode=child.returncode)
        need(child.returncode == 0, label + ' failed; retain partial files and do not retry')
    except BaseException as error:
        row.update(error_type=type(error).__name__, error=str(error))
        if 'returncode' not in row:
            row['status'] = 'UNKNOWN_OR_TIMEOUT'
        raise
    finally:
        row['ended_at'] = time.time()
        save(value)


def verify_outputs():
    preparation = read(V12 / 'preparation.json')
    chart = read(V12 / 'chartdata.json')
    rendered = read(V12 / 'verification.json')
    need(preparation['schema'] == 'D512-progress-v12-preparation-v1'
         and preparation['status'] == 'PREPARED' and preparation['owner_plan_sha256'] == PLAN_SHA
         and preparation['prepare_sha256'] == SOURCES['prepare'][1]
         and preparation['renderer_sha256'] == SOURCES['render'][1]
         and preparation['chartdata_sha256'] == sha(V12 / 'chartdata.json')
         and preparation['manifest_sha256'] == sha(V12 / 'sources/manifest.json')
         and preparation['current_link_updated'] is False and preparation['visual_review_required'] is True,
         'actual prepared data/source identity')
    need(chart['schema'] == 'D512-throughput-latency-progress-v12'
         and chart['performance_point_count'] == len(chart['points']) == 26
         and chart['historical_points_exactly_unchanged'] is True
         and chart['native_historical_pooled_with_off'] is False, '26 separate actual observations')
    for row in (preparation, rendered):
        need(row['performance_points'] == 26 and row['historical_points'] == 14 and row['new_points'] == 12,
             'actual render coverage')
    need(rendered['status'] == 'RENDERED' and rendered['chartdata_sha256'] == preparation['chartdata_sha256']
         and rendered['renderer_sha256'] == SOURCES['render'][1]
         and rendered['visual_review_completed'] is False and rendered['current_link_updated'] is False,
         'render receipt is not visual review or current-link promotion')
    names = {'D512-throughput-latency.png', 'D512-throughput-latency.svg'}
    need(set(rendered['figures']) == names, 'exact two rendered figures')
    for name in names:
        path = V12 / name
        need(path.is_file() and not path.is_symlink() and path.stat().st_size > 0
             and rendered['figures'][name] == dict(sha256=sha(path), bytes=path.stat().st_size),
             'actual figure bytes: ' + name)
    return {name: retain(V12 / name, name) for name in
            ('preparation.json', 'chartdata.json', 'verification.json')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if not args.run:
        print('DESCRIPTION_ONLY: wait for actual offline CPU exit; complete route PASS alone can render v12; no actions performed.')
        return 0
    need(Path(sys.executable) == PYTHON and sys.flags.isolated == 1 and sys.dont_write_bytecode is True,
         'actual outer Python -I -B required')
    need(not OUT.exists(), 'fresh render queue only')
    for label, (path, digest) in SOURCES.items():
        need(sha(path) == digest, 'reviewed v12 source changed: ' + label)
    OUT.mkdir()
    retain(Path(__file__), 'runner.py')
    for label, (path, _) in SOURCES.items():
        retain(path, label + '.py')
    value = dict(schema='D512-v12-offline-render-queue-v1', status='WAITING_FOR_OFFLINE_CPU',
        started_at=time.time(), jobs=[], service_actions=False, model_actions=False,
        visual_review_completed=False, current_link_updated=False, rendered=False)
    save(value)
    try:
        cpu = actual_cpu_exit()
        value['cpu_queue'] = retain(CPU_QUEUE, 'glm-cpu-queue.json')
        value['cpu_status'] = cpu['status']
        closure = read(CLOSURE)
        value['formal_closure'] = retain(CLOSURE, 'formal-closure.json')
        need(finite(closure.get('ended_at')) and closure.get('service_actions') is False,
             'formal closure must be terminal')
        if closure.get('status') != 'PASS':
            value.update(status='NO_NEW_SPEED_CHART', reason='Formal closure is not PASS; keep v11 current.')
        else:
            need(type(closure.get('audit_returncode')) is int and closure['audit_returncode'] == 0
                 and closure.get('audit_status') == 'PASS' and closure.get('performance_qualified') is True,
                 'closed formal eligibility before render')
            need(not any((V12 / name).exists() for name in TARGETS), 'new v12 outputs only; retain all earlier attempts')
            value['status'] = 'PREPARING_V12'
            invoke('prepare', value)
            value['status'] = 'RENDERING_V12'
            invoke('render', value)
            value['outputs'] = verify_outputs()
            need(sha(CPU_QUEUE) == value['cpu_queue']['sha256']
                 and sha(CLOSURE) == value['formal_closure']['sha256'], 'closed prerequisite bytes changed')
            for label, (path, digest) in SOURCES.items():
                need(sha(path) == digest, 'v12 source changed during invocation: ' + label)
            value.update(status='RENDERED_REQUIRES_VISUAL_QA', rendered=True)
    except BaseException as error:
        value.update(status='FAIL', error=dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
    finally:
        if value['status'] != 'RENDERED_REQUIRES_VISUAL_QA':
            value['rendered'] = False
        value.update(ended_at=time.time(), visual_review_completed=False, current_link_updated=False)
        save(value)
    print(json.dumps(value))
    return 0 if value['status'] in ('RENDERED_REQUIRES_VISUAL_QA', 'NO_NEW_SPEED_CHART') else 1


if __name__ == '__main__':
    raise SystemExit(main())
