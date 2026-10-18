"""C2/C4 actual HTTP replay adapter, explicit --prepare-only/--run only.

Original candidate/math remain frozen. One real C2/C4 model pair is constructed;
recorded arrivals are inputs, recorded tokens/schedules only comparison targets.
Resource/service admission belongs to a separate owner, never this producer.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import traceback
HERE=Path(__file__).resolve().parent
DONOR=HERE.parent/'gdn_graph_combined/http_qualification.py'
DONOR_SHA='33b86cb57bf3c4616762c1bd1cfea0abb8f7f063325e9dfe815642e8f591bc7d'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def local(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
if sha(DONOR)!=DONOR_SHA:raise ValueError('frozen combined replay changed')
donor=local(DONOR,'_c24_combined_replay')
domain=local(HERE/'domain.py','_c24_domain_contract')
need,same,read,parse,write,pinned=donor.need,donor.same,donor.read,donor.parse,donor.write,donor.pinned
BASE,BASE_PLAN_SHA,NATIVE,ROUTE,OLD=donor.BASE,donor.BASE_PLAN_SHA,donor.NATIVE,donor.ROUTE,donor.OLD
CAPTURE_NAMES,INPUT_NAMES=donor.CAPTURE_NAMES,donor.INPUT_NAMES
CapturedRoutes,adapted,serving_gate=donor.CapturedRoutes,donor.adapted,donor.serving_gate
FIXED={**donor.FIXED,DONOR:DONOR_SHA}
NEW=('domain.py','http_server.py','http_capture.py','pair.py','replay.py','qualify.py','audit.py')

def source_pins():
    pins=donor.source_pins()
    return {**pins,**{str(HERE/n):sha(HERE/n) for n in NEW}}

def load(path,name):
    path=Path(path)
    if path.parent==HERE and path.name in NEW:return local(path,name)
    if path==DONOR:return local(path,name)
    return donor.load(path,name)

def capture_gate(plan):
    need(set(plan['capture']) == set(CAPTURE_NAMES) and set(plan['inputs']) == set(INPUT_NAMES),
         'exact captured specification, termination and raw inputs required')
    paths = {k: pinned(v) for k, v in plan['capture'].items()}
    inputs = {k: pinned(v) for k, v in plan['inputs'].items()}
    directory = paths['spec'].parent
    for table, names in ((paths, CAPTURE_NAMES), (inputs, INPUT_NAMES)):
        for key, path in table.items():
            need(path == directory / names[key], 'mixed or renamed capture inputs: ' + key)
    spec, complete, child, release = [read(paths[k]) for k in ('spec', 'complete', 'child', 'release')]
    capture = load(HERE / 'http_capture.py', '_combined_replay_capture_contract')
    capture.validate(spec)
    need(spec['concurrency'] == plan['concurrency'] and spec['base_plan'] == plan['qualification'],
         'capture concurrency/base plan')
    need(complete['schema'] == 'GDN-graph-combined-c24-capture-result-v1' and complete['status'] == 'PASS'
         and all(complete.get(k) is True for k in ('capture_complete', 'server_reaped', 'cleanup_complete',
             'cleanup_task_resolved', 'client_task_resolved', 'small_pins_rechecked', 'route_events_retained'))
         and complete['errors'] == [] and complete['numerical_qualified'] is False
         and complete['performance_qualified'] is False, 'complete observed capture with closed cleanup required')
    same(complete['lifecycle_source'],dict(path=str(capture.LIFECYCLE_PATH),sha256=capture.LIFECYCLE_SHA),'frozen lifecycle source')
    same(complete['capture_adapter_source'],dict(path=str(HERE/'http_capture.py'),sha256=sha(HERE/'http_capture.py')),'actual capture source')
    pid = child['pid']
    need(type(pid) is int and pid > 1 and pid != os.getpid() and release['pid'] == pid
         and type(release['returncode']) is int
         and all(release[k] is True for k in ('waited_reaped', 'strict_absence', 'port_absent')),
         'separate captured server must have waited/reaped release')
    need(capture.no_process(read(paths['absence'])) and capture.no_process(read(paths['port'])),
         'captured server and port absence receipts required')
    need(type(release['time']) in (int, float) and math.isfinite(release['time'])
         and child['time'] <= release['time'] <= complete['ended_at'], 'capture/release time order')
    # Retained receipts are not a replacement for the outer owner's current
    # process/resource check. Bind them to the complete producer's actual bytes.
    for key, path in {**paths, **inputs}.items():
        if key == 'complete':
            continue
        same(complete['artifacts'][path.name], dict(sha256=sha(path), bytes=path.stat().st_size),
             'capture raw artifact binding: ' + path.name)
    header = dict(event='route-startup', schema=capture.ROUTE_SCHEMA, pid=pid,
        phase=f"HTTP-C{plan['concurrency']}", identity=spec['identity'])
    with inputs['route_events'].open('rb') as stream:
        raw = stream.readline()
        need(raw.endswith(b'\n'), 'complete route startup line required')
        same(parse(raw), header, 'route startup PID/source/phase binding')
    return spec, header

def validate(path, expected_sha):
    need(sha(path) == expected_sha, 'new HTTP replay plan SHA')
    plan = read(path)
    need(set(plan) == {'schema', 'qualifier_sha256', 'qualification', 'serving', 'concurrency',
        'capture', 'inputs', 'sources'} and plan['schema'] == 'GDN-graph-combined-c24-recorded-http-plan-v1',
        'exact route replay plan required')
    need(plan['qualifier_sha256'] == sha(__file__), 'actual adapter source identity')
    domain.concurrency(plan['concurrency'])
    same(plan['sources'], source_pins(), 'complete fixed adapter/dependency source map')
    for source, digest in plan['sources'].items():
        need(sha(source) == digest, 'source changed: ' + source)
    need(plan['qualification']['sha256'] == BASE_PLAN_SHA, 'unchanged original model/workload plan')
    numeric = read(pinned(plan['qualification']))
    need(numeric['candidate'] == dict(compile_decode=True, fuse_decode=True, gates_compile=True, norm_reuse=False)
         and numeric['configuration']['max_len'] == 2048 and numeric['deadline_s'] == 1800,
         'original flags/context/deadline')
    domain.configuration(numeric, plan['concurrency'])
    serving_gate(plan['serving'])
    spec, header = capture_gate(plan)
    return plan, numeric, spec, header

def prepare(plan, numeric, spec):
    """Join unchanged real wire/trace; no candidate/runtime import here."""
    q = load(NATIVE / 'http_qualification.py', '_e024_replay_original_jobs')
    join = load(NATIVE / 'http_join.py', '_e024_replay_original_join')
    fixtures = read(pinned(numeric['fixtures']))['fixtures']
    model = read(pinned(numeric['model_manifest']))
    tokenizer_path = Path(numeric['model']) / 'tokenizer.json'
    need(sha(tokenizer_path) == model['files']['tokenizer.json'], 'original tokenizer SHA')
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    for fixture in fixtures:
        same(tokenizer.encode(fixture['prompt_text'], add_special_tokens=False).ids, fixture['prompt_ids'],
             'full actual HTTP prompt IDs: ' + fixture['fixture_id'])
    jobs = domain.jobs_for(fixtures, plan['concurrency'])
    same(jobs, spec['jobs'], 'same actual captured request bodies/order')
    group = read(pinned(plan['inputs']['group']))
    with pinned(plan['inputs']['trace']).open('rb') as trace:
        recording = join.join(group, trace, jobs, {f['fixture_id']: f for f in fixtures},
            spec['identity'], lambda ids: tokenizer.decode(ids, skip_special_tokens=False))
    return recording, fixtures

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--plan-sha256')
    parser.add_argument('--out', type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--prepare-only', action='store_true')
    modes.add_argument('--run', action='store_true')
    args = parser.parse_args(argv)
    if not (args.prepare_only or args.run):
        print('DESCRIPTION_ONLY: explicit --prepare-only or --run is required; no files or runtime loaded.')
        return 0
    need(args.plan is not None and args.plan_sha256 is not None and args.out is not None,
         'explicit mode requires plan, plan SHA and fresh output')
    args.out.mkdir(parents=True, exist_ok=False)
    result = dict(schema='GDN-graph-combined-c24-recorded-http-result-v1', status='INCOMPLETE',
        plan_sha256=args.plan_sha256, prepare_only=args.prepare_only,
        numerical_qualified=False, http_qualified=False, performance_qualified=False,
        replay_numerical_checks_passed=False, independent_audit_required=True,
        owner_resources_restore_qualified=False,
        scope='Only this captured HTTP history; original arithmetic comparisons are producer receipts. Independent audit and current outer owner admission remain required.')
    write(args.out / 'complete.json', result)
    try:
        plan, numeric, spec, header = validate(args.plan, args.plan_sha256)
        write(args.out / 'plan.json', plan)
        result.update(sources=plan['sources'], inputs=plan['inputs'], capture=plan['capture'],
            serving=plan['serving'], concurrency=plan['concurrency'])
        recording, fixtures = prepare(plan, numeric, spec)
        write(args.out / 'recording.json', recording)
        result['recording_sha256'] = sha(args.out / 'recording.json')
        if args.prepare_only:
            result['status'] = 'PREPARED'
        else:
            base = load(BASE, '_combined_route_http_private_base')
            replay = load(HERE / 'replay.py', '_c24_private_replay')
            integration = donor.load(donor.HERE / 'integration.py', '_c24_original_install')
            witness = load(ROUTE / 'serving_qualification.py', '_combined_route_http_private_witness')
            counter = load(OLD / 'qualify.py', '_combined_route_http_private_counter')
            with pinned(plan['inputs']['route_events']).open('rb') as captured, \
                 (args.out / 'replay-route-events.jsonl').open('x') as emitted:
                routes = CapturedRoutes(captured, header, emitted, plan['inputs']['route_events'],
                    f"HTTP-C{plan['concurrency']}")
                with adapted(base, replay, integration, witness, counter, recording, fixtures, routes) as coverage:
                    pair = load(HERE / 'pair.py', '_c24_actual_pair')
                    code = pair.run(base, plan['qualification'], numeric, fixtures,
                        plan['concurrency'], args.out / 'base')
                result['coverage'] = coverage
                inner = read(args.out / 'base/complete.json')
                result['base_result'] = dict(path='base/complete.json', sha256=sha(args.out / 'base/complete.json'),
                    returncode=code, status=inner['status'])
                need(code == 0 and inner['status'] == 'PASS' and inner['source_model_pins_rechecked'] is True,
                     'original full-logit/state/ahead replay did not pass')
                count = domain.request_count(plan['concurrency'])
                need(inner['total_requests'] == count and len(inner['cases']) == 1
                     and inner['cases'][0]['name'] == f"HTTP-C{plan['concurrency']}"
                     and len(inner['cases'][0]['completed']) == count, 'complete one-C actual HTTP request set')
                need(coverage['installs'] == 1 and coverage['restores'] == 2, 'fresh pair installation/release')
                stats=coverage['combined']
                need(stats['installs']==stats['restores']==1 and stats['layer_installs']==stats['layer_restores']==48,
                     'combined scope and48 layers restored')
                result['route_comparison'] = routes.finish(plan['inputs']['route_events']['sha256'])
                result.update(status='PASS', replay_numerical_checks_passed=True, total_requests=count)
            result['replay_route_events'] = dict(path='replay-route-events.jsonl',
                sha256=sha(args.out / 'replay-route-events.jsonl'))
        same(validate(args.plan, args.plan_sha256), (plan, numeric, spec, header), 'post-work source/input binding')
        result['input_source_pins_rechecked'] = True
    except BaseException as error:
        result.update(status='FAIL', replay_numerical_checks_passed=False,
            error=dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
    finally:
        write(args.out / 'complete.json', result)
    print(json.dumps(dict(status=result['status'], out=str(args.out), independent_audit_required=True)), flush=True)
    return int(result['status'] not in ('PASS', 'PREPARED'))

if __name__=='__main__':raise SystemExit(main())
