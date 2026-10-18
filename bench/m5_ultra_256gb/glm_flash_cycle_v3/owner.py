"""GLM owner v3: two observation cadences, one unchanged direct model child.

No backend/service action on import. Explicit --run and a separately reviewed
adjacent controller are required. Frozen v1 preparation, spawn, reap, postflight
and physical gates are reused in this real module namespace. Only the explicit
source substitutions below change their control flow; donor files/globals stay
unchanged. This is not the old checkpoint-per-physical-sample resource contract.
"""
import ast as _ast
import hashlib as _hashlib
import importlib.util as _importlib
import inspect as _inspect
from pathlib import Path as _Path

_reuse_path = _Path(__file__).resolve().parents[1]/'glm_flash_cycle_v2/_reuse.py'
_reuse_sha = 'fbf523c586629ce6e5e2030d629b00c9f1cd99dfec9edef6c35ac8e054468f97'
if _hashlib.sha256(_reuse_path.read_bytes()).hexdigest() != _reuse_sha:
    raise ValueError('fixed namespace reuse source changed')
_spec = _importlib.spec_from_file_location('_glm_v3_existing_code_reuse', _reuse_path)
_reuse = _importlib.module_from_spec(_spec); _spec.loader.exec_module(_reuse)
_original = _reuse.install(globals(), 'owner.py', dict(SCHEMA='GLM-cold-load-owner-plan-v3'))

MONITOR_PATH = HERE.parent/'glm_flash/live_monitor.py'
MONITOR_SHA = '10f9f09108257cd95808d98b3392c516ecd5208e567299f7e67e128453e2e9f9'
MONITOR_POLICY = dict(schema='GLM-dual-cadence-policy-v1', physical_max_gap_seconds=1.0,
    metadata_max_gap_seconds=5.0, callback_seconds=5.0, target_gap_fraction=.5,
    join_seconds=7.0, metadata_per_physical_sample=False)
_monitor = module(MONITOR_PATH, MONITOR_SHA)
_original_source_pins = source_pins


def source_pins(plan):
    pins = _original_source_pins(plan)
    pins.update(_reuse.support_pins()); pins[str(MONITOR_PATH)] = MONITOR_SHA
    need(sha(MONITOR_PATH) == MONITOR_SHA, 'dual monitor source changed')
    return pins


def _live(args, plan, guard, child, identity, prior, owner_pid, cold, child_started,
          began, live, checkpoints, state, check, write):
    """Retain both frozen validators; only their scheduling is independent."""
    state.update(started=True, read_commands_settled=False)

    def physical(index, commands, deadline):
        check()
        # Instance-local injection; shared System/subprocess modules are untouched.
        local = System(); local.read = commands.read
        need(local.identity(child.pid) == identity, 'child identity changed before physical sample')
        remaining = min(plan['physical_policy']['probe_seconds'], deadline-time.monotonic())
        need(remaining > 0, 'physical callback deadline before measure')
        row = guard.measure(args.out/f'physical-{index:04d}', child.pid,
                            timeout_seconds=remaining, run=commands.run)
        guard.observe(row, plan['physical_policy'], cold, child_pid=child.pid,
                      child_started_monotonic=child_started, previous=live[-1] if live else None)
        live.append(row)
        # An already-exited unreadable PID does not become a successful sample.
        need(local.identity(child.pid) == identity, 'child identity changed after physical sample')
        check()
        return dict(sample=ref(args.out/f'physical-{index:04d}'/'sample.json'), observed_within_budget=True)

    def metadata(index, commands, deadline):
        check()
        def retain(row):
            row.update(label='running-metadata', lane_sample_index=index)
            write(args.out/f'checkpoint-{len(checkpoints):04d}.json', row); checkpoints.append(row)
        checkpoint(commands, old_pid=prior['replacement']['pid'], child_pid=child.pid,
                   owner_pid=owner_pid, deadline=deadline, record=retain)
        check()
        return dict(checkpoint=ref(args.out/f'checkpoint-{len(checkpoints)-1:04d}.json'))

    try:
        result = _monitor.monitor(child, out=args.out/'monitor', physical=physical, metadata=metadata,
            work_deadline=min(began+1830, child_started+plan['physical_policy']['work_seconds']),
            physical_interval=MONITOR_POLICY['physical_max_gap_seconds'],
            metadata_interval=MONITOR_POLICY['metadata_max_gap_seconds'],
            probe_seconds=MONITOR_POLICY['callback_seconds'], join_seconds=MONITOR_POLICY['join_seconds'])
    except _monitor.MonitorError as error:
        state.update(read_commands_settled=error.report['read_commands_settled'],
                     receipt=ref(args.out/'monitor/monitor.json'), status=error.report['status'])
        raise
    else:
        state.update(read_commands_settled=result['read_commands_settled'],
                     receipt=ref(args.out/'monitor/monitor.json'), status=result['status'])


def monitor_release_allowed(state):
    # Even if the model child was reaped, an unknown/unsettled probe blocks the
    # canonical all-children release. Failed-but-settled observations can restore.
    return not state.get('started', False) or state.get('read_commands_settled') is True


def _variant(name, replacements):
    """Compile only a SHA-bound original function with enumerated exact deltas."""
    source = _inspect.getsource(getattr(_original, name))
    for before, after in replacements:
        need(source.count(before) == 1, 'frozen function delta anchor changed: '+name)
        source = source.replace(before, after, 1)
    tree = _ast.parse(source)
    need(len(tree.body) == 1 and isinstance(tree.body[0], _ast.FunctionDef)
         and tree.body[0].name == name, 'one explicit owner function variant')
    exec(compile(tree, __file__, 'exec'), globals())
    return source


_schema_delta = ("'GLM-cold-load-journal-v1'", "'GLM-cold-load-journal-v3'")
_validate_source = _variant('validate_plan', [
    ("'physical_policy','cold_restart','outputs','budgets'", "'physical_policy','cold_restart','outputs','budgets','monitor_policy'"),
    _schema_delta])
_validated_original_contract = validate_plan


def validate_plan(args):
    plan, probe, guard, prior = _validated_original_contract(args)
    need(plan['monitor_policy'] == MONITOR_POLICY, 'explicit independent physical/metadata cadence required')
    need(plan['physical_policy']['interval_seconds'] == 1.0
         and plan['physical_policy']['probe_seconds'] == 5.0
         and plan['physical_policy']['child_limit_bytes'] == 220*(1 << 30),
         'unchanged one-second physical gap/five-second probe/220GiB child ceiling required')
    return plan, probe, guard, prior


_stopped_source = _variant('stopped', [_schema_delta])
_old_loop = '''            while child.poll() is None:
                check();need(system.identity(child.pid)==ci,'child identity changed')
                row=guard.measure(args.out/f'physical-{len(live):04d}',child.pid,timeout_seconds=plan['physical_policy']['probe_seconds'])
                guard.observe(row,plan['physical_policy'],cold,child_pid=child.pid,child_started_monotonic=child_started,previous=live[-1] if live else None)
                live.append(row);point('running',child.pid)
                need(child.poll() is not None or system.identity(child.pid)==ci,'child identity changed after sample')
                remaining=min(1830-(time.monotonic()-began),plan['physical_policy']['work_seconds']-(time.monotonic()-child_started))
                need(remaining>0,'child elapsed deadline')
                gap_deadline=row['ended_monotonic']+plan['physical_policy']['interval_seconds']
                need(time.monotonic()<=gap_deadline,'resource checkpoint exceeded observation cadence')
                try:child.wait(timeout=min(0.1,remaining,max(0,gap_deadline-time.monotonic())))
                except subprocess.TimeoutExpired:pass'''
_new_loop = '''            _live(args,plan,guard,child,ci,prior,owner_pid,cold,child_started,
                  began,live,checkpoints,monitor_state,check,write)'''
RUN_DELTAS = [
    ('    binding = attempt = None;', '    monitor_state = {}\n    binding = attempt = None;'),
    (_old_loop, _new_loop),
    ("elif len(children)==1 and children[0]['release'] is not None:",
     "elif len(children)==1 and children[0]['release'] is not None and monitor_release_allowed(monitor_state):"),
    ("schema='GLM-cold-load-owner-result-v1'", "schema='GLM-cold-load-owner-result-v3'"),
    ("live_physical_samples=len(live),checkpoints=len(checkpoints),diagnostic=ref(args.out/'diagnostic.json'),",
     "live_physical_samples=len(live),checkpoints=len(checkpoints),diagnostic=ref(args.out/'diagnostic.json'),\n"
     "                monitor=monitor_state,monitor_policy=plan['monitor_policy'],\n"
     "                metadata_live_samples=sum(r.get('label')=='running-metadata' for r in checkpoints),\n"
     "                legacy_checkpoint_per_sample_contract=False,")]
_run_source = _variant('run', RUN_DELTAS)


if __name__ == '__main__':
    raise SystemExit(main())
