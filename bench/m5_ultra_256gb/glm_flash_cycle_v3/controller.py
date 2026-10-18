"""GLM v3 recovery: unchanged cold lifecycle, explicit dual-monitor reap proof.

The predecessor manifest is deliberately unsealed until a real Qwen cycle has
closed RESTORED with its independent audit. No observed LOAD_INTENT exception is
accepted here. This real module remains the actual recovery load-worker entry.
"""
import importlib.util as _importlib
from pathlib import Path as _Path
import hashlib as _hashlib

_reuse_path = _Path(__file__).resolve().parents[1]/'glm_flash_cycle_v2/_reuse.py'
if _hashlib.sha256(_reuse_path.read_bytes()).hexdigest() != 'fbf523c586629ce6e5e2030d629b00c9f1cd99dfec9edef6c35ac8e054468f97':
    raise ValueError('fixed namespace reuse source changed')
_spec = _importlib.spec_from_file_location('_glm_v3_lifecycle_reuse', _reuse_path)
_reuse = _importlib.module_from_spec(_spec); _spec.loader.exec_module(_reuse)
_original = _reuse.install(globals(), 'controller.py', dict(
    OWNER_SHA='9d69c5a64dc50b7d65330ad130b8af9f757b91e884aa197d1e828775b27c88a8',
    JOURNAL_SCHEMA='GLM-cold-load-journal-v3', OWNER_SCHEMA='GLM-cold-load-owner-plan-v3',
    PRIOR_PATH=_Path(__file__).resolve().parents[1]/'cold_observed_cycle/controller.py',
    PRIOR_SHA='5ef9cf1c576eac223a6e82831b443296758f6a45a9fdd8a3486c47ac5e1e8196',
    PRIOR_AUDIT_PATH=_Path(__file__).resolve().parents[1]/'cold_observed_cycle/audit_restore.py',
    PRIOR_AUDIT_SHA='f7e5e7e751b52520a47facfdb65c33db40fb4c103e8ee272b5d154607087ce56'))
BOUND_MANIFEST_SHA = 'UNSEALED'
PRIOR_LIFECYCLE = dict(path=str(PRIOR_PATH.parent/'lifecycle.json'),
    sha256='01a3befda92835ecc49ab58e3acd99c50f0077aeeed5fda96b2b95fdb99fdd88')
PRIOR_REUSE_SHA = 'b22ee6d9c505fefb3f111760096eb9d6542fc2370336ea30941ccae9baa82ecb'
_original_source_pins = source_pins
_original_phase_reaps = phase_reaps


def predecessor_manifest():
    need(type(BOUND_MANIFEST_SHA) is str and re.fullmatch('[0-9a-f]{64}', BOUND_MANIFEST_SHA),
         'actual RESTORED predecessor manifest is not sealed')
    path = HERE/'predecessor-manifest.json'
    need(not path.is_symlink() and b.sha(path) == BOUND_MANIFEST_SHA, 'sealed predecessor manifest changed')
    value = b.read_json(path)
    need(set(value) == {'schema','status','binding','trials'}
         and value['schema'] == 'GLM-v3-predecessor-binding-v1' and value['status'] == 'SEALED'
         and type(value['trials']) is list and len(value['trials']) == 1
         and type(value['trials'][0]) is str and bool(value['trials'][0]), 'actual single-child predecessor manifest')
    return value


def predecessor(binding):
    manifest = predecessor_manifest()
    need(binding == manifest['binding'] and set(binding) == {'path','sha256','audit'}, 'exact archived predecessor pair')
    refs = (binding, binding['audit'])
    for item, filename in zip(refs, ('restored-journal.json', 'restoration-audit.json')):
        path = Path(item['path'])
        need(path.is_absolute() and str(path.resolve()) == str(path) and not path.is_symlink()
             and path.name == filename and path.is_relative_to(ARCHIVE_ROOT.resolve()), 'immutable predecessor path')
    need(Path(refs[0]['path']).parent == Path(refs[1]['path']).parent, 'same predecessor archive')
    prior = b.read_json(binding['path'])
    need(prior.get('schema') == 'cold-single-child-journal-v3' and prior.get('state') == 'RESTORED'
         and prior.get('trials') == manifest['trials'], 'only closed normal Qwen predecessor; no observed bridge')
    need(b.sha(PRIOR_AUDIT_PATH) == PRIOR_AUDIT_SHA and b.sha(PRIOR_LIFECYCLE['path']) == PRIOR_LIFECYCLE['sha256']
         and b.sha(PRIOR_PATH.parent/'_reuse.py') == PRIOR_REUSE_SHA, 'prior audited lifecycle changed')
    prior_controller = load(PRIOR_PATH, PRIOR_SHA, '_glm_v3_closed_qwen_predecessor')
    # Call the frozen normal path directly. Never call its observed-failure branch.
    return prior_controller._original_predecessor(binding, PRIOR_LIFECYCLE)


def source_pins(plan):
    pins = _original_source_pins(plan); pins.update(_reuse.support_pins())
    predecessor_manifest()
    pins.update({str(HERE/'predecessor-manifest.json'):BOUND_MANIFEST_SHA,
                 PRIOR_LIFECYCLE['path']:PRIOR_LIFECYCLE['sha256'],
                 str(PRIOR_PATH.parent/'_reuse.py'):PRIOR_REUSE_SHA})
    return pins


def monitor_receipts(directory, owner_pid, child_pid):
    """Offline command ownership closure; does not grant resource qualification."""
    directory = Path(directory); complete = b.read_json(directory/'complete.json')
    need(complete.get('schema') == 'GLM-cold-load-owner-result-v3'
         and complete.get('owner_pid') == owner_pid, 'actual v3 monitor owner identity')
    state = complete.get('monitor'); need(type(state) is dict, 'explicit monitor state')
    folder = directory/'monitor'
    if not state.get('started', False):
        need(state == {} and not folder.exists(), 'unrecorded monitor activity')
        return []
    need(state.get('read_commands_settled') is True, 'unsettled monitor blocks recovery')
    path = folder/'monitor.json'; reference = state['receipt']
    need(reference == dict(path=str(path),sha256=b.sha(path)), 'exact monitor receipt SHA/path')
    result = b.read_json(path)
    need(result.get('schema') == 'GLM-dual-cadence-monitor-v1'
         and result.get('status') in ('COMPLETE','INCOMPLETE') and result['status'] == state['status']
         and result.get('owner_pid') == owner_pid and result.get('pid') == child_pid
         and result.get('read_commands_settled') is True and result.get('unjoined_workers') == []
         and result.get('unresolved_read_commands') == []
         and set(result.get('commands',{})) == {'physical','metadata'}, 'monitor/probe closure')
    files = [path]
    for lane, rows in result['commands'].items():
        need(type(rows) is list, 'command receipt sequence')
        expected = set()
        for i, row in enumerate(rows):
            stem = folder/lane/f'command-{i:04d}'
            intent, child, final = (stem.with_suffix(s) for s in ('.intent.json','.child.json','.json'))
            expected.update((intent,child,final)); files.extend((intent,child,final))
            need(all(p.is_file() and not p.is_symlink() for p in (intent,child,final)), 'original probe receipt files')
            need(b.read_json(final) == row and row.get('waited_reaped') is True
                 and row.get('unknown_launch') is False and type(row.get('pid')) is int
                 and row['pid'] > 1 and row['pid'] not in (owner_pid, child_pid)
                 and type(row.get('returncode')) is int, 'actual known read-command wait')
            original = b.read_json(intent)
            need(all(original.get(k) == row.get(k) for k in
                ('command','started_monotonic','started_at','deadline_monotonic')),
                'probe intent/terminal command/time identity')
            need(b.read_json(child) == dict(pid=row['pid'],owner_pid=owner_pid,
                 ownership='direct-Popen-returned',command=row['command']), 'probe returned Popen proof')
        actual = set((folder/lane).glob('command-*.json'))
        need(actual == expected, 'unindexed or missing probe intent/child/terminal receipt')
    return files


def phase_reaps(j, directory, attempt):
    paths, pids = _original_phase_reaps(j, directory, attempt)
    need(len(pids) == 1, 'one actual model child')
    paths.extend(monitor_receipts(directory, attempt['owner_pid'], pids[0]))
    return paths, pids


# Original lifecycle code objects, explicitly supplied this real protocol.
stop = _bound('stop', dict(authorize=authorize))
reaped = _bound('reaped', dict(registered_trials=registered_trials,phase_reaps=phase_reaps,no_popen_proof=no_popen_proof))
recover = _bound('recover', dict(authorize=authorize,reaped=reaped,monitored_load=monitored_load))

if __name__ == '__main__': raise SystemExit(main())
