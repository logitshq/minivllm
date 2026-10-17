"""Offline E028 owner audit v2: change only the pinned numerical auditor.

The v1 owner/resource/recovery predicates execute unchanged in a private module.
The v2 numerical auditor fixes the reused scheduler's free-slot stack boundary.
Original v1 source, raw evidence and failed audit receipt remain untouched.
No __file__/HERE substitution, services, model imports or device work.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
FROZEN_OWNER = HERE / 'audit_model_owner.py'
FROZEN_OWNER_SHA = 'b37901424d5bcce9b20b89b19e2cea065a56cf6138bdd6653f922d0a14bc1482'
NUMERICAL = HERE / 'audit_model_v2.py'
NUMERICAL_SHA = '3e7a0d786271c8fd25bb31e470f03b2030b12ef5bdb2ba3626d6694fb271acb0'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, digest, name):
    if sha(path) != digest:
        raise ValueError('independent auditor source changed: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configured_owner():
    """Two private bindings only; original guard functions/path identity stay intact."""
    module = load(FROZEN_OWNER, FROZEN_OWNER_SHA, '_e028_private_owner_audit_v1')
    module.numerical_auditor = lambda: load(NUMERICAL, NUMERICAL_SHA, '_e028_private_numerical_audit_v2')
    # The unchanged v1 owner audit reports this constant on PASS; bind the true
    # selected source SHA instead of allowing an obsolete v1 provenance claim.
    module.MODEL_AUDIT_SHA = NUMERICAL_SHA
    return module


def audit(owner, journal):
    try:
        result = configured_owner().audit(owner, journal)
    except Exception as error:
        result = dict(status='FAIL', qualified=False, ledger_opened=False,
            performance_qualified=False,
            first_failure={'type': type(error).__name__, 'message': str(error)})
    result.update(schema='E028-model-owner-audit-v2',
        frozen_owner_auditor_sha256=FROZEN_OWNER_SHA,
        numerical_auditor_sha256=NUMERICAL_SHA,
        adapter_scope='Only numerical auditor selection/provenance changed; frozen owner, resource, restoration and preflight predicates unchanged.')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--owner', required=True, type=Path)
    parser.add_argument('--journal', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists() or not args.out.parent.is_dir():
        raise ValueError('new audit output required; preserve the v1 failure receipt')
    result = audit(args.owner, args.journal)
    result['audit_source_sha256'] = sha(__file__)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    return {'PASS': 0, 'FAIL': 1, 'INCOMPLETE': 2}[result['status']]


if __name__ == '__main__':
    raise SystemExit(main())
