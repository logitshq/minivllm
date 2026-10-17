"""CPU inventory for the two-phase Lithos qualification owner.

Inherits exact existing Lithos/source/model/pack hashes, checking file stats only.
Hashes the active outer HTTP/tokenizer runtime and dependency distributions without
importing those packages. Execution owner independently hashes every payload twice.
No service/model/GPU imports, downloads, repacking, or package installation.
"""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OLD=HERE.parent/'external_spec_http/prereg-v1/plan.json'
OLD_SHA='e3f0c3e24d4ea841d5e6c0b0094652e65e463c02367836e765ef6d9e3c6e2bdc'
NUMERIC=HERE.parent/'pareto_realistic/prereg-qualification-v1/plan.json'
NUMERIC_SHA='21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb'
LITHOS=Path('/Users/ryanli/code/lithos-metal')


def need(ok, message):
    if not ok: raise ValueError(message)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(1<<20): h.update(chunk)
    return h.hexdigest()


def read_pinned(path, expected):
    raw=Path(path).read_bytes()
    need(hashlib.sha256(raw).hexdigest()==expected, 'frozen inventory changed: '+str(path))
    return json.loads(raw)


def stat(path):
    s=Path(path).stat()
    return dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns)


def runtime_distributions():
    # Packaging is CPU-only; distribution metadata traversal ignores optional extras.
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    pending=['httpx','httpcore','transformers','tokenizers','packaging']
    found={}
    while pending:
        name=canonicalize_name(pending.pop())
        if name in found: continue
        dist=metadata.distribution(name)
        found[name]=dist
        for raw in dist.requires or []:
            req=Requirement(raw)
            if req.marker is None or req.marker.evaluate({'extra':''}):
                dependency=metadata.distribution(req.name)
                need(not req.specifier or dependency.version in req.specifier,
                     'installed dependency violates metadata: '+raw)
                pending.append(req.name)
    return found


def prepare(out):
    need(not out.exists() and out.parent.is_dir(), 'fresh output in existing directory')
    old=read_pinned(OLD,OLD_SHA); numeric=read_pinned(NUMERIC,NUMERIC_SHA)
    model=read_pinned(**dict(path=numeric['model_manifest']['path'],expected=numeric['model_manifest']['sha256']))
    groups={name:{} for name in ('lithos_runtime','lithos_source','target_model','pack','outer_runtime')}
    def inherit(group,path,expected):
        need(path in old['pin_stats'] and old['pins'][path]==expected,'missing inherited stat/hash: '+path)
        before=stat(path)
        need(before==old['pin_stats'][path], 'inherited file stat changed: '+path)
        groups[group][path]=dict(sha256=expected,**before)
    pack=old['packs']['lithos-none']['target']['path']
    for path,sha in old['pins'].items():
        if path.startswith(str(LITHOS/'.venv')+'/'): inherit('lithos_runtime',path,sha)
        elif path.startswith(str(LITHOS)+'/'): inherit('lithos_source',path,sha)
        elif path.startswith(pack+'/'): inherit('pack',path,sha)
    python=str((LITHOS/'.venv/bin/python').resolve())
    inherit('lithos_runtime',python,old['pins'][python])
    for name,sha in model['files'].items(): inherit('target_model',str(Path(model['model'])/name),sha)
    distributions=runtime_distributions(); files={Path(sys.executable).resolve()}
    for dist in distributions.values():
        for file in dist.files or []:
            path=Path(dist.locate_file(file)).resolve()
            if path.is_file() and str(file).endswith(('.py','.so','.dylib','.metallib','METADATA')): files.add(path)
    for path in sorted(files):
        before=stat(path); value=digest(path)
        need(stat(path)==before,'runtime changed while hashing: '+str(path))
        if str(path) in old['pins']: need(value==old['pins'][str(path)],'previously pinned runtime changed: '+str(path))
        groups['outer_runtime'][str(path)]=dict(sha256=value,**before)
    # Recheck inherited metadata after runtime hashing too; no full payload claim.
    for group,rows in groups.items():
        for path,record in rows.items(): need(stat(path)=={k:record[k] for k in ('bytes','mtime_ns')},'inventory changed: '+path)
    out.mkdir()
    def save(name,value):
        (out/name).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    save('verification.json',dict(python=str(Path(sys.executable).absolute()),groups=groups))
    save('receipt.json',dict(schema='Lithos-D512-verification-prepare-v1',status='PREPARED',
        source_sha256=digest(__file__),e020_sha256=OLD_SHA,numerical_plan_sha256=NUMERIC_SHA,
        verification_sha256=digest(out/'verification.json'),
        counts={k:len(v) for k,v in groups.items()},bytes_by_group={k:sum(r['bytes'] for r in v.values()) for k,v in groups.items()},
        outer_versions={n:d.version for n,d in sorted(distributions.items())},
        inherited_payloads_rehashed=False,
        scope='Inherited payload SHA/stat inventory, current outer runtime SHA. Execution owner must hash all payloads before/after.'))
    print(json.dumps(dict(status='PREPARED',counts={k:len(v) for k,v in groups.items()},verification_sha256=digest(out/'verification.json'))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();prepare(args.out.resolve())
