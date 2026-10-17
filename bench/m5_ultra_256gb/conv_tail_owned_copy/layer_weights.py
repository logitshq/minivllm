"""CPU-only, fixed layer-0 weight preparation; default prints the contract.

--prepare --model-dir ABS --out NEW verifies the fixed first shard SHA using one
open fd, then extracts only the 14 approved tensors into weights.bin. Header,
config/index copies and manifest retain provenance. No MLX/NumPy imports here.
Execution screens read this bundle, never mx.load the original multi-GB shard.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import time

REVISION = '5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6'
PREFIX = 'language_model.model.layers.0.linear_attn.'
SHARD = 'model-00001-of-00003.safetensors'
SHARD_SHA = 'a9de796bfb40fbe757ff3e567d83d5d100906d6dd231c8b11c431cd97d3dc288'
SHARD_BYTES = 5343250458
HEADER_BYTES = 86450
HEADER_SHA = '5ee4fe36fedff37be5707fae8d414da5f4fbf11f23c74dd0c3a97d7384f497e7'
CONFIG_SHA = '7380cfc6eb77891381ee24e944e6bd0caed7a957707f2512ddd6effc82c7e706'
INDEX_SHA = '8a3f70b51e57843efcb417829dc8be081e414517995234e71fc7cf508ae01f90'
ITEM_BYTES = {'BF16': 2, 'U8': 1, 'U32': 4}
# Source-header order is irrelevant; bundle order is this fixed lexicographic list.
ROWS = (
 ('A_log', 'BF16', (48,), (3164064448,3164064544)),
 ('conv1d.weight', 'BF16', (10240,4,1), (2268569984,2268651904)),
 ('dt_bias', 'BF16', (48,), (2216994336,2216994432)),
 ('in_proj_a.scales', 'U8', (48,320), (3665864576,3665879936)),
 ('in_proj_a.weight', 'U32', (48,640), (2216994432,2217117312)),
 ('in_proj_b.scales', 'U8', (48,320), (2032936640,2032952000)),
 ('in_proj_b.weight', 'U32', (48,640), (5276869472,5276992352)),
 ('in_proj_qkv.scales', 'U8', (10240,320), (5329961568,5333238368)),
 ('in_proj_qkv.weight', 'U32', (10240,640), (2242355584,2268569984)),
 ('in_proj_z.scales', 'U8', (6144,320), (2217117312,2219083392)),
 ('in_proj_z.weight', 'U32', (6144,640), (2219083392,2234812032)),
 ('norm.weight', 'BF16', (128,), (2211423520,2211423776)),
 ('out_proj.scales', 'U8', (5120,384), (2193728800,2195694880)),
 ('out_proj.weight', 'U32', (5120,768), (2195694880,2211423520)),
)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def sha(path, check=lambda: None):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8 << 20):
            check(); h.update(block)
    check()
    return h.hexdigest()


def descriptors():
    result, cursor = [], 0
    for name, dtype, shape, offsets in ROWS:
        size = math.prod(shape) * ITEM_BYTES[dtype]
        need(offsets[1] - offsets[0] == size, 'internal source extent mismatch')
        result.append(dict(key=PREFIX + name, parameter=name.replace('conv1d.weight', 'conv1d_weight'),
                           dtype=dtype, shape=list(shape), source_offsets=list(offsets),
                           bundle_offsets=[cursor, cursor + size], bytes=size))
        cursor += size
    return result


def contract():
    return dict(schema='E026-layer-weights-v1', revision=REVISION, shard=SHARD,
                shard_sha256=SHARD_SHA, tensor_count=len(ROWS),
                bundle_bytes=sum(r['bytes'] for r in descriptors()),
                scope='Fixed real layer-0 weights; no model/device execution or D-input qualification')


def stat_identity(fd):
    st = os.fstat(fd)
    return {name: getattr(st, name) for name in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')}


def validate_header(header, index, size):
    selected = {k for k in header if k.startswith(PREFIX)}
    need(selected == {r['key'] for r in descriptors()}, 'header layer key set mismatch')
    for row in descriptors():
        h = header[row['key']]
        need(h['dtype'] == row['dtype'] and h['shape'] == row['shape']
             and h['data_offsets'] == row['source_offsets'], 'tensor source metadata mismatch: ' + row['key'])
        need(index['weight_map'].get(row['key']) == SHARD, 'tensor shard mapping mismatch')
        a, b = row['source_offsets']
        need(type(a) is int and type(b) is int and 0 <= a < b <= size - 8 - HEADER_BYTES,
             'tensor source bounds')


def prepare(args):
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    record = dict(status='INCOMPLETE', contract=contract(), model_dir=str(args.model_dir.resolve()))
    def check():
        if time.monotonic() - started > args.deadline_seconds:
            raise TimeoutError('weight preparation deadline')
    def save():
        write(args.out / 'preparation.json', record)
    save()
    try:
        config = (args.model_dir / 'config.json').read_bytes()
        index = (args.model_dir / 'model.safetensors.index.json').read_bytes()
        need(digest(config) == CONFIG_SHA and digest(index) == INDEX_SHA, 'config/index SHA mismatch')
        with (args.model_dir / SHARD).open('rb') as source:
            before = stat_identity(source.fileno())
            need(before['st_size'] == SHARD_BYTES, 'source shard size mismatch')
            h = hashlib.sha256()
            while block := source.read(8 << 20):
                check(); h.update(block)
            del block
            need(h.hexdigest() == SHARD_SHA, 'complete source shard SHA mismatch')
            record.update(full_shard_verified=True, fd_before=before); save()
            source.seek(0)
            prefix = source.read(8)
            need(len(prefix) == 8 and struct.unpack('<Q', prefix)[0] == HEADER_BYTES, 'header length mismatch')
            header_raw = source.read(HEADER_BYTES)
            need(digest(header_raw) == HEADER_SHA, 'header SHA mismatch')
            validate_header(json.loads(header_raw), json.loads(index), before['st_size'])
            rows, whole = [], hashlib.sha256()
            with (args.out / 'weights.bin').open('xb') as output:
                for expected in descriptors():
                    check()
                    source.seek(8 + HEADER_BYTES + expected['source_offsets'][0])
                    raw = source.read(expected['bytes'])
                    need(len(raw) == expected['bytes'], 'short tensor read')
                    output.write(raw); whole.update(raw)
                    rows.append(dict(expected, sha256=digest(raw)))
                    del raw
            after = stat_identity(source.fileno())
            need(after == before, 'source fd identity/size/timestamps changed while hashing/extracting')
            # Resolve pathname once more: replacement must not redirect later provenance.
            with (args.model_dir / SHARD).open('rb') as check_fd:
                need(stat_identity(check_fd.fileno()) == before, 'source pathname replaced')
        check()
        for name, raw in (('config.json', config), ('model.safetensors.index.json', index),
                          ('shard-header.json', header_raw)):
            (args.out / name).write_bytes(raw)
        result = dict(schema='E026-layer-weights-v1',
            source=dict(model_dir=str(args.model_dir.resolve()), revision=REVISION,
                        config_sha256=CONFIG_SHA, index_sha256=INDEX_SHA, shard=SHARD,
                        shard_sha256=SHARD_SHA, header_sha256=HEADER_SHA, header_bytes=HEADER_BYTES,
                        fully_hashed_before_extract=True, fd_before=before, fd_after=after),
            bundle=dict(filename='weights.bin', bytes=sum(r['bytes'] for r in rows), sha256=whole.hexdigest()),
            tensors=rows)
        write(args.out / 'manifest.json', result)
        record.update(status='PASS', manifest_sha256=sha(args.out / 'manifest.json'),
                      bundle_sha256=whole.hexdigest(), fd_after=after)
    except BaseException as error:
        record.update(status='FAIL', error_type=type(error).__name__, error=str(error))
    finally:
        record['elapsed_wall_s'] = time.monotonic() - started
        save()
    print(json.dumps(record))
    return int(record['status'] != 'PASS')


def validate_manifest(manifest):
    need(manifest.get('schema') == 'E026-layer-weights-v1', 'bundle schema')
    source = manifest['source']
    for key, expected in dict(revision=REVISION, config_sha256=CONFIG_SHA, index_sha256=INDEX_SHA,
                             shard=SHARD, shard_sha256=SHARD_SHA, header_sha256=HEADER_SHA,
                             header_bytes=HEADER_BYTES, fully_hashed_before_extract=True).items():
        need(source.get(key) == expected, 'bundle source identity: ' + key)
    need(source['fd_before'] == source['fd_after'] and source['fd_before']['st_size'] == SHARD_BYTES,
         'bundle source fd continuity')
    rows = manifest['tensors']
    need(len(rows) == len(ROWS), 'bundle tensor count')
    for actual, expected in zip(rows, descriptors()):
        need(set(actual) == set(expected) | {'sha256'}, 'tensor descriptor fields')
        need({k: actual[k] for k in expected} == expected, 'tensor descriptor mismatch')
        need(re.fullmatch('[0-9a-f]{64}', actual['sha256']) is not None, 'tensor SHA format')
    bundle = manifest['bundle']
    need(bundle['filename'] == 'weights.bin' and bundle['bytes'] == sum(r['bytes'] for r in rows), 'bundle extent')
    need(re.fullmatch('[0-9a-f]{64}', bundle['sha256']) is not None, 'bundle SHA format')


def verify_bundle(path, expected_sha, check=lambda: None):
    path = Path(path).resolve()
    need(sha(path, check) == expected_sha, 'bundle manifest SHA')
    manifest = json.loads(path.read_text())
    validate_manifest(manifest)
    for name, expected in (('config.json', CONFIG_SHA), ('model.safetensors.index.json', INDEX_SHA),
                           ('shard-header.json', HEADER_SHA)):
        need(sha(path.parent / name, check) == expected, 'bundle metadata SHA: ' + name)
    bundle = path.parent / 'weights.bin'
    need(bundle.stat().st_size == manifest['bundle']['bytes'], 'bundle file size')
    whole = hashlib.sha256()
    with bundle.open('rb') as stream:
        for row in manifest['tensors']:
            check(); raw = stream.read(row['bytes'])
            need(digest(raw) == row['sha256'], 'bundle tensor SHA: ' + row['parameter'])
            whole.update(raw)
        need(not stream.read(1), 'trailing bundle bytes')
    need(whole.hexdigest() == manifest['bundle']['sha256'], 'whole bundle SHA')
    return manifest


def load_arrays(mx, np, directory, manifest):
    """Exact frozen loader conversions, only after bundle verification by caller."""
    weights = []
    with (Path(directory) / 'weights.bin').open('rb') as stream:
        for row in manifest['tensors']:
            raw = stream.read(row['bytes'])
            need(digest(raw) == row['sha256'], 'tensor changed after bundle verification')
            dtype = {'BF16': '<u2', 'U32': '<u4', 'U8': 'u1'}[row['dtype']]
            host = np.frombuffer(raw, dtype=dtype).reshape(row['shape'])
            value = mx.array(host)
            if row['dtype'] == 'BF16':
                value = value.view(mx.bfloat16)
            name = row['parameter']
            if name == 'conv1d_weight':
                value = value.reshape(value.shape[0], -1)
            if name in ('A_log', 'dt_bias'):
                value = value.astype(mx.float32)
            elif row['dtype'] == 'BF16':
                value = value.astype(mx.bfloat16)
            weights.append((name, value))
            del raw, host
    return weights


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--deadline-seconds', type=int, default=600, choices=range(1, 901), metavar='1..900')
    args = parser.parse_args(argv)
    if not args.prepare:
        if args.out or args.model_dir:
            parser.error('--model-dir/--out require --prepare')
        print(json.dumps(contract(), indent=2)); return 0
    if args.model_dir is None or args.out is None:
        parser.error('--prepare requires --model-dir and --out')
    return prepare(args)


if __name__ == '__main__':
    raise SystemExit(main())
