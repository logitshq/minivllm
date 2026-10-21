"""Independent stdlib-only raw safetensors byte auditor for E058 feature pairs.

Reads a manifest of schema E058-feature-pairs-v1 naming 24 reference/candidate
safetensors pairs (labels T{T}/k{k}/{phase} for T in (4,8), k=1..T, phase in
(selected, continuation)). Every referenced file must be a relative path inside
the audit root, a regular file without symlinks, with an exact SHA256 match.

Each safetensors file must contain exactly the 130 E058 arrays (logits,
features, state/{i}/K,V for i%4==3 else state/{i}/conv,S) with fixed dtype and
shape per phase, a well-formed header (8-byte little-endian length + JSON, no
duplicate keys), and a payload whose data_offsets tile it exactly with no gaps
or overlaps. The single reserved top-level key "__metadata__" is not a tensor:
it may be absent, null, or an object mapping string keys to string values, and
is skipped when enumerating the 130 arrays (any other metadata type or
non-string value is rejected). BF16/F32 payload bytes are scanned for nonfinite
bit patterns. Reference and candidate bytes must be identical per array (so
signed-zero differences fail). Any violation raises ValueError; audit()
returns a PASS report only after every pair checks out.
"""
import argparse
import hashlib
import json
import os
import stat
import struct
import sys

try:
    import numpy as _np
except ImportError:  # pragma: no cover - exercised only on stdlib-only hosts
    _np = None

SCHEMA = 'E058-feature-pairs-v1'

DTYPES = {'BF16': 2, 'F32': 4}

WIDTHS = (4, 8)
PHASES = ('selected', 'continuation')
NUM_LAYERS = 64
LOGITS_COLS = 248320
FEATURES_COLS = 25600
KV_SHAPE = (1, 4, 1536, 256)
CONV_SHAPE = (1, 3, 10240)
S_SHAPE = (1, 48, 128, 128)


def expected_labels():
    labels = []
    for width in WIDTHS:
        for k in range(1, width + 1):
            for phase in PHASES:
                labels.append('T{}/k{}/{}'.format(width, k, phase))
    return tuple(labels)


def parse_label(label):
    """Return (width, k, phase) for a required label, else raise ValueError."""
    if not isinstance(label, str):
        raise ValueError('malformed label: {!r}'.format(label))
    parts = label.split('/')
    if len(parts) != 3:
        raise ValueError('malformed label: {!r}'.format(label))
    tok_w, tok_k, phase = parts
    if (len(tok_w) != 2 or tok_w[0] != 'T' or tok_w[1] not in '48' or
            len(tok_k) != 2 or tok_k[0] != 'k' or
            tok_k[1] not in '12345678' or phase not in PHASES):
        raise ValueError('malformed label: {!r}'.format(label))
    width = int(tok_w[1])
    k = int(tok_k[1])
    if width not in WIDTHS or not (1 <= k <= width):
        raise ValueError('malformed label: {!r}'.format(label))
    return width, k, phase


def expected_arrays(width, k, phase):
    """Return {name: (dtype, shape)} for the EXACT 130 E058 array names."""
    rows = k if phase == 'selected' else 1
    spec = {
        'logits': ('BF16', (rows, LOGITS_COLS)),
        'features': ('BF16', (rows, FEATURES_COLS)),
    }
    for i in range(NUM_LAYERS):
        if i % 4 == 3:
            spec['state/{}/K'.format(i)] = ('BF16', KV_SHAPE)
            spec['state/{}/V'.format(i)] = ('BF16', KV_SHAPE)
        else:
            spec['state/{}/conv'.format(i)] = ('BF16', CONV_SHAPE)
            spec['state/{}/S'.format(i)] = ('F32', S_SHAPE)
    expected = 2 + 2 * NUM_LAYERS
    if NUM_LAYERS == 64 and expected != 130:
        raise AssertionError('internal contract drift: {}'.format(expected))
    if len(spec) != expected:
        raise AssertionError('internal contract drift: {}'.format(len(spec)))
    return spec


def _numel(shape):
    total = 1
    for dim in shape:
        if not (isinstance(dim, int) and not isinstance(dim, bool) and dim >= 1):
            raise ValueError('invalid shape dimension: {!r}'.format(dim))
        total *= dim
    return total


def _bf16_nonfinite_scalar(payload):
    for (bits,) in struct.iter_unpack('<H', payload):
        if bits & 0x7F80 == 0x7F80:
            return True
    return False


def _f32_nonfinite_scalar(payload):
    for (bits,) in struct.iter_unpack('<I', payload):
        if bits & 0x7F800000 == 0x7F800000:
            return True
    return False


def _bf16_nonfinite_numpy(payload):
    words = _np.frombuffer(payload, dtype='<u2')
    return bool(_np.any((words & 0x7F80) == 0x7F80))


def _f32_nonfinite_numpy(payload):
    words = _np.frombuffer(payload, dtype='<u4')
    return bool(_np.any((words & 0x7F800000) == 0x7F800000))


if _np is not None:
    _bf16_nonfinite = _bf16_nonfinite_numpy
    _f32_nonfinite = _f32_nonfinite_numpy
else:
    _bf16_nonfinite = _bf16_nonfinite_scalar
    _f32_nonfinite = _f32_nonfinite_scalar


def parse_safetensors(data):
    """Parse raw safetensors bytes into {name: (dtype, shape, payload)}.

    Rejects malformed headers, duplicate keys, unknown dtypes, invalid shapes
    or lengths, and offset ranges that overlap, gap or miss the payload end.
    Nonfinite BF16/F32 bit patterns in any payload are rejected.
    """
    if not isinstance(data, (bytes, bytearray)) or len(data) < 8:
        raise ValueError('safetensors blob too short')
    (header_len,) = struct.unpack('<Q', data[:8])
    if header_len < 2 or 8 + header_len > len(data):
        raise ValueError('malformed header length: {}'.format(header_len))
    try:
        header_text = data[8:8 + header_len].decode('utf-8')
    except UnicodeDecodeError:
        raise ValueError('header is not utf-8')
    stripped = header_text.strip()
    if not (stripped.startswith('{') and stripped.endswith('}')):
        raise ValueError('header is not a JSON object')
    def hook(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError('duplicate header key: {!r}'.format(key))
            obj[key] = value
        return obj
    try:
        header = json.loads(header_text, object_pairs_hook=hook)
    except json.JSONDecodeError as exc:
        raise ValueError('malformed header JSON: {}'.format(exc))
    if not isinstance(header, dict):
        raise ValueError('header is not a JSON object')
    payload_base = 8 + header_len
    payload_size = len(data) - payload_base
    tensors = []
    total_bytes = 0
    for name, meta in header.items():
        if name == '__metadata__':
            if meta is None:
                continue
            if not isinstance(meta, dict):
                raise ValueError(
                    'invalid __metadata__ type: {!r}'.format(type(meta)))
            for mkey, mval in meta.items():
                if not isinstance(mkey, str) or not isinstance(mval, str):
                    raise ValueError(
                        '__metadata__ entries must map strings to strings')
            continue
        if not isinstance(meta, dict):
            raise ValueError('malformed tensor metadata for {!r}'.format(name))
        dtype = meta.get('dtype')
        if dtype not in DTYPES:
            raise ValueError('invalid dtype for {!r}: {!r}'.format(name, dtype))
        shape = meta.get('shape')
        if not isinstance(shape, list) or not shape:
            raise ValueError('invalid shape for {!r}: {!r}'.format(name, shape))
        count = _numel(shape)
        offsets = meta.get('data_offsets')
        if not (isinstance(offsets, list) and len(offsets) == 2 and
                all(isinstance(o, int) and not isinstance(o, bool)
                    for o in offsets)):
            raise ValueError('malformed data_offsets for {!r}'.format(name))
        begin, end = offsets
        if not (0 <= begin <= end):
            raise ValueError('invalid data_offsets for {!r}'.format(name))
        if end - begin != count * DTYPES[dtype]:
            raise ValueError('byte length mismatch for {!r}: {} != {}'.format(
                name, end - begin, count * DTYPES[dtype]))
        if set(meta) != {'dtype', 'shape', 'data_offsets'}:
            raise ValueError('unexpected metadata keys for {!r}'.format(name))
        tensors.append((name, dtype, tuple(shape), begin, end))
        total_bytes += end - begin
    tensors.sort(key=lambda t: (t[3], t[0]))
    cursor = 0
    for name, dtype, shape, begin, end in tensors:
        if begin != cursor:
            raise ValueError(
                'overlapping or gapped payload offsets at {!r}'.format(name))
        cursor = end
    if cursor != payload_size:
        raise ValueError('payload offsets do not cover the data buffer')
    if total_bytes != payload_size:
        raise ValueError('payload length mismatch')
    out = {}
    for name, dtype, shape, begin, end in tensors:
        payload = bytes(data[payload_base + begin:payload_base + end])
        if dtype == 'BF16':
            if _bf16_nonfinite(payload):
                raise ValueError('nonfinite BF16 values in {!r}'.format(name))
        elif dtype == 'F32':
            if _f32_nonfinite(payload):
                raise ValueError('nonfinite F32 values in {!r}'.format(name))
        out[name] = (dtype, shape, payload)
    return out


def compare_arrays(expected_spec, name, ref, cand):
    """Validate one array against the contract and byte-compare the pair.

    expected_spec maps name -> (dtype, shape). ref/cand are the parsed
    (dtype, shape, payload) tuples. Returns the payload byte count checked.
    """
    need_dtype, need_shape = expected_spec[name]
    for side, (dtype, shape, payload) in (('reference', ref),
                                          ('candidate', cand)):
        if dtype != need_dtype:
            raise ValueError('{} {} dtype {} != {}'.format(
                side, name, dtype, need_dtype))
        if shape != need_shape:
            raise ValueError('{} {} shape {} != {}'.format(
                side, name, shape, need_shape))
    if ref[2] != cand[2]:
        raise ValueError('array bytes differ for {}'.format(name))
    return len(ref[2])


def _resolve_member(root, rel_path):
    """Resolve a manifest member path safely inside root; return abs path."""
    if not isinstance(rel_path, str) or not rel_path:
        raise ValueError('member path must be a nonempty string')
    if os.path.isabs(rel_path) or '\\' in rel_path or rel_path.startswith('~'):
        raise ValueError('member path must be relative: {!r}'.format(rel_path))
    parts = rel_path.split('/')
    if any(p in ('', '.', '..') for p in parts):
        raise ValueError('member path has bad components: {!r}'.format(rel_path))
    root_real = os.path.realpath(root)
    current = root_real
    for part in parts:
        current = os.path.join(current, part)
    try:
        st = os.lstat(current)
    except FileNotFoundError:
        raise ValueError('member path does not exist: {!r}'.format(rel_path))
    for depth in range(len(parts)):
        prefix = os.path.join(root_real, *parts[:depth + 1])
        try:
            st = os.lstat(prefix)
        except FileNotFoundError:
            raise ValueError(
                'member path does not exist: {!r}'.format(rel_path))
        if stat.S_ISLNK(st.st_mode):
            raise ValueError(
                'member path crosses a symlink: {!r}'.format(rel_path))
    if not stat.S_ISREG(st.st_mode):
        raise ValueError('member path is not a regular file: {!r}'.format(rel_path))
    real = os.path.realpath(current)
    if real != current:
        raise ValueError('member path escapes root: {!r}'.format(rel_path))
    return current


def _load_member(root, side, info):
    if not isinstance(info, dict) or set(info) != {'path', 'sha256'}:
        raise ValueError('malformed {} member: {!r}'.format(side, info))
    sha = info['sha256']
    if not (isinstance(sha, str) and len(sha) == 64 and
            all(c in '0123456789abcdef' for c in sha)):
        raise ValueError('malformed sha256 for {}: {!r}'.format(side, sha))
    path = _resolve_member(root, info['path'])
    with open(path, 'rb') as handle:
        data = handle.read()
    actual = hashlib.sha256(data).hexdigest()
    if actual != sha:
        raise ValueError('sha256 mismatch for {} {}: {} != {}'.format(
            side, info['path'], actual, sha))
    return info['path'], parse_safetensors(data), len(data)


def _load_manifest(manifest):
    if isinstance(manifest, (str, os.PathLike)):
        with open(manifest, 'rb') as handle:
            try:
                manifest = json.loads(handle.read().decode('utf-8'))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError('manifest is not valid JSON: {}'.format(exc))
    if not isinstance(manifest, dict):
        raise ValueError('manifest must be a JSON object')
    if set(manifest) != {'schema', 'pairs'}:
        raise ValueError(
            'manifest must contain only schema and pairs fields')
    if manifest.get('schema') != SCHEMA:
        raise ValueError('manifest schema must be {!r}'.format(SCHEMA))
    pairs = manifest.get('pairs')
    if not isinstance(pairs, list):
        raise ValueError('manifest pairs must be a list')
    return pairs


def _contract_check(label, arrays, spec):
    names = set(arrays)
    expected_names = set(spec)
    if names != expected_names:
        raise ValueError('{} arrays != expected 130: missing={} extra={}'.format(
            label, sorted(expected_names - names)[:4],
            sorted(names - expected_names)[:4]))


def audit(root, manifest):
    """Audit all 24 E058 safetensors pairs; return a PASS report dict.

    Raises ValueError on any invalid evidence: bad manifest, unsafe path,
    SHA256 mismatch, malformed safetensors, contract drift, nonfinite values
    or byte differences between reference and candidate arrays.
    """
    pairs = _load_manifest(manifest)
    labels = [p.get('label') if isinstance(p, dict) else None for p in pairs]
    if labels != list(expected_labels()):
        raise ValueError(
            'manifest labels must be exactly the 24 required labels in order')
    checked_pairs = []
    total_bytes = 0
    total_arrays = 0
    for pair in pairs:
        label = pair['label']
        if not isinstance(pair, dict) or set(pair) != {
                'label', 'reference', 'candidate'}:
            raise ValueError('malformed pair record for {!r}'.format(label))
        width, k, phase = parse_label(label)
        spec = expected_arrays(width, k, phase)
        pair_bytes = 0
        parsed = {}
        for side in ('reference', 'candidate'):
            rel_path, arrays, file_bytes = _load_member(root, side, pair[side])
            _contract_check(label, arrays, spec)
            parsed[side] = arrays
            pair_bytes += file_bytes
        for name in spec:
            total_arrays += 1
            compare_arrays(spec, name, parsed['reference'][name],
                           parsed['candidate'][name])
        total_bytes += pair_bytes
        checked_pairs.append({
            'label': label,
            'arrays': len(spec),
            'bytes': pair_bytes,
            'reference': pair['reference']['path'],
            'candidate': pair['candidate']['path'],
        })
    return {
        'schema': SCHEMA,
        'status': 'PASS',
        'pairs_checked': len(checked_pairs),
        'arrays_checked': total_arrays,
        'bytes_checked': total_bytes,
        'pairs': checked_pairs,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Independent raw safetensors byte auditor for E058 pairs.')
    parser.add_argument('--root', required=True,
                        help='audit root containing all member files')
    parser.add_argument('--manifest', required=True,
                        help='path to the E058-feature-pairs-v1 manifest JSON')
    parser.add_argument('--out',
                        help='optional path to write the JSON report')
    args = parser.parse_args(argv)
    report = audit(args.root, args.manifest)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, 'w') as handle:
            handle.write(text + '\n')
    print(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
