"""Small source banks; stdlib import only. No model construction or MLX imports."""
from pathlib import Path
import hashlib
import json
import math
import struct

C = 10240
CONTRACTS = {'raw': ((1, C), 'bfloat16'), 'old_tail': ((1, 3, C), 'bfloat16'),
             'w': ((C, 4), 'bfloat16'), 'g': ((1, 48), 'float32'),
             'beta': ((1, 48), 'float32'), 'S': ((1, 48, 128, 128), 'float32')}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def saved_array(directory, name, mx, np):
    directory = Path(directory)
    meta = json.loads((directory / 'bank.json').read_text())['arrays'][name]
    filename = meta['file']
    if Path(filename).name != filename:
        raise ValueError('unsafe saved bank path')
    path = directory / filename
    shape, dtype = CONTRACTS[name]
    if tuple(meta['shape']) != shape or meta['dtype'] != dtype:
        raise ValueError('saved bank shape/dtype: ' + name)
    width = 2 if dtype == 'bfloat16' else 4
    if path.stat().st_size != math.prod(shape) * width or sha(path) != meta['sha256']:
        raise ValueError('saved bank length/hash: ' + name)
    data = np.fromfile(path, dtype='<u2' if width == 2 else '<f4').reshape(shape)
    result = mx.array(data)
    return result.view(mx.bfloat16) if width == 2 else result


def checkpoint_conv(checkpoint_dir, layer, mx, np):
    """Read ONLY the BF16 convolution tensor byte range, never mx.load a shard."""
    checkpoint_dir = Path(checkpoint_dir)
    index = checkpoint_dir / 'model.safetensors.index.json'
    mapping = json.loads(index.read_text())['weight_map']
    keys = [k for k in mapping if f'.layers.{layer}.' in k and k.endswith('.conv1d.weight')]
    if len(keys) != 1:
        raise ValueError('expected one checkpoint conv weight key')
    key = keys[0]
    filename = mapping[key]
    if Path(filename).name != filename:
        raise ValueError('unsafe checkpoint shard path')
    path = checkpoint_dir / filename
    with path.open('rb') as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError('short safetensors prefix')
        n = struct.unpack('<Q', prefix)[0]
        if n > 64 << 20:
            raise ValueError('oversized safetensors header')
        header = json.loads(stream.read(n))
        tensor = header[key]
        if tensor['dtype'] != 'BF16' or tuple(tensor['shape']) not in ((C, 1, 4), (C, 4, 1), (C, 4)):
            raise ValueError('checkpoint conv tensor contract')
        start, end = tensor['data_offsets']
        if type(start) is not int or type(end) is not int or start < 0 or end - start != C * 4 * 2:
            raise ValueError('checkpoint conv offsets')
        stream.seek(8 + n + start)
        payload = stream.read(end - start)
    if len(payload) != C * 4 * 2:
        raise ValueError('short checkpoint conv tensor')
    data = np.frombuffer(payload, dtype='<u2').reshape(C, 4).copy()
    result = mx.array(data).view(mx.bfloat16)
    provenance = {'kind': 'real_checkpoint_conv_only', 'index': str(index), 'key': key,
                  'shard': str(path), 'stored_shape': tensor['shape'], 'header_bytes': n,
                  'tensor_bytes_read': len(payload), 'tensor_sha256': hashlib.sha256(payload).hexdigest(),
                  'mapping': 'reshape(C,-1), identical to accepted15 loader.py:39-40; no shard/model load'}
    return result, provenance


def synthetic(np, mx, real_w):
    """Synthetic preconv inputs; checkpoint weights used only where declared."""
    def arrays(name, tail, raw, weights, weight_kind):
        return {'name': name, 'kind': 'synthetic_preconv', 'weight_kind': weight_kind,
                'raw': mx.array(np.asarray(raw, dtype=np.float32).reshape(1, C)).astype(mx.bfloat16),
                'old_tail': mx.array(np.asarray(tail, dtype=np.float32).reshape(1, 3, C)).astype(mx.bfloat16),
                'w': weights}
    zeros = np.zeros((3, C), np.float32)
    yield arrays('zero-tail-zero-raw', zeros, np.zeros(C), real_w, 'real_checkpoint')
    channels = np.arange(C, dtype=np.float32)
    raw = ((channels % 31) - 15) / np.float32(16)
    tail = np.stack([((channels % p) - p // 2) / np.float32(32) for p in (17, 23, 29)])
    yield arrays('zero-tail-nonzero-raw', zeros, raw, real_w, 'real_checkpoint')
    yield arrays('nonzero-tail', tail, raw, real_w, 'real_checkpoint')
    ones_w = mx.ones((C, 4), mx.bfloat16)
    cancel = np.broadcast_to(np.array([256, 1, -256], np.float32)[:, None], (3, C)).copy()
    yield arrays('cancellation', cancel, np.ones(C), ones_w, 'synthetic_ones')
    rounding = np.broadcast_to(np.array([1.0078125, 1, 0], np.float32)[:, None], (3, C)).copy()
    rw = np.broadcast_to(np.array([1.0078125, -1.015625, 0, 0], np.float32), (C, 4)).copy()
    yield arrays('product-rounding', rounding, np.zeros(C), mx.array(rw).astype(mx.bfloat16), 'synthetic_rounding')
    boundary = ((channels // 128) - 39) / np.float32(64) + (channels % 128) / np.float32(4096)
    yield arrays('head-boundaries', np.stack([boundary, -boundary, boundary / 2]), -boundary, real_w, 'real_checkpoint')
