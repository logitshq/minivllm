"""Default OFF: tiny CPU-only MLX safetensors filename/file-like byte smoke."""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import time
import traceback

LIMIT = 1 << 20
F_NOCACHE = 48  # Darwin SDK sys/fcntl.h; per descriptor, not global cache control.
MODES = ('filename', 'cached-filelike', 'nocache-filelike')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    need(path.stat().st_size < LIMIT, 'only small smoke files may be hashed')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2); f.write('\n')


def fixture(path):
    # Raw finite bit patterns include both signed zeros, subnormals, and extremes.
    values = {
        'half': ('F16', [0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x0400,
                         0x3c00, 0xbc00, 0x3555, 0x7bff, 0xfbff], 'H'),
        'brain': ('BF16', [0x0000, 0x8000, 0x0001, 0x8001, 0x007f, 0x0080,
                          0x3f80, 0xbf80, 0x3eab, 0x7f7f, 0xff7f, 0x4000, 0xc000], 'H'),
        'packed': ('U32', [0, 1, 0xffffffff, 0x80000000, 0x01234567,
                           0x89abcdef, 0x55555555, 0xaaaaaaaa, 0xdeadbeef], 'I'),
    }
    header, payload = {}, bytearray()
    for key, (dtype, words, fmt) in values.items():
        data = struct.pack('<' + fmt * len(words), *words)
        header[key] = dict(dtype=dtype, shape=[len(words)], data_offsets=[len(payload), len(payload)+len(data)])
        payload.extend(data)
    raw_header = json.dumps(header, separators=(',', ':')).encode()
    raw_header += b' ' * (-len(raw_header) % 8)
    raw = struct.pack('<Q', len(raw_header)) + raw_header + payload
    need(len(raw) < LIMIT, 'tiny fixture bound')
    with path.open('xb') as f:
        f.write(raw)
    return parse(path)


def parse(path):
    need(path.stat().st_size < LIMIT, 'tiny file bound')
    raw = path.read_bytes(); n = struct.unpack('<Q', raw[:8])[0]
    header = json.loads(raw[8:8+n]); out = {}
    for key, meta in header.items():
        if key == '__metadata__':
            continue
        lo, hi = meta['data_offsets']; data = raw[8+n+lo:8+n+hi]
        width = {'F16': 2, 'BF16': 2, 'U32': 4}[meta['dtype']]
        need(meta['shape'] and len(data) == width * meta['shape'][0] == hi-lo, 'tensor extent')
        out[key] = dict(dtype=meta['dtype'], shape=meta['shape'], raw=data,
                        absolute_data_offset=8+n+lo)
    return out


class Reader:
    """The actual fd used by MLX, recording IO before/after explicit closure."""
    def __init__(self, path, uncached):
        self.file = path.open('rb', buffering=0)
        self.name = str(path)
        self.events = []
        self.policy = 'ordinary-file-descriptor'
        try:
            if uncached:
                need(sys.platform == 'darwin', 'actual F_NOCACHE smoke requires Darwin')
                result = fcntl.fcntl(self.file.fileno(), F_NOCACHE, 1)
                need(result == 0, 'F_NOCACHE returned nonzero')
                self.policy = 'Darwin F_NOCACHE=48 arg=1 applied before first IO'
                self.events.append(dict(operation='fcntl', command=F_NOCACHE, argument=1, result=result))
        except BaseException:
            self.file.close()
            raise

    @property
    def closed(self):
        return self.file.closed

    def tell(self):
        return self.file.tell()

    def seek(self, *args):
        result = self.file.seek(*args)
        self.events.append(dict(operation='seek', args=args, result=result))
        return result

    def readinto(self, buffer):
        before = self.file.tell()
        count = self.file.readinto(buffer)
        self.events.append(dict(operation='readinto', offset=before, requested=len(buffer), returned=count))
        return count

    def close(self):
        self.file.close()
        self.events.append(dict(operation='explicit-close', closed=self.closed))


def one(mx, mode, source, expected, out):
    reader = None
    result = dict(mode=mode, status='FAIL')
    try:
        need(mx.default_device().type == mx.cpu, 'CPU default before load')
        if mode == 'filename':
            arrays = mx.load(str(source), format='safetensors', stream=mx.cpu)
        else:
            reader = Reader(source, mode == 'nocache-filelike')
            try:
                arrays = mx.load(reader, format='safetensors', stream=mx.cpu)
                result['load_returned_before_close'] = True
            finally:
                reader.close()
            result['fd_closed_before_eval'] = reader.closed
            event_count = len(reader.events)
        need(set(arrays) == set(expected), 'exact tensor keys')
        types = {'U32': mx.uint32, 'F16': mx.float16, 'BF16': mx.bfloat16}
        need(all(a.dtype == types[expected[k]['dtype']] and list(a.shape) == expected[k]['shape']
                 for k, a in arrays.items()), 'actual array dtype and shape')
        # No arithmetic or cast. CPU evaluation and raw safetensors roundtrip occur AFTER close.
        mx.eval(*arrays.values())
        target = out / (mode + '.safetensors')
        mx.save_safetensors(str(target), arrays)
        actual = parse(target)
        need(set(actual) == set(expected), 'roundtrip keys')
        comparisons = {}
        for key, original in expected.items():
            value = actual[key]
            exact = all(value[field] == original[field] for field in ('dtype', 'shape', 'raw'))
            finite = signed_zeros = None
            if original['dtype'] in ('F16', 'BF16'):
                words = [x[0] for x in struct.iter_unpack('<H', value['raw'])]
                mask = 0x7c00 if original['dtype'] == 'F16' else 0x7f80
                finite = all(w & mask != mask for w in words)
                signed_zeros = words[:2] == [0x0000, 0x8000]
                need(finite and signed_zeros, 'finite values and both signed zeros preserved')
            comparisons[key] = dict(dtype=value['dtype'], shape=value['shape'], bytes=len(value['raw']),
                exact_original_bytes=exact, finite=finite, signed_zeros=signed_zeros,
                payload_sha256=hashlib.sha256(value['raw']).hexdigest())
            need(exact, 'raw dtype/shape/payload mismatch: '+key)
        if reader:
            need(reader.closed and len(reader.events) == event_count, 'post-close IO occurred')
            result['post_close_io_events'] = len(reader.events)-event_count
        need(mx.default_device().type == mx.cpu, 'CPU default after verification')
        result.update(status='PASS_BYTES_ONLY', comparisons=comparisons,
                      output=dict(path=str(target), bytes=target.stat().st_size, sha256=sha(target)))
    except BaseException as exc:
        result.update(error=repr(exc), traceback=traceback.format_exc())
    finally:
        if reader:
            if not reader.closed:
                reader.close()
            result.update(fd_closed=reader.closed, policy=reader.policy, io_events=reader.events)
        write(out / (mode + '.json'), result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', action='store_true'); p.add_argument('--out', type=Path)
    args = p.parse_args(argv)
    if not args.run:
        need(args.out is None, '--out requires --run')
        print('DESCRIPTION ONLY: three tiny safetensors paths, CPU only, exact U32/F16/BF16 bytes; explicit --run required.')
        return 0
    out = args.out
    need(out and out.is_absolute() and out.parent.resolve() == out.parent and not out.exists(), 'fresh absolute output required')
    need(__name__ == '__main__', 'standalone CLI execution required')
    out.mkdir(); script = Path(__file__).resolve(); start = time.monotonic()
    write(out/'invocation.json', dict(argv=sys.argv, executable=sys.executable, source=str(script), source_sha256=sha(script),
                                    scope='tiny CPU byte/API smoke; no model, GPU, service or resource-setting changes'))
    shutil.copyfile(script, out/'smoke.py')
    summary = dict(status='FAIL', modes=list(MODES), model_qualified=False, performance_qualified=False)
    try:
        import mlx.core as mx
        mx.set_default_device(mx.cpu)  # Before the first array, load, eval or save.
        need(mx.default_device().type == mx.cpu, 'CPU device selection')
        version = importlib.metadata.version('mlx')
        need(version == '0.32.3', 'fixed actual MLX runtime')
        write(out/'runtime.json', dict(mlx_version=version, python=sys.version, platform=sys.platform,
                                      core_path=mx.__file__, default_device=str(mx.default_device())))
        source = out/'input.safetensors'; expected = fixture(source)
        write(out/'input.json', dict(bytes=source.stat().st_size, sha256=sha(source), tensors={
            k: {**{field: value for field, value in v.items() if field != 'raw'},
                'bytes': len(v['raw']), 'payload_sha256': hashlib.sha256(v['raw']).hexdigest()} for k,v in expected.items()}))
        results = [one(mx, mode, source, expected, out) for mode in MODES]
        passed = sum(r['status'] == 'PASS_BYTES_ONLY' for r in results)
        summary.update(status='PASS_BYTES_ONLY' if passed == 3 else 'FAIL', passed=passed,
                       failed=3-passed, fixture_bytes=source.stat().st_size,
                       source_unchanged=sha(script) == sha(out/'smoke.py'))
        need(summary['source_unchanged'], 'source changed during smoke')
    except BaseException as exc:
        summary.update(status='FAIL', error=repr(exc), traceback=traceback.format_exc())
    finally:
        summary['elapsed_seconds'] = time.monotonic()-start
        write(out/'summary.json', summary); print(json.dumps(summary))
    return 0 if summary['status'] == 'PASS_BYTES_ONLY' else 1


if __name__ == '__main__':
    raise SystemExit(main())
