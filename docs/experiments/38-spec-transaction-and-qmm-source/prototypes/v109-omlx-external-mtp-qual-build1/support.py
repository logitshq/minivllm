"""CPU-only frozen input and checkpoint-header verification."""
import hashlib
import json
from pathlib import Path
import struct

D = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def header(path):
    path = Path(path)
    with path.open('rb') as stream:
        prefix = stream.read(8)
        size = struct.unpack('<Q', prefix)[0]
        assert size < 2**24
        raw = stream.read(size)
    return {'file_bytes': path.stat().st_size, 'prefix_and_header_sha256': hashlib.sha256(prefix + raw).hexdigest(),
            'header_bytes': size, 'resolved_blob_basename': path.resolve().name}

def verify(expected):
    assert sha(D / 'pins.json') == expected, 'pins SHA'
    pins = json.loads((D / 'pins.json').read_text())
    for path, digest in pins['files'].items():
        assert sha(path) == digest, path
    config = json.loads((D / 'config.json').read_text())
    for path, expected_header in config['checkpoint_headers'].items():
        assert header(path) == expected_header, path
    return config
