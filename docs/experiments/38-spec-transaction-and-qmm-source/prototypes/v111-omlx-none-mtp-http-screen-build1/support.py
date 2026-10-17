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
    assert sha(D/'pins.json')==expected,'pins SHA'
    pins=json.loads((D/'pins.json').read_text())['files']
    for path,digest in pins.items():assert sha(path)==digest,path
    config=json.loads((D/'config.json').read_text())
    for path,expected_header in config['checkpoint_headers'].items():assert header(path)==expected_header,path
    q=config['qualification'];assert sha(q['receipt'])==q['receipt_sha256'] and sha(q['release'])==q['release_sha256'] and sha(q['source_pins'])==q['source_pins_sha256']
    qualified_sources=json.loads(Path(q['source_pins']).read_text())['files']
    assert len(qualified_sources)==50 and all(pins[path]==digest for path,digest in qualified_sources.items()),'qualified source subset changed'
    receipt=json.loads(Path(q['receipt']).read_text());release=json.loads(Path(q['release']).read_text())
    assert receipt['status']=='PASS_EXTERNAL_MTP_GREEDY32_RAW_TARGET_CONTROL_RESOURCES' and receipt['actual_fresh_servers']==2 and receipt['actual_requests']==6 and receipt['terminal_cleanup']
    assert receipt['ordinary_control_raw32']==config['qualified_control_raw32'] and not receipt['performance_qualified']
    assert release['status']=='PASS_EXTERNAL_RELEASE' and release['GPU_released'] and release['tool_exit']==0
    assert all(x['returncode']==1 and x['stdout']==x['stderr']=='' for x in release['strict_absence'])
    for review in config['independent_reviews']:
        assert sha(review['receipt'])==review['receipt_sha256'] and sha(review['report'])==review['report_sha256']
        r=json.loads(Path(review['receipt']).read_text());assert r['status'].startswith('PASS_')
    assert config['default_draft_block_size'] is None and config['draft_config_block_size']==3
    assert config['fixed_cooling_seconds']==20 and config['cooling_gaps']==5
    assert [x['mode'] for x in config['plan']]==['none','mtp','mtp','none','none','mtp']
    return config
