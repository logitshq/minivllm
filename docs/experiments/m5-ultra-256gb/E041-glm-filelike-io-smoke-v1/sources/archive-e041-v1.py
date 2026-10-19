import hashlib, json, shutil, struct
from pathlib import Path
R=Path('/Users/ryanli/code/minivllm')
A=R/'.artifacts/m5-ultra-256gb'
D=R/'docs/experiments/m5-ultra-256gb/E041-glm-filelike-io-smoke-v1'
D.mkdir()
origins={}
def sha(p):
    assert p.is_file() and not p.is_symlink() and p.stat().st_size < 1<<20
    return hashlib.sha256(p.read_bytes()).hexdigest()
def copy(src, rel):
    dst=D/rel; dst.parent.mkdir(parents=True,exist_ok=True)
    assert not dst.exists() and not src.is_symlink() and src.is_file() and src.stat().st_size < 1<<20
    shutil.copyfile(src,dst); assert sha(src)==sha(dst)
    origins[rel]=str(src)
for name, folder in [('actual','glm-filelike-io-smoke-v1'),('execution','glm-filelike-io-smoke-execution-v1')]:
    for src in sorted((A/folder).iterdir()):
        copy(src,name+'/'+src.name)
copy(R/'bench/m5_ultra_256gb/glm_filelike_io_smoke_v1/smoke.py','sources/smoke.py')
copy(R/'bench/reviews/2026-10-19-glm-filelike-nocache.md','sources/prior-source-review.md')
copy(Path('/tmp/run-glm-filelike-io-e041-v1.py'),'sources/run-e041-v1.py')
copy(Path(__file__),'sources/archive-e041-v1.py')
# Independent stdlib-only parsing of the copied, actual tiny outputs.
def tensors(path):
    raw=path.read_bytes(); n=struct.unpack('<Q',raw[:8])[0]; h=json.loads(raw[8:8+n])
    return {k:(v['dtype'],v['shape'],raw[8+n+v['data_offsets'][0]:8+n+v['data_offsets'][1]]) for k,v in h.items() if k!='__metadata__'}
original=tensors(D/'actual/input.safetensors')
assert set(original)=={'half','brain','packed'}
checks={}
for mode in ('filename','cached-filelike','nocache-filelike'):
    assert tensors(D/'actual'/f'{mode}.safetensors') == original
    receipt=json.loads((D/'actual'/f'{mode}.json').read_text())
    assert receipt['status']=='PASS_BYTES_ONLY'
    if mode != 'filename':
        assert receipt['fd_closed_before_eval'] and receipt['load_returned_before_close'] and receipt['post_close_io_events']==0
        assert receipt['io_events'][-1]['operation']=='explicit-close'
    if mode == 'nocache-filelike':
        assert receipt['io_events'][0]==dict(operation='fcntl',command=48,argument=1,result=0)
    checks[mode]='exact keys/dtype/shape/payload PASS'
summary=json.loads((D/'actual/summary.json').read_text()); assert summary['passed']==3 and summary['failed']==0
execution=json.loads((D/'execution/execution.json').read_text()); assert all(x['returncode']==0 for x in execution['attempts'])
(D/'independent-byte-check.json').write_text(json.dumps(dict(status='PASS',method='stdlib reparse copied input and three output safetensors; exact keys/dtype/shape/payload; filelike closure receipts',modes=checks),indent=2)+'\n')
(D/'README.md').write_text('''# E041 — actual CPU-only file-like IO byte smoke

Result: **PASS_BYTES_ONLY, 3/3 paths, zero failed attempts**. The default command
returned description only (exit 0); the single explicit actual run returned exit 0.
Actual MLX 0.32.3 / Python 3.13.16 on Darwin used `Device(cpu, 0)`; the script sets
CPU before any array/load/eval and passes CPU explicitly to each load. No model,
GPU workload, service action, memory/cache limit, or global cache control was used.

The manually encoded input is **276 bytes** with 84 payload bytes: U32[9],
F16[11], BF16[13]. Float fixtures include positive and negative zero, finite
normal/subnormal values and maximum finite magnitudes. The raw tensor offsets
(192, 214, 240) and small final tensor exercise reads not aligned to pages.

| Actual input path | Raw dtype/shape/payload | FD closed before later eval/save | IO after close |
| --- | --- | --- | --- |
| Filename | PASS | Reader belongs to MLX | Not instrumented |
| Cached binary file-like | PASS | Yes | 0 events |
| F_NOCACHE binary file-like | PASS | Yes | 0 events |

The uncached path used the **same descriptor passed into MLX**, with Darwin
`fcntl(fd, 48, 1)` returning 0 before its first read. Both file-like loads returned
before their descriptors were closed. CPU eval and a safetensors save then
succeeded, and all original tensor bytes were identical, including signed zeros.
No arithmetic or dtype cast was used for the comparison. An independent stdlib
parse rechecked the archived input against all three actual outputs.

This qualifies this tiny CPU API/byte path and its descriptor lifetime only.
It does not prove large-shard behavior, bounded total memory, less swap, full-model
loader success, performance, or Pareto improvement. File-like MLX safetensors
loads evaluate before returning; ordinary filename loading is lazy, so a future
cached-file-like versus uncached-file-like comparison is needed to isolate the
cache-policy hypothesis. The previous source investigation is copied under
`sources/` with official MLX v0.32.3 links.

## Reproduction and evidence

Source: `bench/m5_ultra_256gb/glm_filelike_io_smoke_v1/smoke.py`. Default is OFF.
`execution/execution.json` retains both exact commands, true exits and elapsed
wall times. `actual/` contains the actual runtime, invocation, per-mode IO events,
original fixture and tiny output files. `sources/` retains the executed source,
runner and archive script. All attempt outputs are retained; no failed attempt
occurred. `manifest.json` and `verification.json` verify each copied member
against its original and re-read archive member, without model payloads.
''')
members=[]
for p in sorted(D.rglob('*')):
    if p.is_file():
        rel=p.relative_to(D).as_posix()
        members.append(dict(path=rel,bytes=p.stat().st_size,sha256=sha(p),original=origins.get(rel)))
manifest=dict(schema='small-evidence-members-v1',members=members)
(D/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for m in members:
    assert sha(D/m['path'])==m['sha256']
    if m['original']:
        assert sha(Path(m['original']))==m['sha256']
verification=dict(status='PASS',member_count=len(members),copied_member_count=len(origins),bytes=sum(m['bytes'] for m in members),manifest_sha256=sha(D/'manifest.json'),all_members_reread=True,all_copied_originals_reread=True,independent_original_bytes_pass=True)
(D/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
print(json.dumps(dict(archive=str(D),**verification,script_sha256=sha(D/'sources/smoke.py'))))
