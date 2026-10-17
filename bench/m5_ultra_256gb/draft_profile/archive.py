"""CPU-only independent reconstruction of E013 medians and immutable archive."""
import hashlib
import io
import json
from pathlib import Path
import statistics
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'.artifacts/m5-ultra-256gb'
OUT=ROOT/'docs/experiments/m5-ultra-256gb/E013-draft-profile-v2'


def read(p):return json.loads(p.read_text())
def sha(b):return hashlib.sha256(b).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')


def main():
    OUT.mkdir()
    run=BASE/'draft-profile-v2';rows=read(run/'profile.json');plan=read(run/'plan.json')
    assert len(rows)==9 and {(r['history_length'],r['block']) for r in rows}=={(n,b) for n in (32,256,1024) for b in (3,5,8)}
    assert read(run/'complete.json')['status']=='PASS'
    for r in rows:
        assert len(r['hidden_logits_kv_bit_equal'])==12 and all(r['hidden_logits_kv_bit_equal'])
        assert len(r['samples'])==49
        for name,median in r['median_ms'].items():
            samples=[s for s in r['samples'] if s['component']==name]
            assert len(samples)==7 and {s['repeat'] for s in samples}==set(range(7))
            assert all(s['total_ms']>0 and abs(s['total_ms']-s['call_ms']-s['eval_wait_ms'])<1e-8 for s in samples)
            assert statistics.median(s['total_ms'] for s in samples)==median
        assert r['head_fraction_of_full']==r['median_ms']['lm_head']/r['median_ms']['full_draft']
    files={}
    def collect(p):files[str(p.relative_to(ROOT))]=p.read_bytes()
    for source,digest in plan['source_sha256'].items():
        p=ROOT/source;assert sha(p.read_bytes())==digest;collect(p)
    for version in ('v1','v2'):
        owner=BASE/f'draft-profile-owner-{version}';lease=BASE/f'draft-profile-service-lease-{version}'
        release=read(owner/'release.json');resume=read(lease/'resume.json')
        assert release['waited_reaped'] and release['strict_absence'] and release['returncode']==(1 if version=='v1' else 0)
        assert resume['identity_match'] and 'T' not in resume['state_after']
        for folder in (owner,lease):
            for p in folder.rglob('*'):
                if p.is_file():collect(p)
        collect(BASE/f'draft-profile-jobs-{version}.json')
    resources=[read(p) for p in (BASE/'draft-profile-owner-v2').glob('resource-*.json')]
    for r in resources:
        assert r['pressure_level_value']==1 and r['processinfo_thermal_metadata']['code']==0
        assert r['no_active_foreign_native_processes']
    assert len({r['swap_used_mb'] for r in resources})==1
    assert len({r['vm_counters']['Swapouts'] for r in resources})==1
    for p in run.rglob('*'):
        if p.is_file():collect(p)
    files['failed-source/bench/m5_ultra_256gb/draft_profile/profile.py']=subprocess.check_output(['git','show','5e65885:bench/m5_ultra_256gb/draft_profile/profile.py'])
    collect(Path(__file__))
    for name in ('guarded_v2.py','resources_v2.py','suspended_service.py'):collect(ROOT/'bench/m5_ultra_256gb'/name)
    summary={'status':'PASS','cases':9,'array_checks':108,'resource_samples':len(resources),
        'previous_attempt':'Import failure from stdlib profile name collision, no model load or timing; preserved',
        'decision':'Backbone has largest component time in every case; investigate graph overhead and long-history KV copying',
        'scope':'Synchronized components are not additive; call_ms and eval_wait_ms are not separate CPU/GPU hardware timings',
        'rows':[{'history_length':r['history_length'],'block':r['block'],'median_ms':r['median_ms'],'head_fraction_of_full':r['head_fraction_of_full']} for r in rows]}
    assert all(max((n for n in r['median_ms'] if n!='full_draft'),key=r['median_ms'].get)=='backbone' for r in rows)
    assert all(r['head_fraction_of_full']<.4 for r in rows)
    write(OUT/'summary.json',summary)
    with tarfile.open(OUT/'evidence.tar.gz','w:gz') as t:
        for name,data in sorted(files.items()):
            info=tarfile.TarInfo(name);info.size=len(data);t.addfile(info,io.BytesIO(data))
    manifest={n:{'bytes':len(d),'sha256':sha(d)} for n,d in sorted(files.items())}
    with tarfile.open(OUT/'evidence.tar.gz') as t:
        assert set(t.getnames())==set(manifest)
        for name,m in manifest.items():assert sha(t.extractfile(name).read())==m['sha256']
    write(OUT/'manifest.json',{'archive_sha256':sha((OUT/'evidence.tar.gz').read_bytes()),'files':manifest})
    print(json.dumps({'status':'PASS','files':len(files),'resource_samples':len(resources),
        'head_fraction_range':[min(r['head_fraction_of_full'] for r in rows),max(r['head_fraction_of_full'] for r in rows)]}))


if __name__=='__main__':main()
