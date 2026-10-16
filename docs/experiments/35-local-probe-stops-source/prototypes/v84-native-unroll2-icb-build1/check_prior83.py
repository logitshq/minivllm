"""CPU-only exact prior83 source + actual qualification binding, not generic result-only reuse."""
from pathlib import Path
import hashlib,json
D=Path(__file__).resolve().parent;P=D.parent;H=P/'v83-nvfp4-kloop-unroll2-build1';R=P/'v23-root-control-build1/numeric-v83-attempt1';O=R/'child-output'
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
j=lambda p:json.loads(p.read_text())
def validate():
 manifest=j(H/'source-manifest.json');pins=j(H/'source-pins.json')
 for n,s in manifest['files'].items():assert h(H/n)==s,('prior83file',n)
 for p,s in pins.items():assert h(Path(p))==s,('prior83pin',p)
 for row in manifest['upstream_sources'].values():assert h(Path(row['path']))==row['sha256']
 identity=j(O/'identity.json');owner=j(R/'owner-receipt.json');child=j(O/'child.json');cleanup=j(O/'cleanup.json');release=j(R/'external-release.json');q=j(O/'qualification.json');result=j(O/'result.json');checks=j(O/'bytes.json')
 assert identity['driver_sha256']==owner['driver_sha256']==h(H/'micro.py')
 assert identity['prototype_manifest_sha256']==h(H/'source-manifest.json') and identity['protocol_sha256']==h(H/'protocol.json')
 assert owner['waited_reaped'] and owner['returncode']==0 and owner['stage']=='reaped' and owner['absence']['absent']
 assert str(H/'micro.py') in owner['command'] and '--qualify-only' in owner['command']
 assert child['owner_pid']==owner['driver_pid'] and str(H/'micro.py') in child['argv'] and '--qualify-only' in child['argv']
 assert cleanup['pid']==child['pid'] and cleanup['returncode']==0 and cleanup['wait_completed'] and cleanup['exact_pid_absence']['absent']
 assert j(O/'child-finally.json')=={'exiting':True,'pid':child['pid']}
 assert release['status']=='PASS_EXTERNAL_RELEASE' and release['GPU_released'] and release['tool_exit']==0
 assert {r['pid'] for r in release['strict_absence']}=={owner['owner_pid'],owner['driver_pid'],child['pid']}
 assert all(r['returncode']==1 and not r['stdout'] for r in release['strict_absence'])
 assert result['status']=='STRICT-NUMERIC-QUALIFICATION-PASS' and result['checks']==36 and result['all_bytes_exact'] and result['readonly'] and result['timing_run'] is False and result['adopted'] is False
 assert q['status']=='STRICT-SG2-R4-AND-COMPILED-UNROLL2-R4-SG2-QMV-BYTES-PASS' and q['checks']==36 and q['readonly'] and q['same_immutable_weight_handles'] and q['both_arms_mx_compile'] and q['timing_allowed'] is False
 assert q['candidate_UNROLL2-R4-SG2_grid']==[64,2176,1] and q['candidate_threadgroup']==[64,1,1]
 assert q['trace_counts_after_cold']==q['trace_counts_after_qualification']=={'A':1,'B':1}
 assert all(r['before']==r['after'] for r in j(O/'readonly.json'))
 assert len(checks)==36 and all(r['exact'] and r['finite'] and r['differing_bytes']==0 and r['left_sha256']==r['right_sha256'] for r in checks)
 header=(H/'kernel_header.metal').read_text();pragma='#pragma clang loop unroll_count(2)\n'
 assert header.count(pragma)==1 and header.replace(pragma,'')==(P/'v66-gdn-four-projection-build2/kernel_header.metal').read_text()
 assert 'fp4pair' not in header and 'half2' not in header
 return {'status':'PASS_CPU_SOURCE_BOUND_PRIOR83_36_CHECKS_AND_RELEASE','driver_sha256':h(H/'micro.py'),'manifest_sha256':h(H/'source-manifest.json'),'identity_sha256':h(O/'identity.json'),'qualification_sha256':h(O/'qualification.json'),'result_sha256':h(O/'result.json'),'release_sha256':h(R/'external-release.json'),'result_SHA_alone_sufficient':False,'GPU_used':False}
if __name__=='__main__':print(json.dumps(validate(),sort_keys=True))
