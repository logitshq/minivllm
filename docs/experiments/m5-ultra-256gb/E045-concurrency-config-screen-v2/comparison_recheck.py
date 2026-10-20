"""Independent stdlib Base64/SSE comparison, reproducing tool 1e3468 offline.

No import of the original raw parser or auditor comparison functions; no model,
GPU, HTTP, subprocess or service operations. Does not identify mismatch cause.
"""
import base64
import hashlib
import json
from pathlib import Path
import sys

A=Path('/Users/ryanli/code/minivllm/.artifacts/m5-ultra-256gb')
LABELS=('c1-start50','c1-mb200','c1-mb16','c1-mb800','c1-end50',
        'c8-start50','c8-mb800','c8-mb16','c8-mb200','c8-end50')

def run():
    ap=A/'concurrency-config-v2-independent-audit/audit.json';audit=json.loads(ap.read_bytes())
    result=dict(status='INCOMPLETE',errors=[],raw_groups=0,raw_requests=0,
        audit_source=dict(path=str(ap),sha256=hashlib.sha256(ap.read_bytes()).hexdigest()),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        canonical_sources={str(c):f'c{c}-start50/warm-group.json' for c in (1,8)},groups=[],
        scope='Saved complete output-ID equality only. No new runtime execution, full-state or performance qualification.')
    canonical={}; bodies={}
    try:
        for label in LABELS:
            C=int(label[1]); mismatches=[]
            for phase in ('warm','formal'):
                p=A/'concurrency-config-v2-screen/trials'/label/(phase+'-group.json'); raw=p.read_bytes()
                assert hashlib.sha256(raw).hexdigest()==audit['input_sha256'][str(p)]
                group=json.loads(raw); assert group['phase']==('warm' if phase=='warm' else 'admission_probe')
                rows={r['fixture_id']:r for r in group['requests']}; assert len(rows)==len(group['requests'])==8
                if C not in bodies: bodies[C]={fid:(r['path'],r['body']) for fid,r in rows.items()}
                assert {fid:(r['path'],r['body']) for fid,r in rows.items()}==bodies[C]
                spec=json.loads((p.parent/'spec.json').read_bytes()); fixtures={f['fixture_id']:f for f in spec['fixtures']}
                observed={}
                for fid,row in rows.items():
                    data=b''.join(base64.b64decode(c['body_bytes_base64'],validate=True) for c in row['raw_sse_chunks'])
                    assert len(data)==row['raw_body_bytes'] and data.endswith(b'data: [DONE]\n\n')
                    events=[json.loads(line[6:]) for line in data.splitlines()
                            if line.startswith(b'data: ') and line!=b'data: [DONE]']
                    terms=[e for e in events if 'benchmark_output_ids' in e]; assert len(terms)==1
                    terminal=terms[0]; assert events[-1]==terminal
                    observed[fid]=terminal['benchmark_output_ids']; assert len(observed[fid])==512
                    assert observed[fid]==terminal['benchmark_committed_output_ids']
                    assert terminal['benchmark_terminal_eos_id'] is None
                    assert terminal['benchmark_prompt_ids']==fixtures[fid]['prompt_ids']
                    result['raw_requests']+=1
                if C not in canonical: canonical[C]={k:list(v) for k,v in observed.items()}
                difference=[]
                for fid,expected in canonical[C].items():
                    actual=observed[fid]
                    if actual!=expected:
                        index=next(i for i,(e,v) in enumerate(zip(expected,actual)) if e!=v)
                        difference.append(dict(phase=phase,fixture_id=fid,expected_length=len(expected),actual_length=len(actual),
                            first_index=index,expected_id=expected[index],actual_id=actual[index]))
                mismatches.extend(difference);result['raw_groups']+=1
                result['groups'].append(dict(label=label,raw_phase=group['phase'],body_path_equal_within_C=True,
                    body_path_sha256=hashlib.sha256(json.dumps(bodies[C],sort_keys=True).encode()).hexdigest(),
                    raw_source=dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest()),mismatches=difference))
            assert mismatches==audit['token_checks'][label]['mismatches']
        result.update(status='PASS_INDEPENDENT_COMPARISON_RECHECK',auditor_mismatches_exactly_reproduced=True,
            all_body_path_equal_within_C=True,prompt_ids_match_spec=True)
    except Exception as exc:
        result.update(status='FAIL');result['errors'].append(dict(type=type(exc).__name__,message=str(exc)))
    return result

if __name__=='__main__':
    assert len(sys.argv)==2,'fresh result JSON path required'
    result=run()
    with Path(sys.argv[1]).open('x') as out:json.dump(result,out,indent=2);out.write('\n')
    print(json.dumps({k:result[k] for k in ('status','errors','raw_groups','raw_requests')}))
    raise SystemExit(0 if result['status']=='PASS_INDEPENDENT_COMPARISON_RECHECK' else 1)
