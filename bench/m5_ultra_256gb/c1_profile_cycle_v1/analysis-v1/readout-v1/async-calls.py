"""Refine existing accepted profile readout by sequential async_eval call ordinal."""
import hashlib, importlib.util, json
from collections import Counter, defaultdict
from pathlib import Path
H=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('readout_helpers',H/'analyze.py'); a=importlib.util.module_from_spec(s); s.loader.exec_module(a)
r=a.read(H/'readout.json'); result=dict(schema='C1-async-call-ordinal-readout-v1',performance_qualified=False,modes={})
for mode in ('runner','full'):
    p=a.read(a.O/'phases'/mode/'trial/server/profile.json'); rids={x['rid'] for x in r['modes'][mode]['requests']}
    steps={x['step']:x for x in p['events'] if x['kind']=='runner_step' and set(x['rids']+x['prefill_rids']) & rids}
    calls=defaultdict(list); grouped=defaultdict(list); counts=Counter()
    for event in p['events']:
        if event['kind']=='existing_async_eval' and event.get('step') in steps: calls[event['step']].append(event)
    for step,event in steps.items():
        group='prefill' if event['prefill_rids'] else 'ahead_consumed' if event['ahead_consumed'] else 'current_required'
        ordered=sorted(calls[step],key=lambda x:x['start_ns']); counts[group+'_calls_'+str(len(ordered))]+=1
        for ordinal,call in enumerate(ordered,1): grouped[group+'_ordinal_'+str(ordinal)].append((call['end_ns']-call['start_ns'])/1e6)
    result['modes'][mode]=dict(step_counts=dict(counts),call_ms={k:a.stats(v) for k,v in sorted(grouped.items())},
        service_seconds=r['modes'][mode]['profile']['local_service_s'])
result.update(input_files=a.INPUTS,scope='Sequential call ordinal inside an actual runner step, not GPU-time attribution. Candidate ahead_submit_once only targets the first already-submitted call on an eligible ahead hit; no claim that second-call waiting can be deleted.')
with (H/'async-calls.json').open('x') as f: json.dump(result,f,indent=2,allow_nan=False); f.write('\n')
print(json.dumps(result['modes'],indent=2))
