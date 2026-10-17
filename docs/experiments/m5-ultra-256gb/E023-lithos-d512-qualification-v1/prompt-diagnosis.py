"""CPU tokenizer-only reproduction; no MLX, model payload, server, or mutation."""
import builtins
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

REPO=Path('/Users/ryanli/code/minivllm')
TRIAL=REPO/'.artifacts/m5-ultra-256gb/lithos-d512-qualification-v1-owner/phases/lithos-c1/trial'
OLD=REPO/'bench/m5_ultra_256gb/lithos_d512'
NEW=REPO/'bench/m5_ultra_256gb/lithos_d512_retry'
original_import=builtins.__import__
def no_device(name,*args,**kwargs):
    if name.split('.')[0] in ('mlx','torch','tensorflow','jax'):
        raise RuntimeError('actual device import forbidden: '+name)
    return original_import(name,*args,**kwargs)
builtins.__import__=no_device
from transformers import AutoTokenizer
from importlib.metadata import version

def pin(path):
    path=Path(path);raw=path.read_bytes()
    return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))

spec=json.loads((TRIAL/'spec.json').read_text())
events=[json.loads(line) for line in (TRIAL/'probe/observer.jsonl').read_text().splitlines()]
starts=[e for e in events if e['event']=='generation_start']
returned=[e for e in events if e['event']=='generation_return']
fixtures=spec['fixtures'];by_id={f['fixture_id']:f for f in fixtures}
module_spec=importlib.util.spec_from_file_location('_retry_probe_cpu',NEW/'serve_probe.py')
probe=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(probe)
tokenizer=AutoTokenizer.from_pretrained(spec['model_directory'],local_files_only=True,trust_remote_code=False)
kw=dict(tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False)
rows=[]
for f,e,job in zip(fixtures,starts,spec['jobs']):
    corrected=tokenizer.apply_chat_template(f['messages'],**kw)
    double=tokenizer.apply_chat_template([dict(role='user',content=f['prompt_text'])],**kw)
    first=next(i for i,(a,b) in enumerate(zip(e['prompt_ids'],f['prompt_ids'])) if a!=b)
    rows.append(dict(fixture_id=f['fixture_id'],actual_length=len(e['prompt_ids']),
        expected_length=len(f['prompt_ids']),first_difference=first,
        actual_at_first=e['prompt_ids'][first],expected_at_first=f['prompt_ids'][first],
        original_messages_match_all_ids=corrected==f['prompt_ids'],
        double_template_matches_actual_all_ids=double==e['prompt_ids'],
        old_sent_body_is_double_template=job['body']['messages']==[dict(role='user',content=f['prompt_text'])],
        actual_prompt_ids=e['prompt_ids'],fixed_prompt_ids=f['prompt_ids']))
corrected_jobs=[]
for C in (1,8):
    jobs=probe.jobs_for(fixtures,C,spec['served_model'])
    checks=[]
    for job in jobs:
        f=by_id[job['fixture_id']]
        checks.append(dict(request_id=job['request_id'],fixture_id=f['fixture_id'],
            exact_messages=job['body']['messages']==f['messages'],
            all_prompt_ids_equal=tokenizer.apply_chat_template(job['body']['messages'],**kw)==f['prompt_ids']))
    corrected_jobs.append(dict(concurrency=C,requests=len(jobs),checks=checks))
paths=[TRIAL/n for n in ('spec.json','group.json','complete.json','probe/observer.jsonl')]
paths += [REPO/'bench/m5_ultra_256gb/pareto_realistic/fixtures/prepared-D-v1/fixtures.json',
          REPO/'bench/m5_ultra_256gb/pareto_realistic/prepare_fixtures.py',OLD/'serve_probe.py',OLD/'join.py',
          NEW/'serve_probe.py',NEW/'join.py',NEW/'test_cpu.py',
          Path('/Users/ryanli/code/lithos-metal/monolith/serve.py'),
          Path('/Users/ryanli/code/lithos-metal/monolith/serving/protocol.py')]
model=Path(spec['model_directory'])
raw_count=sum(len(e['generation']['tokens']) for e in returned)
eos=sum(bool(e['generation']['tokens']) and e['generation']['tokens'][-1] in (248046,248044) for e in returned)
passed=(len(rows)==8 and all(r['original_messages_match_all_ids'] and r['double_template_matches_actual_all_ids']
    and r['old_sent_body_is_double_template'] for r in rows)
    and all(x['exact_messages'] and x['all_prompt_ids_equal'] for r in corrected_jobs for x in r['checks']))
result=dict(schema='lithos-D512-double-template-cpu-diagnosis-v1',status='PASS' if passed else 'FAIL',
    cause='Already-rendered fixture.prompt_text was wrapped as user content and templated again by official /api/chat.',
    minimal_fix='Use original fixture.messages in job body and exact body gate; preserve frozen prompt_ids.',
    versions={x:version(x) for x in ('transformers','tokenizers','jinja2')},
    sources_and_artifacts=[pin(p) for p in paths],tokenizer_inputs=[pin(model/n) for n in
        ('tokenizer.json','tokenizer_config.json','chat_template.jinja')],script=pin(__file__),
    rows=rows,corrected_jobs=corrected_jobs,event_counts={k:sum(e['event']==k for e in events)
        for k in sorted({e['event'] for e in events})},
    actual_prompt_tokens_total=sum(r['actual_length'] for r in rows),fixed_prompt_tokens_total=sum(r['expected_length'] for r in rows),
    actual_raw_output_tokens=raw_count,actual_committed_output_tokens=raw_count-eos,observed_terminal_eos=eos,
    device_modules_loaded=sorted(x for x in sys.modules if x.split('.')[0] in ('mlx','torch','tensorflow','jax')),
    qualification_pass=False,performance_qualified=False,full_state_qualified=False,old_evidence_modified=False,
    attempts=[dict(chunk='1af917',status='FAIL',reason='Diagnostic MetaPathFinder intercepted torch availability lookup before tokenizer; no model/device import.'),
              dict(chunk='4881b1',status='PASS',reason='Actual import guard; all 8 official tokenizer results match corrected original IDs and double template reproduces actual IDs.'),
              dict(chunk='230f44',status='FAIL',tests_passed=17,tests_total=18,reason='Copied test referenced absent local E020 log fixture; test-only path fixed to original retained fixture.'),
              dict(chunk='99a1be',status='PASS',tests_passed=18,tests_total=18,elapsed_s=.083)],
    patch=pin('/tmp/lithos-d512-prompt-minimal.patch'))
Path('/tmp/lithos-d512-v1-prompt-diagnosis.json').write_text(json.dumps(result,indent=2)+'\n')
table='\n'.join(f"| {r['fixture_id']} | {r['actual_length']} | {r['expected_length']} | {r['first_difference']} |" for r in rows)
md=f'''# Lithos D512 v1: confirmed double-template input failure

CPU diagnostic only. The eight completed C1 requests are not qualified; C8 was
not executed. No old source, input, output, or failure receipt was modified.

The builder used rendered `fixture.prompt_text` as `/api/chat` user content.
Official `monolith/serve.py:293` applies the same chat template again with
`add_generation_prompt=True, enable_thinking=False`. Frozen fixtures were already
rendered with those exact options (`prepare_fixtures.py:53–57`). This is an input
construction bug, not a different official template default or inference result.

| Fixture | Actual IDs | Frozen IDs | First differing index (zero based) |
| --- | ---: | ---: | ---: |
{table}

All eight differ first at index3: actual248045 (`<|im_start|>`) versus expected60856
(`Implement`). Each actual prompt is11 tokens longer:7549 total versus7461.
The observer closed all34 events (open1, eight each start/return/body/wire,
sealed1). Actual raw/committed output counts are4096/4096 and observedEOS0.
These outputs cannot become fair-workload benchmark points by changing the oracle.

Official CPU tokenizer versions are transformers5.18.0, tokenizers0.23.2 and
jinja2 3.1.6. For all eight fixtures, applying the official template to original
`fixture.messages` exactly reproduces every frozen prompt ID; applying it to the
incorrect old body exactly reproduces every observed old prompt ID. New retry
`jobs_for` was also invoked for actual8/C1 and16/C8 jobs, and all24 new request
bodies reproduce the frozen IDs exactly. No MLX/Torch/JAX/TensorFlow import.

Minimal proposed patch changes only job messages and the exact join body check;
the original full prompt-ID gate and all route/count/cache/EOS gates remain.
Retry regression suite18/18 PASS. Its first run passed17/18 and failed only on
the copied saved-log fixture's relative location; only the test's path changed.
The earlier diagnostic guard failed before tokenization on a package-availability
lookup; the actual-import guard then succeeded. All attempts are retained in JSON.

Artifacts: `/tmp/lithos-d512-v1-prompt-diagnosis.py`, `.json`, this `.md`, and
`/tmp/lithos-d512-prompt-minimal.patch`. JSON pins original raw artifacts, original
and retry sources, tokenizer inputs, versions, full actual/frozen prompt IDs,
and the diagnostic script itself. No large model payload was read.
'''
Path('/tmp/lithos-d512-v1-prompt-diagnosis.md').write_text(md)
print(json.dumps(dict(status=result['status'],actual_prompt_tokens=result['actual_prompt_tokens_total'],
    fixed_prompt_tokens=result['fixed_prompt_tokens_total'],corrected_jobs=sum(r['requests'] for r in corrected_jobs),
    device_modules_loaded=result['device_modules_loaded'])))
raise SystemExit(0 if passed else 1)
