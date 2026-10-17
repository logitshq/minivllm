"""V130 fresh9-arm official Ollama queued-load slice; root-only after qualification."""
import argparse,json,signal,statistics,time
from pathlib import Path
from qualify import D,control,run_fresh,verify_pins

def qualification_gate(path,digest,release_path,release_digest,power,pins):
 assert control.sha(path)==digest and control.sha(release_path)==release_digest
 q=json.loads(Path(path).read_text());assert q['status']=='PASS_V130_OFFICIAL_OLLAMA_QUEUE_C1_C2_C4_OUTPUT_AND_CALIBRATION'
 assert q['fixed_power']==power and q['source_pins_sha256']==pins and q['HTTP_requests']==8 and q['completion_tokens']==256
 assert len(q['groups'])==4 and all(x['all_correctness_checks_passed'] and x['all_requested_POSTs_before_first_terminal'] for x in q['groups'])
 assert q['owned_exit']['wait_completed'] and q['owned_exit']['exact_pid_absence']['absent']
 release=json.loads(Path(release_path).read_text());assert release['status'].startswith('PASS')
 def absent_fields(v):
  if isinstance(v,dict):
   for k,x in v.items():
    if k in ('absent','strict_absent','strict_refused'):yield x
    yield from absent_fields(x)
  elif isinstance(v,list):
   for x in v:yield from absent_fields(x)
 checks=list(absent_fields(release));assert checks and all(x is True for x in checks)
 return {'receipt':str(path),'receipt_sha256':digest,'external_release':str(release_path),'external_release_sha256':release_digest}

def main():
 ap=argparse.ArgumentParser(description=__doc__);m=ap.add_mutually_exclusive_group(required=True);m.add_argument('--cpu-dry-run',action='store_true');m.add_argument('--run',action='store_true');ap.add_argument('--out',type=Path);ap.add_argument('--pins-sha256',required=True);ap.add_argument('--power',choices=['AC','Battery'],required=True)
 ap.add_argument('--qualification',type=Path);ap.add_argument('--qualification-sha256');ap.add_argument('--qualification-release',type=Path);ap.add_argument('--qualification-release-sha256');a=ap.parse_args()
 verify_pins(a.pins_sha256);plan=json.loads((D/'plan-nine-arms.json').read_text())['arms']
 if a.cpu_dry_run:print(json.dumps({'status':'CPU_PLAN_ONLY_QUALIFICATION_NOT_BYPASSED_FOR_RUN','fresh_arms':9,'HTTP_requests':126,'formal_requests':63,'fixed_power':a.power,'fixed_unloaded_gaps':8,'cooling_s':480,'protocol_all_arms':'official Ollama NDJSON','GPU_batch_claim':False}));return
 assert all((a.qualification,a.qualification_sha256,a.qualification_release,a.qualification_release_sha256))
 qualified=qualification_gate(a.qualification,a.qualification_sha256,a.qualification_release,a.qualification_release_sha256,a.power,a.pins_sha256)
 assert a.out and a.out.is_absolute();O=a.out;O.mkdir();control.write_new(D/'screen.attempt.consumed.json',{'out':str(O),'fixed_power':a.power,'time':time.time(),'qualification':qualified})
 samples=[];completed=[];first_error=None
 def sample(label):
  control.probe(samples,O/'campaign-resources.json',label,a.power);assert samples[-1]['processinfo_thermal_metadata']['code']==0
 try:
  sample('before-campaign')
  for i,item in enumerate(plan):
   sample('before-'+item['label'])
   r=run_fresh(O/item['label'],a.power,a.pins_sha256,screen_C=item['concurrency'])
   formal=[g for g in r['groups'] if '-measure-' in g['group_id']];assert len(formal)==3
   arm={'label':item['label'],'concurrency':item['concurrency'],'replica':item['replica'],'latency_p50_ms':statistics.median(g['latency_p50_ms'] for g in formal),'total_throughput_tok_s':statistics.median(g['total_throughput_tok_s'] for g in formal),'HTTP_requests':r['HTTP_requests'],'completion_tokens':r['completion_tokens']}
   completed.append(arm);control.write_new(O/(item['label']+'-validated.json'),arm);sample('after-owned-unload-'+item['label'])
   if i+1<len(plan):
    begin=time.monotonic();time.sleep(60);end=time.monotonic();control.write_new(O/f'cool-gap-{i+1:02d}.json',{'requested_seconds':60,'start_monotonic_s':begin,'end_monotonic_s':end,'outside_HTTP_clocks':True,'prior_owned_cleanup_passed':True,'adaptive':False});sample('after-cool-before-next')
  medians={str(C):{'latency_p50_ms':statistics.median(r['latency_p50_ms'] for r in completed if r['concurrency']==C),'total_throughput_tok_s':statistics.median(r['total_throughput_tok_s'] for r in completed if r['concurrency']==C)} for C in (1,2,4)}
  control.write_new(O/'suite-complete.json',{'status':'PASS_V130_OFFICIAL_OLLAMA_QUEUED_LOAD_SLICE_AWAIT_INDEPENDENT_AUDIT_AND_EXTERNAL_RELEASE','fixed_power':a.power,'arms':completed,'medians':medians,'HTTP_requests':126,'completion_tokens':4032,'formal_requests':63,'formal_tokens':2016,'source_pins_sha256':a.pins_sha256,'qualification':qualified,'all_same_protocol':True,'GPU_batch_claim':False,'HTTP_arrival_fairness_claim':False,'automatic_retry':False,'prior_AC_campaign_pooled':False,'full_27_arm_comparison':False,'automatic_adoption':False})
 except BaseException as exc:
  first_error={'type':type(exc).__name__,'message':str(exc),'completed_arms':completed};control.write_new(O/'suite-failure.json',first_error);raise
 finally:
  control.write_new(O/'driver-exit.json',{'error':first_error,'completed_arms':len(completed),'no_retry':True})
if __name__=='__main__':
 signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('owner SIGTERM cleanup')))
 main()
