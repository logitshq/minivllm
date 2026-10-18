"""Read-only probe; never terminates apps or exposes command arguments/secrets."""
import re,subprocess,time
from datetime import datetime,timezone

def snapshot(label,owned_pid=None):
 out={'label':label,'recorded_at':datetime.now(timezone.utc).isoformat(),'monotonic_s':time.monotonic(),'owned_model_pid':owned_pid}
 for name,cmd in [('memory_pressure',['memory_pressure','-Q']),('pressure_level',['sysctl','-n','kern.memorystatus_vm_pressure_level']),('swap',['sysctl','vm.swapusage']),('vm_stat',['vm_stat']),('thermal',['pmset','-g','therm'])]:
  r=subprocess.run(cmd,text=True,capture_output=True);out[name]={'returncode':r.returncode,'text':r.stdout.strip(),'error':r.stderr.strip()}
 m=re.search(r'free percentage:\s*(\d+)',out['memory_pressure']['text']);out['free_percent']=int(m[1]) if m else None
 m=re.search(r'used = ([0-9.]+)M',out['swap']['text']);out['swap_used_mb']=float(m[1]) if m else None
 text=out['pressure_level']['text'];out['pressure_level_value']=int(text) if text.isdigit() else None
 out['vm_counters']={name:int(re.search(r'^'+name+r':\s*([0-9]+)',out['vm_stat']['text'],re.M)[1]) for name in ['Swapouts','Swapins','Pageouts','Compressions','Decompressions'] if re.search(r'^'+name+r':\s*([0-9]+)',out['vm_stat']['text'],re.M)}
 r=subprocess.run(['ps','-axo','pid=,ppid=,rss=,args='],text=True,capture_output=True);engines=[]
 if r.returncode==0:
  for line in r.stdout.splitlines():
   fields=line.strip().split(None,3)
   if len(fields)!=4:continue
   pid,ppid,rss,args=fields
   for engine,needle in [('tutorial','core_http'),('omlx','/bin/omlx serve'),('lithos','-m monolith.serve'),('minivllm','-m minivllm.server'),('mlx-lm','-m mlx_lm.server'),('lithos-cli','/lithos-metal serve'),('dflash','/dflash benchmark')]:
    if needle in args and 'resource_snapshot.py' not in args and 'paired_sweep.py' not in args:
     engines.append({'pid':int(pid),'ppid':int(ppid),'rss_kib':int(rss),'engine':engine,'owned':int(pid)==owned_pid});break
 out['process_inventory_ok']=r.returncode==0
 out['known_inference_processes']=engines;out['known_inference_process_count']=len(engines)
 out['only_owned_known_inference_processes']=all(e['owned'] for e in engines)
 out['exactly_one_owned_known_model']=len(engines)==1 and engines[0]['owned']
 out['process_scope']='known benchmark command matches only; GUI GPU use and unrecognized runtimes unknown'
 if owned_pid is not None:
  r=subprocess.run(['ps','-p',str(owned_pid),'-o','rss='],text=True,capture_output=True)
  out['owned_pid_rss_kib']=int(r.stdout.strip()) if r.returncode==0 and r.stdout.strip().isdigit() else None
  out['owned_pid_rss_scope']='exact owned PID, independent of process title; unavailable is unknown'
 return out

def deltas(samples):
 for i,s in enumerate(samples):
  if i:
   prev=samples[i-1]
   s['swap_used_mb_delta']=s['swap_used_mb']-prev['swap_used_mb'] if s['swap_used_mb'] is not None and prev['swap_used_mb'] is not None else None
   s['vm_counter_delta']={k:v-prev['vm_counters'][k] for k,v in s['vm_counters'].items() if k in prev['vm_counters']}
 return samples

def guard(samples):
 s=samples[-1]
 for key in ['pressure_level_value','free_percent','swap_used_mb']:
  if s.get(key) is None:raise RuntimeError('Required resource probe unknown: '+key)
 if not s.get('process_inventory_ok'):raise RuntimeError('Required process inventory unavailable')
 if any(s.get('vm_counters',{}).get(k) is None for k in ['Swapouts','Pageouts']):raise RuntimeError('Required paging counters unavailable')
 if s['pressure_level_value'] is not None and s['pressure_level_value']>=2:raise RuntimeError('Resource pressure warning: unload owned model and pause')
 if not s['only_owned_known_inference_processes']:raise RuntimeError('Unexpected other known inference process: unload owned model and pause')
 if len(samples)>=3:
  d=deltas(samples)
  if all(x.get('swap_used_mb_delta') is not None and x['swap_used_mb_delta']>64 for x in d[-2:]):raise RuntimeError('Sustained swap growth: unload owned model and pause')
  if all(x.get('vm_counter_delta',{}).get('Swapouts',0)>0 for x in d[-2:]):raise RuntimeError('Sustained Swapouts: unload owned model and pause')
