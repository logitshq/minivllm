"""Retain closed GDN-only twelve-process formal evidence, including failed prefixes.

Default inert; --run requires actual RESTORED plus independent restoration PASS.
This retains evidence and grants no eligibility. Frozen auditors run separately.
No process/service/runtime/model action or model-weight payload traversal.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
import tempfile

R=Path('/Users/ryanli/code/minivllm');B=R/'bench/m5_ultra_256gb';A=R/'.artifacts/m5-ultra-256gb'
NAME='gdn-http-formal-v1';P=B/'gdn_prework_formal/prereg-formal-v1'
OWNER=A/(NAME+'-owner');JOURNAL=A/(NAME+'-cold-journal');EXECUTION=A/(NAME+'-execution')
OUT=R/'docs/experiments/m5-ultra-256gb/E029-gdn-http-formal-v1'
PLAN_SHA='02337f0267a327ad7b025fd90ae18dec124cfe139811a767e9597c2b27af52f5'
RESTORE=B/'gdn_prework_formal/audit_restore.py';FORMAL=B/'gdn_prework_formal/audit_formal.py'
FIXED={str(RESTORE):'8ece5748f8328a110651981f40d3c7ba158194a9479052c544083d45a5621fc0',
       str(FORMAL):'70705b8ad9e2a1948306d801662302a4cdfb38a7a30370d5961c27fbbeb3f976'}
PHASES=['r1-c1-off','r1-c1-on','r1-c8-on','r1-c8-off','r2-c8-off','r2-c8-on',
        'r2-c1-on','r2-c1-off','r3-c1-off','r3-c1-on','r3-c8-on','r3-c8-off']
MAX_PART=48*(1<<20)
FORBIDDEN={'.safetensors','.bin','.pack','.so','.dylib','.metallib'}

def need(ok,message):
    if not ok:raise ValueError(message)
def read(path):
    def pairs(items):
        out={}
        for k,v in items:need(k not in out,'duplicate JSON key');out[k]=v
        return out
    return json.loads(Path(path).read_bytes(),object_pairs_hook=pairs,
        parse_constant=lambda v:(_ for _ in ()).throw(ValueError(v)))
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def info(path):
    path=Path(path);need(path.is_file() and not path.is_symlink(),'regular file '+str(path))
    key=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
    before=path.stat();h=hashlib.sha256();size=0
    with path.open('rb') as stream:
        need(key(os.fstat(stream.fileno()))==key(before),'replaced before read')
        while chunk:=stream.read(1<<20):h.update(chunk);size+=len(chunk)
        need(key(os.fstat(stream.fileno()))==key(before),'changed while read')
    need(key(path.stat())==key(before) and size==before.st_size,'changed after read')
    return dict(sha256=h.hexdigest(),bytes=size)
def sha(path):return info(path)['sha256']
def listing(root):
    need(root.is_dir() and not root.is_symlink(),'actual evidence tree '+str(root));files=[]
    for folder,dirs,names in os.walk(root,followlinks=False):
        for name in sorted(dirs+names):
            p=Path(folder)/name;need(not p.is_symlink(),'symlink refused '+str(p))
            if p.is_file():
                need(p.suffix not in FORBIDDEN,'no model/runtime/tensor payload in observer-OFF timing '+str(p));files.append(p)
            else:need(p.is_dir(),'special file '+str(p))
    return sorted(files)
class Parts(io.RawIOBase):
    def __init__(self,paths):self.paths=iter(paths);self.current=None
    def readable(self):return True
    def readinto(self,buffer):
        while True:
            if self.current is None:
                try:self.current=next(self.paths).open('rb')
                except StopIteration:return 0
            n=self.current.readinto(buffer)
            if n:return n
            self.current.close();self.current=None
    def close(self):
        if self.current is not None:self.current.close()
        super().close()
def attempts():
    return sorted(p for p in A.glob(NAME+'-analysis*') if p.is_dir()
                  and re.fullmatch(NAME+r'-analysis(?:-repeat-[1-9][0-9]*)?',p.name))

def archive(analysis):
    analysis=Path(analysis).absolute()
    need(analysis.parent==A and re.fullmatch(NAME+r'-analysis(?:-repeat-[1-9][0-9]*)?',analysis.name),'fixed selected analysis')
    need(not OUT.exists(),'exclusive archive destination; retain previous attempts')
    receipt=read(P/'receipt.json');plan=read(P/'owner-plan.json')
    final=read(EXECUTION/'final.json');journal=read(JOURNAL/'journal.json')
    need(receipt['schema']=='GDN-prework-formal-prereg-receipt-v1' and receipt['status']=='PREPARED'
         and receipt['owner_plan_sha256']==sha(P/'owner-plan.json')==PLAN_SHA,'fixed actual registered plan')
    need(final['journal_state']==journal['state']=='RESTORED' and type(final['recovery_returncode']) is int
         and final['recovery_returncode']==0,'actual driver finally-recovery must close')
    need(plan['schema']=='GDN-prework-http-formal-owner-plan-v1' and plan['trials']==PHASES==journal['trials']
         and plan['cold_restart']['journal_path']==str(JOURNAL) and journal['owner_directory']==str(OWNER)
         and journal['owner_plan']==dict(path=str(P/'owner-plan.json'),sha256=PLAN_SHA)
         and journal['owner_source']==plan['execution']['owner'],'actual plan/journal/output namespace')
    restore=read(analysis/'restoration-audit.json');summary=read(analysis/'execution-summary.json')
    need(restore['schema']=='cold-service-restoration-audit-v1' and restore['status']=='PASS' and restore['qualified'] is True
         and restore['performance_qualified'] is False and restore['numerical_qualified'] is False
         and restore['journal_sha256']==sha(JOURNAL/'journal.json') and restore['owner_plan_sha256']==PLAN_SHA
         and restore['audit_source_sha256']==FIXED[str(RESTORE)]
         and restore['controller_sha256']==journal['controller_sha256']
         and restore['service_restore']['replacement']==journal['replacement'],'independent restoration PASS required')
    need(summary['schema']=='GDN-prework-formal-closed-audits-v1' and summary['status'] in ('PASS','FAIL')
         and summary['owner_plan_sha256']==PLAN_SHA and summary['restoration_qualified'] is True
         and summary.get('ended_at'),'closed actual audit attempt')
    jobs=summary['jobs'];need('restoration' in jobs and set(jobs)<={'restoration','formal'},'actual audit jobs')
    audits={}
    for label,source,filename,middle in (
      ('restoration',RESTORE,'restoration-audit.json',['--journal',str(JOURNAL)]),
      ('formal',FORMAL,'independent-formal.json',['--owner',str(OWNER),'--journal',str(JOURNAL),'--plan-sha256',PLAN_SHA])):
        target=analysis/filename
        if label not in jobs:
            need(label=='formal' and summary['status']=='FAIL' and not target.exists(),'unattempted audit must remain absent')
            audits[label]=None;continue
        job=jobs[label]
        need(read(analysis/(label+'-execution.json'))==job and job['status'] in ('RETURNED','ERROR')
             and job['source']==dict(path=str(source),**info(source)) and sha(source)==FIXED[str(source)]
             and job['argv']==[str(R/'.venv/bin/python'),'-I','-B',str(source),*middle,'--out',str(target)]
             and job['cwd']==str(R) and job.get('ended_at'),'real frozen offline command/source')
        if target.exists():
            value=read(target);need(job['output']==dict(path=str(target),**info(target))
                and job['audit_status']==value['status'],'audit result bytes/status')
            need(value['status']!='PASS' or job.get('returncode')==0,'PASS requires actual success exit');audits[label]=value
        else:
            need(job['status']=='ERROR' or job.get('returncode')!=0,'missing successful audit output');audits[label]=None
    need(audits['restoration']==restore and jobs['restoration'].get('returncode')==0,'actual restore invocation')
    for name,item in summary['inputs'].items():
        need(info(item['path'])=={k:item[k] for k in ('sha256','bytes')},'audit input changed '+name)
        if (analysis/name).exists():need(info(analysis/name)==info(item['path']),'audit input snapshot changed')
    formal=audits['formal'];complete=read(OWNER/'complete.json') if (OWNER/'complete.json').exists() else None
    if complete is not None:need(complete['owner_plan_sha256']==PLAN_SHA and complete['planned_trials']==PHASES,'actual owner terminal')
    if formal is not None:
        need(formal['schema']=='GDN-prework-http-formal-independent-audit-v1'
             and formal['audit_source_sha256']==FIXED[str(FORMAL)] and formal['owner_directory']==str(OWNER),'actual formal audit identity')
        if formal['status']=='PASS':
            need(formal['qualified'] is formal['performance_qualified'] is True and formal['owner_plan_sha256']==PLAN_SHA
                 and formal['restoration']==restore and formal['journal_sha256']==sha(JOURNAL/'journal.json')
                 and formal['owner_complete_sha256']==sha(OWNER/'complete.json')
                 and formal['requests']==dict(total=864,warm=96,formal=768)
                 and [p['phase'] for p in formal['points']]==PHASES
                 and formal['http_raw_opened'] is True and formal['actual_prework_calls_observed'] is False,
                 'actual complete independent12 formal/resource PASS')
        else:need(formal['qualified'] is formal['performance_qualified'] is False,'failed formal audit cannot qualify')
    if summary['status']=='PASS':
        need(formal is not None and formal['status']=='PASS' and summary['execution_pass'] is True
             and summary['performance_qualified'] is summary['resources_verified_with_formal'] is True
             and summary['sources_unchanged'] is True,'closed summary PASS gates')
    else:need(summary['performance_qualified'] is False,'failed execution summary cannot grant performance')
    # Exact raw trees include all trial wire bytes, source/pre/post inventories,
    # resource commands, failed prefixes and complete restoration evidence.
    selected_attempts=attempts();need(analysis in selected_attempts,'selected attempt is present')
    trees={'owner':OWNER,'cold-journal':JOURNAL,'execution':EXECUTION,
           'cpu-prepare':A/(NAME+'-cpu-prepare'),'registration':A/(NAME+'-registration'),
           'prepared-specs':A/'gdn-http-formal-prepare-v1','preregistration':P}
    trees.update({'analysis/'+p.name:p for p in selected_attempts})
    inputs={};memberships={};missing=[]
    for prefix,root in trees.items():
        if prefix=='owner' and not root.exists():
            need(restore['owner_reap']['status']=='NO_EXPERIMENT','missing owner requires independent NO_EXPERIMENT')
            missing.append(prefix);continue
        files=listing(root);memberships[prefix]=[str(p.relative_to(root)) for p in files]
        inputs.update({prefix+'/'+str(p.relative_to(root)):p for p in files})
    # Registered small files only; never follow model manifests into weights.
    pins=dict(receipt['source_pins'])
    for path,digest in summary['source_pins'].items():
        need(path not in pins or pins[path]==digest,'registered/audit source conflict');pins[path]=digest
    for source,digest in pins.items():
        path=Path(source);need(path.is_absolute() and path.suffix in ('.py','.json','.jsonl','.md','.metal','.h')
             and path.stat().st_size<32*(1<<20) and sha(path)==digest,'small registered source changed '+source)
        inputs['registered-source/'+source.lstrip('/')]=path
        if source in summary['source_pins']:
            need(info(analysis/'frozen-source'/source.lstrip('/'))==info(path),'frozen audit source copy')
    for attempt in selected_attempts:
        if (attempt/'manifest.json').exists():
            declared=read(attempt/'manifest.json')
            need(set(declared)=={str(p.relative_to(attempt)) for p in listing(attempt) if p!=attempt/'manifest.json'},'closed attempt inventory')
            for name,pin in declared.items():need(info(attempt/name)==pin,'closed audit artifact changed')
    OUT.mkdir();(OUT/'archive.py').write_bytes(Path(__file__).read_bytes())
    write(OUT/'pack-command.json',dict(argv=sys.argv,scope='retention only; no audits/tests/device operations'))
    sidecars={'restored-journal.json':JOURNAL/'journal.json','restoration-audit.json':analysis/'restoration-audit.json',
              'audit-execution-summary.json':analysis/'execution-summary.json'}
    if (analysis/'independent-formal.json').exists():sidecars['independent-formal.json']=analysis/'independent-formal.json'
    for name,path in sidecars.items():shutil.copyfile(path,OUT/name);need(info(path)==info(OUT/name),'exact sidecar bytes')
    value=dict(schema='GDN-prework-formal-archive-summary-v1',owner_plan_sha256=PLAN_SHA,
        owner_status=complete['status'] if complete else 'NOT_STARTED',audit_status=summary['status'],
        restoration_qualified=True,performance_qualified=summary['status']=='PASS' and bool(formal and formal.get('performance_qualified')),
        planned_trials=PHASES,actual_closed_trial_names=[p for p in PHASES if (OWNER/'phases'/p/'trial/complete.json').exists()],
        independently_checked_requests=formal.get('requests') if formal else None,
        original_identity=journal['original'],replacement_identity=journal['replacement'],
        actual_prework_calls_observed=False,archive_grants_qualification=False,
        scope='Observer-OFF OFF/ON process timing with prior actual-history qualification. No proof of all future arrivals, observed formal prework calls, or combined candidate qualification.')
    write(OUT/'summary.json',value)
    (OUT/'README.md').write_text('# E029 GDN prework: paired native HTTP formal evidence\n\n'
        'This package requires independently verified restoration. Experiment, performance and retention outcomes remain separate. '
        'The registered cycle has twelve fresh servers: three OFF/ON pairs for C1 and C8, eight warm and 64 formal requests per server '
        '(96 warm plus 768 formal). Only complete independent formal audit PASS establishes performance eligibility; failures and partial prefixes are preserved. '
        'POST-to-DONE latency and committed-token common-window throughput are recomputed by the frozen auditor. '
        'The observer-OFF timing does not observe prework calls or prove all future request schedules.\n\n'
        'All actual owner/trial raw wire, logs, source/pre/post preparation, resource/reap, journal, execution, CPU preparation, registration, '
        'prepared specifications and every analysis attempt are retained. Original outer absence receipts have three fields; inner stderr evidence is '
        'a separate earlier observation. No absent outer stderr is fabricated. Model weights and runtime binaries are not included or rehashed.\n\n'
        'manifest.json maps each source path to its exact member SHA256 and size. All regular members are extracted safely to a temporary directory '
        'and verified; original bytes and directory membership are rechecked. If split, concatenate evidence.tar.xz.part-* in numeric order; '
        'archive-stream.json binds the complete stream and every part (at most48MiB). Packaging grants no new qualification.\n')
    for name in ('archive.py','pack-command.json','summary.json','README.md'):inputs['archive-metadata/'+name]=OUT/name
    for name in sidecars:inputs['archive-sidecars/'+name]=OUT/name
    manifest={name:dict(source=str(path),**info(path)) for name,path in sorted(inputs.items())}
    if formal is not None and formal['status']=='PASS':
        need(type(formal['raw_file_pins']) is dict and formal['raw_file_pins'],'complete audited wire/resource inventory')
        for name,pin in formal['raw_file_pins'].items():
            relative=PurePosixPath(name);need(not relative.is_absolute() and '..' not in relative.parts and str(relative)==name,'safe audit path')
            need('owner/'+name in manifest and {k:manifest['owner/'+name][k] for k in ('sha256','bytes')}==pin,'audited raw missing/changed '+name)
    write(OUT/'manifest.json',dict(schema='GDN-prework-formal-archive-manifest-v1',files=manifest,source_pins=pins,
        missing_unstarted_trees=missing,analysis_attempts=[str(p) for p in selected_attempts],model_payloads_included=False))
    target=OUT/'evidence.tar.xz'
    with target.open('xb') as stream,tarfile.open(fileobj=stream,mode='w:xz',preset=1,dereference=True) as tar:
        for name,path in sorted(inputs.items()):tar.add(path,arcname=name,recursive=False)
    whole=info(target);parts=[]
    if whole['bytes']>MAX_PART:
        with target.open('rb') as source:
            number=0
            while data:=source.read(MAX_PART):
                need(number<1000,'part limit');part=OUT/f'evidence.tar.xz.part-{number:03d}'
                with part.open('xb') as stream:stream.write(data)
                parts.append(part);number+=1
        h=hashlib.sha256();size=0
        for part in parts:
            with part.open('rb') as stream:
                while chunk:=stream.read(1<<20):h.update(chunk);size+=len(chunk)
        need(dict(sha256=h.hexdigest(),bytes=size)==whole,'split stream differs');target.unlink()
    else:parts=[target]
    write(OUT/'archive-stream.json',dict(archive=whole,split=len(parts)>1,max_part_bytes=MAX_PART,
        parts=[dict(path=p.name,**info(p)) for p in parts]))
    actual={}
    with tempfile.TemporaryDirectory(prefix='gdn-formal-verify-') as temp,io.BufferedReader(Parts(parts)) as joined:
        base=Path(temp).resolve()
        with tarfile.open(fileobj=joined,mode='r|xz') as tar:
            for member in tar:
                name=PurePosixPath(member.name)
                need(member.isfile() and not name.is_absolute() and '..' not in name.parts and str(name)==member.name
                     and member.name in manifest and member.name not in actual,'unsafe/unexpected archive member')
                dest=base.joinpath(*name.parts);need(dest.resolve().is_relative_to(base),'unsafe extraction path')
                dest.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(member) as src,dest.open('xb') as dst:shutil.copyfileobj(src,dst,1<<20)
                actual[member.name]=info(dest)
                need(actual[member.name]=={k:manifest[member.name][k] for k in ('sha256','bytes')},'extracted member differs')
                dest.unlink()
    need(set(actual)==set(manifest),'missing archive members')
    for name,path in inputs.items():need(info(path)==actual[name],'original changed during archive '+name)
    for prefix,names in memberships.items():need([str(p.relative_to(trees[prefix])) for p in listing(trees[prefix])]==names,'source tree changed')
    for prefix in missing:need(not trees[prefix].exists(),'unstarted tree appeared')
    need(attempts()==selected_attempts and all(info(p)['bytes']<=MAX_PART for p in parts),'analysis attempts/part bounds changed')
    verified=dict(schema='GDN-prework-formal-archive-verification-v1',status='PASS',archive=whole,
        parts=[dict(path=p.name,**info(p)) for p in parts],members=len(actual),raw_bytes=sum(v['bytes'] for v in actual.values()),
        all_safe_extracted_members_verified=True,all_original_files_rehashed=True,all_tree_members_rechecked=True,
        model_payloads_rehashed=False,runtime_payloads_rehashed=False,packaging_grants_qualification=False)
    write(OUT/'archive-verification.json',verified);print(json.dumps(dict(output=str(OUT),**verified)))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true')
    parser.add_argument('--analysis',type=Path,default=A/(NAME+'-analysis'));args=parser.parse_args()
    if args.run:archive(args.analysis)
    else:print('DESCRIPTION_ONLY: no archive; independent restoration PASS required.')
