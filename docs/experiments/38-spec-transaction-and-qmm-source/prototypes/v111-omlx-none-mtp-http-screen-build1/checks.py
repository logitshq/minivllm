"""Pure official-log participation and paired-statistic checks; no inference imports."""
import re
import statistics

def inspect_log(text,mode,expected_requests,config,model_alias):
    assert mode in ('none','mtp') and expected_requests in (3,6)
    load_marker='VLM MTP drafter loaded: path='+config['draft_model_dir']+' kind=mtp model_type=qwen3_5_mtp'
    loads=[x for x in text.splitlines() if 'VLM MTP drafter loaded:' in x]
    scheduler_attach=[x for x in text.splitlines() if 'VLM MTP drafter attached to scheduler' in x]
    engine_attach=[x for x in text.splitlines() if 'VLM MTP drafter attached to engine:' in x]
    enabled=[x for x in text.splitlines() if 'VLM MTP enabled for ' in x]
    routes=re.findall(r'vlm_mtp decode started: request=(\S+) uid=(-?\d+) block_size=(\S+)',text)
    raw_stats=re.findall(r'vlm_mtp stats: request=(\S+) finish=(\S+) rounds=(\d+) accepted=(\d+)/(\d+) \([^)]*\) tokens_per_round=([\d.]+) emitted=(\d+) block_size=(\d+)',text)
    if mode=='none':
        assert not loads and not scheduler_attach and not engine_attach and not enabled and not routes and not raw_stats,'none control unexpectedly engaged external MTP'
        return {'mode':'none','operational_MTP_events':0,'requests_checked_by_client':expected_requests,'general_MTP_model_patch_logs_are_allowed':True}
    assert len(loads)==1 and load_marker in loads[0],'external MTP exact loading evidence absent'
    assert len(scheduler_attach)==len(engine_attach)==len(enabled)==1,'MTP attach/enable evidence absent'
    assert '(block_size=None)' in scheduler_attach[0] and ('VLM MTP drafter attached to engine: '+model_alias+' (block_size=None)') in engine_attach[0]
    assert ('VLM MTP enabled for '+config['target_revision']+', drafter='+config['draft_model_dir']) in enabled[0]
    assert not any(word in text for word in ['VLM MTP toggle on for','VLM MTP drafter load raised','vlm_mtp routing skipped for']),'MTP soft fallback/route skip observed'
    assert len(routes)==len(raw_stats)==expected_requests,'every request must have native MTP route and terminal stats'
    assert len({x[0] for x in routes})==expected_requests and [x[0] for x in routes]==[x[0] for x in raw_stats]
    assert all(int(uid)<0 and block=='None' for _,uid,block in routes)
    stats=[]
    for rid,finish,rounds,accepted,denominator,tpr,emitted,block in raw_stats:
        rounds,accepted,denominator,emitted,block=map(int,(rounds,accepted,denominator,emitted,block))
        assert finish=='length' and emitted==32 and block==3 and rounds>0 and 0<=accepted<=denominator
        assert denominator==rounds*max(1,block-1),'native log denominator formula changed'
        stats.append({'request_id':rid,'rounds_logged':rounds,'accepted_logged':accepted,'acceptance_denominator_logged':denominator,'emitted':emitted,'block_size_logged':block})
    return {'mode':'mtp','loaded_exact_path':True,'attached_scheduler_and_engine':True,'default_block_override':None,'decode_routes':expected_requests,'terminal_stats':stats,
            'measurement_observers':False,'measurement_raw_IDs_available':False,'proposal_count_from_measurement_verified':False,
            'denominator_scope':'native logger rounds*max(1,configured-or-checkpoint-block-size-1); not observed per-verify proposal count',
            'actual_verify_participation_qualification':config['qualification']['receipt_sha256']}

def summarize(arms):
    assert len(arms)==6
    pairs=[]
    for number in (1,2,3):
        rows=[r for r in arms if r['pair']==number];assert len(rows)==2 and {r['mode'] for r in rows}=={'none','mtp'}
        a=next(r for r in rows if r['mode']=='none');b=next(r for r in rows if r['mode']=='mtp')
        pairs.append({'pair':number,'order':[r['mode'] for r in rows],'none':a,'mtp':b,'x_signed_MTP_pct':100*(b['x']/a['x']-1),'y_signed_MTP_pct':100*(b['y']/a['y']-1)})
    aggregate={mode:{key:statistics.median(r[key] for r in arms if r['mode']==mode) for key in ('x','y','ttft_s','E2E_s')} for mode in ('none','mtp')}
    return {'pairs':pairs,'arm_medians':aggregate,'paired_signed_medians':{key:statistics.median(x[key] for x in pairs) for key in ('x_signed_MTP_pct','y_signed_MTP_pct')},
            'all_signed_pairs_retained':True,'none_baseline_retained':True,'negative_or_slower_arms_discarded':False,
            'scope':'fixed same-official two-configuration C1 screen, no fullfrontier/globalbest/adoption claim'}


def owned_reap_state(output):
    """Truthful saved cleanup evidence, including a failed arm outside completed[]."""
    import json
    from pathlib import Path
    recorded=[];unreaped=[]
    for pidfile in sorted(Path(output).glob('*/owned-pid.json')):
        arm=pidfile.parent;recorded.append(arm.name);exitfile=arm/'owned-process-exit.json'
        if not exitfile.exists():unreaped.append(arm.name);continue
        pid=json.loads(pidfile.read_text())['pid'];value=json.loads(exitfile.read_text());absence=value.get('exact_pid_absence',{})
        good=(value.get('pid')==pid and value.get('wait_completed') is True and absence.get('pid')==pid and absence.get('returncode')==1 and absence.get('stdout')==absence.get('stderr')=='')
        if not good:unreaped.append(arm.name)
    return {'recorded_owned_server_arms':recorded,'all_owned_servers_reaped':not unreaped,'unreaped_or_unproven_owned_server_arms':unreaped,'proof_scope':'saved exact owned wait+strict PIDabsence receipts; root additionally checks current PID/portabsence'}
