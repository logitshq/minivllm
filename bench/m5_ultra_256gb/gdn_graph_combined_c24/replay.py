"""Replay a joined native HTTP trace using two independently executing engines.

Input is http_join.join()'s E023-recorded-http-v1 dictionary. Only arrivals,
prompt IDs and caps are inputs to the engines; recorded schedules/tokens are
comparison targets, never scheduler replacements or sampled-token inputs.
The caller supplies the source-pinned qualify_native module as ``base`` and
owns model/config/source validation, the absolute alarm, Observer restoration
and pair destruction. This new domain changes only admission/coverage; its full replay math is frozen.
This module imports only stdlib and starts no device.

PASS covers this recorded schedule's numerical replay, not HTTP performance or
unrecorded formal arrivals. A failure writes the partial current step and is
re-raised; the outer qualifier retains its failure/cleanup evidence. Successful
steps drop Observer's current snapshots, retaining only its actual future for
the next semantic-key comparison. No observer is installed/restored here.
"""
import json


def run_recorded_case(recording, engines, observers, sampling, runner_module,
                      cmp, ledger, deadline, *, base):
    """Finite actual-engine replay; engine/observer order is reference, candidate."""
    if recording.get('schema') != 'E023-recorded-http-v1' or len(engines) != 2 or len(observers) != 2:
        raise ValueError('joined HTTP recording and exactly two engines/observers required')
    jobs, steps, concurrency = recording['jobs'], recording['steps'], recording['concurrency']
    if type(concurrency) is not int or concurrency not in (2, 4) or not jobs or not steps:
        raise ValueError('nonempty finite C2/C4 recording required')
    for rid, job in jobs.items():
        if (job['rid'] != rid or not job['prompt_ids'] or type(job['cap']) is not int
                or job['cap'] <= 0):
            raise ValueError('invalid joined request input')
    limit = sum(len(j['prompt_ids']) + j['cap'] for j in jobs.values()) + 1
    if len(steps) > limit:
        raise ValueError('recording exceeds the finite progress step bound')
    for engine, observer in zip(engines, observers):
        if observer.engine is not engine:
            raise ValueError('Observer must belong to its independently executing engine')
        base.assert_empty(engine, cmp)
        cmp.meta('recorded concurrency', engine.scheduler.max_seqs, concurrency)
        cmp.meta('recorded EOS set', sorted(engine.eos_ids), recording['eos_token_ids'])
    if engines[0] is engines[1] or engines[0].runner is engines[1].runner:
        raise ValueError('replay requires independent engines and runners')
    name = f'HTTP-C{concurrency}'
    reference, candidate = observers
    active, admitted, completed, delivered, admissions, histogram = set(), set(), [], {}, [], {}
    previous_decode, tail_continuation = [], False
    for number, expected in enumerate(steps, 1):
        row = {'case': name, 'step': number, 'status': 'RUNNING', 'recorded': expected}
        cmp.rows = []
        try:
            deadline()
            cmp.meta('recorded step order', expected['step'], number)
            for rid in expected['arrivals']:
                if rid not in jobs or rid in admitted:
                    cmp.fail('unknown or repeated HTTP admission', {'rid': rid})
                job = jobs[rid]
                for engine in engines:
                    actual_rid = engine.add_request(list(job['prompt_ids']), sampling(max_tokens=job['cap'],
                        temperature=0.0, ignore_eos=False), req_id=rid)
                    cmp.meta('HTTP admission returned rid', actual_rid, rid)
                admitted.add(rid); active.add(rid); delivered[rid] = []
                admissions.append({'rid': rid, 'fixture_id': job['fixture_id'],
                    'client_request_id': job['client_request_id'], 'before_step': number, 'cap': job['cap']})
            cmp.compare('before-step-state', base.state(engines[0]), base.state(engines[1]))
            for arm, engine in zip(('reference', 'candidate'), engines):
                cmp.meta(arm + '/recorded before', {
                    'running': [r.req_id for r in engine.scheduler.running],
                    'waiting': [r.req_id for r in engine.scheduler.waiting]}, expected['before'])
            before_progress = [(r.req_id, r.num_computed_tokens, len(r.output_token_ids))
                               for r in engines[0].requests.values()]
            reference.begin()
            discarded = base.ahead_semantic_key(engines[0].runner._ahead)
            engines[0].runner._ahead = None  # Previous ahead has already been compared.
            deadline()
            with base.grouping_disabled(runner_module):
                left = engines[0].step()
            row.update(schedule=reference.schedule, reference_discarded_ahead=base.plain(discarded),
                       reference_delivered=base.plain(left))
            cmp.meta('reference/recorded schedule',
                     {k: reference.schedule[k] for k in ('decode', 'prefill', 'finished_ids')}, expected['schedule'])
            candidate.begin(reference.schedule)  # Checks its actual schedule before candidate model work.
            deadline()
            right = engines[1].step()
            row.update(candidate_schedule=candidate.schedule, candidate_delivered=base.plain(right))
            cmp.meta('candidate/recorded schedule',
                     {k: candidate.schedule[k] for k in ('decode', 'prefill', 'finished_ids')}, expected['schedule'])
            current_decode = list(reference.schedule['decode'])
            if len(previous_decode) > 1 and len(current_decode) == 1 and current_decode[0] in previous_decode:
                tail_continuation = True
            previous_decode = current_decode
            B = len(current_decode)
            histogram[str(B)] = histogram.get(str(B), 0) + 1
            cmp.meta('delivered tuples', left, right)
            cmp.meta('recorded delivered tuples', left, expected['results'])
            row['call_keys'] = base.pair_calls(reference, candidate, cmp)
            if reference.postrun is None or candidate.postrun is None:
                cmp.fail('step with work lacked runner observation', {})
            for arm, observer in zip(('reference', 'candidate'), observers):
                cmp.meta(arm + '/recorded postrun rows',
                         observer.postrun['metadata']['engine']['batch_rids'], expected['postrun_decode_rows'])
            cmp.compare('before-retirement-state', reference.postrun, candidate.postrun)
            cmp.compare('after-retirement-state', base.state(engines[0]), base.state(engines[1]))
            row['after'] = base.state(engines[0])['metadata']
            cmp.meta('recorded retirement', [rid for rid, _, done in left if done], expected['retired'])
            for arm, engine in zip(('reference', 'candidate'), engines):
                cmp.meta(arm + '/recorded surviving rows', list(engine.runner.batch.req_ids), expected['surviving_decode_rows'])
            for rid, token, done in left:
                if rid not in active:
                    cmp.fail('output for inactive/retired HTTP rid', {'rid': rid})
                ids, job = delivered[rid], jobs[rid]
                ids.append(token)
                terminal = token in engines[0].eos_ids or len(ids) == job['cap']
                if len(ids) > job['cap'] or any(x in engines[0].eos_ids for x in ids[:-1]) or done is not terminal:
                    cmp.fail('HTTP output violates normal EOS/cap retirement', {'rid': rid, 'ids': ids, 'done': done})
                if done:
                    active.remove(rid)
                    committed = ids[:-1] if token in engines[0].eos_ids else list(ids)
                    cmp.meta('final HTTP raw IDs ' + rid, ids, job['raw_ids'])
                    cmp.meta('final HTTP committed IDs ' + rid, committed, job['committed_ids'])
                    completed.append({'rid': rid, 'raw_ids': list(ids), 'committed_ids': committed})
            after_progress = [(r.req_id, r.num_computed_tokens, len(r.output_token_ids))
                              for r in engines[0].requests.values()]
            if engines[0].has_work() and before_progress == after_progress and not left:
                cmp.fail('work made no computed/output progress', {'step': number})
            row['ahead'] = {'reference': reference.finish(), 'candidate': candidate.finish()}
            deadline()
            row['status'] = 'PASS'
        except BaseException as exc:
            row['error'] = {'type': type(exc).__name__, 'message': str(exc)}
            raise
        finally:
            row['arrays'] = cmp.rows
            if row['status'] != 'PASS':
                row['failure'] = cmp.first_failure
            ledger.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            ledger.flush()
            if row['status'] == 'PASS':
                for observer in observers:
                    observer.calls.clear()
                    observer.postrun = None
    deadline()
    for engine in engines:
        base.assert_empty(engine, cmp)
    cmp.meta('final pending retire buffer', engines[0].scheduler._finished_since_last_step,
             engines[1].scheduler._finished_since_last_step)
    cmp.meta('all HTTP jobs admitted', sorted(admitted), sorted(jobs))
    cmp.meta('all HTTP jobs completed', sorted(x['rid'] for x in completed), sorted(jobs))
    cmp.meta('no HTTP jobs remain active', sorted(active), [])
    cmp.meta('recorded decode histogram', histogram, recording['histogram'])
    if max(map(int, histogram), default=0) != concurrency or not tail_continuation:
        cmp.fail('C2/C4 requires full B=C and continuing multirow-to-B1 tail coverage', histogram)
    return {'name': name, 'status': 'PASS', 'steps': len(steps), 'step_limit': limit,
            'admissions': admissions, 'completed': completed, 'decode_batch_histogram': histogram,
            'raw_output_count': sum(len(x['raw_ids']) for x in completed),
            'committed_output_count': sum(len(x['committed_ids']) for x in completed),
            'scope': 'Exact recorded-arrival numerical replay only; not future formal schedules.',
            'performance_qualified': False,
            'domain_coverage': {'full_batch': concurrency, 'continuing_tail_B1': tail_continuation}}
