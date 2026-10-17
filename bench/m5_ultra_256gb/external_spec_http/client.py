"""C1 adaptation of lithos/client.py, retaining partial and failed SSE requests.

Only the external HTTP client's clock measures latency. Server logs and previous
observer timings never enter the performance numerators or denominators.
"""
import json
import signal
import time
import traceback


def body_for(arm, model, prompt, user_text):
    if arm.startswith('lithos-'):
        return '/v1/chat/completions', {
            'model': model, 'messages': [{'role': 'user', 'content': user_text}],
            'temperature': 0, 'top_p': 1, 'max_completion_tokens': 32,
            'stream': True, 'stream_options': {'include_usage': True}}
    return '/v1/completions', {
        'model': model, 'prompt': prompt, 'temperature': 0, 'repetition_penalty': 1.0,
        'max_tokens': 32, 'stream': True, 'stream_options': {'include_usage': True}}


def finish_record(row, tok):
    """CPU-only finalization, also used by pure fixture tests."""
    text = ''.join(event['text'] for event in row['text_events'])
    ids = tok.encode(text, add_special_tokens=False).ids
    first_piece = row['text_events'][0]['text'] if row['text_events'] else ''
    first_ids = tok.encode(first_piece, add_special_tokens=False).ids
    usage = row.get('usage') or {}
    cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens')
    first, end, start = (row.get(key) for key in
                         ('first_text_perf_counter_s', 'done_perf_counter_s', 'start_perf_counter_s'))
    prefix = bool(first_piece) and ids[:len(first_ids)] == first_ids and text.startswith(first_piece)
    wall = end - start if end is not None else None
    count = usage.get('completion_tokens')
    remaining = count - len(first_ids) if isinstance(count, int) else None
    row.update(text=text, visible_roundtrip_ids=ids, actual_visible_canonical_count=len(ids),
        actual_usage_output_count=count, cached_prompt_tokens_when_available=cached,
        cache_usage_field_present=cached is not None, first_chunk_text=first_piece,
        first_chunk_reconstructed_tokens=len(first_ids), first_chunk_prefix_verified=prefix,
        full_visible_roundtrip_count_matches_usage=len(ids) == count,
        client_total_s=wall, client_ttft_text_s=first - start if first is not None else None,
        client_post_first_chunk_s=end - first if end is not None and first is not None else None,
        actual_request_count=1, group_window_s=wall, total_completion_tokens=count,
        output_tok_s=count / wall if wall and isinstance(count, int) else None,
        aggregate_output_tok_s=count / wall if wall and isinstance(count, int) else None,
        stream_cleanup_after_done_s=row['cleanup_complete_perf_counter_s'] - end if end is not None else None,
        client_estimated_interactivity_tok_s=(remaining / (end - first)
            if prefix and len(ids) == count and remaining and remaining > 0
            and end is not None and first is not None and end > first else None))
    return row


def request(base, headers, path, body, tok, request_id):
    import httpx
    row = {'request_id': request_id, 'method': 'POST', 'path': path, 'body': body,
           'http_status': None, 'done': False, 'usage': None, 'finish_reason': None,
           'first_text_perf_counter_s': None, 'done_perf_counter_s': None,
           'output_ids_when_available': None, 'sse_events': [], 'text_events': [], 'error': None,
           'absolute_request_timeout_s': 300}
    # Client construction is outside the timer, matching the old C1 harness.
    with httpx.Client(headers=headers, trust_env=False, timeout=300) as client:
        # httpx's read timeout is per read, not total request time. A hard alarm
        # keeps a trickling/stalled stream inside the outer pause lease budget.
        # E020 is serial C1 on macOS and always calls this on the main thread.
        assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), 'unexpected existing alarm'
        def deadline(signum, frame):
            raise TimeoutError('absolute HTTP request deadline: 300 seconds')
        previous_handler = signal.signal(signal.SIGALRM, deadline)
        signal.setitimer(signal.ITIMER_REAL, 300)
        row['start_perf_counter_s'] = time.perf_counter()
        try:
            with client.stream('POST', base + path, json=body) as response:
                row['http_status'] = response.status_code
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith('data:'):
                        continue
                    received = time.perf_counter()
                    payload = line[5:].strip()
                    row['sse_events'].append({'client_perf_counter_s': received, 'data': payload})
                    if payload == '[DONE]':
                        row['done'] = True
                        row['done_perf_counter_s'] = received
                        break
                    data = json.loads(payload)
                    if 'error' in data:
                        raise RuntimeError(str(data['error']))
                    row['usage'] = data.get('usage') or row['usage']
                    if 'benchmark_output_ids' in data:
                        row['output_ids_when_available'] = data['benchmark_output_ids']
                    choices = data.get('choices', [])
                    if len(choices) > 1:
                        raise ValueError('unexpected multi-choice response')
                    for choice in choices:
                        row['finish_reason'] = choice.get('finish_reason') or row['finish_reason']
                        piece = choice.get('text', choice.get('delta', {}).get('content', ''))
                        if piece:
                            if row['first_text_perf_counter_s'] is None:
                                row['first_text_perf_counter_s'] = received
                            row['text_events'].append({'client_perf_counter_s': received, 'text': piece})
        except BaseException as error:
            # Preserve partial SSE on controller cancellation too; run.py saves
            # this error row and aborts the matrix, then reaps its actual server.
            row['error'] = {'type': type(error).__name__, 'text': str(error), 'traceback': traceback.format_exc()}
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
            row['cleanup_complete_perf_counter_s'] = time.perf_counter()
    try:
        return finish_record(row, tok)
    except Exception as error:
        row['finalization_error'] = {'type': type(error).__name__, 'text': str(error)}
        row['error'] = row['error'] or row['finalization_error']
        return row
