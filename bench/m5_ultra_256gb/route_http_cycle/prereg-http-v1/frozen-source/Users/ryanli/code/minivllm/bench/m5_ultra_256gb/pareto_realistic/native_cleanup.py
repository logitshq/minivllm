"""Versioned cleanup-only adapter for the frozen E023 native SSE client.

Use this module's run_group and GroupCancelled instead of client.py's exports.
The API and E023-http-group-v1 result remain unchanged; cleanup gains a version
receipt. A private byte-pinned client receives only the _cleanup replacement.
No public module is patched. Requests, raw SSE, DONE clocks, immediate refill,
qualification validation and count/statistics code are the original functions.

Like the reviewed Lithos cleanup fix, close cancelled independently of the
worker is failure, even after valid DONE. One absolute cleanup budget covers
workers, then response closers, then client close. Unresolved response tasks
block client close and invalidate the group; the external owner must reap the
server. This adapter neither drains native SSE past DONE nor grants correctness
qualification. It starts no listener, model or device.
"""
import asyncio
import hashlib
from pathlib import Path
import time
import types


BASE_PATH = Path(__file__).with_name('client.py')
BASE_SHA256 = 'ef64a114f4f7d90ec5044d6bdcf058ca05f98dea445d6639db706a4fc75217dc'
CLEANUP_REVISION = 'native-close-cancellation-v1'


def _private_base():
    raw = BASE_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != BASE_SHA256:
        raise ValueError('frozen HTTP client source changed; review cleanup adapter')
    module = types.ModuleType(__name__ + '._private_frozen_client')
    module.__file__ = str(BASE_PATH)
    exec(compile(raw, str(BASE_PATH), 'exec'), module.__dict__)
    return module


_BASE = _private_base()


async def _cleanup(workers, closers, client, budget_s):
    started = time.perf_counter()
    loop = asyncio.get_running_loop()
    deadline = loop.time() + budget_s
    errors, all_tasks = [], list(workers)
    cancelled_again = False

    async def bounded(tasks):
        nonlocal cancelled_again
        pending = {task for task in tasks if not task.done()}
        while pending and loop.time() < deadline:
            try:
                _, pending = await asyncio.wait(pending, timeout=max(0, deadline - loop.time()))
            except asyncio.CancelledError:
                cancelled_again = True
                for task in pending:
                    task.cancel()
        return pending

    for task in workers:
        if not task.done():
            task.cancel()
    pending = await bounded(workers)
    all_tasks.extend(closers)  # Completed workers have registered response closes.
    pending |= await bounded(closers)
    timed_out = bool(pending)
    for task in pending:
        task.cancel()
    try:
        await asyncio.sleep(0)
    except asyncio.CancelledError:
        cancelled_again = True
    remaining = [task for task in all_tasks if not task.done()]
    client_close_started = not remaining
    if client_close_started:
        closer = asyncio.create_task(client.aclose(), name='close:client')
        all_tasks.append(closer)
        late = await bounded([closer])
        timed_out |= bool(late)
        for task in late:
            task.cancel()
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            cancelled_again = True
    else:
        errors.append({'task': 'close:client', 'type': 'BlockedByActiveResponse',
                       'message': 'client close not raced against unresolved response tasks'})
    for task in all_tasks:
        if task.cancelled() and task not in workers:
            errors.append({'task': task.get_name(), 'type': 'CancelledError',
                           'message': 'independent response/client close task cancelled'})
        elif task.done() and not task.cancelled() and task.exception() is not None:
            errors.append({'task': task.get_name(), **_BASE._error(task.exception())})
    unresolved = [task.get_name() for task in all_tasks if not task.done()]
    return {'start_perf_counter_s': started, 'end_perf_counter_s': time.perf_counter(),
            'budget_s': budget_s, 'complete': not timed_out and not errors and not unresolved,
            'deadline_exceeded': timed_out, 'pending_tasks': unresolved, 'errors': errors,
            'cancelled_again': cancelled_again, 'client_close_started': client_close_started,
            'order': 'workers, response-close, client-close; one absolute budget',
            'adapter': CLEANUP_REVISION, 'frozen_client_sha256': BASE_SHA256}


_BASE._cleanup = _cleanup
run_group = _BASE.run_group
GroupCancelled = _BASE.GroupCancelled
StreamProtocolError = _BASE.StreamProtocolError
canonical_body_sha256 = _BASE.canonical_body_sha256
COUNT_SEMANTICS = _BASE.COUNT_SEMANTICS
