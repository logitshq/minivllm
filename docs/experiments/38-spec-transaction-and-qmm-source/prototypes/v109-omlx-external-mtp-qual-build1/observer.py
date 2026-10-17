"""Qualification-only observers. Original functions execute once, unchanged.
No imports or model activity until install() is called by the GPU owner's entry.
Vendor sources are never written; this instrumented run is not a speed result.
"""
import functools
import json
import os
import threading


def install(path, mode, expected_draft):
    import mlx.core as mx
    import omlx.scheduler as scheduler
    import omlx.speculative.vlm_mtp as vlm
    import omlx.engine.vlm as engine_vlm
    from mlx_vlm.speculative import mtp
    from mlx_vlm.speculative.common import speculative_stats_snapshot
    lock = threading.RLock()
    active = {'id': None, 'last_verify': None}
    seen = set()

    def emit(kind, **data):
        with lock, path.open('a') as stream:
            stream.write(json.dumps({'kind': kind, 'pid': os.getpid(), **data}, allow_nan=False) + '\n')
            stream.flush()

    def memory():
        return {name: int(getattr(mx, name)()) for name in
                ('get_active_memory', 'get_peak_memory', 'get_cache_memory')}

    def wrap(owner, name, body):
        original = getattr(owner, name)
        @functools.wraps(original)
        def call(*args, **kwargs):
            return body(original, *args, **kwargs)
        setattr(owner, name, call)

    def loaded(original, source):
        result = original(source)
        emit('load', requested_path=str(source), loaded=result is not None,
             source_path=getattr(result, 'source_path', None),
             draft_kind=getattr(result, 'draft_kind', None),
             model_class=None if result is None else type(result.model).__module__ + '.' + type(result.model).__name__,
             config_model_type=None if result is None else getattr(result.model.config, 'model_type', None),
             block_size=None if result is None else int(result.model.config.block_size), memory=memory())
        if mode == 'mtp':
            assert result is not None and str(result.source_path) == expected_draft and result.draft_kind == 'mtp', 'MTP actual loading proof failed'
        return result

    def attached(original, self, drafter):
        result = original(self, drafter)
        fields = ('vlm_mtp_enabled', 'vlm_mtp_draft_model', 'vlm_mtp_draft_block_size',
                  'mtp_enabled', 'dflash_enabled', 'specprefill_enabled', 'repetition_penalty',
                  'presence_penalty', 'frequency_penalty')
        settings = getattr(self, '_model_settings', None)
        emit('attach', actual_model_name=self._model_name,
             drafter_source=str(drafter.source_path), draft_kind=drafter.draft_kind,
             engine_same_object=self._vlm_mtp_drafter is drafter,
             flags={name: getattr(settings, name, None) for name in fields}, memory=memory())
        return result

    def route(original, self, request, *args, **kwargs):
        result = original(self, request, *args, **kwargs)
        emit('route', request_id=request.request_id, routed=result is not None, uid=result,
             prompt_ids=list(request.prompt_token_ids), max_tokens=request.sampling_params.max_tokens,
             stop_ids=sorted(self._get_stop_tokens()),
             sampler_temperature=request.sampling_params.temperature,
             target_class=type(self.model).__module__ + '.' + type(self.model).__name__,
             rollback_capability=callable(getattr(getattr(self.model, '_language_model', None), 'rollback_speculative_cache', None)),
             scheduler_drafter_source=getattr(self._vlm_mtp_drafter, 'source_path', None))
        assert mode == 'mtp' and result is not None, 'MTP routing fell back to ordinary decode'
        assert active['id'] is None, 'qualification requires one active MTP request'
        active['id'] = request.request_id
        active['last_verify'] = None
        return result

    def verify(original, lm, verify_input, *args, **kwargs):
        assert mode == 'mtp' and active['id'] is not None
        shape = list(verify_input.shape)
        assert len(shape) == 2 and shape[0] == 1 and shape[1] >= 2
        result = original(lm, verify_input, *args, **kwargs)
        row = {'request_id': active['id'], 'input_shape': shape, 'input_dtype': str(verify_input.dtype),
               'actual_draft_token_count': shape[1] - 1, 'target_model_class': type(lm).__module__ + '.' + type(lm).__name__}
        assert active['last_verify'] is None, 'unpaired verify'
        active['last_verify'] = row
        emit('verify', **row)
        return result

    def record(original, draft_model, accepted, draft_count):
        assert active['last_verify'] is not None
        assert int(draft_count) == active['last_verify']['actual_draft_token_count']
        assert float(accepted).is_integer() and 0 <= int(accepted) <= int(draft_count)
        result = original(draft_model, accepted, draft_count)
        emit('round', request_id=active['id'], accepted=int(accepted), drafted=int(draft_count),
             lifetime=list(speculative_stats_snapshot(draft_model)))
        active['last_verify'] = None
        return result

    def stats(original, self, state, finish_reason):
        result = original(self, state, finish_reason)
        model = self._vlm_mtp_drafter.model
        emit('stats', request_id=state.request.request_id, finish_reason=finish_reason,
             accept_lens=[int(x) for x in model.accept_lens], draft_lens=[int(x) for x in model.draft_lens],
             lifetime=list(speculative_stats_snapshot(model)), emitted=state.emitted,
             configured_block_size=self._vlm_mtp_draft_block_size,
             actual_config_block_size=int(model.config.block_size), memory=memory())
        assert state.request.request_id == active['id'] and active['last_verify'] is None
        active['id'] = None
        return result

    def process(original, self, responses):
        result = original(self, responses)
        for output in result[0]:
            if output.finished:
                assert output.request_id not in seen, 'duplicate terminal request'
                seen.add(output.request_id)
                emit('terminal', request_id=output.request_id, raw_output_ids=list(output.output_token_ids),
                     prompt_tokens=output.prompt_tokens, completion_tokens=output.completion_tokens,
                     cached_tokens=output.cached_tokens, finish_reason=output.finish_reason,
                     error=output.error, memory=memory())
        return result

    wrap(vlm, 'load_vlm_mtp_drafter', loaded)
    wrap(engine_vlm.VLMBatchedEngine, 'set_vlm_mtp_drafter', attached)
    # Do not invoke route at all in mode=none. Ordinary routing source decides this.
    wrap(scheduler.Scheduler, '_route_to_vlm_mtp', route)
    wrap(mtp, '_mtp_verify_target', verify)
    wrap(mtp, '_record_speculative_round', record)
    wrap(scheduler.Scheduler, '_log_vlm_mtp_stats', stats)
    wrap(scheduler.Scheduler, '_process_batch_responses', process)
    emit('observer_installed', mode=mode, no_added_array_eval=True, timing_eligible=False)
