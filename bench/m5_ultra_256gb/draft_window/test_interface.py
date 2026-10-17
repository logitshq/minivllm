"""CPU-only E014 configuration/lifecycle contracts; no models or arrays loaded."""
from contextlib import ExitStack
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tutorial_code.miniengine.engine.llm_engine import LLMEngine
from tutorial_code.miniengine.spec.dflash2 import DFlash2Drafter, DraftAttention, load_drafter

SPEC = 'tutorial_code.miniengine.spec.dflash2'
ENGINE = 'tutorial_code.miniengine.engine.llm_engine'


def empty_drafter():
    draft = DFlash2Drafter.__new__(DFlash2Drafter)
    layers = []
    for _ in range(5):
        attention = DraftAttention.__new__(DraftAttention)
        attention.reset()
        object.__setattr__(attention, '_bounded_kv_enabled', False)
        layers.append(SimpleNamespace(self_attn=attention))
    draft.layers = layers
    draft.set_device_selection(False)
    return draft


class InterfaceContract(unittest.TestCase):
    def test_public_default_off(self):
        self.assertIs(inspect.signature(load_drafter).parameters['bounded_kv'].default, False)
        self.assertIs(inspect.signature(LLMEngine).parameters['draft_bounded_kv'].default, False)
        self.assertIs(inspect.signature(DFlash2Drafter.set_bounded_kv).parameters['enabled'].default, False)

    def test_switches_are_independent_before_caching(self):
        draft = empty_drafter()
        for selector in (False, True):
            for bounded in (False, True):
                draft.set_device_selection(selector)
                draft.set_bounded_kv(bounded)
                self.assertIs(draft._device_selection_enabled, selector)
                self.assertTrue(all(layer.self_attn._bounded_kv_enabled is bounded for layer in draft.layers))
        for value in (1, 'yes', None):
            with self.assertRaises(TypeError):
                draft.set_bounded_kv(value)

    def test_live_mode_change_is_atomic_and_requires_reset(self):
        draft = empty_drafter()
        last = draft.layers[-1].self_attn
        last.ctx_len, last.ctx_k = 1, object()
        for value in (True, False):
            with self.assertRaisesRegex(RuntimeError, 'reset'):
                draft.set_bounded_kv(value)
            self.assertTrue(all(not layer.self_attn._bounded_kv_enabled for layer in draft.layers))
        draft.reset()
        draft.set_bounded_kv(True)
        last.ctx_len, last.ctx_k, last.ctx_v = 5000, object(), object()
        with self.assertRaises(RuntimeError):
            draft.set_bounded_kv(False)
        self.assertTrue(all(layer.self_attn._bounded_kv_enabled for layer in draft.layers))
        draft.reset()
        draft.set_bounded_kv(False)

    def test_invalid_options_fail_before_io_or_runtime_mutation(self):
        with patch(SPEC + '.DraftConfig.from_dir') as config, patch(ENGINE + '.load_model') as load, patch(ENGINE + '.mx.set_cache_limit') as limit:
            for value in (1, 'yes', None):
                with self.assertRaises(TypeError):
                    load_drafter('unused', bounded_kv=value)
                with self.assertRaises(TypeError):
                    load_drafter('unused', device_selection=value)
                with self.assertRaises(TypeError):
                    LLMEngine('unused', drafter_dir='unused', draft_bounded_kv=value)
            with self.assertRaises(ValueError):
                LLMEngine('unused', draft_bounded_kv=True)
            with self.assertRaises(TypeError):
                load_drafter('unused', bound_kv=True)
            with self.assertRaises(TypeError):
                LLMEngine('unused', draft_bound_kv=True)
            config.assert_not_called()
            load.assert_not_called()
            limit.assert_not_called()

    def test_loader_forwards_all_four_combinations(self):
        with patch(SPEC + '.DraftConfig.from_dir'), patch(SPEC + '.DFlash2Drafter') as constructor, patch(SPEC + '.mx.load', return_value={}), patch(SPEC + '.mx.eval'):
            for selector in (False, True):
                for bounded in (False, True):
                    model = Mock()
                    constructor.return_value = model
                    self.assertIs(load_drafter('unused', device_selection=selector, bounded_kv=bounded), model)
                    model.set_device_selection.assert_called_once_with(selector)
                    model.set_bounded_kv.assert_called_once_with(bounded)

    def test_engine_forwards_all_four_combinations(self):
        with ExitStack() as stack:
            stack.enter_context(patch(ENGINE + '.load_model', return_value=(Mock(), Mock())))
            for name in ('read_eos_ids', 'Scheduler', 'ModelRunner', 'mx.set_cache_limit'):
                stack.enter_context(patch(ENGINE + '.' + name))
            draft_load = stack.enter_context(patch(SPEC + '.load_drafter'))
            for selector in (False, True):
                for bounded in (False, True):
                    draft_load.reset_mock()
                    LLMEngine('unused', drafter_dir='draft', draft_device_selection=selector, draft_bounded_kv=bounded)
                    draft_load.assert_called_once_with('draft', device_selection=selector, bounded_kv=bounded)


if __name__ == '__main__':
    unittest.main()
