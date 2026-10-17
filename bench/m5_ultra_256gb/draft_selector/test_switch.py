"""Invalid experiment options must fail before loading target weights."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tutorial_code.miniengine.engine.llm_engine import LLMEngine
from tutorial_code.miniengine.spec.dflash2 import DFlash2Drafter


class SwitchContract(unittest.TestCase):
    def test_bool_only_and_reversible(self):
        draft = DFlash2Drafter.__new__(DFlash2Drafter)
        for value in (True, False, True, False):
            draft.set_device_selection(value)
            self.assertIs(draft._device_selection_enabled, value)
        with self.assertRaises(TypeError):
            draft.set_device_selection(1)

    def test_engine_rejects_before_model_load(self):
        with patch('tutorial_code.miniengine.engine.llm_engine.load_model') as load:
            with self.assertRaises(ValueError):
                LLMEngine('unused', draft_device_selection=True)
            with self.assertRaises(TypeError):
                LLMEngine('unused', draft_device_selection='yes')
            load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
