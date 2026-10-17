"""Rerun E012 on the actual opt-in module; no prototype selection installed."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import screen
from tutorial_code.miniengine.spec.dflash2 import DFlash2Drafter

ACTUAL = DFlash2Drafter._select


def integrated_select(drafter, h, logits, anchor, *, device=True, trace=False):
    previous = drafter._device_selection_enabled
    drafter.set_device_selection(device)
    scores = []
    argmax = screen.mx.argmax
    def capture(value, *args, **kwargs):
        scores.append(value)
        return argmax(value, *args, **kwargs)
    try:
        if trace:
            with patch.object(screen.mx, 'argmax', capture):
                result = ACTUAL(drafter, h, logits, anchor)
            return result, scores
        return ACTUAL(drafter, h, logits, anchor)
    finally:
        drafter.set_device_selection(previous)


@contextmanager
def integrated_enabled(device, observe=None):
    def route(self, h, logits, anchor):
        if observe is not None:
            observe(h, logits, anchor)
        return integrated_select(self, h, logits, anchor, device=device)
    DFlash2Drafter._select = route
    try:
        yield
    finally:
        DFlash2Drafter._select = ACTUAL


if __name__ == '__main__':
    out = Path(sys.argv[sys.argv.index('--out') + 1])
    manifest = out.with_suffix('.adapter.json')
    assert not manifest.exists()
    manifest.write_text(json.dumps({'adapter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'actual_module': 'tutorial_code/miniengine/spec/device_selector.py',
        'switch': 'DFlash2Drafter.set_device_selection(bool)',
        'prototype_select_installed': False,
        'score_capture': 'Intercept argmax input during diagnostic calls only; full-draft timing has no score capture'}, indent=2) + '\n')
    screen.select = integrated_select
    screen.enabled = integrated_enabled
    screen.main()
