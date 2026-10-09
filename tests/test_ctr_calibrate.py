from __future__ import annotations

import numpy as np

from searchpilot.ctr.calibrate import apply_temperature, fit_temperature
from searchpilot.ctr.metrics import expected_calibration_error


def test_temperature_moves_overconfident_probabilities() -> None:
    labels = np.array([1] * 70 + [0] * 30)
    logits = np.full(100, 4.0)
    raw = 1.0 / (1.0 + np.exp(-logits))
    temperature = fit_temperature(logits, labels, steps=40)
    calibrated = apply_temperature(logits, temperature)
    raw_ece, _curve = expected_calibration_error(labels, raw)
    cal_ece, _curve = expected_calibration_error(labels, calibrated)
    assert temperature > 1.0
    assert cal_ece < raw_ece
