from __future__ import annotations

import numpy as np

from searchpilot.ctr.features import feature_vector, fit_standardizer
from searchpilot.ctr.metrics import binary_auc, expected_calibration_error
from searchpilot.ctr.model import LogisticModel, predict_logits, train_binary


def test_feature_vector_ignores_nothing_about_clicks_as_input() -> None:
    cold = feature_vector(history_len=0, category_hits=3, train_clicks=10, title_chars=0)
    warm = feature_vector(history_len=4, category_hits=2, train_clicks=0, title_chars=0)
    assert cold[1] == 0.0
    assert warm[1] == 0.5
    assert cold[2] > warm[2]


def test_ece_is_zero_when_probabilities_match_labels() -> None:
    labels = [0, 0, 1, 1]
    probabilities = [0.0, 0.0, 1.0, 1.0]
    ece, curve = expected_calibration_error(labels, probabilities, bins=2)
    assert ece == 0.0
    assert curve


def test_auc_none_for_one_class() -> None:
    assert binary_auc([0, 0, 0], [0.2, 0.3, 0.4]) is None


def test_linear_model_separates_a_simple_pattern() -> None:
    features = np.array([[0.0], [0.0], [1.0], [1.0]], dtype=np.float64)
    labels = np.array([0, 0, 1, 1])
    mean, std = fit_standardizer(features)
    scaled = (features - mean) / std
    model = train_binary(LogisticModel(1), scaled, labels, epochs=40, learning_rate=0.2, seed=1)
    logits = predict_logits(model, scaled)
    assert logits[2] > logits[0]
    assert logits[3] > logits[1]
