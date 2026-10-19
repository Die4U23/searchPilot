from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from searchpilot.ctr.runtime import FEATURE_NAMES, CtrNotReadyError, load_ctr


class MlpArtifactValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directory = self.root / "ctr"
        self.directory.mkdir()
        self.write_json(
            "meta.json",
            {
                "feature_names": list(FEATURE_NAMES),
                "mean": [0.0] * 4,
                "std": [1.0] * 4,
                "model_version": "test-model",
                "calibrator_version": "test-temperature",
            },
        )
        self.write_json("lr.json", {"weight": [0.0] * 4, "bias": 0.0})
        self.write_json("temperature.json", {"T": 1.0})
        self.valid = {
            "w1": [[1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0]],
            "b1": [0.5, -0.5],
            "w2": [[2.0, -1.0]],
            "b2": [0.25],
        }

    def write_json(self, name: str, value: object) -> None:
        (self.directory / name).write_text(json.dumps(value), encoding="utf-8")

    def test_valid_export_preserves_logits(self) -> None:
        self.write_json("mlp.json", self.valid)
        runtime = load_ctr(self.root)
        inputs = np.array([[1.0, 0.0, 1.0, 0.0], [-2.0, 0.0, 0.0, 0.0]])
        np.testing.assert_allclose(runtime._logits(inputs), [1.75, 0.25])
        self.assertTrue(runtime.ctr_ready)

    def test_legacy_lr_only_still_loads(self) -> None:
        runtime = load_ctr(self.root)
        self.assertIsNone(runtime.mlp)
        np.testing.assert_array_equal(runtime._logits(np.zeros((2, 4))), [0.0, 0.0])

    def test_wrong_shapes_rejected_before_ready(self) -> None:
        cases = {
            "w1": [[], [1.0] * 4, [[1.0] * 3], [[1.0] * 5], [[[1.0] * 4]]],
            "b1": [[], [0.0], [[0.0, 0.0]]],
            "w2": [[], [1.0, 1.0], [[1.0]], [[1.0, 1.0], [1.0, 1.0]]],
            "b2": [[], 0.0, [0.0, 0.0]],
        }
        for name, variants in cases.items():
            for value in variants:
                with self.subTest(name=name, value=value):
                    payload = dict(self.valid)
                    payload[name] = value
                    self.write_json("mlp.json", payload)
                    with self.assertRaises(CtrNotReadyError):
                        load_ctr(self.root)

    def test_nonfinite_weights_rejected_in_each_array(self) -> None:
        for name in self.valid:
            for value in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(name=name, value=value):
                    payload = copy.deepcopy(self.valid)
                    if name.startswith("w"):
                        payload[name][0][0] = value
                    else:
                        payload[name][0] = value
                    self.write_json("mlp.json", payload)
                    with self.assertRaisesRegex(CtrNotReadyError, "finite"):
                        load_ctr(self.root)

    def test_missing_ragged_and_nonnumeric_weights_rejected(self) -> None:
        for name in self.valid:
            payload = dict(self.valid)
            del payload[name]
            with self.subTest(missing=name):
                self.write_json("mlp.json", payload)
                with self.assertRaises(CtrNotReadyError):
                    load_ctr(self.root)
        for value in ([[1.0] * 4, [1.0]], [["bad"] * 4], [[10**400] * 4]):
            with self.subTest(w1=value):
                payload = dict(self.valid)
                payload["w1"] = value
                self.write_json("mlp.json", payload)
                with self.assertRaises(CtrNotReadyError):
                    load_ctr(self.root)
