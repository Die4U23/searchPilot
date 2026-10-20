from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from searchpilot.ctr import artifacts
from searchpilot.ctr.artifacts import export_bundle
from searchpilot.ctr.runtime import FEATURE_NAMES, CtrNotReadyError, load_ctr


def inputs() -> dict:
    return {
        "meta": {"feature_names": list(FEATURE_NAMES), "mean": [0.0] * 4, "std": [1.0] * 4},
        "lr": {"weight": [0.0] * 4, "bias": 0.0},
        "mlp": {"w1": [[1.0, 0.0, 0.0, 0.0]], "b1": [0.5], "w2": [[2.0]], "b2": [0.1]},
        "temperature": 2.0,
        "feature_tables": {
            "item_clicks": {"i": 1},
            "item_category": {"i": "news"},
            "item_title_chars": {"i": 10},
            "user_history_len": {"u": 2},
            "user_category_counts": {"u": {"news": 1}},
        },
    }


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


class CtrBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directory = self.root / "ctr"
        self.args = inputs()
        self.versions = export_bundle(self.directory, **self.args)

    def test_round_trip_matches_mlp_and_temperature(self) -> None:
        runtime = load_ctr(self.root)
        self.assertEqual((runtime.model_version, runtime.calibrator_version), self.versions)
        self.assertIsNotNone(runtime.mlp)
        hit = runtime.score("u", [("i", None)])[0]
        logit = (np.log1p(2) + 0.5) * 2 + 0.1
        self.assertAlmostEqual(hit.pctr, float(1 / (1 + np.exp(-logit))))
        self.assertAlmostEqual(hit.pctr_calibrated, float(1 / (1 + np.exp(-logit / 2))))

    def test_missing_each_inference_file_is_not_ready(self) -> None:
        for name in ("meta.json", "lr.json", "mlp.json", "temperature.json", "feature_tables.json"):
            with self.subTest(name=name):
                path = self.directory / name
                contents = path.read_bytes()
                path.unlink()
                try:
                    with self.assertRaises(CtrNotReadyError):
                        load_ctr(self.root)
                finally:
                    path.write_bytes(contents)

    def test_manifest_cannot_be_removed_from_new_bundle(self) -> None:
        (self.directory / "manifest.json").unlink()
        with self.assertRaises(CtrNotReadyError):
            load_ctr(self.root)

    def test_modified_content_is_not_ready(self) -> None:
        path = self.directory / "mlp.json"
        value = json.loads(path.read_text())
        value["w1"][0][0] += 1
        write(path, value)
        with self.assertRaises(CtrNotReadyError):
            load_ctr(self.root)

    def test_checksums_alone_cannot_hide_stale_versions(self) -> None:
        path = self.directory / "mlp.json"
        value = json.loads(path.read_text())
        value["w2"][0][0] += 1
        write(path, value)
        manifest_path = self.directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"]["mlp.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
        write(manifest_path, manifest)
        with self.assertRaises(CtrNotReadyError):
            load_ctr(self.root)

    def test_every_inference_component_changes_both_versions(self) -> None:
        cases = [
            ("lr", "weight", [1.0, 0, 0, 0]),
            ("lr", "bias", 1.0),
            ("mlp", "w1", [[2.0, 0, 0, 0]]),
            ("mlp", "b1", [1.5]),
            ("mlp", "w2", [[3.0]]),
            ("mlp", "b2", [1.1]),
            ("meta", "mean", [1.0, 0, 0, 0]),
            ("meta", "std", [2.0, 1, 1, 1]),
            ("feature_tables", "item_clicks", {"i": 2}),
            ("feature_tables", "item_category", {"i": "other"}),
            ("feature_tables", "item_title_chars", {"i": 20}),
            ("feature_tables", "user_history_len", {"u": 3}),
            ("feature_tables", "user_category_counts", {"u": {"news": 2}}),
        ]
        for section, key, value in cases:
            with self.subTest(section=section, key=key):
                args = copy.deepcopy(self.args)
                args[section][key] = value
                versions = export_bundle(self.directory, **args)
                self.assertNotEqual(versions[0], self.versions[0])
                self.assertNotEqual(versions[1], self.versions[1])
                load_ctr(self.root)
        args = copy.deepcopy(self.args)
        args["temperature"] = 3.0
        changed = export_bundle(self.directory, **args)
        self.assertNotEqual(changed[0], self.versions[0])
        self.assertNotEqual(changed[1], self.versions[1])

    def test_legacy_mlp_exports_never_fall_back(self) -> None:
        for explicit in (True, False):
            with self.subTest(explicit=explicit):
                export_bundle(self.directory, **self.args)
                (self.directory / "manifest.json").unlink()
                (self.directory / "mlp.json").unlink()
                meta = json.loads((self.directory / "meta.json").read_text())
                meta.pop("artifact_schema")
                if not explicit:
                    meta.pop("model_type")
                    meta.update(epochs=10, learning_rate=0.01)
                write(self.directory / "meta.json", meta)
                with self.assertRaises(CtrNotReadyError):
                    load_ctr(self.root)

    def test_explicit_lr_and_untyped_legacy_lr_remain_supported(self) -> None:
        (self.directory / "manifest.json").unlink()
        meta = json.loads((self.directory / "meta.json").read_text())
        meta.pop("artifact_schema")
        meta["model_type"] = "lr"
        write(self.directory / "meta.json", meta)
        self.assertIsNone(load_ctr(self.root).mlp)
        (self.directory / "mlp.json").unlink()
        meta.pop("model_type")
        write(self.directory / "meta.json", meta)
        self.assertIsNone(load_ctr(self.root).mlp)

    def test_interrupted_publication_does_not_load_a_mixed_bundle(self) -> None:
        old_runtime = load_ctr(self.root)
        original_write = artifacts._atomic_write
        args = copy.deepcopy(self.args)
        args["mlp"]["b1"] = [99.0]

        def fail_on_mlp(path: Path, data: bytes) -> None:
            if path.name == "mlp.json":
                raise OSError("injected write failure")
            original_write(path, data)

        with (
            patch.object(artifacts, "_atomic_write", side_effect=fail_on_mlp),
            self.assertRaises(OSError),
        ):
            export_bundle(self.directory, **args)
        with self.assertRaises(CtrNotReadyError):
            load_ctr(self.root)
        self.assertEqual(old_runtime.model_version, self.versions[0])
        export_bundle(self.directory, **args)
        self.assertNotEqual(load_ctr(self.root).model_version, self.versions[0])

    def test_initial_publication_writes_not_ready_marker_before_files(self) -> None:
        (self.directory / "manifest.json").unlink()
        original_write = artifacts._atomic_write
        seen = []

        def record(path: Path, data: bytes) -> None:
            seen.append(path.name)
            original_write(path, data)
            if path.name != "manifest.json":
                with self.assertRaises(CtrNotReadyError):
                    load_ctr(self.root)

        with patch.object(artifacts, "_atomic_write", side_effect=record):
            export_bundle(self.directory, **self.args)
        self.assertEqual(seen[0], "manifest.json")
        self.assertEqual(seen[-1], "manifest.json")
        load_ctr(self.root)

    def test_manifest_rejects_unexpected_paths_and_schema_types(self) -> None:
        path = self.directory / "manifest.json"
        original = json.loads(path.read_text())
        for schema in (True, "1", 2):
            with self.subTest(schema=schema):
                manifest = copy.deepcopy(original)
                manifest["artifact_schema"] = schema
                write(path, manifest)
                with self.assertRaises(CtrNotReadyError):
                    load_ctr(self.root)
        manifest = copy.deepcopy(original)
        manifest["files"]["../extra.json"] = "irrelevant"
        write(path, manifest)
        with self.assertRaises(CtrNotReadyError):
            load_ctr(self.root)


if __name__ == "__main__":
    unittest.main()
