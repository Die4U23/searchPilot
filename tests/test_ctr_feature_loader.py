"""Exercise the actual loader function with in-memory Parquet-reader fixtures.

Only the data-loading functions are compiled, so these tests do not import Torch
or train models. Feature calculations and split helpers are the real code.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from searchpilot.ctr.features import rows_to_matrix
from searchpilot.ctr.split import split_dev_by_time
from searchpilot.ctr.temporal_features import training_features


def load_fixture(labels: list[int]) -> tuple:
    impressions = pd.DataFrame(
        {
            "impression_id": ["a", "b", "c", "d", "e", "f"],
            "user_id": ["u"] * 6,
            "item_id": ["i"] * 6,
            "clicked": [1, 1, 0, 1] + labels,
            "split": ["train"] * 4 + ["dev"] * 2,
            "shown_at": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-04",
                    "2026-01-05",
                ],
                utc=True,
            ),
        }
    )
    items = pd.DataFrame({"item_id": ["i"], "title": ["Title"], "category": ["news"]})
    read_names = []

    def read_table(path: Path, columns: list[str] | None = None) -> SimpleNamespace:
        read_names.append(path.name)
        # A final user_history snapshot must never be read into early training rows.
        table = {"impressions.parquet": impressions, "items.parquet": items}[path.name]
        return SimpleNamespace(to_pandas=lambda: table.copy())

    source = Path(__file__).resolve().parents[1] / "scripts" / "train_ctr.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    module = ast.Module(
        body=[
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"_sample_split", "load_split_features"}
        ],
        type_ignores=[],
    )
    namespace = {
        "Path": Path,
        "pd": pd,
        "np": np,
        "pq": SimpleNamespace(read_table=read_table),
        "DATA_VERSION": "fixture",
        "SEED": 20261009,
        "TRAIN_N": 100,
        "VAL_N": 100,
        "TEST_N": 100,
        "rows_to_matrix": rows_to_matrix,
        "split_dev_by_time": split_dev_by_time,
        "training_features": training_features,
    }
    exec(compile(module, str(source), "exec"), namespace)
    result = namespace["load_split_features"](Path("unused"))
    return result, read_names


class CtrFeatureLoaderTests(unittest.TestCase):
    def test_loader_uses_prefix_features_and_never_final_history(self) -> None:
        result, reads = load_fixture([0, 1])
        np.testing.assert_allclose(result[0][:, 0], np.log1p([0, 1, 1, 2]))
        np.testing.assert_allclose(result[0][:, 2], np.log1p([0, 1, 1, 2]))
        np.testing.assert_array_equal(result[1], [1, 1, 0, 1])
        np.testing.assert_allclose(result[2][:, 0], np.log1p([3]))
        self.assertEqual(reads, ["impressions.parquet", "items.parquet"])

    def test_dev_labels_never_change_training_features_or_frozen_tables(self) -> None:
        before, _ = load_fixture([0, 1])
        after, _ = load_fixture([1, 0])
        for index in (0, 1, 2, 4):
            np.testing.assert_array_equal(before[index], after[index])
        self.assertEqual(before[11:], after[11:])


if __name__ == "__main__":
    unittest.main()
