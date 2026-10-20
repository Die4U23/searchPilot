from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from searchpilot.ctr.temporal_features import training_features


def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "impression_id": ["a", "b", "c", "d"],
            "user_id": ["u", "u", "u", "u"],
            "item_id": ["i", "i", "i", "i"],
            "clicked": [1, 1, 0, 1],
            "shown_at": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-01-02", "2026-01-03"], utc=True
            ),
        }
    )


def build(data: pd.DataFrame, positions: list[int]) -> tuple:
    return training_features(
        data, positions, item_category={"i": "news"}, item_title_chars={"i": 10}
    )


class TemporalTrainingFeaturesTests(unittest.TestCase):
    def test_current_and_same_time_labels_do_not_enter_features(self) -> None:
        matrix, tables = build(frame(), [0, 1, 2, 3])
        np.testing.assert_allclose(matrix[:, 0], np.log1p([0, 1, 1, 2]))
        np.testing.assert_allclose(matrix[:, 2], np.log1p([0, 1, 1, 2]))
        np.testing.assert_allclose(matrix[:, 1], [0, 1, 1, 1])
        self.assertEqual(tables["item_clicks"], {"i": 3})
        self.assertEqual(tables["user_history_len"], {"u": 3})

    def test_changing_current_label_changes_only_later_features(self) -> None:
        data = frame()
        before, _ = build(data, [0, 1, 2, 3])
        data.loc[1, "clicked"] = 0
        after, _ = build(data, [0, 1, 2, 3])
        np.testing.assert_array_equal(before[:3], after[:3])
        self.assertNotEqual(before[3, 0], after[3, 0])

    def test_unsampled_events_and_requested_order_are_preserved(self) -> None:
        matrix, tables = build(frame(), [3, 0])
        np.testing.assert_allclose(matrix[:, 0], np.log1p([2, 0]))
        self.assertEqual(tables["item_clicks"], {"i": 3})

    def test_unsorted_input_does_not_change_temporal_semantics(self) -> None:
        baseline, tables = build(frame(), [0, 1, 2, 3])
        data = frame().iloc[[3, 2, 0, 1]].reset_index(drop=True)
        matrix, shuffled_tables = build(data, [2, 3, 1, 0])
        np.testing.assert_array_equal(matrix, baseline)
        self.assertEqual(tables, shuffled_tables)

    def test_empty_sample_still_builds_frozen_training_end_tables(self) -> None:
        matrix, tables = build(frame(), [])
        self.assertEqual(matrix.shape, (0, 4))
        self.assertEqual(tables["user_history_len"], {"u": 3})

    def test_users_have_separate_history_and_items_shared_popularity(self) -> None:
        data = frame()
        data.loc[1:, "user_id"] = "other"
        matrix, _ = build(data, [1, 3])
        np.testing.assert_allclose(matrix[:, 0], np.log1p([0, 1]))
        np.testing.assert_allclose(matrix[:, 2], np.log1p([1, 2]))

    def test_rejects_invalid_rows_and_sample_positions(self) -> None:
        for positions in ([0, 0], [-1], [4], [True], [1.5]):
            with self.subTest(positions=positions), self.assertRaises(ValueError):
                build(frame(), positions)
        for column, value in (("shown_at", None), ("clicked", 2), ("user_id", None)):
            data = frame()
            data.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                build(data, [0])
        data = frame()
        data.loc[1, "impression_id"] = "a"
        with self.assertRaises(ValueError):
            build(data, [0])


if __name__ == "__main__":
    unittest.main()
