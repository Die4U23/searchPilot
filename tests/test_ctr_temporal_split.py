from __future__ import annotations

import unittest

import pandas as pd

from searchpilot.ctr.split import split_dev_by_time


class TemporalCtrSplitTests(unittest.TestCase):
    def fixture(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "impression_id": ["I1", "I2", "I2", "I2", "I2", "I3"],
                "shown_at": pd.to_datetime(["2020-01-01", *(["2020-01-02"] * 4), "2020-01-03"]),
                "item_id": [f"N{i}" for i in range(6)],
            }
        )

    def test_midpoint_regression_keeps_the_whole_impression(self) -> None:
        frame = self.fixture()
        mid = len(frame) // 2
        self.assertEqual(
            set(frame.iloc[:mid].impression_id) & set(frame.iloc[mid:].impression_id), {"I2"}
        )
        val, test = split_dev_by_time(frame)
        self.assertEqual(set(val.impression_id), {"I1"})
        self.assertEqual(set(test.impression_id), {"I2", "I3"})
        self.assertEqual(len(test.loc[test.impression_id == "I2"]), 4)
        self.assertLess(val.shown_at.max(), test.shown_at.min())

    def test_shared_timestamps_stay_together(self) -> None:
        frame = self.fixture()
        frame.loc[frame.impression_id == "I3", "shown_at"] = pd.Timestamp("2020-01-02")
        val, test = split_dev_by_time(frame)
        self.assertEqual(set(val.impression_id), {"I1"})
        self.assertEqual(set(test.impression_id), {"I2", "I3"})
        self.assertFalse(set(val.shown_at) & set(test.shown_at))

    def test_input_order_does_not_change_membership_or_mutate_input(self) -> None:
        frame = self.fixture().iloc[::-1]
        original = frame.copy(deep=True)
        val, test = split_dev_by_time(frame)
        pd.testing.assert_frame_equal(frame, original)
        self.assertEqual(set(val.impression_id), {"I1"})
        self.assertEqual(set(test.impression_id), {"I2", "I3"})

    def test_one_impression_cannot_span_timestamps(self) -> None:
        frame = self.fixture()
        frame.loc[2, "shown_at"] = pd.Timestamp("2020-01-03")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            split_dev_by_time(frame)

    def test_insufficient_or_missing_times_fail_explicitly(self) -> None:
        frame = self.fixture()
        for invalid in (frame.iloc[:0], frame.iloc[:1], frame.assign(shown_at=pd.NaT)):
            with self.subTest(rows=len(invalid)), self.assertRaises(ValueError):
                split_dev_by_time(invalid)
        with self.assertRaises(ValueError):
            split_dev_by_time(frame.assign(impression_id=None))
