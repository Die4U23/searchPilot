"""合成出价 eCPM 对照：校准改 Top-1、同出价排序不变、并列破坏。"""

from __future__ import annotations

import pandas as pd
import pytest

from searchpilot.ctr.ecpm import (
    BID_HIGH_INCLUSIVE,
    BID_LOW_EXCLUSIVE,
    BID_SEED,
    BID_SOURCE,
    compare_ecpm,
    format_ecpm_markdown,
    select_top1,
    synthetic_bid,
    synthetic_bid_map,
)

COLUMNS = ("impression_id", "item_id", "clicked", "pctr_raw", "pctr_calibrated")


def _rows(*records: tuple[object, ...]) -> pd.DataFrame:
    return pd.DataFrame(list(records), columns=COLUMNS)


def test_synthetic_bid_range_and_deterministic() -> None:
    first = synthetic_bid("N58043")
    assert first == synthetic_bid("N58043")
    assert BID_LOW_EXCLUSIVE < first <= BID_HIGH_INCLUSIVE
    other = synthetic_bid("N1")
    assert BID_LOW_EXCLUSIVE < other <= BID_HIGH_INCLUSIVE
    assert first != other
    mapped = synthetic_bid_map(["N2", "N1", "N2"])
    assert list(mapped) == ["N2", "N1"]
    assert mapped["N1"] == synthetic_bid("N1", seed=BID_SEED)


def test_calibration_changes_top1() -> None:
    # pCTR 顺序校准前后都是 A > B（单调），异质出价后 eCPM Top-1 改变。
    # raw:  A 0.9*0.4=0.36 > B 0.4*0.8=0.32 → A（未点击）
    # cal:  A 0.55*0.4=0.22 < B 0.38*0.8=0.304 → B（点击）
    rows = _rows(
        ("imp-1", "A", 0, 0.9, 0.55),
        ("imp-1", "B", 1, 0.4, 0.38),
    )
    result = compare_ecpm(rows, bids={"A": 0.4, "B": 0.8})
    assert result.source == BID_SOURCE
    assert result.impression_count == 1
    assert result.agreement_rate == 0.0
    assert result.raw.proxy_revenue == pytest.approx(0.4 * 0)
    assert result.calibrated.proxy_revenue == pytest.approx(0.8 * 1)
    assert result.raw.click_hits == 0
    assert result.calibrated.click_hits == 1


def test_equal_bids_preserve_ranking() -> None:
    rows = _rows(
        ("imp-1", "B", 0, 0.8, 0.55),
        ("imp-1", "A", 1, 0.4, 0.30),
        ("imp-2", "D", 1, 0.6, 0.42),
        ("imp-2", "C", 0, 0.2, 0.15),
    )
    same = 0.5
    item_ids = ["B", "A"]
    pctr_raw = [0.8, 0.4]
    pctr_cal = [0.55, 0.30]
    assert select_top1(item_ids, pctr_raw) == select_top1(item_ids, [same * p for p in pctr_raw])
    assert select_top1(item_ids, pctr_cal) == select_top1(item_ids, [same * p for p in pctr_cal])
    result = compare_ecpm(rows, bids={"A": same, "B": same, "C": same, "D": same})
    assert result.agreement_rate == 1.0
    assert result.raw.proxy_revenue == pytest.approx(same * 1)  # imp-1 A 未选中；imp-2 D 命中
    assert result.calibrated.proxy_revenue == pytest.approx(result.raw.proxy_revenue)
    assert result.raw.click_hits == 1
    assert result.calibrated.click_hits == 1


def test_tie_break_smaller_item_id() -> None:
    rows = _rows(
        ("imp-1", "N2", 0, 0.5, 0.5),
        ("imp-1", "N1", 1, 0.5, 0.5),
    )
    assert select_top1(["N2", "N1"], [0.5, 0.5]) == "N1"
    result = compare_ecpm(rows, bids={"N1": 1.0, "N2": 1.0})
    assert result.agreement_rate == 1.0
    assert result.raw.proxy_revenue == pytest.approx(1.0)
    assert result.calibrated.proxy_revenue == pytest.approx(1.0)
    assert result.raw.click_hits == 1
    assert result.calibrated.click_hits == 1


def test_default_synthetic_bids_and_markdown_disclaimer() -> None:
    rows = _rows(
        ("imp-1", "N10", 1, 0.2, 0.2),
        ("imp-1", "N2", 0, 0.2, 0.2),
    )
    result = compare_ecpm(rows)
    assert result.source == "synthetic"
    assert result.bid_seed == BID_SEED
    bid_n10 = synthetic_bid("N10")
    bid_n2 = synthetic_bid("N2")
    # 同分 pCTR 下 eCPM 并列按 item_id；若出价不同则选出价更高者。
    expected = "N10" if bid_n10 > bid_n2 or (bid_n10 == bid_n2 and "N10" < "N2") else "N2"
    chosen_bid = bid_n10 if expected == "N10" else bid_n2
    chosen_click = 1 if expected == "N10" else 0
    assert result.raw.proxy_revenue == pytest.approx(chosen_bid * chosen_click)
    assert result.raw.click_hits == chosen_click
    text = format_ecpm_markdown(result)
    assert "source: `synthetic`" in text
    assert "不是真实广告收益" in text
    assert "真实广告收益" in text
    assert str(BID_SEED) in text


def test_select_top1_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        select_top1([], [])
    with pytest.raises(ValueError, match="same length"):
        select_top1(["A"], [0.1, 0.2])


def test_compare_ecpm_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        compare_ecpm(pd.DataFrame({"item_id": ["A"]}))
