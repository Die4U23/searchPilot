from __future__ import annotations

import pandas as pd

from searchpilot.recommend.popular import PopularModel


def _train_impressions() -> pd.DataFrame:
    rows = [
        ("i1", 1, "train"),
        ("i1", 1, "train"),
        ("i2", 1, "train"),
        ("i3", 0, "train"),
        ("i4", 1, "dev"),
    ]
    return pd.DataFrame(
        {
            "item_id": [r[0] for r in rows],
            "clicked": [r[1] for r in rows],
            "split": [r[2] for r in rows],
        }
    )


def test_popular_scores_clicks_only() -> None:
    model = PopularModel()
    model.fit(_train_impressions())
    preds = model.predict(frozenset(), limit=10)
    assert preds == [("i1", 2.0), ("i2", 1.0)]


def test_popular_exclude_and_limit() -> None:
    model = PopularModel()
    model.fit(_train_impressions())
    preds = model.predict(frozenset({"i1"}), limit=1)
    assert preds == [("i2", 1.0)]


def test_popular_tie_break_by_item_id() -> None:
    df = pd.DataFrame(
        {
            "item_id": ["b", "a", "b", "a"],
            "clicked": [1, 1, 1, 1],
            "split": ["train"] * 4,
        }
    )
    model = PopularModel()
    model.fit(df)
    preds = model.predict(frozenset(), limit=2)
    assert preds[0][1] == preds[1][1]
    assert preds == [("a", 2.0), ("b", 2.0)]


def test_popular_roundtrip_dict() -> None:
    model = PopularModel()
    model.fit(_train_impressions())
    restored = PopularModel.from_dict(model.to_dict())
    assert restored.predict(frozenset(), 10) == model.predict(frozenset(), 10)
