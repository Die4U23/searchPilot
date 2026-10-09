from __future__ import annotations

from pathlib import Path

from searchpilot.data.mind import parse_behaviors, parse_news


def test_parse_news_fills_missing_abstract(tmp_path: Path) -> None:
    news = tmp_path / "news.tsv"
    news.write_text(
        "N1\tnews\tworld\tFirst title\t\thttps://example.test/1\t[]\t[]\n",
        encoding="utf-8",
    )

    frame = parse_news(news)

    assert frame.loc[0, "item_id"] == "N1"
    assert frame.loc[0, "abstract"] == ""
    assert str(frame["item_id"].dtype) == "string"


def test_parse_behaviors_expands_impressions_and_uses_utc(tmp_path: Path) -> None:
    behaviors = tmp_path / "behaviors.tsv"
    behaviors.write_text(
        "I1\tU1\t11/11/2019 9:05:58 AM\tN9 N8\tN1-1 N2-0\n",
        encoding="utf-8",
    )

    impressions, raw_behaviors = parse_behaviors(behaviors, "train")

    assert impressions["item_id"].tolist() == ["N1", "N2"]
    assert impressions["clicked"].tolist() == [1, 0]
    assert impressions["split"].tolist() == ["train", "train"]
    assert impressions["source"].tolist() == ["mind", "mind"]
    assert str(impressions["shown_at"].dt.tz) == "UTC"
    assert impressions.loc[0, "shown_at"].isoformat() == "2019-11-11T09:05:58+00:00"
    assert raw_behaviors.loc[0, "history"] == ["N9", "N8"]
