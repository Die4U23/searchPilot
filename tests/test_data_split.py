from __future__ import annotations

import pandas as pd

from searchpilot.data.split import build_user_history, split_by_shown_at_quantile


def test_history_uses_each_users_latest_train_behavior() -> None:
    behaviors = pd.DataFrame(
        {
            "impression_id": ["I2", "I1", "I3"],
            "user_id": ["U1", "U1", "U2"],
            "shown_at": pd.to_datetime(
                [
                    "2019-11-02T00:00:00Z",
                    "2019-11-01T00:00:00Z",
                    "2019-11-03T00:00:00Z",
                ],
                utc=True,
            ),
            "history": [["N1", "N2"], ["N1"], []],
        }
    )

    result = build_user_history(behaviors)

    assert result.to_dict("records") == [
        {"user_id": "U1", "history": ["N1", "N2"]},
        {"user_id": "U2", "history": []},
    ]


def test_quantile_split_includes_boundary_in_train() -> None:
    frame = pd.DataFrame(
        {
            "shown_at": pd.to_datetime(
                [
                    "2019-11-01T00:00:00Z",
                    "2019-11-02T00:00:00Z",
                    "2019-11-03T00:00:00Z",
                    "2019-11-04T00:00:00Z",
                ],
                utc=True,
            )
        }
    )

    result = split_by_shown_at_quantile(frame, train_fraction=0.5)

    assert result["split"].tolist() == ["train", "train", "dev", "dev"]
