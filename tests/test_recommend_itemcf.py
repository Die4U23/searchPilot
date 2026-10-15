from __future__ import annotations

import random
from collections.abc import Mapping, Sequence

import pytest

from searchpilot.recommend.itemcf import ItemCFModel


def test_itemcf_hand_computed_similarity() -> None:
    """三用户四物品：共现余弦与邻居聚合可手算验证。"""
    histories = {
        "u1": ("A", "B"),
        "u2": ("A", "C"),
        "u3": ("B", "D"),
    }
    model = ItemCFModel()
    model.fit(histories, top_n_neighbors=50, min_cooccurrence=1)

    # A-B cooc=1, sim=1/sqrt(2*2)=0.5
    preds = model.predict(("A",), frozenset(), limit=10)
    by_id = dict(preds)
    assert by_id["B"] == 0.5
    assert "A" not in by_id

    preds_from_b = model.predict(("B",), frozenset(), limit=10)
    assert dict(preds_from_b)["A"] == 0.5
    # B 出现在 2 个用户、D 在 1 个用户：sim(B,D)=1/sqrt(2)
    assert dict(preds_from_b)["D"] == pytest.approx(2**-0.5)


def test_itemcf_similarity_is_independent_of_history_order() -> None:
    """时间序历史（非字母序、长度 ≥ 3）与字母序历史必须得到相同的相似度。

    histories u1=(C,A,B), u2=(C,A), u3=(C,B)：
    count(C)=3, count(A)=2, count(B)=2；cooc(A,C)=2, cooc(B,C)=2, cooc(A,B)=1
    sim(A,C)=sim(B,C)=2/sqrt(2*3)，sim(A,B)=1/sqrt(2*2)=0.5
    """
    chronological = {"u1": ("C", "A", "B"), "u2": ("C", "A"), "u3": ("C", "B")}
    alphabetical = {"u1": ("A", "B", "C"), "u2": ("A", "C"), "u3": ("B", "C")}

    expected_ac = 2 / (2 * 3) ** 0.5
    for histories in (chronological, alphabetical):
        model = ItemCFModel()
        model.fit(histories)
        from_c = dict(model.predict(("C",), frozenset(), limit=10))
        assert from_c["A"] == pytest.approx(expected_ac)
        assert from_c["B"] == pytest.approx(expected_ac)
        from_a = dict(model.predict(("A",), frozenset(), limit=10))
        assert from_a["B"] == pytest.approx(0.5)
        assert from_a["C"] == pytest.approx(expected_ac)


def test_itemcf_neighbor_truncation() -> None:
    histories = {f"u{i}": ("seed", f"n{i}") for i in range(60)}
    model = ItemCFModel()
    model.fit(histories, top_n_neighbors=10)
    neighbors = model.to_dict()["neighbors"]["seed"]
    assert len(neighbors) == 10


def test_itemcf_excludes_history() -> None:
    histories = {"u1": ("A", "B", "C")}
    model = ItemCFModel()
    model.fit(histories)
    preds = model.predict(("A", "B"), frozenset(), limit=10)
    assert all(item not in {"A", "B"} for item, _ in preds)


def test_itemcf_empty_history() -> None:
    model = ItemCFModel()
    model.fit({"u1": ("A",)})
    assert model.predict((), frozenset(), 5) == []


def test_itemcf_roundtrip_dict() -> None:
    histories = {"u1": ("A", "B")}
    model = ItemCFModel()
    model.fit(histories)
    restored = ItemCFModel.from_dict(model.to_dict())
    assert restored.predict(("A",), frozenset(), 5) == model.predict(("A",), frozenset(), 5)


def _intersection_oracle(
    histories: Mapping[str, Sequence[str]], top_n: int, min_cooccurrence: int
) -> dict[str, list[tuple[str, float]]]:
    """Independent reference: intersect item user sets, without counting history pairs."""
    items = sorted({item for history in histories.values() for item in history})
    users = {
        item: {user for user, history in histories.items() if item in history} for item in items
    }
    neighbors: dict[str, list[tuple[str, float]]] = {}
    for item in items:
        scores = []
        for other in items:
            if item == other:
                continue
            count = len(users[item] & users[other])
            # Threshold zero still does not create edges between disjoint items.
            if count and count >= min_cooccurrence:
                scores.append((other, count / (len(users[item]) * len(users[other])) ** 0.5))
        if scores:
            neighbors[item] = sorted(scores, key=lambda pair: (-pair[1], pair[0]))[:top_n]
    return neighbors


@pytest.mark.parametrize(
    ("top_n", "min_cooccurrence"),
    [(50, 0), (50, 1), (50, 2), (50, 100), (0, 1), (1, 1), (-1, 1), (-100, 1)],
)
def test_itemcf_matches_independent_intersections(top_n: int, min_cooccurrence: int) -> None:
    rng = random.Random(20261015)
    catalog = [f"item-{index}" for index in range(18)]
    histories = {
        f"u{index}": tuple(rng.choices(catalog, k=rng.randrange(15))) for index in range(35)
    }
    histories.update(
        empty=(),
        singleton=("isolated",),
        duplicate=("repeated", "repeated"),
        disjoint=("left", "right"),
    )
    expected = _intersection_oracle(histories, top_n, min_cooccurrence)
    model = ItemCFModel()
    model.fit(histories, top_n_neighbors=top_n, min_cooccurrence=min_cooccurrence)
    expected_payload = {
        "neighbors": {
            item: [[neighbor, score] for neighbor, score in pairs]
            for item, pairs in expected.items()
        }
    }
    # Exact floats, edge presence, ordering and empty lists after slicing all matter.
    assert model.to_dict() == expected_payload
    restored = ItemCFModel.from_dict(expected_payload)
    assert restored.to_dict() == expected_payload

    history = ("item-0", "item-1", "item-0", "unknown")
    excluded = frozenset({"item-2", "right"})
    candidates = {neighbor for pairs in expected.values() for neighbor, _ in pairs}
    predictions = []
    for candidate in candidates - set(history) - excluded:
        contributions = [
            score
            for seed in history
            for neighbor, score in expected.get(seed, ())
            if neighbor == candidate
        ]
        if contributions:
            # Preserve sequential IEEE additions; Python 3.12 sum uses compensation.
            total = 0.0
            for score in contributions:
                total += score
            predictions.append((candidate, total))
    predictions.sort(key=lambda pair: (-pair[1], pair[0]))
    assert model.predict(history, excluded, 7) == predictions[:7]
    assert restored.predict(history, excluded, 7) == predictions[:7]


def test_itemcf_long_history_lexical_ties_and_duplicate_prediction_seeds() -> None:
    items = tuple(f"item-{index}" for index in range(160))
    histories = {"long": (*reversed(items), items[0], items[1]), "empty": ()}
    model = ItemCFModel()
    model.fit(histories, top_n_neighbors=4)
    expected = _intersection_oracle(histories, 4, 1)
    assert model.to_dict() == {
        "neighbors": {
            item: [[neighbor, score] for neighbor, score in pairs]
            for item, pairs in expected.items()
        }
    }
    # Equal scores use string ordering (item-10 before item-2), including at the cutoff.
    assert model.predict(("item-159", "item-159"), frozenset({"item-1"}), 10) == [
        ("item-0", 2.0),
        ("item-10", 2.0),
        ("item-100", 2.0),
    ]


def test_itemcf_refit_replaces_neighbors_and_clears_empty_models() -> None:
    model = ItemCFModel()
    model.fit({"old": ("old-a", "old-b")})
    model.fit({"new": ("new-a", "new-b")})
    assert model.to_dict() == {"neighbors": {"new-a": [["new-b", 1.0]], "new-b": [["new-a", 1.0]]}}
    assert model.predict(("old-a",), frozenset(), 10) == []
    for histories in ({}, {"empty": (), "single": ("solo", "solo")}):
        model.fit({"old": ("old-a", "old-b")})
        model.fit(histories)
        assert model.to_dict() == {"neighbors": {}}
        assert model.predict(("old-a",), frozenset(), 10) == []
