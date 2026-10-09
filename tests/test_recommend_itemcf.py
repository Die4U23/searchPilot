from __future__ import annotations

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
