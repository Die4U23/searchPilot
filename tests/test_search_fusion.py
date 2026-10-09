from __future__ import annotations

import pytest

from searchpilot.search.fusion import normalized_score_fusion, reciprocal_rank_fusion


def test_rrf_two_rankings_hand_computed() -> None:
    """k=60：
    A: rank1 in r1 (1/61), rank2 in r2 (1/62) → 0.032534
    B: rank2 in r1 (1/62), rank1 in r2 (1/61) → 同 A，按 id 升序 A 在前
    C: rank3 in r1 (1/63) 只出现一次
    D: rank3 in r2 (1/63) 只出现一次，与 C 并列 → C 在前
    """
    fused = reciprocal_rank_fusion([["A", "B", "C"], ["B", "A", "D"]], k=60)
    assert [item for item, _ in fused] == ["A", "B", "C", "D"]
    scores = dict(fused)
    assert scores["A"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["B"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["C"] == pytest.approx(1 / 63)
    assert scores["D"] == pytest.approx(1 / 63)


def test_rrf_small_k_hand_computed() -> None:
    """k=1：r1 = [x, y], r2 = [y, z]
    x: 1/2 = 0.5；y: 1/3 + 1/2 = 0.8333；z: 1/3 = 0.3333 → y, x, z
    """
    fused = reciprocal_rank_fusion([["x", "y"], ["y", "z"]], k=1)
    assert fused[0] == ("y", pytest.approx(1 / 3 + 1 / 2))
    assert fused[1] == ("x", pytest.approx(1 / 2))
    assert fused[2] == ("z", pytest.approx(1 / 3))


def test_rrf_weights() -> None:
    """r1 权重 2、r2 权重 1，k=1：
    x: 2/2 = 1.0；y: 2/3 + 1/2 = 1.1667；z: 1/3 → y, x, z；
    若权重反过来（1, 2）：x: 1/2；y: 1/3 + 2/2 = 1.333；z: 2/3 → y, z, x
    """
    fused = reciprocal_rank_fusion([["x", "y"], ["y", "z"]], k=1, weights=[2.0, 1.0])
    assert [i for i, _ in fused] == ["y", "x", "z"]
    assert dict(fused)["y"] == pytest.approx(2 / 3 + 1 / 2)
    fused2 = reciprocal_rank_fusion([["x", "y"], ["y", "z"]], k=1, weights=[1.0, 2.0])
    assert [i for i, _ in fused2] == ["y", "z", "x"]


def test_rrf_deduplicates_within_a_ranking() -> None:
    fused = reciprocal_rank_fusion([["a", "a", "b"]], k=60)
    assert fused == [("a", pytest.approx(1 / 61)), ("b", pytest.approx(1 / 62))]


def test_rrf_empty_inputs() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_single_ranking_preserves_order() -> None:
    fused = reciprocal_rank_fusion([["c", "a", "b"]])
    assert [i for i, _ in fused] == ["c", "a", "b"]


def test_rrf_invalid_arguments() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=0)
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], weights=[-1.0])


def test_normalized_score_fusion_does_not_add_raw_scores() -> None:
    fused = normalized_score_fusion([[("A", 100.0), ("B", 0.0)], [("B", 1.0), ("A", 0.0)]])
    assert [item for item, _score in fused] == ["A", "B"]
    assert dict(fused)["A"] == pytest.approx(0.5)
    assert dict(fused)["B"] == pytest.approx(0.5)
