"""Production search assembly with tiny artifacts and an offline query encoder."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from searchpilot.ports import Document
from searchpilot.search import service as search_module
from searchpilot.search.artifacts import bm25_artifact_dir, save_index
from searchpilot.search.errors import SearchNotReadyError
from searchpilot.search.ltr import FEATURE_NAMES, assemble_ltr_model, save_ltr
from searchpilot.search.service import InMemorySearchService, build_search_service
from searchpilot.search.vector_index import VectorIndex, make_vector_meta, save_vector_index


@pytest.fixture
def artifacts(tmp_path: Path) -> Path:
    docs = [Document("N1", "alpha news", "alpha report", "news", "world")]
    memory = InMemorySearchService(docs)
    save_index(memory.index, memory.meta, bm25_artifact_dir(tmp_path))
    matrix = np.array([[1.0, 0.0]], dtype=np.float32)
    meta = make_vector_meta(
        matrix=matrix, model_name="offline-fixture", data_version="fixture", input_sha256={}
    )
    save_vector_index(tmp_path / "search" / "vector", VectorIndex(("N1",), ("news",), matrix, meta))
    width = len(FEATURE_NAMES)
    save_ltr(
        tmp_path / "search" / "ltr",
        assemble_ltr_model(
            weights=(0.0,) * width,
            bias=0.0,
            mean=(0.0,) * width,
            std=(1.0,) * width,
            reference_time=None,
            clicks={},
            title_chars={},
            first_seen_at={},
        ),
    )
    return tmp_path


def _patch_encoder(
    monkeypatch: pytest.MonkeyPatch, output: np.ndarray | Exception
) -> tuple[list[tuple[str, str]], list[str]]:
    constructions: list[tuple[str, str]] = []
    queries: list[str] = []

    def encode(query: str) -> np.ndarray:
        queries.append(query)
        if isinstance(output, Exception):
            raise output
        return output

    def construct(name: str, prefix: str) -> Callable[[str], np.ndarray]:
        constructions.append((name, prefix))
        return encode

    monkeypatch.setattr(search_module, "SentenceTransformerQueryEncoder", construct)
    return constructions, queries


def test_default_builder_keeps_encoder_lazy(
    artifacts: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    constructions, queries = _patch_encoder(monkeypatch, np.array([1.0, 0.0]))
    service = build_search_service(artifacts)
    assert len(constructions) == 1
    assert queries == []
    assert service.vector_ready is True
    assert service.search("alpha", 1, "vector").hits[0].item_id == "N1"
    assert queries == ["alpha"]


def test_enabled_warmup_uses_same_encoder_and_makes_ready(
    artifacts: Path,
    monkeypatch: pytest.MonkeyPatch,
    fakes: SimpleNamespace,
    make_client: Callable[..., TestClient],
) -> None:
    constructions, queries = _patch_encoder(monkeypatch, np.array([1.0, 0.0]))
    service = build_search_service(artifacts, warmup_vector=True)
    assert len(constructions) == 1
    assert len(queries) == 1 and queries[0].strip()
    assert service.vector_ready is True
    client = make_client(
        search=service, recommend=fakes.FakeRecommend(), feedback=fakes.FakeFeedbackStore()
    )
    assert client.get("/health/ready").status_code == 200
    assert len(queries) == 1  # Health reads state; it does not run inference.
    assert service.search("alpha", 1, "vector").hits[0].item_id == "N1"
    assert len(constructions) == 1
    assert len(queries) == 2 and queries[-1] == "alpha"


@pytest.mark.parametrize(
    "output",
    [
        pytest.param(ImportError("private-marker dependency missing"), id="missing-dependency"),
        pytest.param(RuntimeError("private-marker model failure"), id="encode-error"),
        pytest.param(np.array([1.0]), id="dimension-mismatch"),
        pytest.param(np.array([[1.0, 0.0]]), id="wrong-shape"),
        pytest.param(np.array([np.nan, 1.0]), id="nan"),
        pytest.param(np.array([np.inf, 1.0]), id="infinity"),
        pytest.param(np.array([0.0, 0.0]), id="zero"),
    ],
)
def test_warmup_failure_retains_bm25_and_reports_not_ready(
    artifacts: Path,
    monkeypatch: pytest.MonkeyPatch,
    fakes: SimpleNamespace,
    output: np.ndarray | Exception,
    caplog: pytest.LogCaptureFixture,
    make_client: Callable[..., TestClient],
) -> None:
    _, queries = _patch_encoder(monkeypatch, output)
    service = build_search_service(artifacts, warmup_vector=True)
    assert len(queries) == 1
    assert service.vector_ready is False
    assert service.ltr_ready is True
    assert "vector warmup failed" in caplog.text
    assert "private-marker" not in caplog.text
    client = make_client(
        search=service, recommend=fakes.FakeRecommend(), feedback=fakes.FakeFeedbackStore()
    )
    assert client.get("/health/live").status_code == 200
    response = client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["checks"] == {
        "search": True,
        "recommend": True,
        "feedback": True,
        "vector_index": False,
        "ltr": True,
    }
    response = client.post("/search", json={"query": "alpha", "mode": "bm25"})
    assert response.status_code == 200
    assert response.json()["results"][0]["item_id"] == "N1"
    for mode in ("vector", "hybrid", "ltr"):
        response = client.post("/search", json={"query": "alpha", "mode": mode})
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "NOT_READY"
    assert len(queries) == 1


def test_warmup_does_not_swallow_encoder_wrapper_construction_failure(
    artifacts: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_encoder(name: str, prefix: str) -> None:
        raise RuntimeError("offline wrapper construction failure")

    monkeypatch.setattr(search_module, "SentenceTransformerQueryEncoder", broken_encoder)
    with pytest.raises(RuntimeError, match="wrapper construction failure"):
        build_search_service(artifacts, warmup_vector=True)


def test_no_vector_artifacts_does_not_construct_encoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = InMemorySearchService([Document("N1", "alpha", "report", "news", "world")])
    save_index(memory.index, memory.meta, bm25_artifact_dir(tmp_path))
    constructions, queries = _patch_encoder(monkeypatch, RuntimeError("must not be called"))
    service = build_search_service(tmp_path, warmup_vector=True)
    assert constructions == []
    assert queries == []
    assert service.vector_ready is False
    assert service.search("alpha", 1, "bm25").hits


def test_warmup_does_not_swallow_corrupt_vector_index(
    artifacts: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    np.save(artifacts / "search" / "vector" / "embeddings.npy", np.array([[0.0, 1.0]]))
    constructions, queries = _patch_encoder(monkeypatch, np.array([1.0, 0.0]))
    with pytest.raises(SearchNotReadyError, match="hash mismatch"):
        build_search_service(artifacts, warmup_vector=True)
    assert constructions == []
    assert queries == []
