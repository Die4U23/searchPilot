from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from searchpilot.data.download import sha256_file
from searchpilot.data.manifest import INPUT_PATHS, build_dataset, compute_data_version
from searchpilot.data.schemas import (
    IMPRESSIONS_SCHEMA,
    ITEMS_SCHEMA,
    USER_HISTORY_SCHEMA,
)


def _write_raw_sample(root: Path) -> None:
    for split in ("train", "dev"):
        (root / split).mkdir(parents=True)
    (root / "train/news.tsv").write_text(
        "N1\tnews\tworld\tOne\tAbstract one\thttps://e/1\t[]\t[]\n"
        "N2\tsports\tfootball\tTwo\t\thttps://e/2\t[]\t[]\n",
        encoding="utf-8",
    )
    (root / "dev/news.tsv").write_text(
        "N2\tsports\tfootball\tTwo\t\thttps://e/2\t[]\t[]\n"
        "N3\tfinance\tmarkets\tThree\tAbstract three\thttps://e/3\t[]\t[]\n",
        encoding="utf-8",
    )
    (root / "train/behaviors.tsv").write_text(
        "I1\tU1\t11/01/2019 9:00:00 AM\tN9\tN1-1 N2-0\n"
        "I2\tU1\t11/02/2019 9:00:00 AM\tN9 N1\tN2-1\n",
        encoding="utf-8",
    )
    (root / "dev/behaviors.tsv").write_text(
        "I3\tU1\t11/03/2019 9:00:00 AM\tN9 N1 N2\tN3-0\n",
        encoding="utf-8",
    )


def test_data_version_is_stable_and_rehashes_input_digests(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_raw_sample(raw)
    inputs = tuple(raw / relative for relative in INPUT_PATHS)
    expected = hashlib.sha256(
        "".join(sha256_file(path) for path in inputs).encode("ascii")
    ).hexdigest()[:12]

    assert compute_data_version(inputs) == expected
    assert compute_data_version(inputs) == compute_data_version(inputs)


def test_manifest_hashes_and_parquet_schemas(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    output = tmp_path / "processed"
    _write_raw_sample(raw)

    manifest = build_dataset(raw, output)
    version_dir = output / manifest.data_version
    stored = json.loads((version_dir / "manifest.json").read_text(encoding="utf-8"))

    assert stored["data_version"] == manifest.data_version
    for entry in stored["source_files"]:
        assert entry["sha256"] == sha256_file(raw / entry["name"])
    for entry in stored["outputs"]:
        assert entry["sha256"] == sha256_file(version_dir / entry["name"])
    assert pq.read_schema(version_dir / "items.parquet").equals(ITEMS_SCHEMA)
    assert pq.read_schema(version_dir / "impressions.parquet").equals(IMPRESSIONS_SCHEMA)
    assert pq.read_schema(version_dir / "user_history.parquet").equals(USER_HISTORY_SCHEMA)
    assert {entry["name"]: entry["rows"] for entry in stored["outputs"]} == {
        "items.parquet": 3,
        "impressions.parquet": 4,
        "user_history.parquet": 1,
    }
