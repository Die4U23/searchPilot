"""Versioned inference bundles; integrity checks are not authenticity checks."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
BUNDLE_FILES = ("meta.json", "lr.json", "mlp.json", "temperature.json", "feature_tables.json")


def _canonical(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _digest(payload: object) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _versions(documents: dict[str, dict[str, Any]]) -> tuple[str, str]:
    meta = documents["meta.json"]
    inference = {
        "artifact_schema": SCHEMA_VERSION,
        "model_type": meta["model_type"],
        "feature_names": meta["feature_names"],
        "mean": meta["mean"],
        "std": meta["std"],
        "lr": documents["lr.json"],
        "mlp": documents["mlp.json"],
        "temperature": documents["temperature.json"]["T"],
        "feature_tables": documents["feature_tables.json"],
    }
    model = "ctr-v1-" + _digest(inference)
    calibrator = "temperature-v1-" + _digest(
        {"model_version": model, "T": documents["temperature.json"]["T"]}
    )
    return model, calibrator


def _atomic_write(path: Path, data: bytes) -> None:
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".ctr-", delete=False) as stream:
            temporary = stream.name
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def export_bundle(
    directory: Path,
    *,
    meta: dict[str, Any],
    lr: dict[str, Any],
    mlp: dict[str, Any],
    temperature: float,
    feature_tables: dict[str, Any],
) -> tuple[str, str]:
    """Write the manifest last. Partial publication is unavailable, never mixed-ready."""
    documents = {
        "meta.json": {**meta, "artifact_schema": SCHEMA_VERSION, "model_type": "mlp"},
        "lr.json": lr,
        "mlp.json": mlp,
        "temperature.json": {"T": temperature},
        "feature_tables.json": feature_tables,
    }
    model, calibrator = _versions(documents)
    identities = {"model_version": model, "calibrator_version": calibrator}
    documents["meta.json"].update(identities)
    documents["temperature.json"].update(identities)
    # Serialize everything before changing any existing file.
    encoded = {name: _canonical(documents[name]) + b"\n" for name in BUNDLE_FILES}
    manifest = {
        "artifact_schema": SCHEMA_VERSION,
        "model_type": "mlp",
        **identities,
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in encoded.items()},
    }
    manifest_bytes = _canonical(manifest) + b"\n"
    directory.mkdir(parents=True, exist_ok=True)
    if not (directory / MANIFEST_FILENAME).exists():
        # Mark an initial legacy-to-versioned publication unavailable before
        # touching inference files. Existing loaded runtimes remain usable.
        _atomic_write(directory / MANIFEST_FILENAME, b'{"publishing":true}\n')
    for name, data in encoded.items():
        _atomic_write(directory / name, data)
    _atomic_write(directory / MANIFEST_FILENAME, manifest_bytes)
    return model, calibrator


def _object(raw: bytes, name: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain an object")
    return value


def read_bundle(directory: Path) -> dict[str, dict[str, Any]] | None:
    """Return the very bytes verified, parsed once; never reread an inference file."""
    manifest_path = directory / MANIFEST_FILENAME
    if not manifest_path.exists():
        return None
    manifest = _object(manifest_path.read_bytes(), MANIFEST_FILENAME)
    if type(manifest.get("artifact_schema")) is not int or manifest["artifact_schema"] != 1:
        raise ValueError("unsupported CTR artifact schema")
    if manifest.get("model_type") != "mlp":
        raise ValueError("schema 1 requires explicit mlp model_type")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(BUNDLE_FILES):
        raise ValueError("manifest must list exactly the five inference files")
    documents = {}
    for name in BUNDLE_FILES:
        raw = (directory / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != files[name]:
            raise ValueError(f"CTR artifact checksum mismatch: {name}")
        documents[name] = _object(raw, name)
    meta = documents["meta.json"]
    if type(meta.get("artifact_schema")) is not int or meta["artifact_schema"] != 1:
        raise ValueError("metadata schema does not match manifest")
    if meta.get("model_type") != "mlp":
        raise ValueError("metadata model_type does not match manifest")
    model, calibrator = _versions(documents)
    for document in (manifest, meta, documents["temperature.json"]):
        if (
            document.get("model_version") != model
            or document.get("calibrator_version") != calibrator
        ):
            raise ValueError("CTR model/calibrator identity mismatch")
    return documents
