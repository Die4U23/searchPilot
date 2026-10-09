"""MIND Small 下载、校验与安全解压。"""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

# 官方 blob（mind201910small.blob.core.windows.net）自 2024-07 起关闭公共访问（HTTP 409
# PublicAccessNotPermitted），见 msnews/MIND#17。2024 年的 z20 镜像
# （recodatasets.z20.web.core.windows.net，recommenders#2145）到 2026-10-09 已是 NXDOMAIN。
# 与 recommenders/datasets/mind.py 当前常量一致，改用 Hugging Face 上的 Recommenders/MIND。
MIND_URLS = {
    "train": "https://huggingface.co/datasets/Recommenders/MIND/resolve/main/MINDsmall_train.zip",
    "dev": "https://huggingface.co/datasets/Recommenders/MIND/resolve/main/MINDsmall_dev.zip",
}
DOWNLOAD_MANIFEST = "download_manifest.json"


def sha256_file(path: Path) -> str:
    """流式计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"files": {}}
    with path.open(encoding="utf-8") as stream:
        loaded = json.load(stream)
    if not isinstance(loaded, dict) or not isinstance(loaded.get("files"), dict):
        raise ValueError(f"invalid download manifest: {path}")
    return loaded


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"unsafe zip member: {member.filename}")
        bundle.extractall(destination)


def download_mind(raw_dir: Path) -> dict[str, Any]:
    """下载 MIND Small train/dev，首次记录实际摘要，后续严格校验。"""
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / DOWNLOAD_MANIFEST
    manifest = _read_manifest(manifest_path)
    files = manifest["files"]

    for split, url in MIND_URLS.items():
        archive = raw_dir / f"MINDsmall_{split}.zip"
        expected = files.get(split, {}).get("sha256")
        if archive.exists():
            actual = sha256_file(archive)
            if expected is not None and actual != expected:
                raise ValueError(f"SHA-256 mismatch for {archive.name}")
        else:
            partial = archive.with_suffix(".zip.part")
            try:
                with (
                    urllib.request.urlopen(url, timeout=60) as response,  # noqa: S310
                    partial.open("wb") as stream,
                ):
                    shutil.copyfileobj(response, stream)
                actual = sha256_file(partial)
                if expected is not None and actual != expected:
                    raise ValueError(f"SHA-256 mismatch for {archive.name}")
                partial.replace(archive)
            finally:
                partial.unlink(missing_ok=True)
        files[split] = {
            "url": url,
            "name": archive.name,
            "sha256": actual,
            "bytes": archive.stat().st_size,
        }
        _safe_extract(archive, raw_dir / split)

    with manifest_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return manifest
