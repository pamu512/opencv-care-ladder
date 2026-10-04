"""Pinned Kaggle version + sha256 records (bytes stay gitignored)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(root: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    root = Path(root)
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        records[rel] = sha256_file(path)
    return records


def pin_record(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kaggle_id": entry.get("kaggle_id"),
        "last_updated": entry.get("last_updated"),
        "license": entry.get("license"),
        "slug": entry.get("slug"),
        "total_bytes": entry.get("total_bytes"),
        "version": entry.get("version"),
    }


def metadata_sha256(record: Mapping[str, Any]) -> str:
    blob = json.dumps(dict(record), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def write_manifest(path: Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path
