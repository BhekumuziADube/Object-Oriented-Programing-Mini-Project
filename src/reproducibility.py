"""Metadata and checksum helpers for reproducible experiment runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping


def data_checksum(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_data(path: str | Path) -> str:
    data_path = Path(path)
    checksum_path = data_path.with_name(data_path.name + ".sha256")
    expected = checksum_path.read_text(encoding="utf-8").strip().split()[0]
    actual = data_checksum(data_path)
    if actual != expected:
        raise RuntimeError(
            f"Checksum mismatch for {data_path}: expected {expected}, got {actual}"
        )
    return actual


def write_run_metadata(
    output_path: str | Path,
    data_path: str | Path,
    *,
    seeds: Mapping[str, int] | None = None,
    params: Mapping[str, object] | None = None,
) -> Path:
    output_path = Path(output_path)
    data_path = Path(data_path)
    metadata = {
        "data": str(data_path),
        "data_sha256": data_checksum(data_path),
        "seeds": dict(seeds or {}),
        "params": dict(params or {}),
    }
    path = output_path.with_suffix(".meta.json")
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path
