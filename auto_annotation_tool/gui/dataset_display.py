#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Presentation identity for YOLO datasets shown in the GUI."""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime as _dt
import hashlib
from pathlib import Path
import re
from typing import Callable, Mapping

from ..utils import safe_load_yaml


_TIMESTAMP_RE = re.compile(r"(20\d{6})[_-](\d{6})")


@dataclass(frozen=True)
class DatasetDisplayRef:
    """Stable, short UI identity for a dataset without renaming files on disk."""

    id: str
    target: str
    target_label: str
    name: str
    path: str
    yaml_path: str
    counts: dict[str, int] = field(default_factory=dict)
    created_label: str = ""

    @property
    def split_label(self) -> str:
        train = int(self.counts.get("train", 0) or 0)
        val = int(self.counts.get("val", 0) or 0)
        test = int(self.counts.get("test", 0) or 0)
        total = int(self.counts.get("total", 0) or (train + val + test))
        if total <= 0:
            return "bez liczników splitu"
        return f"train {train} | val {val} | test {test} | razem {total}"

    @property
    def compact_label(self) -> str:
        return f"{self.id} | {self.target_label}"

    @property
    def combo_label(self) -> str:
        return f"{self.id} | {self.split_label}"

    @property
    def table_label(self) -> str:
        return self.id

    @property
    def detail_label(self) -> str:
        parts = [self.id, self.target_label, self.split_label]
        if self.created_label:
            parts.append(f"utworzono {self.created_label}")
        if self.name:
            parts.append(self.name)
        return " | ".join(part for part in parts if part)


def build_dataset_display_ref(
    path_like,
    *,
    target_hint: str | None = None,
    counts: Mapping[str, int] | None = None,
    count_loader: Callable[[Path], Mapping[str, int]] | None = None,
) -> DatasetDisplayRef:
    raw = str(path_like or "").strip()
    root = _dataset_root(raw)
    yaml_path = _dataset_yaml_path(root)
    normalized = _normalized_identity_path(root, raw)
    target = _normalize_target(target_hint) or _infer_target(root, yaml_path)
    name = root.name if root is not None else Path(raw).name
    safe_counts = _normalize_counts(counts)
    if not safe_counts and root is not None and callable(count_loader):
        try:
            safe_counts = _normalize_counts(count_loader(root))
        except Exception:
            safe_counts = {}
    created_label, stamp_code = _dataset_timestamp(root, name)
    digest = hashlib.blake2b(normalized.encode("utf-8", errors="ignore"), digest_size=2).hexdigest().upper()
    target_code = _target_code(target)
    if stamp_code:
        dataset_id = f"DS-{target_code}-{stamp_code}-{digest}"
    else:
        dataset_id = f"DS-{target_code}-{digest}"
    return DatasetDisplayRef(
        id=dataset_id,
        target=target or "unknown",
        target_label=_target_label(target),
        name=name or "-",
        path=str(root or raw),
        yaml_path=str(yaml_path or ""),
        counts=safe_counts,
        created_label=created_label,
    )


def dataset_id(path_like, *, target_hint: str | None = None) -> str:
    return build_dataset_display_ref(path_like, target_hint=target_hint).id


def _dataset_root(raw: str) -> Path | None:
    if not raw:
        return None
    try:
        path = Path(raw)
        if path.name.lower() == "data.yaml":
            return path.parent
        return path
    except Exception:
        return None


def _dataset_yaml_path(root: Path | None) -> Path | None:
    if root is None:
        return None
    try:
        if root.is_file() and root.name.lower() == "data.yaml":
            return root
        candidate = root / "data.yaml"
        return candidate if candidate.exists() else None
    except Exception:
        return None


def _normalized_identity_path(root: Path | None, raw: str) -> str:
    try:
        if root is not None:
            return str(root.resolve()).lower()
    except Exception:
        pass
    return str(raw or "").strip().lower()


def _normalize_counts(counts: Mapping[str, int] | None) -> dict[str, int]:
    if not isinstance(counts, Mapping):
        return {}
    result: dict[str, int] = {}
    for key in ("train", "val", "test", "total"):
        try:
            result[key] = int(counts.get(key, 0) or 0)
        except Exception:
            result[key] = 0
    if int(result.get("total", 0) or 0) <= 0:
        result["total"] = int(result.get("train", 0) or 0) + int(result.get("val", 0) or 0) + int(result.get("test", 0) or 0)
    return result


def _normalize_target(target: str | None) -> str:
    raw = str(target or "").strip().lower()
    if raw in {"char", "chars", "character", "characters", "znaki", "znak"}:
        return "char"
    if raw in {"plate", "plates", "tablice", "tablica"}:
        return "plate"
    if raw in {"vehicle", "vehicles", "pojazdy", "pojazd"}:
        return "vehicle"
    return ""


def _infer_target(root: Path | None, yaml_path: Path | None) -> str:
    text = " ".join(part.lower() for part in ((root.name if root is not None else ""), str(root or "")))
    if yaml_path is not None:
        try:
            cfg = safe_load_yaml(yaml_path) or {}
            if isinstance(cfg, dict) and cfg.get("kpt_shape"):
                return "plate"
        except Exception:
            pass
    if any(token in text for token in ("char", "znak", "gold")):
        return "char"
    if any(token in text for token in ("plate", "tablic", "pose")):
        return "plate"
    if any(token in text for token in ("vehicle", "pojazd")):
        return "vehicle"
    return "unknown"


def _target_code(target: str) -> str:
    return {
        "char": "ZN",
        "plate": "TB",
        "vehicle": "PJ",
    }.get(target, "XX")


def _target_label(target: str) -> str:
    return {
        "char": "znaki",
        "plate": "tablice",
        "vehicle": "pojazdy",
    }.get(target, "dataset")


def _dataset_timestamp(root: Path | None, name: str) -> tuple[str, str]:
    match = _TIMESTAMP_RE.search(str(name or ""))
    if match:
        try:
            stamp = _dt.datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")
            return stamp.strftime("%Y-%m-%d %H:%M"), stamp.strftime("%y%m%d-%H%M")
        except Exception:
            pass
    if root is not None:
        try:
            stamp = _dt.datetime.fromtimestamp(root.stat().st_mtime)
            return stamp.strftime("%Y-%m-%d %H:%M"), stamp.strftime("%y%m%d-%H%M")
        except Exception:
            pass
    return "", ""
