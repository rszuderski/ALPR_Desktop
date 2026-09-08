#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Short presentation identities for trained model artifacts shown in the GUI."""

from __future__ import annotations

from dataclasses import dataclass
import datetime as _dt
import hashlib
from pathlib import Path
import re
from typing import Any, Mapping


_TIMESTAMP_RE = re.compile(r"(20\d{6})[_-](\d{6})")


@dataclass(frozen=True)
class ModelDisplayRef:
    """Stable, short UI identity for a model without renaming files on disk."""

    id: str
    target: str
    target_label: str
    file_name: str
    path: str
    source_run_id: str = ""
    source_run_label: str = ""
    created_label: str = ""

    @property
    def compact_label(self) -> str:
        return self.id

    @property
    def combo_label(self) -> str:
        parts = [self.id, self.target_label]
        if self.source_run_label:
            parts.append(self.source_run_label)
        return " | ".join(part for part in parts if part)

    @property
    def detail_label(self) -> str:
        parts = [self.id, self.target_label]
        if self.created_label:
            parts.append(f"utworzono {self.created_label}")
        if self.source_run_label:
            parts.append(f"run {self.source_run_label}")
        if self.file_name:
            parts.append(self.file_name)
        return " | ".join(part for part in parts if part)


def build_model_display_ref(
    path_like: Any = None,
    *,
    run: Any = None,
    target_hint: str | None = None,
    source_run_label: str | None = None,
) -> ModelDisplayRef:
    data = run if isinstance(run, Mapping) else {}
    if not data and run is not None:
        data = _object_to_mapping(run)

    raw_path = _first_text(
        path_like,
        _mapping_text(data, "best_weights"),
        _mapping_text(data, "last_weights"),
        _mapping_text(data, "model_path"),
    )
    path = _safe_path(raw_path)
    file_name = path.name if path is not None else Path(raw_path).name if raw_path else "-"
    target = _normalize_target(target_hint) or _infer_target(raw_path, data)
    source_run_id = _first_text(
        _mapping_text(data, "id"),
        _mapping_text(data, "run_id"),
        _mapping_text(data, "name"),
    )
    created_label, stamp_code = _timestamp_label(
        _mapping_text(data, "finished_at"),
        _mapping_text(data, "started_at"),
        _mapping_text(data, "created_at"),
        source_run_id,
        raw_path,
    )
    if not created_label and path is not None:
        try:
            stamp = _dt.datetime.fromtimestamp(path.stat().st_mtime)
            created_label = stamp.strftime("%Y-%m-%d %H:%M")
            stamp_code = stamp.strftime("%y%m%d-%H%M")
        except Exception:
            pass

    identity = "|".join(part for part in (raw_path, source_run_id, target) if part)
    if not identity:
        identity = repr(path_like or run or "")
    digest = hashlib.blake2b(identity.encode("utf-8", errors="ignore"), digest_size=2).hexdigest().upper()
    code = _target_code(target)
    model_id = f"{code}-{stamp_code}-{digest}" if stamp_code else f"{code}-{digest}"
    return ModelDisplayRef(
        id=model_id,
        target=target or "unknown",
        target_label=_target_label(target),
        file_name=file_name,
        path=str(path or raw_path or ""),
        source_run_id=source_run_id,
        source_run_label=str(source_run_label or "").strip(),
        created_label=created_label,
    )


def model_display_id(path_like: Any = None, *, run: Any = None, target_hint: str | None = None) -> str:
    return build_model_display_ref(path_like, run=run, target_hint=target_hint).id


def _object_to_mapping(source: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in (
        "id",
        "run_id",
        "name",
        "best_weights",
        "last_weights",
        "model_path",
        "dataset_path",
        "base_model",
        "output_dir",
        "started_at",
        "created_at",
        "finished_at",
    ):
        try:
            result[key] = getattr(source, key)
        except Exception:
            pass
    return result


def _mapping_text(data: Mapping[str, Any], key: str) -> str:
    try:
        return str(data.get(key, "") or "").strip()
    except Exception:
        return ""


def _first_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _safe_path(raw: str) -> Path | None:
    if not raw:
        return None
    try:
        return Path(raw)
    except Exception:
        return None


def _normalize_target(target: str | None) -> str:
    raw = str(target or "").strip().lower()
    if raw in {"char", "chars", "character", "characters", "znaki", "znak"}:
        return "char"
    if raw in {"plate", "plates", "tablice", "tablica"}:
        return "plate"
    if raw in {"vehicle", "vehicles", "pojazdy", "pojazd"}:
        return "vehicle"
    return ""


def _infer_target(path_text: str, data: Mapping[str, Any]) -> str:
    merged = " ".join(
        str(value or "").lower()
        for value in (
            path_text,
            _mapping_text(data, "dataset_path"),
            _mapping_text(data, "base_model"),
            _mapping_text(data, "output_dir"),
            _mapping_text(data, "name"),
        )
    )
    if any(token in merged for token in ("char", "znak", "gold")):
        return "char"
    if any(token in merged for token in ("plate", "tablic", "pose")):
        return "plate"
    if any(token in merged for token in ("vehicle", "pojazd")):
        return "vehicle"
    return "unknown"


def _target_code(target: str) -> str:
    return {
        "char": "MZ",
        "plate": "MT",
        "vehicle": "MP",
    }.get(target, "MD")


def _target_label(target: str) -> str:
    return {
        "char": "model znakow",
        "plate": "model tablic",
        "vehicle": "model pojazdow",
    }.get(target, "model")


def _timestamp_label(*texts: str) -> tuple[str, str]:
    for text in texts:
        raw = str(text or "").strip()
        if not raw:
            continue
        match = _TIMESTAMP_RE.search(raw)
        if match:
            try:
                stamp = _dt.datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")
                return stamp.strftime("%Y-%m-%d %H:%M"), stamp.strftime("%y%m%d-%H%M")
            except Exception:
                pass
        try:
            stamp = _dt.datetime.fromisoformat(raw)
            return stamp.strftime("%Y-%m-%d %H:%M"), stamp.strftime("%y%m%d-%H%M")
        except Exception:
            continue
    return "", ""
