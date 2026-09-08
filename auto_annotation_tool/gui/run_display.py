#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Short presentation identities for long run names shown in the GUI."""

from __future__ import annotations

from dataclasses import dataclass
import datetime as _dt
import hashlib
from pathlib import Path
import re
from typing import Any, Mapping


_TIMESTAMP_RE = re.compile(r"(20\d{6})[_-](\d{6})")


@dataclass(frozen=True)
class RunDisplayRef:
    """Stable, short UI identity for a run without changing its real name."""

    id: str
    kind: str
    kind_label: str
    name: str
    path: str
    created_label: str = ""

    @property
    def compact_label(self) -> str:
        return self.id

    @property
    def detail_label(self) -> str:
        parts = [self.id, self.kind_label]
        if self.created_label:
            parts.append(f"utworzono {self.created_label}")
        if self.name:
            parts.append(self.name)
        return " | ".join(part for part in parts if part)


def build_run_display_ref(
    source: Any = None,
    *,
    kind_hint: str | None = None,
    run_id: str | None = None,
    path: str | None = None,
    name: str | None = None,
) -> RunDisplayRef:
    """Build a readable run ID such as TRN-260722-1404-A1B2."""

    data = source if isinstance(source, Mapping) else {}
    if not data and source is not None and not isinstance(source, (str, Path)):
        data = _object_to_mapping(source)

    resolved_path = _first_text(
        path,
        _mapping_text(data, "output_dir"),
        _mapping_text(data, "run_dir"),
        _mapping_text(data, "path"),
        _mapping_text(data, "restore_run_dir"),
        str(source) if isinstance(source, (str, Path)) else "",
    )
    resolved_name = _first_text(
        name,
        _mapping_text(data, "name"),
        _mapping_text(data, "run_name"),
        _mapping_text(data, "display_name"),
        _mapping_text(data, "id"),
        Path(resolved_path).name if resolved_path else "",
    )
    resolved_run_id = _first_text(
        run_id,
        _mapping_text(data, "run_id"),
        _mapping_text(data, "id"),
        resolved_name,
    )
    kind = _normalize_kind(kind_hint) or _infer_kind(resolved_path, resolved_name)
    existing_display_id = _existing_display_id(resolved_run_id) or _existing_display_id(resolved_name)
    if existing_display_id:
        return RunDisplayRef(
            id=existing_display_id,
            kind=kind or "generic",
            kind_label=_kind_label(kind),
            name=resolved_name,
            path=resolved_path,
        )
    created_label, stamp_code = _timestamp_label(
        resolved_run_id,
        resolved_name,
        resolved_path,
        _mapping_text(data, "started_at"),
        _mapping_text(data, "created_at"),
        _mapping_text(data, "finished_at"),
        _mapping_text(data, "updated_at"),
    )
    identity_parts = (resolved_run_id, kind) if resolved_run_id else (resolved_path, resolved_name, kind)
    identity = "|".join(part for part in identity_parts if part)
    if not identity:
        identity = repr(source)
    digest = hashlib.blake2b(identity.encode("utf-8", errors="ignore"), digest_size=2).hexdigest().upper()
    code = _kind_code(kind)
    run_display_id = f"{code}-{stamp_code}-{digest}" if stamp_code else f"{code}-{digest}"
    return RunDisplayRef(
        id=run_display_id,
        kind=kind or "generic",
        kind_label=_kind_label(kind),
        name=resolved_name,
        path=resolved_path,
        created_label=created_label,
    )


def run_display_id(source: Any = None, *, kind_hint: str | None = None) -> str:
    return build_run_display_ref(source, kind_hint=kind_hint).id


def _object_to_mapping(source: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in (
        "id",
        "run_id",
        "name",
        "run_name",
        "display_name",
        "output_dir",
        "run_dir",
        "path",
        "restore_run_dir",
        "started_at",
        "created_at",
        "finished_at",
        "updated_at",
    ):
        try:
            value = getattr(source, key)
        except Exception:
            continue
        result[key] = value
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


def _normalize_kind(kind: str | None) -> str:
    raw = str(kind or "").strip().lower()
    if raw in {"training", "train", "z4", "trn"}:
        return "training"
    if raw in {"annotation", "annotations", "annot", "auto_annotation", "z2", "at"}:
        return "annotation"
    if raw in {"characters", "char", "z3", "az"}:
        return "character"
    if raw in {"dataset", "split", "z4_pz1"}:
        return "dataset"
    return ""


def _existing_display_id(value: str) -> str:
    raw = str(value or "").strip()
    if re.match(r"^(TRN|Z2|Z3|DST|RUN)-[A-Z0-9-]+$", raw, re.IGNORECASE):
        return raw
    return ""


def _infer_kind(path: str, name: str) -> str:
    text = f"{path} {name}".lower()
    if any(token in text for token in ("training_run", "training_runs", "5_training_runs", "\\train", "/train")):
        return "training"
    if any(token in text for token in ("auto_annotation", "auto_annotations", "annotations.xml", "run_")):
        return "annotation"
    if any(token in text for token in ("char", "character", "gold", "z3")):
        return "character"
    if any(token in text for token in ("dataset", "split", "data.yaml")):
        return "dataset"
    return "generic"


def _kind_code(kind: str) -> str:
    return {
        "training": "TRN",
        "annotation": "Z2",
        "character": "Z3",
        "dataset": "DST",
    }.get(kind, "RUN")


def _kind_label(kind: str) -> str:
    return {
        "training": "run treningu",
        "annotation": "run anotacji",
        "character": "run znaków",
        "dataset": "run datasetu",
    }.get(kind, "run")


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
