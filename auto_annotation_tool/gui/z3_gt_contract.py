"""Shared PZ2/GT contract helpers.

Keeps RAW result identity and revision-set normalization consistent across
validation, GT Assist, dataset provenance and training benchmark metadata.
"""

from __future__ import annotations

import hashlib
import json


def normalize_revision_ids(value=None, singular=None) -> list[str]:
    values: list[str] = []

    if isinstance(value, (list, tuple, set)):
        values.extend(
            str(item or "").strip()
            for item in value
            if str(item or "").strip()
        )
    else:
        raw = str(value or "").strip()
        if raw:
            parsed = None
            if raw.startswith("["):
                try:
                    parsed = json.loads(raw)
                except Exception:
                    parsed = None
            if isinstance(parsed, list):
                values.extend(
                    str(item or "").strip()
                    for item in parsed
                    if str(item or "").strip()
                )
            else:
                values.append(raw)

    single = str(singular or "").strip()
    if single:
        values.append(single)

    return sorted(set(values))


def revision_ids_from_data(
    data: dict | None,
    plural_key: str,
    singular_key: str,
) -> list[str]:
    source = data if isinstance(data, dict) else {}
    return normalize_revision_ids(
        source.get(plural_key),
        source.get(singular_key),
    )


def canonical_raw_detection_hash(raw_detection: dict | None) -> str:
    """Content hash of model RAW output; intentionally independent of GT."""
    raw = raw_detection if isinstance(raw_detection, dict) else {}
    characters = raw.get("characters", [])
    if not isinstance(characters, list):
        characters = []

    prepared = []
    for record in characters:
        if not isinstance(record, dict):
            continue

        bbox = record.get("bbox", [])
        safe_bbox = []
        if isinstance(bbox, (list, tuple)):
            for value in list(bbox)[:4]:
                try:
                    safe_bbox.append(round(float(value), 6))
                except Exception:
                    safe_bbox.append(0.0)

        try:
            confidence = round(
                float(record.get("confidence", 0.0) or 0.0),
                6,
            )
        except Exception:
            confidence = 0.0

        prepared.append(
            {
                "character": str(
                    record.get("character", "") or ""
                ),
                "bbox": safe_bbox,
                "confidence": confidence,
                "method": str(
                    record.get("method", "") or ""
                ),
            }
        )

    core = {
        "contract": str(raw.get("contract", "") or ""),
        "prediction_text": str(
            raw.get("prediction_text", "") or ""
        ).strip().upper(),
        "characters": prepared,
    }
    payload = json.dumps(
        core,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
