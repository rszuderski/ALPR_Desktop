#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Adapter realnego metadata PZ2 do przenośnego canonical AZ payload."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .az_revision_store import canonicalize_az_payload


@dataclass(frozen=True)
class PZ2AZRevisionContext:
    source_kind: str
    trust_state: str
    source_status: str
    effective_status: str


def pz2_metadata_to_az_payload(
    data: Mapping[str, Any],
    *,
    image_width: float | int | None = None,
    image_height: float | int | None = None,
) -> dict[str, Any]:
    """Przekształć jeden rekord metadata.json PZ2 do canonical AZ payload.

    Bboxy PZ2 są w pikselach. Adapter normalizuje je do [0, 1].
    Wymiary można podać jawnie albo pobrać z plate_image_width/height.
    """
    if not isinstance(data, Mapping):
        raise TypeError("PZ2 metadata row musi być mapowaniem.")

    crop_identity_sha256 = str(
        data.get("crop_identity_sha256") or ""
    ).strip().lower()
    if not crop_identity_sha256:
        raise ValueError(
            "Brak crop_identity_sha256. Rekord musi pochodzić z AZ003C."
        )

    chars = data.get("characters")
    chars = chars if isinstance(chars, list) else []

    width = _dimension(
        image_width if image_width is not None else data.get("plate_image_width"),
        "image_width",
        required=bool(chars),
    )
    height = _dimension(
        image_height if image_height is not None else data.get("plate_image_height"),
        "image_height",
        required=bool(chars),
    )

    canonical_chars = []
    for index, rec in enumerate(chars):
        if not isinstance(rec, Mapping):
            raise ValueError(f"characters[{index}] musi być mapowaniem.")

        bbox = rec.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            raise ValueError(f"characters[{index}].bbox musi mieć 4 wartości.")

        try:
            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"characters[{index}].bbox zawiera nieprawidłowe liczby."
            ) from exc

        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        row = _int_or_zero(rec.get("reading_row"))
        if row <= 0:
            row = 0

        canonical_chars.append(
            {
                "character": str(
                    rec.get("character")
                    or rec.get("text")
                    or rec.get("char")
                    or ""
                ).strip().upper(),
                "bbox": [
                    x1 / width,
                    y1 / height,
                    x2 / width,
                    y2 / height,
                ],
                "row": row,
                "method": str(rec.get("method") or "").strip(),
                "source_kind": _character_source_kind(rec),
                "confidence": _float_or_default(
                    rec.get("confidence"),
                    1.0,
                ),
            }
        )

    gold_raw = data.get("gold_state")
    gold_raw = gold_raw if isinstance(gold_raw, Mapping) else {}

    payload = {
        "crop_identity_sha256": crop_identity_sha256,
        "characters": canonical_chars,
        "layout": {
            "kind": _layout_kind(data),
            "confirmed": _layout_confirmed(data),
        },
        "gold_state": {
            "approved": bool(gold_raw.get("approved", False)),
            "excluded": bool(gold_raw.get("excluded", False)),
            "candidate": bool(gold_raw.get("candidate", False)),
        },
        "status": str(data.get("status") or "unknown").strip().lower()
        or "unknown",
    }

    expected_text = _expected_text(data)
    if expected_text:
        payload["expected_text"] = expected_text

    return canonicalize_az_payload(payload)


def derive_pz2_revision_context(
    data: Mapping[str, Any],
) -> PZ2AZRevisionContext:
    """Wyznacz kontekst rewizji bez zależności od GUI."""
    if not isinstance(data, Mapping):
        raise TypeError("PZ2 metadata row musi być mapowaniem.")

    status = str(data.get("status") or "unknown").strip().lower() or "unknown"

    gold_raw = data.get("gold_state")
    gold_raw = gold_raw if isinstance(gold_raw, Mapping) else {}
    excluded = bool(gold_raw.get("excluded", False))
    approved = bool(gold_raw.get("approved", False))

    source_info = data.get("source_info")
    source_info = source_info if isinstance(source_info, Mapping) else {}
    bucket = str(source_info.get("bucket") or "").strip().lower()
    origin = str(source_info.get("origin") or "").strip().lower()

    char_kinds = {
        _character_source_kind(rec)
        for rec in (data.get("characters") or [])
        if isinstance(rec, Mapping)
    }

    if "cvat_manual" in char_kinds or bucket == "cvat_manual" or origin == "cvat_import":
        source_kind = "cvat_manual"
        trust_state = "external_reviewed"
    elif "local_manual" in char_kinds or bucket == "local_manual" or origin == "preview_editor":
        source_kind = "local_manual"
        trust_state = "local_manual"
    else:
        source_kind = "pz2_detect"
        trust_state = "auto"

    if excluded:
        effective_status = "excluded"
    elif approved:
        effective_status = "approved"
    elif status == "perfect":
        effective_status = "ready"
    elif status in {"needs_fix", "unknown"}:
        effective_status = status
    else:
        effective_status = status

    return PZ2AZRevisionContext(
        source_kind=source_kind,
        trust_state=trust_state,
        source_status=status,
        effective_status=effective_status,
    )


def az_revision_to_pz2_metadata(
    base_data: Mapping[str, Any],
    revision: Mapping[str, Any],
    *,
    image_width: float | int,
    image_height: float | int,
) -> dict[str, Any]:
    """Nałóż trwałą rewizję AZ na kopię rekordu PZ2.

    Funkcja nie mutuje ``base_data``. Warstwa pochodzenia cropa/PZ1 zostaje
    zachowana, natomiast semantyczny stan anotacji znaków pochodzi z AZ.

    ``review_state`` nie jest odtwarzany z AZ, bo canonical AZ nie przechowuje
    sesyjnego kontraktu REVIEW (approved_reference, timestamps itd.). Dzięki
    temu nie fabrykujemy nowej decyzji człowieka podczas reuse.
    """
    import copy

    if not isinstance(base_data, Mapping):
        raise TypeError("base_data musi być mapowaniem.")
    if not isinstance(revision, Mapping):
        raise TypeError("revision musi być mapowaniem.")

    payload_raw = revision.get("payload")
    if not isinstance(payload_raw, Mapping):
        raise ValueError("Rewizja AZ nie zawiera payload.")

    payload = canonicalize_az_payload(payload_raw)
    width = _dimension(image_width, "image_width", required=True)
    height = _dimension(image_height, "image_height", required=True)

    base_crop_id = str(base_data.get("crop_id") or "").strip()
    revision_crop_id = str(revision.get("crop_id") or "").strip()
    if base_crop_id and revision_crop_id and base_crop_id != revision_crop_id:
        raise ValueError("Rewizja AZ należy do innego crop_id.")

    base_identity = str(
        base_data.get("crop_identity_sha256") or ""
    ).strip().lower()
    payload_identity = str(
        payload.get("crop_identity_sha256") or ""
    ).strip().lower()
    if base_identity and base_identity != payload_identity:
        raise ValueError(
            "crop_identity_sha256 rewizji AZ nie odpowiada rekordowi PZ2."
        )

    result = copy.deepcopy(dict(base_data))

    layout_raw = payload.get("layout")
    layout_raw = layout_raw if isinstance(layout_raw, Mapping) else {}
    layout_kind = _az_layout_kind_to_pz2(layout_raw.get("kind"))
    layout_confirmed = bool(layout_raw.get("confirmed", False))

    chars = []
    row_counts: dict[int, int] = {}
    for reading_index, rec in enumerate(payload.get("characters") or [], start=1):
        if not isinstance(rec, Mapping):
            raise ValueError("Nieprawidłowy rekord characters w AZ.")

        bbox = rec.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            raise ValueError("Rekord AZ characters nie ma poprawnego bbox.")

        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        pixel_bbox = [
            round(x1 * width, 6),
            round(y1 * height, 6),
            round(x2 * width, 6),
            round(y2 * height, 6),
        ]

        row = _int_or_zero(rec.get("row"))
        if row <= 0:
            row = 1
        row_counts[row] = row_counts.get(row, 0) + 1

        chars.append(
            {
                "character": str(rec.get("character") or "").strip().upper(),
                "bbox": pixel_bbox,
                "confidence": _float_or_default(rec.get("confidence"), 1.0),
                "method": str(rec.get("method") or "").strip(),
                "source_kind": str(rec.get("source_kind") or "").strip(),
                "reading_row": row,
                "reading_col": row_counts[row],
                "reading_index": reading_index,
            }
        )

    result["characters"] = chars
    result["plate_image_width"] = width
    result["plate_image_height"] = height

    if layout_kind:
        result["plate_layout"] = layout_kind
        result["layout_row_count"] = 2 if layout_kind == "two_row" else 1
        if layout_confirmed:
            result["plate_layout_override"] = layout_kind
            result["layout_source"] = "manual_override"
            result["layout_confidence"] = 1.0
        else:
            result.pop("plate_layout_override", None)
            result.pop("layout_override_source", None)
            result["layout_source"] = "az_reuse"
    else:
        result.pop("plate_layout_override", None)
        result.pop("layout_override_source", None)

    gold_raw = payload.get("gold_state")
    gold_raw = gold_raw if isinstance(gold_raw, Mapping) else {}
    effective_status = str(
        revision.get("effective_status") or ""
    ).strip().lower()
    imported_pending_review = effective_status == "imported_pending_review"

    if imported_pending_review:
        # Import między projektami przenosi zawartość anotacji, ale nie
        # lokalną decyzję zaufania projektu źródłowego. Target musi wykonać
        # własny REVIEW/N/O.
        result["gold_state"] = {
            "approved": False,
            "excluded": False,
            "candidate": False,
        }
        result["status"] = "needs_fix"
    else:
        result["gold_state"] = {
            "approved": bool(gold_raw.get("approved", False)),
            "excluded": bool(gold_raw.get("excluded", False)),
            "candidate": bool(gold_raw.get("candidate", False)),
        }
        result["status"] = str(
            payload.get("status") or "unknown"
        ).strip().lower() or "unknown"

    expected_text = str(payload.get("expected_text") or "").strip().upper()
    if expected_text:
        result["ground_truth_text"] = expected_text

    # REVIEW nie jest odtwarzany z decyzji projektu źródłowego.
    # Dla cross-project importu tworzymy lokalny stan TARGET=in_progress.
    # To nie jest odziedziczone OK, tylko obowiązek lokalnej kontroli.
    if imported_pending_review:
        result["review_state"] = {
            "schema": "alpr.pz2.review.v1",
            "status": "in_progress",
            "source": "az_project_import",
            "human_edited": False,
            "approved_at": None,
        }
    else:
        result.pop("review_state", None)

    result["fusion_strategy"] = "az_reuse"
    result["fusion_details"] = {
        "source": (
            "az_project_import"
            if imported_pending_review
            else "az_registry"
        ),
        "az_revision_id": str(revision.get("az_revision_id") or "").strip(),
        "effective_status": effective_status or None,
    }
    result["az_reuse"] = {
        "schema": "alpr.az_reuse.v1",
        "az_revision_id": str(revision.get("az_revision_id") or "").strip(),
        "payload_sha256": str(revision.get("payload_sha256") or "").strip(),
        "source_kind": str(revision.get("source_kind") or "").strip(),
        "trust_state": str(revision.get("trust_state") or "").strip(),
        "origin_project_id": str(
            revision.get("origin_project_id") or ""
        ).strip() or None,
        "origin_iteration": revision.get("origin_iteration"),
        "created_at": str(revision.get("created_at") or "").strip(),
        "project_id": str(revision.get("project_id") or "").strip() or None,
        "effective_status": effective_status or None,
        "requires_review": bool(imported_pending_review),
    }

    return result


def _az_layout_kind_to_pz2(value: Any) -> str:
    kind = str(value or "").strip().lower()
    if kind in {"single_row", "1r", "1_row"}:
        return "single_row"
    if kind in {"two_row", "2r", "2_row", "two_row_candidate"}:
        return "two_row"
    return ""



def _dimension(value: Any, name: str, *, required: bool) -> float:
    if value is None or str(value).strip() == "":
        if required:
            raise ValueError(
                f"Brak {name}; nie można normalizować bboxów PZ2."
            )
        return 1.0
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} musi być liczbą dodatnią.") from exc
    if number <= 0:
        raise ValueError(f"{name} musi być > 0.")
    return number


def _character_source_kind(rec: Mapping[str, Any]) -> str:
    explicit = str(rec.get("source_kind") or "").strip().lower().replace("-", "_")
    if explicit:
        return explicit

    method = str(rec.get("method") or "").strip().lower().replace("-", "_")
    tag = str(rec.get("source_tag") or "").strip().lower().replace("-", "_")

    if method == "cvat_manual":
        return "cvat_manual"
    if method == "manual" or tag in {"manual", "manual_correction"}:
        return "local_manual"
    if method in {"yolo", "yolo_box", "yolo_symbol", "yolo_ocr"}:
        return "yolo_box_ocr" if method == "yolo_ocr" else method
    if method == "ocr" or tag == "ocr":
        return "ocr"
    return explicit or method or tag or "unknown"


def _layout_kind(data: Mapping[str, Any]) -> str:
    override = str(data.get("plate_layout_override") or "").strip().lower()
    if override in {"single_row", "two_row"}:
        return override

    layout = str(data.get("plate_layout") or "").strip().lower()
    if layout in {"single_row", "two_row"}:
        return layout
    if layout in {"1r", "1_row"}:
        return "single_row"
    if layout in {"2r", "2_row", "two_row_candidate"}:
        return "two_row"
    return "unknown"


def _layout_confirmed(data: Mapping[str, Any]) -> bool:
    override = str(data.get("plate_layout_override") or "").strip().lower()
    source = str(data.get("layout_source") or "").strip().lower()
    return bool(
        override in {"single_row", "two_row"}
        or source == "manual_override"
    )


def _expected_text(data: Mapping[str, Any]) -> str:
    direct = str(data.get("ground_truth_text") or "").strip()
    if direct:
        return direct.upper()

    attrs = data.get("plate_attributes")
    if isinstance(attrs, Mapping):
        nested = str(attrs.get("ground_truth_text") or "").strip()
        if nested:
            return nested.upper()

    source_expected = str(data.get("source_expected_text") or "").strip()
    return source_expected.upper()


def _int_or_zero(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)
