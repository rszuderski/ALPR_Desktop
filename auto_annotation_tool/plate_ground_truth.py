"""Ground-truth contract for plate annotations.

Ground truth belongs to a concrete plate annotation (polygon), not to the
source image filename. The module deliberately contains no Z2/Z3 UI logic.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any
from uuid import uuid4

from .registration_text import normalize_registration


PLATE_ANNOTATION_ID_ATTR = "plate_annotation_id"
GROUND_TRUTH_TEXT_ATTR = "ground_truth_text"
GROUND_TRUTH_SOURCE_ATTR = "ground_truth_source"

GROUND_TRUTH_SOURCE_MANUAL_Z2 = "manual_z2"

PLATE_LAYOUT_GT_ATTR = "plate_layout_gt"
PLATE_LAYOUT_SINGLE_ROW = "single_row"
PLATE_LAYOUT_TWO_ROW = "two_row"
PLATE_LAYOUT_GT_VALUES = frozenset({
    PLATE_LAYOUT_SINGLE_ROW,
    PLATE_LAYOUT_TWO_ROW,
})


def normalize_plate_ground_truth_text(value: Any) -> str:
    """Return canonical registration text stored as plate ground truth."""
    return normalize_registration(value)


def ensure_plate_annotation_id(
    attributes: MutableMapping[str, Any],
    *,
    id_factory=None,
) -> str:
    """Ensure that one plate polygon has a stable annotation identifier."""
    existing = str(attributes.get(PLATE_ANNOTATION_ID_ATTR) or "").strip()
    if existing:
        return existing

    factory = id_factory or (lambda: uuid4().hex)
    raw_id = str(factory() or "").strip()
    if not raw_id:
        raw_id = uuid4().hex
    annotation_id = f"plate-ann-{raw_id}"
    attributes[PLATE_ANNOTATION_ID_ATTR] = annotation_id
    return annotation_id



def normalize_plate_layout_gt(
    value: Any,
    *,
    default: str = PLATE_LAYOUT_SINGLE_ROW,
) -> str:
    """Normalize Z2 row-layout GT to the vocabulary already used by Z3."""
    raw = str(value or "").strip().lower()
    aliases = {
        "1": PLATE_LAYOUT_SINGLE_ROW,
        "1r": PLATE_LAYOUT_SINGLE_ROW,
        "1-row": PLATE_LAYOUT_SINGLE_ROW,
        "single": PLATE_LAYOUT_SINGLE_ROW,
        "single_row": PLATE_LAYOUT_SINGLE_ROW,
        "single-row": PLATE_LAYOUT_SINGLE_ROW,
        "2": PLATE_LAYOUT_TWO_ROW,
        "2r": PLATE_LAYOUT_TWO_ROW,
        "2-row": PLATE_LAYOUT_TWO_ROW,
        "double": PLATE_LAYOUT_TWO_ROW,
        "two_row": PLATE_LAYOUT_TWO_ROW,
        "two-row": PLATE_LAYOUT_TWO_ROW,
    }
    normalized = aliases.get(raw, raw)
    if normalized in PLATE_LAYOUT_GT_VALUES:
        return normalized

    fallback = str(default or "").strip().lower()
    if fallback in PLATE_LAYOUT_GT_VALUES:
        return fallback
    return ""


def get_plate_layout_gt(
    attributes: MutableMapping[str, Any] | None,
    *,
    default: str = PLATE_LAYOUT_SINGLE_ROW,
) -> str:
    if not isinstance(attributes, MutableMapping):
        return normalize_plate_layout_gt("", default=default)
    return normalize_plate_layout_gt(
        attributes.get(PLATE_LAYOUT_GT_ATTR),
        default=default,
    )


def has_explicit_plate_layout_gt(
    attributes: MutableMapping[str, Any] | None,
) -> bool:
    if not isinstance(attributes, MutableMapping):
        return False
    raw = str(attributes.get(PLATE_LAYOUT_GT_ATTR) or "").strip()
    return bool(
        raw
        and normalize_plate_layout_gt(raw, default="")
        in PLATE_LAYOUT_GT_VALUES
    )


def set_plate_layout_gt(
    attributes: MutableMapping[str, Any],
    layout: Any,
) -> str:
    ensure_plate_annotation_id(attributes)
    normalized = normalize_plate_layout_gt(layout)
    attributes[PLATE_LAYOUT_GT_ATTR] = normalized
    return normalized


def ensure_plate_layout_gt(
    attributes: MutableMapping[str, Any],
) -> str:
    """Materialize Z2's visible default 1R when this plate is edited."""
    return set_plate_layout_gt(
        attributes,
        get_plate_layout_gt(attributes),
    )


def get_plate_ground_truth(attributes: MutableMapping[str, Any] | None) -> str:
    if not isinstance(attributes, MutableMapping):
        return ""
    return normalize_plate_ground_truth_text(
        attributes.get(GROUND_TRUTH_TEXT_ATTR)
    )


def set_plate_ground_truth(
    attributes: MutableMapping[str, Any],
    text: Any,
    *,
    source: str = GROUND_TRUTH_SOURCE_MANUAL_Z2,
) -> str:
    """Assign or clear GT on one concrete plate annotation."""
    ensure_plate_annotation_id(attributes)
    normalized = normalize_plate_ground_truth_text(text)
    if not normalized:
        attributes.pop(GROUND_TRUTH_TEXT_ATTR, None)
        attributes.pop(GROUND_TRUTH_SOURCE_ATTR, None)
        return ""

    attributes[GROUND_TRUTH_TEXT_ATTR] = normalized
    attributes[GROUND_TRUTH_SOURCE_ATTR] = str(
        source or GROUND_TRUTH_SOURCE_MANUAL_Z2
    ).strip()
    return normalized


def ensure_plate_detection_contract(detection) -> dict[str, Any]:
    """Ensure identity and normalize already-present GT on a Detection."""
    attributes = dict(getattr(detection, "attributes", {}) or {})
    ensure_plate_annotation_id(attributes)

    if GROUND_TRUTH_TEXT_ATTR in attributes:
        normalized = normalize_plate_ground_truth_text(
            attributes.get(GROUND_TRUTH_TEXT_ATTR)
        )
        if normalized:
            attributes[GROUND_TRUTH_TEXT_ATTR] = normalized
        else:
            attributes.pop(GROUND_TRUTH_TEXT_ATTR, None)
            attributes.pop(GROUND_TRUTH_SOURCE_ATTR, None)

    attributes[PLATE_LAYOUT_GT_ATTR] = get_plate_layout_gt(attributes)

    detection.attributes = attributes
    return attributes
