from __future__ import annotations

import copy
from pathlib import Path


def fusion_details_yolo_box_backend_positions(fusion_details) -> set[int]:
    if not isinstance(fusion_details, dict):
        return set()
    backend = str(fusion_details.get("box_backend", "") or "").strip().lower().replace("-", "_")
    if backend not in {"yolo", "yolo_box", "yolo_backend", "yolo_filtered"}:
        return set()
    positions = set()
    for raw_idx in fusion_details.get("box_backend_positions", []) or []:
        try:
            positions.add(int(raw_idx))
        except Exception:
            continue
    return positions


def normalize_character_source_kind(host, rec, data=None, plate_source_bucket: str = "") -> str:
    raw_kind = ""
    raw_tag = ""
    method_name = ""
    if isinstance(rec, dict):
        raw_kind = str(rec.get("source_kind", "") or "").strip().lower().replace("-", "_")
        raw_tag = rec.get("source_tag")
        method_name = str(rec.get("method", "") or "").strip().lower()
    else:
        raw_kind = str(getattr(rec, "source_kind", "") or "").strip().lower().replace("-", "_")
        raw_tag = getattr(rec, "source_tag", None)
        method_name = str(getattr(rec, "method", "") or "").strip().lower()

    explicit_box = _normalized_record_text(rec, "box_source")
    explicit_sign = _normalized_record_text(rec, "sign_source")
    if explicit_box or explicit_sign:
        if explicit_box == "manual_box" or explicit_sign == "manual_sign":
            if str(plate_source_bucket or "").strip().lower() == "cvat_manual":
                return "cvat_manual"
            return "local_manual"
        if explicit_box == "yolo_box" and explicit_sign == "yolo_symbol":
            return "yolo"
        if explicit_box == "yolo_box" and explicit_sign == "ocr_symbol":
            return "yolo_box_ocr"
        if explicit_box == "yolo_box":
            return "yolo_box"
        if explicit_sign == "yolo_symbol":
            return "yolo_symbol"
        return "ocr"

    if raw_kind == "ocr" and host._character_record_uses_yolo_box_backend(rec):
        return "yolo_box_ocr"
    if raw_kind in {"cvat_manual", "local_manual", "yolo_box", "yolo_symbol", "yolo_box_ocr", "yolo_rescue", "yolo", "ocr"}:
        return raw_kind

    if method_name == "cvat_manual":
        return "cvat_manual"
    if method_name == "manual":
        if str(plate_source_bucket or "").strip().lower() == "cvat_manual":
            return "cvat_manual"
        return "local_manual"

    normalized_tag = host._normalize_character_source_tag(raw_tag=raw_tag, method=method_name)
    if normalized_tag == "manual":
        if str(plate_source_bucket or "").strip().lower() == "cvat_manual":
            return "cvat_manual"
        if isinstance(data, dict):
            source_info = data.get("source_info", {})
            if isinstance(source_info, dict) and str(source_info.get("bucket", "") or "").strip().lower() == "cvat_manual":
                return "cvat_manual"
        return "local_manual"
    if normalized_tag == "ocr" and host._character_record_uses_yolo_box_backend(rec):
        return "yolo_box_ocr"
    if normalized_tag in {"yolo_box", "yolo_symbol", "yolo_box_ocr", "yolo_rescue", "yolo", "ocr"}:
        return normalized_tag
    return "ocr"


def _record_text_value(rec, key: str) -> str:
    try:
        value = rec.get(key) if isinstance(rec, dict) else getattr(rec, key, "")
    except Exception:
        value = ""
    return str(value or "").strip()


def _normalized_record_text(rec, key: str) -> str:
    return _record_text_value(rec, key).lower().replace("-", "_")


def _record_has_symbol(host, rec) -> bool:
    try:
        value = rec.get("character", "") if isinstance(rec, dict) else getattr(rec, "character", "")
        return bool(host._sanitize_preview_char_symbol(value))
    except Exception:
        return bool(_record_text_value(rec, "character"))


def normalize_character_box_source(host, rec, data=None, fallback_index: int = 0) -> str:
    explicit = _normalized_record_text(rec, "box_source")
    explicit_sign = _normalized_record_text(rec, "sign_source")
    if explicit in {"manual_box", "manual", "local_manual", "cvat_manual", "preview_editor"}:
        return "manual_box"
    if explicit in {"yolo_box", "yolo", "yolo_backend", "yolo_filtered", "yolo_box_only", "yolo_box_ocr"}:
        return "yolo_box"
    if explicit in {"generated_box", "generated", "ocr", "segment", "segmented", "ocr_segment"}:
        return "generated_box"

    geometry_values = {
        _normalized_record_text(rec, "bbox_source"),
        _normalized_record_text(rec, "geometry_source"),
        _normalized_record_text(rec, "geometry_method"),
        _normalized_record_text(rec, "box_backend"),
        _normalized_record_text(rec, "box_backend_source"),
    }
    if geometry_values.intersection({"manual", "manual_box", "local_manual", "cvat_manual", "preview_editor"}):
        return "manual_box"
    if geometry_values.intersection({"yolo", "yolo_box", "yolo_backend", "yolo_filtered", "yolo_box_only", "yolo_box_ocr"}):
        return "yolo_box"
    if geometry_values.intersection({"generated", "generated_box", "ocr", "segment", "segmented", "ocr_segment"}):
        return "generated_box"

    method_name = _normalized_record_text(rec, "method")
    source_kind = _normalized_record_text(rec, "source_kind")
    if explicit_sign and not explicit:
        source_tag = host._normalize_character_source_tag(
            raw_tag=_record_text_value(rec, "source_tag"),
            method=method_name,
        )
    else:
        source_tag = get_character_source_tag(host, rec, data=data, fallback_index=fallback_index)
        source_tag = str(source_tag or "").strip().lower().replace("-", "_")

    if source_kind in {"local_manual", "cvat_manual"} or source_tag == "manual" or method_name in {"manual", "cvat_manual"}:
        return "manual_box"
    if (
        host._character_record_uses_yolo_box_backend(rec)
        or source_tag in {"yolo", "yolo_box", "yolo_box_ocr", "yolo_rescue"}
        or source_kind in {"yolo", "yolo_box", "yolo_box_ocr", "yolo_rescue"}
        or method_name in {"yolo", "yolo_box", "yolo_ocr"}
    ):
        return "yolo_box"
    return "generated_box"


def normalize_character_sign_source(host, rec, data=None, fallback_index: int = 0) -> str:
    if not _record_has_symbol(host, rec):
        return ""

    explicit = _normalized_record_text(rec, "sign_source")
    if explicit in {"manual_sign", "manual", "local_manual", "cvat_manual", "preview_editor"}:
        return "manual_sign"
    if explicit in {"yolo_symbol", "yolo", "ys", "yolo_rescue"}:
        return "yolo_symbol"
    if explicit in {"ocr_symbol", "ocr", "os", "yolo_box_ocr"}:
        return "ocr_symbol"

    symbol_values = {
        _normalized_record_text(rec, "symbol_source"),
        _normalized_record_text(rec, "symbol_method"),
        _normalized_record_text(rec, "text_source"),
        _normalized_record_text(rec, "character_source"),
    }
    if symbol_values.intersection({"manual", "manual_sign", "local_manual", "cvat_manual", "preview_editor"}):
        return "manual_sign"
    if symbol_values.intersection({"yolo", "yolo_symbol", "ys", "yolo_rescue"}):
        return "yolo_symbol"
    if symbol_values.intersection({"ocr", "ocr_symbol", "os", "yolo_box_ocr"}):
        return "ocr_symbol"

    method_name = _normalized_record_text(rec, "method")
    source_kind = _normalized_record_text(rec, "source_kind")
    explicit_box = _normalized_record_text(rec, "box_source")
    if explicit_box and not explicit:
        source_tag = host._normalize_character_source_tag(
            raw_tag=_record_text_value(rec, "source_tag"),
            method=method_name,
        )
    else:
        source_tag = get_character_source_tag(host, rec, data=data, fallback_index=fallback_index)
        source_tag = str(source_tag or "").strip().lower().replace("-", "_")

    if source_kind in {"local_manual", "cvat_manual"} or source_tag == "manual" or method_name in {"manual", "cvat_manual"}:
        return "manual_sign"
    if source_tag in {"yolo", "yolo_symbol", "yolo_rescue"} or source_kind in {"yolo", "yolo_symbol", "yolo_rescue"} or method_name in {"yolo", "yolo_symbol"}:
        return "yolo_symbol"
    if source_tag in {"ocr", "yolo_box_ocr"} or source_kind in {"ocr", "yolo_box_ocr"} or method_name in {"ocr", "yolo_ocr"}:
        return "ocr_symbol"
    return "ocr_symbol"


def compose_character_source_tag(box_source: str = "", sign_source: str = "") -> str:
    box = str(box_source or "").strip().lower().replace("-", "_")
    sign = str(sign_source or "").strip().lower().replace("-", "_")
    if box == "manual_box" and sign == "manual_sign":
        return "manual"
    if box == "manual_box":
        return "manual"
    if sign == "manual_sign":
        return "manual"
    if box == "yolo_box" and sign == "yolo_symbol":
        return "yolo"
    if box == "yolo_box" and sign == "ocr_symbol":
        return "yolo_box_ocr"
    if box == "yolo_box":
        return "yolo_box"
    if sign == "yolo_symbol":
        return "yolo_symbol"
    return "ocr"


def normalize_plate_source_bucket(host, raw_bucket=None, data=None, meta_path: Path | None = None) -> str:
    bucket = str(raw_bucket or "").strip().lower().replace("-", "_")
    if bucket in {"auto_preview", "auto", "preview", "pz2_detect"}:
        return "auto_preview"
    if bucket in {"local_manual", "manual", "preview_editor", "manual_correction"}:
        return "local_manual"
    if bucket in {"cvat_manual", "cvat", "cvat_import", "manual_pool"}:
        return "cvat_manual"

    if isinstance(data, dict):
        source_info = data.get("source_info", {})
        if isinstance(source_info, dict):
            nested_bucket = str(source_info.get("bucket", "") or "").strip()
            if nested_bucket:
                normalized_nested = normalize_plate_source_bucket(host, nested_bucket, data=None, meta_path=meta_path)
                if normalized_nested:
                    return normalized_nested

        chars = list(data.get("characters", []) or [])
        if any(str((rec or {}).get("method", "") or "").strip().lower() == "cvat_manual" for rec in chars if isinstance(rec, dict)):
            return "cvat_manual"
        if str(data.get("fusion_strategy", "") or "").strip().lower() == "manual_correction":
            return "local_manual"
        if any(normalize_character_source_kind(host, rec, data=data) in {"local_manual", "cvat_manual"} for rec in chars if isinstance(rec, dict)):
            if any(normalize_character_source_kind(host, rec, data=data) == "cvat_manual" for rec in chars if isinstance(rec, dict)):
                return "cvat_manual"
            return "local_manual"

    if meta_path is not None:
        path_parts = {part.lower() for part in meta_path.parts}
        if "manual_char_pool" in path_parts:
            return "cvat_manual"

    return "auto_preview"


def normalize_plate_source_origin(host, raw_origin=None, bucket: str = "") -> str:
    origin = str(raw_origin or "").strip().lower().replace("-", "_")
    if origin in {"pz2_detect", "detect", "auto_detect"}:
        return "pz2_detect"
    if origin in {"preview_editor", "manual_correction", "local_manual"}:
        return "preview_editor"
    if origin in {"cvat_import", "cvat", "manual_pool"}:
        return "cvat_import"

    normalized_bucket = normalize_plate_source_bucket(host, bucket)
    if normalized_bucket == "cvat_manual":
        return "cvat_import"
    if normalized_bucket == "local_manual":
        return "preview_editor"
    return "pz2_detect"


def build_character_source_tags(host, chars, fusion_strategy="", fusion_details=None):
    ordered = host._sort_character_records_by_x(list(chars or []))
    rescue_positions = set()
    if str(fusion_strategy or "").strip().lower() == "ocr_yolo_rescue" and isinstance(fusion_details, dict):
        try:
            rescue_positions = {
                int(idx) for idx in (fusion_details.get("mismatch_positions", []) or [])
                if str(idx).strip() != ""
            }
        except Exception:
            rescue_positions = set()
    backend_positions = fusion_details_yolo_box_backend_positions(fusion_details)

    tags = []
    for idx, rec in enumerate(ordered):
        method_name = ""
        raw_tag = None
        if isinstance(rec, dict):
            raw_tag = rec.get("source_tag")
            method_name = rec.get("method", "")
        else:
            raw_tag = getattr(rec, "source_tag", None)
            method_name = getattr(rec, "method", "")

        explicit_box = _normalized_record_text(rec, "box_source")
        explicit_sign = _normalized_record_text(rec, "sign_source")
        if explicit_box or explicit_sign:
            normalized = compose_character_source_tag(
                normalize_character_box_source(host, rec, fallback_index=idx),
                normalize_character_sign_source(host, rec, fallback_index=idx),
            )
        else:
            normalized = host._normalize_character_source_tag(raw_tag=raw_tag, method=method_name)
        if idx in rescue_positions:
            normalized = "yolo_rescue"
        elif normalized == "ocr" and (
            host._character_record_uses_yolo_box_backend(rec)
            or idx in backend_positions
        ):
            normalized = "yolo_box_ocr"
        tags.append(normalized)

    return tags


def get_character_source_tag(host, rec, data=None, fallback_index: int = 0) -> str:
    raw_tag = None
    method_name = ""

    if isinstance(rec, dict):
        raw_tag = rec.get("source_tag")
        method_name = rec.get("method", "")
    else:
        raw_tag = getattr(rec, "source_tag", None)
        method_name = getattr(rec, "method", "")

    explicit_box = _normalized_record_text(rec, "box_source")
    explicit_sign = _normalized_record_text(rec, "sign_source")
    if explicit_box or explicit_sign:
        return compose_character_source_tag(
            normalize_character_box_source(host, rec, data=data, fallback_index=fallback_index),
            normalize_character_sign_source(host, rec, data=data, fallback_index=fallback_index),
        )

    normalized = host._normalize_character_source_tag(raw_tag=raw_tag, method=method_name)
    if normalized == "ocr" and host._character_record_uses_yolo_box_backend(rec):
        return "yolo_box_ocr"
    if isinstance(data, dict) and normalized == "ocr":
        backend_positions = fusion_details_yolo_box_backend_positions(data.get("fusion_details", {}))
        try:
            if int(fallback_index) in backend_positions:
                return "yolo_box_ocr"
        except Exception:
            pass
    if raw_tag:
        return normalized

    if isinstance(data, dict):
        strategy = str(data.get("fusion_strategy", "") or "").strip().lower()
        if strategy == "ocr_yolo_rescue":
            details = data.get("fusion_details", {})
            if isinstance(details, dict):
                mismatch_positions = details.get("mismatch_positions", []) or []
                try:
                    if int(fallback_index) in {int(idx) for idx in mismatch_positions}:
                        return "yolo_rescue"
                except Exception:
                    pass
        elif strategy in {"yolo_box_ocr", "yolo_box_ocr_no_gt", "yolo_box_ocr_fallback"}:
            return "yolo_box_ocr"

    return normalized


def build_plate_layout_export_metadata(host, data: dict | None) -> dict:
    if not isinstance(data, dict):
        return {
            "plate_layout": "unknown",
            "plate_layout_label": "?",
            "layout_row_count": 0,
            "layout_confidence": 0.0,
            "layout_source": "none",
            "plate_layout_override": "",
            "layout_separator": None,
            "plate_text_reading_order": "",
        }

    layout_label, _tone = host._get_preview_plate_layout_label(data)
    try:
        row_count = int(data.get("layout_row_count", 0) or 0)
    except Exception:
        row_count = 0
    try:
        confidence = float(data.get("layout_confidence", 0.0) or 0.0)
    except Exception:
        confidence = 0.0

    try:
        layout_separator = host._normalize_preview_layout_separator(data.get("layout_separator"))
    except Exception:
        layout_separator = None

    return {
        "plate_layout": str(data.get("plate_layout", "unknown") or "unknown"),
        "plate_layout_label": str(layout_label or "?"),
        "layout_row_count": int(row_count),
        "layout_confidence": float(confidence),
        "layout_source": str(data.get("layout_source", "none") or "none"),
        "plate_layout_override": str(data.get("plate_layout_override", "") or ""),
        "layout_separator": layout_separator,
        "plate_text_reading_order": host._characters_to_text(data.get("characters", []), data=data),
    }


def character_record_overlap_score(host, left, right) -> float:
    left_bbox = host._char_record_bbox(left)
    right_bbox = host._char_record_bbox(right)
    if not left_bbox or not right_bbox:
        return 0.0

    lx1, ly1, lx2, ly2 = left_bbox
    rx1, ry1, rx2, ry2 = right_bbox
    inter_x1 = max(float(lx1), float(rx1))
    inter_y1 = max(float(ly1), float(ry1))
    inter_x2 = min(float(lx2), float(rx2))
    inter_y2 = min(float(ly2), float(ry2))
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    left_area = max(1.0, (float(lx2) - float(lx1)) * (float(ly2) - float(ly1)))
    right_area = max(1.0, (float(rx2) - float(rx1)) * (float(ry2) - float(ry1)))
    union_area = max(1.0, left_area + right_area - inter_area)
    return float(inter_area / union_area)


def character_record_collides_with_manual(host, rec, manual_records) -> bool:
    bbox = host._char_record_bbox(rec)
    if not bbox:
        return False

    try:
        x1, y1, x2, y2 = (float(value) for value in bbox[:4])
    except Exception:
        return False

    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    area = width * height
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    for manual_rec in list(manual_records or []):
        manual_bbox = host._char_record_bbox(manual_rec)
        if not manual_bbox:
            continue
        try:
            mx1, my1, mx2, my2 = (float(value) for value in manual_bbox[:4])
        except Exception:
            continue

        inter_x1 = max(x1, mx1)
        inter_y1 = max(y1, my1)
        inter_x2 = min(x2, mx2)
        inter_y2 = min(y2, my2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0.0:
            continue

        mw = max(1.0, mx2 - mx1)
        mh = max(1.0, my2 - my1)
        manual_area = mw * mh
        union_area = max(1.0, area + manual_area - inter_area)
        iou = inter_area / union_area
        min_coverage = inter_area / max(1.0, min(area, manual_area))
        auto_coverage = inter_area / max(1.0, area)
        manual_coverage = inter_area / max(1.0, manual_area)
        horizontal_overlap = inter_w / max(1.0, min(width, mw))
        vertical_overlap = inter_h / max(1.0, min(height, mh))
        center_dx = abs(cx - ((mx1 + mx2) / 2.0)) / max(width, mw)
        center_dy = abs(cy - ((my1 + my2) / 2.0)) / max(height, mh)

        if iou >= 0.28:
            return True
        if min_coverage >= 0.62 and horizontal_overlap >= 0.50 and vertical_overlap >= 0.45:
            return True
        if auto_coverage >= 0.42 and center_dx <= 0.42 and center_dy <= 0.52:
            return True
        if manual_coverage >= 0.42 and center_dx <= 0.42 and center_dy <= 0.52:
            return True

    return False


def clone_base_record_with_candidate_bbox(host, base_rec, candidate_rec):
    candidate_bbox = host._char_record_bbox(candidate_rec)
    if not candidate_bbox:
        return copy.deepcopy(base_rec) if isinstance(base_rec, dict) else host._clone_character_detection(base_rec)

    candidate_confidence = host._char_record_confidence(candidate_rec)
    if isinstance(base_rec, dict):
        updated = copy.deepcopy(base_rec)
        updated["bbox"] = [float(v) for v in candidate_bbox[:4]]
        try:
            existing_confidence = float(updated.get("confidence", 0.0) or 0.0)
        except Exception:
            existing_confidence = 0.0
        updated["confidence"] = float(max(existing_confidence, candidate_confidence))
        updated["box_backend"] = "yolo"
        updated["box_backend_source"] = "yolo_filtered"
        updated["geometry_source"] = "yolo"
        updated["geometry_method"] = "yolo_box"
        updated["box_backend_confidence"] = float(candidate_confidence)
        return updated

    symbol, _ = host._char_record_to_symbol_and_x(base_rec)
    method_name = str(getattr(base_rec, "method", "ocr") or "ocr")
    source_tag = str(getattr(base_rec, "source_tag", "") or "")
    return {
        "character": str(symbol or ""),
        "bbox": [float(v) for v in candidate_bbox[:4]],
        "confidence": float(max(host._char_record_confidence(base_rec), candidate_confidence)),
        "method": method_name,
        "source_tag": source_tag,
        "box_backend": "yolo",
        "box_backend_source": "yolo_filtered",
        "geometry_source": "yolo",
        "geometry_method": "yolo_box",
        "box_backend_confidence": float(candidate_confidence),
    }


def merge_detected_characters_preserving_manual(host, existing_chars, detected_chars, data=None) -> tuple[list[dict], dict]:
    existing_ordered = host._sort_character_records_by_x(list(existing_chars or []))
    detected_ordered = host._sort_character_records_by_x(list(detected_chars or []))

    manual_records = [
        copy.deepcopy(rec)
        for rec in existing_ordered
        if isinstance(rec, dict) and host._is_manual_character_record(rec, data=data)
    ]
    if not manual_records:
        return detected_ordered, {
            "manual_preserved_count": 0,
            "auto_appended_count": int(len(detected_ordered)),
            "auto_skipped_due_manual": 0,
        }

    merged_records = list(manual_records)
    skipped_auto = 0
    appended_auto = 0

    for rec in detected_ordered:
        if not isinstance(rec, dict):
            continue
        if character_record_collides_with_manual(host, rec, manual_records):
            skipped_auto += 1
            continue
        merged_records.append(rec)
        appended_auto += 1

    return host._sort_character_records_by_x(merged_records), {
        "manual_preserved_count": int(len(manual_records)),
        "auto_appended_count": int(appended_auto),
        "auto_skipped_due_manual": int(skipped_auto),
    }


def normalize_preview_source_path_key(raw_value: str) -> str:
    text = str(raw_value or "").strip()
    if not text:
        return ""
    try:
        path_obj = Path(text)
        if path_obj.exists():
            try:
                return path_obj.resolve().as_posix().lower()
            except Exception:
                return path_obj.absolute().as_posix().lower()
        if path_obj.is_absolute():
            try:
                return path_obj.absolute().as_posix().lower()
            except Exception:
                pass
    except Exception:
        pass
    return text.replace("\\", "/").lower()


def _preview_plate_reextract_bbox_key(data: dict | None = None) -> str:
    source_data = data if isinstance(data, dict) else {}
    if not isinstance(source_data, dict):
        return ""

    bbox = list(source_data.get("source_bbox") or [])
    if len(bbox) >= 4:
        try:
            return ":".join(str(int(round(float(value)))) for value in bbox[:4])
        except Exception:
            return ":".join(str(value) for value in bbox[:4])
    return ""


def _preview_plate_reextract_source_variants(data: dict | None = None) -> list[str]:
    source_data = data if isinstance(data, dict) else {}
    if not isinstance(source_data, dict):
        return []

    raw_values = [
        source_data.get("source_image", ""),
        source_data.get("source_name", ""),
        source_data.get("filename", ""),
        source_data.get("source_image_name", ""),
    ]
    variants: list[str] = []
    seen: set[str] = set()

    def _add(value: str) -> None:
        value = str(value or "").strip()
        if not value or value in seen:
            return
        seen.add(value)
        variants.append(value)

    for raw_value in raw_values:
        text = str(raw_value or "").strip()
        if not text:
            continue
        normalized = normalize_preview_source_path_key(text)
        if normalized:
            _add(f"path:{normalized}")
        slash_text = text.replace("\\", "/")
        name = slash_text.rsplit("/", 1)[-1].strip().lower()
        if name:
            _add(f"name:{name}")
            stem = name.rsplit(".", 1)[0].strip()
            if stem:
                _add(f"stem:{stem}")

    expected = str(source_data.get("source_expected_text", "") or "").strip().upper()
    if expected:
        _add(f"text:{expected}")

    return variants


def build_preview_plate_reextract_match_keys(data: dict | None = None) -> list[str]:
    source_data = data if isinstance(data, dict) else {}
    if not isinstance(source_data, dict):
        return []

    source_key = normalize_preview_source_path_key(
        str(source_data.get("source_image", "") or source_data.get("source_name", "") or source_data.get("filename", "") or "").strip()
    )
    bbox_key = _preview_plate_reextract_bbox_key(source_data)
    index_key = ""
    try:
        if "source_plate_index" in source_data:
            index_key = str(int(source_data.get("source_plate_index")))
    except Exception:
        index_key = str(source_data.get("source_plate_index", "") or "").strip()

    keys: list[str] = []
    seen: set[str] = set()

    def _add(key: str) -> None:
        key = str(key or "").strip()
        if not key or key in seen:
            return
        seen.add(key)
        keys.append(key)

    if source_key:
        _add(f"{source_key}|{bbox_key}")

    for variant in _preview_plate_reextract_source_variants(source_data):
        if bbox_key:
            if index_key:
                _add(f"{variant}|idx:{index_key}|bbox:{bbox_key}")
            _add(f"{variant}|bbox:{bbox_key}")
        elif index_key:
            _add(f"{variant}|idx:{index_key}")

    return keys


def build_preview_plate_reextract_match_key(data: dict | None = None) -> str:
    keys = build_preview_plate_reextract_match_keys(data)
    if keys:
        return keys[0]
    return ""


def capture_preview_reextract_seed_metadata(host) -> dict:
    seed_source = {}
    preview_dir = None
    if isinstance(getattr(host, "preview_metadata", None), dict) and host.preview_metadata:
        seed_source = copy.deepcopy(host.preview_metadata)
        try:
            preview_dir_raw = str(host.preview_dir_var.get() or "").strip()
            preview_dir = Path(preview_dir_raw) if preview_dir_raw else None
        except Exception:
            preview_dir = None
    else:
        preview_dir_raw = str(host.preview_dir_var.get() or "").strip()
        if preview_dir_raw:
            meta_path = Path(preview_dir_raw) / "metadata.json"
            seed_source = host._load_json_file_safely(meta_path)
            preview_dir = Path(preview_dir_raw)

    if not isinstance(seed_source, dict):
        return {}

    seed_lookup: dict[str, dict] = {}
    for _pid, raw_data in seed_source.items():
        if not isinstance(raw_data, dict):
            continue
        seed_payload = copy.deepcopy(raw_data)
        try:
            seed_size = host._get_preview_plate_image_size(preview_dir, str(_pid or "").strip())
        except Exception:
            seed_size = None
        if seed_size:
            seed_payload["_seed_image_size"] = [int(seed_size[0]), int(seed_size[1])]
        for match_key in build_preview_plate_reextract_match_keys(raw_data):
            if not match_key or match_key in seed_lookup:
                continue
            seed_lookup[match_key] = seed_payload
    return seed_lookup


def preview_plate_has_renderable_boxes(data: dict | None) -> bool:
    if not isinstance(data, dict):
        return False
    for key_name in ("characters", "yolo_detections", "yolo_nms_detections", "yolo_raw_detections"):
        values = data.get(key_name)
        if isinstance(values, list) and len(values) > 0:
            return True
    return False


def preview_plate_reextract_seed_score(data: dict | None) -> int:
    if not isinstance(data, dict):
        return 0
    characters = data.get("characters")
    char_count = len(characters) if isinstance(characters, list) else 0
    yolo_count = 0
    for key_name in ("yolo_detections", "yolo_nms_detections", "yolo_raw_detections"):
        values = data.get(key_name)
        if isinstance(values, list):
            yolo_count += len(values)

    manual_count = 0
    if isinstance(characters, list):
        for rec in characters:
            if not isinstance(rec, dict):
                continue
            method = str(rec.get("method", "") or "").strip().lower()
            source_kind = str(rec.get("source_kind", "") or "").strip().lower()
            source_tag = str(rec.get("source_tag", "") or "").strip().lower()
            if method in {"manual", "cvat_manual"} or source_kind in {"local_manual", "cvat_manual"} or source_tag == "manual":
                manual_count += 1

    status = str(data.get("status", "") or "").strip().lower()
    status_bonus = 0
    if status == "perfect":
        status_bonus = 500
    elif status in {"ok", "approved"}:
        status_bonus = 250
    return int(char_count * 1000 + manual_count * 50 + yolo_count * 10 + status_bonus)


def collect_historical_preview_reextract_seed_metadata(host, current_preview_dir: Path | None = None) -> dict:
    run_root = None
    if current_preview_dir is not None:
        try:
            run_root = Path(current_preview_dir).parent
        except Exception:
            run_root = None
    if run_root is None:
        try:
            run_root = host._get_step3_chars_root_dir(ensure_exists=False)
        except Exception:
            run_root = None
    if run_root is None or not Path(run_root).exists():
        return {}

    current_meta = None
    if current_preview_dir is not None:
        try:
            current_meta = Path(current_preview_dir) / "metadata.json"
        except Exception:
            current_meta = None

    try:
        meta_files = sorted(
            (
                path for path in Path(run_root).rglob("metadata.json")
                if current_meta is None or path != current_meta
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except Exception:
        return {}

    seed_lookup: dict[str, dict] = {}
    seed_scores: dict[str, int] = {}
    for meta_path in meta_files:
        metadata = host._load_json_file_safely(meta_path)
        if not isinstance(metadata, dict) or not metadata:
            continue
        for _pid, raw_data in metadata.items():
            if not isinstance(raw_data, dict) or not preview_plate_has_renderable_boxes(raw_data):
                continue
            seed_payload = copy.deepcopy(raw_data)
            seed_score = preview_plate_reextract_seed_score(raw_data)
            try:
                seed_size = host._get_preview_plate_image_size(meta_path.parent, str(_pid or "").strip())
            except Exception:
                seed_size = None
            if seed_size:
                seed_payload["_seed_image_size"] = [int(seed_size[0]), int(seed_size[1])]
            for match_key in build_preview_plate_reextract_match_keys(raw_data):
                if not match_key:
                    continue
                if match_key in seed_lookup and seed_score <= int(seed_scores.get(match_key, 0) or 0):
                    continue
                seed_lookup[match_key] = seed_payload
                seed_scores[match_key] = int(seed_score)
    return seed_lookup


def scale_preview_char_records_to_new_size(host, records, from_size, to_size):
    if not isinstance(records, list):
        return copy.deepcopy(records)
    try:
        from_w, from_h = (float(from_size[0]), float(from_size[1]))
        to_w, to_h = (float(to_size[0]), float(to_size[1]))
    except Exception:
        return copy.deepcopy(records)

    if from_w <= 0 or from_h <= 0 or to_w <= 0 or to_h <= 0:
        return copy.deepcopy(records)

    scale_x = to_w / from_w
    scale_y = to_h / from_h
    scaled_records = []
    for rec in list(records or []):
        cloned = copy.deepcopy(rec)
        bbox = host._char_record_bbox(cloned)
        if not bbox:
            scaled_records.append(cloned)
            continue
        x1, y1, x2, y2 = bbox
        scaled_bbox = [
            max(0.0, min(to_w, float(x1) * scale_x)),
            max(0.0, min(to_h, float(y1) * scale_y)),
            max(0.0, min(to_w, float(x2) * scale_x)),
            max(0.0, min(to_h, float(y2) * scale_y)),
        ]
        if scaled_bbox[2] < scaled_bbox[0]:
            scaled_bbox[0], scaled_bbox[2] = scaled_bbox[2], scaled_bbox[0]
        if scaled_bbox[3] < scaled_bbox[1]:
            scaled_bbox[1], scaled_bbox[3] = scaled_bbox[3], scaled_bbox[1]
        if isinstance(cloned, dict):
            cloned["bbox"] = [float(v) for v in scaled_bbox]
        else:
            try:
                cloned.bbox = tuple(float(v) for v in scaled_bbox)
            except Exception:
                pass
        scaled_records.append(cloned)
    return scaled_records


def adapt_reextract_seed_payload_to_current_preview(host, previous: dict, preview_dir: Path, pid: str) -> dict:
    adapted = copy.deepcopy(previous)
    from_size = adapted.get("_seed_image_size")
    to_size = host._get_preview_plate_image_size(preview_dir, pid)
    if from_size and to_size and list(from_size) != [int(to_size[0]), int(to_size[1])]:
        for key_name in ("characters", "yolo_detections", "yolo_nms_detections", "yolo_raw_detections"):
            if key_name in adapted:
                adapted[key_name] = scale_preview_char_records_to_new_size(
                    host,
                    adapted.get(key_name),
                    from_size,
                    to_size,
                )
    adapted.pop("_seed_image_size", None)
    return adapted


def merge_reextract_seed_metadata_into_preview(
    host,
    preview_dir: Path | None,
    seed_lookup: dict | None = None,
    *,
    allow_historical_fallback: bool = False,
) -> dict:
    if preview_dir is None:
        return {"matched": 0, "total": 0}

    meta_path = Path(preview_dir) / "metadata.json"
    if not meta_path.exists():
        return {"matched": 0, "total": 0}

    previous_lookup = dict(seed_lookup or getattr(host, "_campaign_step3_reextract_seed_metadata", {}) or {})
    if (
        allow_historical_fallback
        and (not previous_lookup or not any(preview_plate_has_renderable_boxes(item) for item in previous_lookup.values()))
    ):
        previous_lookup = collect_historical_preview_reextract_seed_metadata(host, preview_dir)
    if not previous_lookup:
        return {"matched": 0, "total": 0}

    metadata = host._load_json_file_safely(meta_path)
    if not isinstance(metadata, dict) or not metadata:
        return {"matched": 0, "total": 0}

    preserve_keys = (
        "characters",
        "status",
        "fusion_strategy",
        "fusion_details",
        "source_info",
        "gold_state",
        "yolo_detections",
        "yolo_nms_detections",
        "yolo_raw_detections",
    )

    matched = 0
    for pid, raw_data in list(metadata.items()):
        if not isinstance(raw_data, dict):
            continue
        previous = None
        for match_key in build_preview_plate_reextract_match_keys(raw_data):
            candidate = previous_lookup.get(match_key)
            if isinstance(candidate, dict):
                previous = candidate
                break
        if not isinstance(previous, dict):
            continue
        previous = adapt_reextract_seed_payload_to_current_preview(
            host,
            previous,
            Path(preview_dir),
            str(pid or "").strip(),
        )
        for key_name in preserve_keys:
            if key_name in previous:
                raw_data[key_name] = copy.deepcopy(previous.get(key_name))
        host._ensure_plate_source_metadata(raw_data, plate_id=str(pid or ""), meta_path=meta_path)
        matched += 1

    metadata = host._recalculate_preview_statuses_in_metadata(metadata)
    host._atomic_write_json(meta_path, metadata)
    return {"matched": int(matched), "total": int(len(metadata))}


def normalize_preview_char_bbox(host, bbox, *, min_size: float = 4.0):
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    except Exception:
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    state = getattr(host, "_preview_render_state", None) or {}
    max_w = max(1.0, float(state.get("orig_w", 1.0)))
    max_h = max(1.0, float(state.get("orig_h", 1.0)))
    x1 = max(0.0, min(max_w, x1))
    y1 = max(0.0, min(max_h, y1))
    x2 = max(0.0, min(max_w, x2))
    y2 = max(0.0, min(max_h, y2))

    if x2 - x1 < float(min_size):
        if x1 + float(min_size) <= max_w:
            x2 = x1 + float(min_size)
        else:
            x1 = max(0.0, x2 - float(min_size))
    if y2 - y1 < float(min_size):
        if y1 + float(min_size) <= max_h:
            y2 = y1 + float(min_size)
        else:
            y1 = max(0.0, y2 - float(min_size))

    normalized = [float(x1), float(y1), float(x2), float(y2)]
    drag_state = getattr(host, "_preview_char_drag_state", None)
    skip_auto_separator_constraint = isinstance(drag_state, dict) and drag_state.get("layout_row") in (1, 2)
    try:
        data = host._get_preview_active_data(create=False)
        if not skip_auto_separator_constraint:
            constrained = host._constrain_preview_char_bbox_to_layout_separator(
                normalized,
                data=data,
                min_size=float(min_size),
            )
            if isinstance(constrained, (list, tuple)) and len(constrained) >= 4:
                normalized = [float(value) for value in constrained[:4]]
    except Exception:
        pass

    return normalized


def char_record_to_symbol_and_x(rec, fallback_index: int = 0):
    symbol = ""
    x_key = float(fallback_index)

    if isinstance(rec, dict):
        symbol = str(
            rec.get("character")
            or rec.get("text")
            or rec.get("char")
            or ""
        )

        bbox = rec.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            try:
                x_key = (float(bbox[0]) + float(bbox[2])) / 2.0
            except Exception:
                pass

        return symbol, x_key

    if isinstance(rec, str):
        return rec, x_key

    try:
        symbol = str(getattr(rec, "character", getattr(rec, "text", "")) or "")
    except Exception:
        symbol = ""

    try:
        bbox = getattr(rec, "bbox", None)
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            x_key = (float(bbox[0]) + float(bbox[2])) / 2.0
    except Exception:
        pass

    return symbol, x_key
