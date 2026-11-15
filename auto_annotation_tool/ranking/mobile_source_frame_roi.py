"""Verified pre-ROI mobile frames and read-only source-coordinate overlays.

The reference frame is the Android pipeline Bitmap *before ROI*, not camera
sensor data, MT network input, independent GT, or an operator correction.
Only a verified source archive entry can be displayed; absence fails closed.
"""
from __future__ import annotations

import io
import math
import zipfile
from pathlib import PurePosixPath


SOURCE_FRAME_SPACE = "mt_source_bitmap_xy"
SOURCE_FRAME_ORIGIN = "pipeline_bitmap_before_roi"
MT_POSE_SPACE = "source_frame_normalized_v1"
MT_CORNER_ORDER = "TL_TR_BR_BL"


def _number(value):
    if value is None or isinstance(value, bool) or str(value).strip() == "":
        raise ValueError("Missing numeric source-frame field")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Nonfinite source-frame field")
    return number


def source_frame_entry(row, available_names):
    """Accept only a declared, written JPEG inside the checked report archive."""
    if not isinstance(row, dict):
        return None
    path = str(row.get("source_frame_entry") or "")
    parts = PurePosixPath(path).parts
    if (len(parts) != 3 or parts[:2] != ("samples", "frames")
            or not parts[-1].lower().endswith((".jpg", ".jpeg"))
            or parts[-1].startswith(".") or "\\" in path or ":" in path
            or path not in set(available_names or ())):
        return None
    if (str(row.get("source_frame_status") or "").upper() != "WRITTEN"
            or str(row.get("source_frame_expected") or "").lower() != "true"
            or row.get("source_frame_coordinate_space") != SOURCE_FRAME_SPACE
            or row.get("source_frame_origin") != SOURCE_FRAME_ORIGIN):
        return None
    return path


def frame_roi(row, image_size):
    """Validate declared ROI in exact source Bitmap coordinates; never clip."""
    try:
        width, height = (_number(value) for value in image_size)
        declared_w = _number(row["source_frame_width_px"])
        declared_h = _number(row["source_frame_height_px"])
        left, top, right, bottom = (_number(row[key]) for key in
            ("roi_left", "roi_top", "roi_right", "roi_bottom"))
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if (width <= 0 or height <= 0 or width != declared_w or height != declared_h
            or not (0 <= left < right <= width and 0 <= top < bottom <= height)):
        return None
    return left, top, right, bottom


def source_invocation_context(records, invocation_id, available_names, image_size=None):
    """One archived Bitmap and ROI for *all* records of one MT invocation.

    Reject contradictory source paths, frame timestamps, dimensions, and ROI.
    An isolated row does not justify combining detections from other calls.
    """
    if not invocation_id or not records:
        return None
    result = None
    for row in records:
        if not isinstance(row, dict) or str(row.get("mt_invocation_id") or "") != str(invocation_id):
            return None
        entry = source_frame_entry(row, available_names)
        if entry is None:
            return None
        try:
            w = _number(row["source_frame_width_px"])
            h = _number(row["source_frame_height_px"])
        except (KeyError, ValueError, TypeError, OverflowError):
            return None
        roi = frame_roi(row, (w, h))
        if roi is None:
            return None
        candidate = (entry, (w, h), roi, str(row.get("source_sequence") or ""),
                     str(row.get("source_timestamp_nanos") or ""))
        if result is not None and candidate != result:
            return None
        result = candidate
    if image_size is not None:
        try:
            if result is None or tuple(_number(v) for v in image_size) != result[1]:
                return None
        except (TypeError, ValueError, OverflowError):
            return None
    return {"entry": result[0], "size": result[1], "roi": result[2],
            "source_sequence": result[3], "source_timestamp_nanos": result[4],
            "invocation_id": str(invocation_id), "records": tuple(records)}


def source_pose_quad(row, image_size):
    """Return only actual, valid four-corner MT Pose in source coordinates."""
    if not isinstance(row, dict):
        return None
    if (row.get("mt_pose_status") != "VALID_QUAD"
            or row.get("mt_pose_coordinate_space") != MT_POSE_SPACE
            or row.get("mt_pose_corner_order") != MT_CORNER_ORDER):
        return None
    try:
        width, height = tuple(_number(value) for value in image_size)
        source_w = _number(row["mt_pose_source_width_px"])
        source_h = _number(row["mt_pose_source_height_px"])
        parts = str(row["mt_pose_corners_norm"]).split(";")
        if len(parts) != 4:
            return None
        pairs = [part.split(":") for part in parts]
        if any(len(pair) != 2 for pair in pairs):
            return None
        points = [(_number(pair[0]), _number(pair[1])) for pair in pairs]
    except (KeyError, ValueError, TypeError, OverflowError):
        return None
    if (width <= 0 or height <= 0 or source_w != width or source_h != height
            or any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in points)):
        return None
    quad = [(x * width, y * height) for x, y in points]
    # Convex polygon with consistent nonzero winding, never repair model data.
    edges = [((quad[(i + 1) % 4][0] - quad[i][0]),
              (quad[(i + 1) % 4][1] - quad[i][1])) for i in range(4)]
    crosses = [edges[i][0] * edges[(i + 1) % 4][1]
               - edges[i][1] * edges[(i + 1) % 4][0] for i in range(4)]
    span = max(max(x for x, _ in quad) - min(x for x, _ in quad),
               max(y for _, y in quad) - min(y for _, y in quad))
    eps = max(1e-10, span * span * 1e-9)
    if not all(value > eps for value in crosses):
        return None
    return tuple(quad)


def verified_invocation_detections(context):
    """All distinct recorded MT Pose polygons for one verified frame."""
    if not isinstance(context, dict):
        return []
    result, seen = [], set()
    for row in context["records"]:
        quad = source_pose_quad(row, context["size"])
        if quad is None:
            continue
        try:
            raw_index = row["mt_detection_index"]
            if raw_index is None or str(raw_index).strip() == "":
                continue
            index = int(raw_index)
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if index < 0 or index in seen:
            continue
        seen.add(index)
        result.append({"attempt_id": str(row.get("id") or row.get("attempt_id") or ""),
                       "detection_index": index, "quad": quad})
    return sorted(result, key=lambda item: item["detection_index"])


def quad_contains_point(quad, x, y):
    """Hit test for selecting one of several detections on the source view."""
    try:
        point = (float(x), float(y))
        if not all(math.isfinite(v) for v in point):
            return False
        cross = [((quad[(i + 1) % 4][0] - quad[i][0]) * (point[1] - quad[i][1])
                  - (quad[(i + 1) % 4][1] - quad[i][1]) * (point[0] - quad[i][0]))
                 for i in range(4)]
    except (TypeError, ValueError, IndexError, OverflowError):
        return False
    return all(v >= -1e-5 for v in cross) or all(v <= 1e-5 for v in cross)


def read_mobile_source_frame(bundle_or_path, entry_name, *, max_entry_bytes=20*1024*1024,
                             max_dimension=12000, max_pixels=40_000_000):
    """Bounded image read via the existing archive validation and name checks.

    No extraction to filesystem and no unchecked `ZipFile.read`; the same
    normalized ZIP index that governs crops/evidence is reused for frames.
    """
    from PIL import Image
    from .mobile_package_experiments import (
        _archive_name_safe, _full_report_source_path,
        _normalized_zip_entries_for_full_read,
    )
    safe, path = _archive_name_safe(str(entry_name))
    parts = PurePosixPath(path).parts
    if (not safe or len(parts) != 3 or parts[:2] != ("samples", "frames")
            or not parts[-1].lower().endswith((".jpg", ".jpeg"))):
        raise ValueError("Plik nie należy do zapisanych pełnych klatek źródłowych MT.")
    archive_path = _full_report_source_path(bundle_or_path)
    with zipfile.ZipFile(archive_path, "r") as archive:
        entries = _normalized_zip_entries_for_full_read(archive)
        info = entries.get(path)
        if info is None:
            raise FileNotFoundError(f"Brak pełnej klatki w archiwum: {path}")
        if info.file_size > max_entry_bytes:
            raise ValueError("Plik klatki przekracza limit rozmiaru.")
        with archive.open(info, "r") as handle:
            data = handle.read(max_entry_bytes + 1)
        if len(data) > max_entry_bytes:
            raise ValueError("Plik klatki przekracza limit rozmiaru.")
    with Image.open(io.BytesIO(data)) as source:
        width, height = source.size
        if (width <= 0 or height <= 0 or max(width, height) > max_dimension
                or width * height > max_pixels):
            raise ValueError("Klatka przekracza limit wymiarów.")
        source.load()
        return source.convert("RGB")
