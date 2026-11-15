"""Ordered, incremental background writes for canvas corrections."""
from __future__ import annotations

import copy
import datetime
from pathlib import Path
import threading
import xml.etree.ElementTree as ET

from ..config import CONFIG, logger
from ..exporters.cvat_exporter import CVATExporter, write_cvat_tree_atomic
from ..plate_ground_truth import ensure_plate_detection_contract


def queue_gt_sync(host, xml_path, item):
    if item is None:
        return
    pending = getattr(host, "_preview_gt_sync_pending", None)
    if not isinstance(pending, dict):
        pending = {}
        host._preview_gt_sync_pending = pending
    host._preview_gt_sync_version = int(getattr(host, "_preview_gt_sync_version", 0) or 0) + 1
    key = (str(Path(xml_path)), item["plate_id"], item.get("field", "ground_truth"))
    pending[key] = {"version": host._preview_gt_sync_version, "item": copy.deepcopy(item)}


def _pending_gt_snapshot(host, xml_path, names=None):
    return {key: copy.deepcopy(value)
            for key, value in (getattr(host, "_preview_gt_sync_pending", {}) or {}).items()
            if key[0] == str(Path(xml_path))
            and (names is None or value["item"]["image_name"] in names)}


def _acknowledge_gt_snapshot(host, snapshot):
    pending = getattr(host, "_preview_gt_sync_pending", {}) or {}
    for key, saved in snapshot.items():
        if pending.get(key) == saved:
            pending.pop(key, None)


def flush_pending_gt_after_xml_save(host, xml_path):
    """Used by explicit save/exit after writing the newest full XML state."""
    from .z2_gt_pack_runtime import sync_gt_items_after_xml_save, invalidate_gt_pack_status_cache
    snapshot = _pending_gt_snapshot(host, xml_path)
    if snapshot:
        result = sync_gt_items_after_xml_save(xml_path, [v["item"] for v in snapshot.values()])
        _acknowledge_gt_snapshot(host, snapshot)
        invalidate_gt_pack_status_cache(host)
        return result
    return {"ok": True, "queued": 0}


def update_saved_images(xml_path: Path, annotations: list) -> None:
    """Replace dirty image elements while preserving all untouched XML data."""
    root = ET.parse(xml_path).getroot()
    replacements = {ann.filename: ann for ann in annotations}
    exporter = CVATExporter()
    found = set()
    for index, element in enumerate(list(root)):
        if element.tag != "image" or element.get("name") not in replacements:
            continue
        name = element.get("name")
        holder = ET.Element("annotations")
        exporter._add_image(holder, element.get("id", str(index)), replacements[name], True)
        replacement = holder[0]
        # Keep any image-level provenance that is outside the editor's model.
        for key, value in element.attrib.items():
            replacement.attrib.setdefault(key, value)
        replacement.tail = element.tail
        root[index] = replacement
        found.add(name)
    missing = set(replacements) - found
    if missing:
        raise ValueError(f"Brak obrazów w docelowym XML: {', '.join(sorted(missing)[:3])}")
    write_cvat_tree_atomic(root, xml_path)


def wait_for_pending_save(host) -> None:
    """An explicit save/export must be ordered after an earlier autosave."""
    job = getattr(host, "_preview_background_save", None)
    if isinstance(job, dict):
        job["done"].wait()
        if job.get("error") is None:
            _acknowledge_gt_snapshot(host, job.get("gt_snapshot", {}))
        # The synchronous save owns metadata/dirty flags from this point on.
        host._preview_background_save = None


def start_preview_autosave(host, *, status_message: str = "") -> bool:
    pending = getattr(host, "_preview_background_save", None)
    if isinstance(pending, dict):
        return True
    dirty = set(getattr(host, "_preview_dirty_images", set()) or set())
    if not dirty:
        return True
    xml_path = host._get_current_annotation_xml_path()
    if xml_path is None or bool(getattr(host, "_preview_draw_mode", False)):
        return False
    versions = dict(getattr(host, "_preview_image_edit_versions", {}) or {})
    source = [ann for ann in host.current_annotations if ann.filename in dirty]
    if not source:
        return False
    # Allocate missing stable IDs on the live objects once, before copying.
    # Doing this only in the worker would generate new IDs on every autosave.
    for ann in source:
        for detection in ann.detections:
            if str(detection.label or "").strip().lower() in CONFIG.PLATE_LABELS:
                ensure_plate_detection_contract(detection)
    job = {"done": threading.Event(), "path": Path(xml_path),
           "annotations": copy.deepcopy(source), "versions": versions, "error": None,
           "gt_snapshot": _pending_gt_snapshot(host, xml_path, dirty), "gt_result": None}
    host._preview_background_save = job

    def complete():
        if getattr(host, "_preview_background_save", None) is not job:
            return
        try:
            quiet = int(host._preview_user_interaction_quiet_remaining_ms(padding_ms=250))
        except (AttributeError, TypeError, ValueError):
            quiet = 0
        if quiet > 0 or getattr(host, "_preview_drag_state", None):
            host.frame.after(max(250, quiet), complete)
            return
        host._preview_background_save = None
        if job["error"] is not None:
            logger.error("[Z2 AUTOSAVE] Nie zapisano poprawek: %s", job["error"])
            host._update_preview_edit_status("Nie udało się automatycznie zapisać poprawek. Użyj Zapisz.",
                                             refresh_toolbar=False, refresh_debug=False)
            return
        _acknowledge_gt_snapshot(host, job["gt_snapshot"])
        if job["gt_snapshot"]:
            from .z2_gt_pack_runtime import invalidate_gt_pack_status_cache
            invalidate_gt_pack_status_cache(host)
        if host._get_current_annotation_xml_path() != job["path"]:
            return
        saved_names = {ann.filename for ann in job["annotations"]}
        manifest = host._load_annotation_run_manifest(job["path"].parent)
        manual_names = set(manifest.get("manual_touched_filenames") or []) | saved_names
        host._update_annotation_run_manifest(
            job["path"].parent, has_manual_edits=True,
            last_manual_edit_at=datetime.datetime.now().isoformat(timespec="seconds"),
            last_manual_edit_kind="preview_save", manual_touched_filenames=sorted(manual_names),
            **host._collect_preview_resume_manifest_fields(),
        )
        live_versions = getattr(host, "_preview_image_edit_versions", {}) or {}
        for ann in job["annotations"]:
            if live_versions.get(ann.filename, 0) == versions.get(ann.filename, 0):
                host._preview_dirty_images.discard(ann.filename)
        host._campaign_iteration_manual_filenames = set(
            getattr(host, "_campaign_iteration_manual_filenames", set()) or set()) | saved_names
        host._campaign_manual_touched_cache = None
        host._remember_campaign_manual_plate_source(
            run_dir=job["path"].parent, xml_path=job["path"], input_dir=host.current_input_dir)
        host._queue_free_mode_session_save(include_preview_approved=False)
        if not getattr(host, "_preview_drag_state", None):
            host._update_preview_toolbar_state(refresh_summary=False)
            message = status_message or "Zapisano poprawki ramek."
            if (job.get("gt_result") or {}).get("queued"):
                message = "GT zapisano w XML; synchronizacja pakietu GT oczekuje na ponowienie."
            host._update_preview_edit_status(message,
                                             refresh_toolbar=False, refresh_debug=False)
        if host._preview_dirty_images:
            host._schedule_preview_autosave()

    def worker():
        try:
            update_saved_images(job["path"], job["annotations"])
            if job["gt_snapshot"]:
                from .z2_gt_pack_runtime import sync_gt_items_after_xml_save
                job["gt_result"] = sync_gt_items_after_xml_save(
                    job["path"], [v["item"] for v in job["gt_snapshot"].values()])
        except Exception as exc:
            job["error"] = exc
        finally:
            job["done"].set()
        host._post_to_ui(complete)

    try:
        threading.Thread(target=worker, name="z2-preview-autosave", daemon=False).start()
    except Exception:
        job["done"].set()
        host._preview_background_save = None
        raise
    return True
