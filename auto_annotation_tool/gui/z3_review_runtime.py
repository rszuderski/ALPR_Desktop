#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Jawny przepływ Z3/PZ2: RAW -> REVIEW -> GOLD."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from datetime import datetime
from tkinter import messagebox

from .z3_gt_contract import revision_ids_from_data
from .z3_gt_box_policy import plate_gt
from .z3_metadata_cache import mark_preview_metadata_changed

REVIEW_SCHEMA = "alpr.pz2.review.v1"
REVIEW_IN_PROGRESS = "in_progress"
REVIEW_APPROVED = "approved"
PIPELINE_PREPARATION_VERSION = "pipeline_blocks_only.v1"

def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")

def get_review_state_status(data: dict | None) -> str:
    if not isinstance(data, dict):
        return ""
    state = data.get("review_state")
    if not isinstance(state, dict):
        return ""
    return str(state.get("status", "") or "").strip().lower()


def _normalize_reference_text(value) -> str:
    return str(value or "").strip().upper()


def build_review_reference_snapshot(data: dict | None) -> dict:
    source = data if isinstance(data, dict) else {}
    attrs = source.get("plate_attributes")
    if not isinstance(attrs, dict):
        attrs = {}

    ground_truth_text = _normalize_reference_text(
        source.get("ground_truth_text")
        or attrs.get("ground_truth_text")
    )

    source_annotation_id = str(
        source.get("source_annotation_id")
        or attrs.get("source_annotation_id")
        or ""
    ).strip()
    plate_annotation_id = str(
        source.get("plate_annotation_id")
        or attrs.get("plate_annotation_id")
        or ""
    ).strip()
    ground_truth_source = str(
        source.get("ground_truth_source")
        or attrs.get("ground_truth_source")
        or ""
    ).strip()

    return {
        "gt_hash": str(source.get("source_gt_hash") or "").strip(),
        "gt_revision_ids": revision_ids_from_data(
            source,
            "source_gt_revision_ids",
            "source_gt_revision_id",
        ),
        "geometry_revision_ids": revision_ids_from_data(
            source,
            "source_geometry_revision_ids",
            "source_geometry_revision_id",
        ),
        "ground_truth_text": ground_truth_text,
        "source_annotation_id": source_annotation_id,
        "plate_annotation_id": plate_annotation_id,
        "ground_truth_source": ground_truth_source,
    }


def review_approval_is_current(data: dict | None) -> bool:
    if not isinstance(data, dict):
        return False

    state = data.get("review_state")
    if not isinstance(state, dict):
        return False
    if str(state.get("status", "") or "").strip().lower() != REVIEW_APPROVED:
        return False

    # GOLD/perfect wymaga jawnego GT. Sam REVIEW=approved nie może
    # podtrzymywać zatwierdzenia, jeśli GT nie istnieje albo zostało usunięte.
    if not plate_gt(data):
        return False

    approved_reference = state.get("approved_reference")
    if not isinstance(approved_reference, dict):
        return False

    return approved_reference == build_review_reference_snapshot(data)


def reconcile_review_approval(data: dict | None) -> bool:
    """
    Reopen REVIEW when GT/reference identity changed after human GOLD approval.
    A new RAW run is intentionally ignored here.
    """
    if not isinstance(data, dict):
        return False

    state = data.get("review_state")
    if not isinstance(state, dict):
        return False
    if str(state.get("status", "") or "").strip().lower() != REVIEW_APPROVED:
        return False
    if review_approval_is_current(data):
        return False

    now = _now_iso()
    previous_reference = copy.deepcopy(state.get("approved_reference"))

    state["status"] = REVIEW_IN_PROGRESS
    state["approved_at"] = None
    state["modified_at"] = now
    state["approval_invalidated_at"] = now
    state["approval_invalidated_reason"] = "reference_changed"
    state["invalidated_approved_reference"] = previous_reference
    state["current_reference"] = build_review_reference_snapshot(data)

    data["status"] = "needs_fix"

    gold_state = data.get("gold_state")
    if not isinstance(gold_state, dict):
        gold_state = {}
        data["gold_state"] = gold_state
    gold_state["candidate"] = False
    gold_state["approved"] = False

    return True



def reconcile_review_gold_integrity(data: dict | None) -> bool:
    """
    Keep persisted REVIEW/GOLD state fail-closed and internally consistent.

    Legacy records without review_state are intentionally untouched.
    """
    if not isinstance(data, dict):
        return False

    state = data.get("review_state")
    if not isinstance(state, dict):
        return False

    changed = False
    review_status = str(state.get("status", "") or "").strip().lower()

    gold_state = data.get("gold_state")
    if not isinstance(gold_state, dict):
        gold_state = {}
        data["gold_state"] = gold_state
        changed = True

    if review_status == REVIEW_APPROVED:
        if not review_approval_is_current(data):
            return bool(reconcile_review_approval(data) or changed)

        persisted_status = str(
            data.get("status", "unknown") or "unknown"
        ).strip().lower()
        gold_approved = bool(gold_state.get("approved", False))
        if persisted_status != "perfect" or not gold_approved:
            now = _now_iso()
            previous_reference = copy.deepcopy(
                state.get("approved_reference")
            )
            state["status"] = REVIEW_IN_PROGRESS
            state["approved_at"] = None
            state["modified_at"] = now
            state["approval_invalidated_at"] = now
            state["approval_invalidated_reason"] = (
                "approval_state_incomplete"
            )
            state["invalidated_approved_reference"] = previous_reference
            state["current_reference"] = build_review_reference_snapshot(data)
            data["status"] = "needs_fix"
            gold_state["candidate"] = False
            gold_state["approved"] = False
            return True

        if not bool(gold_state.get("candidate", False)):
            gold_state["candidate"] = True
            changed = True
        return changed

    if review_status == REVIEW_IN_PROGRESS:
        if str(data.get("status", "") or "").strip().lower() != "needs_fix":
            data["status"] = "needs_fix"
            changed = True
        if state.get("approved_at") is not None:
            state["approved_at"] = None
            changed = True
        if "approved_reference" in state:
            state.pop("approved_reference", None)
            changed = True
        if bool(gold_state.get("candidate", False)):
            gold_state["candidate"] = False
            changed = True
        if bool(gold_state.get("approved", False)):
            gold_state["approved"] = False
            changed = True
        return changed

    now = _now_iso()
    state["schema"] = REVIEW_SCHEMA
    state["status"] = REVIEW_IN_PROGRESS
    state["approved_at"] = None
    state["modified_at"] = now
    state["approval_invalidated_at"] = now
    state["approval_invalidated_reason"] = "invalid_review_state"
    state.pop("approved_reference", None)
    data["status"] = "needs_fix"
    gold_state["candidate"] = False
    gold_state["approved"] = False
    return True


def _resolve_plate(host, plate_id=None):
    pid = str(plate_id or getattr(host, "_preview_active_pid", "") or "").strip()
    metadata = getattr(host, "preview_metadata", None)
    if not pid or not isinstance(metadata, dict):
        return "", None
    data = metadata.get(pid)
    return pid, data if isinstance(data, dict) else None

def _refresh_after_change(host, pid: str, *, persist: bool, message: str = "") -> None:
    mark_preview_metadata_changed(host)
    if persist:
        host._persist_preview_metadata(
            success_message=None,
            refresh_list=False,
            sync_access=True,
        )
    try:
        host._refresh_preview_listbox_row(pid)
    except Exception:
        pass
    try:
        if host._get_preview_box_mode_key() != "AUTO":
            label = (
                host._get_preview_box_mode_label("AUTO")
                if hasattr(host, "_get_preview_box_mode_label")
                else "AUTO"
            )
            host.preview_box_mode_var.set(label)
    except Exception:
        pass
    try:
        host._refresh_detection_review_controls()
    except Exception:
        pass
    try:
        host._update_preview_info_label()
    except Exception:
        pass
    try:
        host._on_preview_select(None)
    except Exception:
        pass
    if message:
        try:
            host._update_preview_edit_status(message, tone="info")
        except Exception:
            pass



def _persist_review_az_revision_best_effort(
    host,
    plate_id: str,
    data: dict,
    *,
    event: str,
) -> dict:
    """Zapisz trwałą rewizję AZ po jawnej decyzji PZ2; nigdy nie blokuj UI."""
    try:
        from ..campaign_manager import CAMPAIGN
        from ..config import CONFIG, logger
        from ..registry.az_registry import AZRegistry, ensure_campaign_project
        from ..registry.pz2_revision_registry import save_pz2_revision

        registry = AZRegistry.for_workspace(CONFIG.WORKSPACE_DIR)

        project_id = None
        iteration_num = None
        active_project = str(
            CAMPAIGN.get_active_project_name() or ""
        ).strip()
        in_campaign = bool(
            getattr(host, "_step3_linear_mode", False)
            and active_project
        )

        if in_campaign:
            project_data = dict(
                (CAMPAIGN.state.get("projects", {}) or {}).get(
                    active_project,
                    {},
                )
                or {}
            )
            folder_name = str(
                project_data.get("folder_name") or ""
            ).strip()
            if not folder_name:
                raise RuntimeError(
                    "Aktywny projekt kampanii nie ma folder_name."
                )

            project_id = ensure_campaign_project(
                registry,
                project_name=active_project,
                folder_name=folder_name,
            )
            iteration_num = int(
                CAMPAIGN.get_current_iteration_num() or 1
            )

        saved = save_pz2_revision(
            registry,
            data,
            project_id=project_id,
            iteration_num=iteration_num,
        )

        logger.info(
            "[AZ][PZ2] revision event=%s plate=%s revision=%s created=%s "
            "project=%s iteration=%s status=%s",
            str(event or ""),
            str(plate_id or ""),
            saved.revision.az_revision_id,
            bool(saved.revision.created),
            project_id or "FREE",
            iteration_num or 0,
            saved.effective_status,
        )
        return {
            "ok": True,
            "event": str(event or ""),
            "plate_id": str(plate_id or ""),
            "az_revision_id": saved.revision.az_revision_id,
            "created": bool(saved.revision.created),
            "project_id": project_id,
            "iteration_num": iteration_num,
            "effective_status": saved.effective_status,
        }
    except Exception as exc:
        try:
            from ..config import logger
            logger.warning(
                "[AZ][PZ2] Nie udało się zapisać rewizji event=%s plate=%s: %s",
                str(event or ""),
                str(plate_id or ""),
                exc,
            )
        except Exception:
            pass
        return {
            "ok": False,
            "event": str(event or ""),
            "plate_id": str(plate_id or ""),
            "error": str(exc),
        }

def reopen_review_gold(
    host,
    plate_id=None,
    *,
    persist: bool = True,
    quiet: bool = False,
    refresh: bool = True,
):
    """Cofnij świadome OK bez usuwania GT, boxów, znaków ani wykluczenia."""
    pid, data = _resolve_plate(host, plate_id)
    if not isinstance(data, dict):
        return {
            "ok": False,
            "reason": "no_active_plate",
            "plate_id": pid,
        }

    state = data.get("review_state")
    gold_state = data.get("gold_state")
    approved = bool(
        isinstance(state, dict)
        and get_review_state_status(data) == REVIEW_APPROVED
        and isinstance(gold_state, dict)
        and bool(gold_state.get("approved", False))
        and str(data.get("status", "") or "").strip().lower() == "perfect"
    )
    if not approved:
        return {
            "ok": False,
            "reason": "not_approved",
            "plate_id": pid,
        }

    mark_review_edit_started(host, data)
    data["status"] = "needs_fix"

    if refresh:
        _refresh_after_change(
            host,
            pid,
            persist=persist,
            message="Cofnięto OK. Tablica wróciła do kontroli.",
        )

    if persist:
        _persist_review_az_revision_best_effort(
            host,
            pid,
            data,
            event="approval_reverted",
        )

    if not quiet:
        try:
            messagebox.showinfo(
                "Cofnięto OK",
                "Tablica wróciła do kontroli. GT, boxy i znaki pozostają bez zmian.",
            )
        except Exception:
            pass

    return {
        "ok": True,
        "reason": "",
        "plate_id": pid,
        "review_status": REVIEW_IN_PROGRESS,
        "status": "needs_fix",
    }


def toggle_review_excluded(
    host,
    plate_id=None,
    *,
    reason: str = "unreadable",
    persist: bool = True,
    refresh: bool = True,
):
    # Toggle manual PZ2 exclusion without deleting annotations or REVIEW/GOLD history.
    # Excluded records stay visible in PZ2 but must never be exportable to PZ3/Z4.
    pid, data = _resolve_plate(host, plate_id)
    if not isinstance(data, dict):
        return {
            "ok": False,
            "reason": "no_active_plate",
            "plate_id": pid,
            "excluded": False,
        }

    gold_state = data.get("gold_state")
    if not isinstance(gold_state, dict):
        gold_state = {}
        data["gold_state"] = gold_state

    excluded = not bool(gold_state.get("excluded", False))
    gold_state["excluded"] = excluded

    if excluded:
        gold_state["excluded_reason"] = str(reason or "unreadable").strip() or "unreadable"
        gold_state["excluded_at"] = _now_iso()
        gold_state["excluded_by"] = "human"
        gold_state["candidate"] = False
    else:
        gold_state["excluded_reason"] = ""
        gold_state.pop("excluded_at", None)
        gold_state.pop("excluded_by", None)

        status = str(data.get("status", "unknown") or "unknown").strip().lower()
        review_state = data.get("review_state")
        if not isinstance(review_state, dict):
            candidate = status == "perfect"
        else:
            candidate = bool(
                status == "perfect"
                and gold_state.get("approved", False)
                and review_approval_is_current(data)
            )
        gold_state["candidate"] = candidate

    if refresh:
        _refresh_after_change(host, pid, persist=persist, message="")
    else:
        # Hot-key F must stay lightweight: mutate one row now and let the
        # incremental metadata autosave persist it outside the key event.
        mark_preview_metadata_changed(host)
        if persist:
            try:
                host._schedule_preview_metadata_save(delay_ms=90)
            except Exception:
                pass

    if persist:
        _persist_review_az_revision_best_effort(
            host,
            pid,
            data,
            event="exclude_toggle",
        )
    return {
        "ok": True,
        "reason": "",
        "plate_id": pid,
        "excluded": excluded,
        "excluded_reason": str(gold_state.get("excluded_reason", "") or ""),
    }


def _working_annotation_fingerprint(data):
    """Hash editable content, excluding display caches and serialization backfills."""
    payload = {key: data.get(key) for key in (
        "ground_truth_text", "plate_layout", "plate_layout_override",
        "plate_layout_separator_y", "layout_override_source", "layout_override_updated_at",
    )}
    payload["characters"] = [{
        "bbox": [float(value) for value in rec.get("bbox", [])],
        **{key: rec.get(key) for key in ("character", "box_source", "sign_source")},
    } for rec in data.get("characters", [])]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def can_refresh_automatic_working_annotation(data):
    """Only replace a stale, untouched automatic preview; unknown history stays protected."""
    raw = data.get("raw_detection") or {}
    working = data.get("working_annotation") or {}
    state = data.get("review_state") or {}
    source = data.get("source_info") or {}
    if not all(isinstance(value, dict) for value in (raw, working, state, source)):
        return False
    if (data.get("last_detection") or {}).get("execution_mode") == "raw_evidence":
        return False
    old_hash = working.get("source_raw_result_hash")
    needs_pipeline_preparation = bool(
        working.get("automatic_content_hash")
        and (working.get("pipeline_preparation_version") != PIPELINE_PREPARATION_VERSION
             or working.get("pipeline_blocks") != raw.get("pipeline_blocks"))
    )
    if (not old_hash or not raw.get("result_hash")
            or (raw["result_hash"] == old_hash and not needs_pipeline_preparation)
            or state.get("raw_result_hash") != old_hash
            or state.get("status") != REVIEW_IN_PROGRESS
            or state.get("source") != "raw_detection"
            or state.get("human_edited") or state.get("reopened_at")
            or state.get("approved_at") or state.get("approved_reference")
            or (data.get("gold_state") or {}).get("approved")
            or data.get("status") == "perfect"
            or data.get("fusion_strategy") != "review_from_raw"):
        return False
    if source and (source.get("last_modified_by") != "system"
                   or source.get("bucket") != "auto_preview"
                   or source.get("origin") != "pz2_detect"):
        return False
    baseline = working.get("automatic_content_hash")
    if baseline:
        try:
            return baseline == _working_annotation_fingerprint(data)
        except (TypeError, ValueError, AttributeError):
            return False
    # Legacy materializations did not store a content hash. Require all their
    # independent provenance markers to confirm no review action took place.
    prepared = working.get("prepared_at")
    if (not source or not prepared or state.get("started_at") != prepared
            or state.get("modified_at") != prepared
            or data.get("layout_override_updated_at")
            or data.get("layout_override_source")
            or not data.get("characters")):
        return False
    try:
        if datetime.fromisoformat(source["last_modified_at"]).replace(tzinfo=None) != datetime.fromisoformat(prepared).replace(tzinfo=None):
            return False
    except (KeyError, TypeError, ValueError):
        return False
    return all(isinstance(rec, dict) and not any(str(rec.get(key) or "").lower().startswith(("manual", "cvat", "human"))
                       for key in ("method", "source_tag", "box_source", "sign_source", "correction_source"))
               for rec in data["characters"])


def refresh_stale_automatic_working_annotations(host, metadata):
    """Recover stale previews on loading a run, without rerunning inference."""
    changed = False
    for pid, data in metadata.items():
        if (isinstance(data, dict)
                and (data.get("last_detection") or {}).get("execution_mode") == "annotation"
                and can_refresh_automatic_working_annotation(data)):
            changed = prepare_working_annotation_from_raw(host, data, plate_id=pid) or changed
    return changed


def prepare_working_annotation_from_raw(host, data, *, plate_id="", overwrite=False, plate_image=None,
                                      geometry_settings=None, from_detection=False):
    """Refresh untouched automatic results; preserve human work and approvals."""
    if not isinstance(data, dict) or not isinstance(data.get("raw_detection"), dict):
        return False
    # Clearing automatic boxes is a human edit, but explicitly running the
    # pipeline again authorizes filling the cleared slots. Loading/rendering
    # the old RAW does not. Other human corrections remain protected.
    cleared = data.get("automatic_boxes_cleared")
    current_chars = data.get("characters") or []
    refill_cleared = bool(
        from_detection and isinstance(cleared, dict) and cleared.get("removed_count", 0) > 0
        and get_review_state_status(data) == REVIEW_IN_PROGRESS
        and not (data.get("gold_state") or {}).get("approved") and data.get("status") != "perfect"
        and all(host._get_character_box_source_tag(rec, data=data, fallback_index=index) == "manual_box"
                for index, rec in enumerate(current_chars))
    )
    preserved_manuals = copy.deepcopy(current_chars) if refill_cleared else []
    if not overwrite and not refill_cleared and (get_review_state_status(data) or data.get("characters")
                          or data.get("working_annotation")
                          or data.get("fusion_strategy") in {"manual_correction", "manual", "cvat_import"}
                          or (data.get("gold_state") or {}).get("approved")
                          or str(data.get("status") or "") == "perfect"):
        if not can_refresh_automatic_working_annotation(data):
            return False
    pid = plate_id
    raw_detection = data["raw_detection"]
    raw_chars = raw_detection.get("characters") or []
    review_chars = copy.deepcopy(raw_chars)
    # Geometry comes only from the declared pipeline's RAW result. GT may
    # select/label existing boxes, but must not add an implicit segmentation.
    data.pop("gt_geometry_recovery", None)
    if preserved_manuals:
        from .z3_preview_records import merge_detected_characters_preserving_manual
        review_chars, _merge_info = merge_detected_characters_preserving_manual(
            host, preserved_manuals, review_chars, data=data,
        )
    try:
        host._update_preview_plate_layout_metadata(data, review_chars)
        review_chars = host._apply_preview_layout_separator_constraints_to_chars(
            data,
            review_chars,
        )
    except Exception:
        pass
    try:
        review_chars = host._annotate_preview_character_reading_positions(
            host._sort_character_records_by_x(review_chars, data=data),
            data=data,
        )
    except Exception:
        try:
            review_chars = host._sort_character_records_by_x(review_chars, data=data)
        except Exception:
            pass

    now = _now_iso()
    raw_hash = str(raw_detection.get("result_hash", "") or "").strip()

    data["characters"] = review_chars
    data["status"] = "needs_fix"
    data["fusion_strategy"] = "review_from_raw"
    data["fusion_details"] = {
        "source": "raw_detection",
        "raw_result_hash": raw_hash,
    }
    data["review_state"] = {
        "schema": REVIEW_SCHEMA,
        "status": REVIEW_IN_PROGRESS,
        "source": "raw_detection",
        "raw_result_hash": raw_hash,
        "started_at": now,
        "modified_at": now,
        "approved_at": None,
    }

    gold_state = data.get("gold_state")
    if not isinstance(gold_state, dict):
        gold_state = {}
        data["gold_state"] = gold_state
    gold_state["candidate"] = False
    gold_state["approved"] = False
    gold_state["approved_from_bucket"] = str(
        gold_state.get("approved_from_bucket", "") or ""
    )

    try:
        host._ensure_plate_source_metadata(
            data,
            plate_id=pid,
            default_bucket="auto_preview",
            default_origin="pz2_detect",
            modified_by="system",
        )
    except Exception:
        pass

    assist = getattr(host, "_apply_live_gt_assist", None)
    if callable(assist):
        assist(data, prepare=True)
    if preserved_manuals:
        # GT count selection must not delete any manual geometry, even if
        # the operator intentionally placed more boxes than the reference.
        automatic = [rec for index, rec in enumerate(data["characters"])
                     if host._get_character_box_source_tag(rec, data=data, fallback_index=index) != "manual_box"]
        data["characters"] = host._annotate_preview_character_reading_positions(
            host._sort_character_records_by_x(automatic + preserved_manuals, data=data), data=data,
        )
        if callable(assist):
            assist(data, prepare=False)
    if refill_cleared:
        automatic_count = sum(host._get_character_box_source_tag(rec, data=data, fallback_index=index) != "manual_box"
                              for index, rec in enumerate(data["characters"]))
        if automatic_count:
            data.pop("automatic_boxes_cleared", None)
        if preserved_manuals or not automatic_count:
            data["review_state"]["human_edited"] = True
            data["fusion_strategy"] = "manual_correction"
    data["working_annotation"] = {
        "schema": "alpr.pz2.working.v1", "prepared_at": now,
        "source_raw_result_hash": raw_hash,
        "preparation": "gt_assisted" if plate_gt(data) else "pipeline",
        "automatic_content_hash": _working_annotation_fingerprint(data),
        "pipeline_preparation_version": PIPELINE_PREPARATION_VERSION,
        "pipeline_blocks": copy.deepcopy(raw_detection.get("pipeline_blocks")),
    }
    return True


def start_review_from_raw(
    host,
    plate_id=None,
    *,
    overwrite: bool = False,
    persist: bool = True,
    quiet: bool = False,
    refresh: bool = True,
):
    """Create mutable REVIEW as a deep copy of frozen RAW."""
    pid, data = _resolve_plate(host, plate_id)
    if not isinstance(data, dict):
        result = {"ok": False, "reason": "no_active_plate", "plate_id": pid}
        if not quiet:
            messagebox.showinfo("Sprawdzanie znaków", "Wybierz tablicę do sprawdzenia.")
        return result

    raw_detection = data.get("raw_detection")
    if not isinstance(raw_detection, dict):
        result = {"ok": False, "reason": "missing_raw_detection", "plate_id": pid}
        if not quiet:
            messagebox.showinfo(
                "Brak wyniku wykrywania",
                "Ta tablica nie ma jeszcze wyniku wykrywania. Najpierw uruchom wykrywanie znaków.",
            )
        return result

    raw_chars = raw_detection.get("characters", [])
    if not isinstance(raw_chars, list):
        raw_chars = []

    current_review_status = get_review_state_status(data)
    existing_chars = data.get("characters", [])
    existing_chars = existing_chars if isinstance(existing_chars, list) else []

    if not overwrite and (current_review_status or existing_chars):
        result = {
            "ok": False,
            "reason": "review_exists",
            "plate_id": pid,
            "review_status": current_review_status,
            "character_count": len(existing_chars),
        }
        if not quiet:
            messagebox.showinfo(
                "Tablica jest już w trakcie sprawdzania",
                "Nie nadpisano wcześniejszych poprawek ani zatwierdzenia.",
            )
        return result

    prepare_working_annotation_from_raw(host, data, plate_id=pid, overwrite=True)
    raw_hash = str(raw_detection.get("result_hash") or "")
    if refresh:
        _refresh_after_change(
            host, pid, persist=persist,
            message="Wynik modelu został otwarty do sprawdzenia i korekty.",
        )
    return {
        "ok": True,
        "reason": "",
        "plate_id": pid,
        "review_status": REVIEW_IN_PROGRESS,
        "character_count": len(data["characters"]),
        "raw_result_hash": raw_hash,
    }

def can_restore_empty_review(data: dict | None) -> bool:
    """Only an explicit recovery action may refill an already empty REVIEW."""
    return bool(
        isinstance(data, dict)
        and get_review_state_status(data) == REVIEW_IN_PROGRESS
        and not data.get("characters")
        and isinstance(data.get("raw_detection"), dict)
        and data["raw_detection"].get("characters")
    )


def open_or_restore_review(host):
    """Toolbar action: open a prediction or recover boxes in an empty review."""
    if any(getattr(host, flag, False) for flag in (
        "fast_test_running", "is_processing", "_preview_review_batch_running"
    )):
        return {"ok": False, "reason": "busy"}
    pid, data = _resolve_plate(host)
    recover = can_restore_empty_review(data)
    if recover:
        host._push_preview_history_snapshot(pid)
    return start_review_from_raw(host, pid, overwrite=recover)


def prepare_layout_review(host, data: dict) -> bool:
    """Materialize the prediction before a layout gesture opens REVIEW.

    The caller owns history, the layout mutation and the final redraw/save.
    Existing corrections, including an intentionally empty one, are preserved.
    """
    if any(getattr(host, flag, False) for flag in (
        "fast_test_running", "is_processing", "_preview_review_batch_running"
    )):
        return False
    if not get_review_state_status(data) and not data.get("characters"):
        if isinstance(data.get("raw_detection"), dict):
            result = start_review_from_raw(host, persist=False, quiet=True, refresh=False)
            if not result.get("ok"):
                return False
        else:
            from .z3_gt_box_policy import working_characters
            data["characters"] = copy.deepcopy(working_characters(host, data))
    mode_var = getattr(host, "preview_box_mode_var", None)
    if mode_var is not None:
        mode_var.set(host._get_preview_box_mode_label("AUTO"))
    return True


def prepare_active_preview_review(host) -> bool:
    """An editing gesture opens the displayed prediction without another CTA."""
    if getattr(host, "fast_test_running", False) or getattr(host, "is_processing", False):
        return False
    pid, data = _resolve_plate(host)
    if not isinstance(data, dict):
        return False
    created = False
    if not get_review_state_status(data) and not data.get("characters") and isinstance(data.get("raw_detection"), dict):
        host._push_preview_history_snapshot(pid)
        result = start_review_from_raw(host, pid, persist=False, quiet=True, refresh=False)
        if not result.get("ok"):
            return False
        created = True
    if not get_review_state_status(data):
        mark_review_edit_started(host, data)
        created = True
    previous_mode = host._get_preview_box_mode_key()
    if previous_mode != "AUTO":
        host.preview_box_mode_var.set(host._get_preview_box_mode_label("AUTO"))
    assist = getattr(host, "_apply_live_gt_assist", None)
    changed = bool(assist(data).get("changed")) if callable(assist) else False
    if created or changed:
        host._schedule_preview_metadata_save(delay_ms=350)
        host._refresh_preview_listbox_row(pid)
        host._refresh_detection_review_controls()
    if created or changed or previous_mode != "AUTO":
        host._redraw_preview_character_overlays_light()
    return True


def mark_review_edit_started(host, data: dict | None):
    """Any human edit opens/reopens REVIEW and invalidates GOLD approval."""
    if not isinstance(data, dict):
        return None

    now = _now_iso()
    raw = data.get("raw_detection")
    raw_hash = str(raw.get("result_hash", "") or "").strip() if isinstance(raw, dict) else ""

    state = data.get("review_state")
    previous_status = (
        str(state.get("status", "") or "").strip().lower()
        if isinstance(state, dict)
        else ""
    )
    if not isinstance(state, dict):
        state = {
            "schema": REVIEW_SCHEMA,
            "source": "raw_detection" if isinstance(raw, dict) else "manual_editor",
            "raw_result_hash": raw_hash,
            "started_at": now,
        }
        data["review_state"] = state

    state["schema"] = REVIEW_SCHEMA
    state["status"] = REVIEW_IN_PROGRESS
    state["modified_at"] = now
    state["human_edited"] = True
    state["approved_at"] = None
    state.pop("approved_reference", None)
    if not str(state.get("source", "") or "").strip():
        state["source"] = "raw_detection" if isinstance(raw, dict) else "manual_editor"
    if raw_hash and not str(state.get("raw_result_hash", "") or "").strip():
        state["raw_result_hash"] = raw_hash
    if not str(state.get("started_at", "") or "").strip():
        state["started_at"] = now
    if previous_status == REVIEW_APPROVED:
        state["reopened_at"] = now

    gold_state = data.get("gold_state")
    if not isinstance(gold_state, dict):
        gold_state = {}
        data["gold_state"] = gold_state
    gold_state["candidate"] = False
    gold_state["approved"] = False
    return state

def get_review_quality_status(host, data: dict, chars=None) -> str:
    """Validate current geometry/GT before committing a human approval."""
    if data.get("gt_revision_conflict"):
        return "needs_fix"
    probe = dict(data)
    records = data.get("characters", []) if chars is None else chars
    if data.get("plate_layout") == "two_row_candidate" and not data.get("plate_layout_override"):
        return "needs_fix"
    try:
        for record in records:
            box = record.get("bbox") if isinstance(record, dict) else getattr(record, "bbox", None)
            x1, y1, x2, y2 = map(float, box)
            if not all(math.isfinite(value) for value in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
                return "needs_fix"
    except (TypeError, ValueError):
        return "needs_fix"
    # Tylko GT odziedziczone z Z2 jest niezależną referencją, względem
    # której PZ2 może samodzielnie oceniać kompletność/liczbę znaków.
    # Bez takiego GT kompletność ramek jest decyzją operatora. Program
    # sprawdza jedynie, czy istniejące ramki mają poprawną geometrię i znak.
    attrs = data.get("plate_attributes")
    attrs = attrs if isinstance(attrs, dict) else {}
    number_source = str(
        data.get("ground_truth_source")
        or attrs.get("ground_truth_source")
        or ""
    ).strip().lower()
    protected_z2_gt = bool(plate_gt(data) and number_source == "manual_z2")

    if not protected_z2_gt:
        from ..plate_ground_truth import normalize_plate_ground_truth_text
        text = normalize_plate_ground_truth_text(
            host._characters_to_text(records, data=data)
        )
        if not text or len(text) != len(records):
            return "needs_fix"

        # Do oceny "czy można nacisnąć O" używamy bieżącego, ręcznie
        # sprawdzonego odczytu. To nie zapisuje GT - zapis następuje dopiero
        # po świadomym O w confirm_review_gold().
        probe["ground_truth_text"] = text
        probe["ground_truth_source"] = "manual_z3"
        probe_attrs = dict(probe.get("plate_attributes") or {})
        probe_attrs["ground_truth_text"] = text
        probe_attrs["ground_truth_source"] = "manual_z3"
        probe["plate_attributes"] = probe_attrs
    probe_state = dict(probe.get("review_state") or {})
    probe_state["status"] = REVIEW_APPROVED
    probe_state["approved_reference"] = build_review_reference_snapshot(probe)
    probe["review_state"] = probe_state
    return str(host._derive_preview_status_from_data(probe, records) or "needs_fix").strip().lower()


def confirm_review_gold(
    host,
    plate_id=None,
    *,
    persist: bool = True,
    quiet: bool = False,
    refresh: bool = True,
):
    """Explicit human approval. Only a valid REVIEW may become GOLD/perfect."""
    pid, data = _resolve_plate(host, plate_id)
    if not isinstance(data, dict):
        result = {"ok": False, "reason": "no_active_plate", "plate_id": pid}
        if not quiet:
            messagebox.showinfo("Zatwierdzanie tablicy", "Wybierz tablicę do zatwierdzenia.")
        return result

    state = data.get("review_state")
    if not isinstance(state, dict) or get_review_state_status(data) != REVIEW_IN_PROGRESS:
        result = {"ok": False, "reason": "review_not_in_progress", "plate_id": pid}
        if not quiet:
            messagebox.showinfo("Najpierw sprawdź tablicę", "Otwórz wynik modelu do sprawdzenia i wprowadź potrzebne poprawki.")
        return result

    chars = data.get("characters", [])
    if not isinstance(chars, list) or not chars:
        result = {"ok": False, "reason": "empty_review", "plate_id": pid}
        if not quiet:
            messagebox.showwarning(
                "Nie można zatwierdzić tablicy",
                "Ta tablica nie zawiera jeszcze żadnych ramek znaków.",
            )
        return result

    from ..plate_ground_truth import normalize_plate_ground_truth_text

    candidate_text = normalize_plate_ground_truth_text(
        host._characters_to_text(chars, data=data)
    )
    saved_text = plate_gt(data)

    attrs = data.get("plate_attributes")
    attrs = attrs if isinstance(attrs, dict) else {}
    number_source = str(
        data.get("ground_truth_source")
        or attrs.get("ground_truth_source")
        or ""
    ).strip().lower()
    inherited_from_z2 = bool(saved_text and number_source == "manual_z2")

    if saved_text and candidate_text and candidate_text != saved_text:
        # Tylko GT odziedziczone z Z2 jest twardą referencją. Każdy inny
        # zapisany numer jest lokalnym/fallbackowym stanem roboczym PZ2 i
        # świadome R operatora może go skorygować do aktualnego odczytu.
        if not inherited_from_z2:
            try:
                from .z3_plate_gt_runtime import save_plate_ground_truth
                save_plate_ground_truth(host, data, candidate_text, prepare=False)
            except Exception as exc:
                if not quiet:
                    messagebox.showerror("Nie zapisano numeru tablicy", str(exc))
                return {
                    "ok": False,
                    "reason": "number_write_failed",
                    "error": str(exc),
                    "plate_id": pid,
                }
            saved_text = candidate_text
            number_source = "manual_z3"
        else:
            data["status"] = "needs_fix"
            result = {
                "ok": False,
                "reason": "number_mismatch",
                "plate_id": pid,
                "resolved_status": "needs_fix",
                "saved_text": saved_text,
                "candidate_text": candidate_text,
                "inherited_from_z2": inherited_from_z2,
            }
            if not quiet:
                source_label = "numerem z Z2" if inherited_from_z2 else "zapisanym numerem"
                messagebox.showwarning(
                    "Odczyt różni się od zapisanego numeru",
                    f"Wpisane znaki nie zgadzają się z {source_label}: {saved_text}. "
                    "Popraw znaki albo użyj „Zmień numer”, jeśli zapisany numer jest błędny.",
                )
            return result

    try:
        resolved_status = get_review_quality_status(host, data, chars)
    except Exception:
        resolved_status = "needs_fix"

    if resolved_status != "perfect":
        data["status"] = "needs_fix"
        result = {
            "ok": False,
            "reason": "review_not_perfect",
            "plate_id": pid,
            "resolved_status": resolved_status,
        }
        if not quiet:
            messagebox.showwarning(
                "Tablica nadal wymaga poprawy",
                "Aktualne ramki lub znaki nie są jeszcze poprawne. "
                "Wprowadź poprawki i spróbuj zatwierdzić ponownie.",
            )
        return result

    # Pierwsze O jest jawną decyzją operatora: potwierdza, że sprawdzone boxy
    # i wpisane znaki przedstawiają prawidłowy numer tej tablicy.
    if not saved_text:
        if not candidate_text:
            return {
                "ok": False,
                "reason": "empty_number",
                "plate_id": pid,
                "resolved_status": "needs_fix",
            }
        try:
            from .z3_plate_gt_runtime import save_plate_ground_truth
            save_plate_ground_truth(host, data, candidate_text, prepare=False)
        except Exception as exc:
            if not quiet:
                messagebox.showerror("Nie zapisano numeru tablicy", str(exc))
            return {
                "ok": False,
                "reason": "number_write_failed",
                "error": str(exc),
                "plate_id": pid,
            }

    now = _now_iso()
    state["schema"] = REVIEW_SCHEMA
    state["status"] = REVIEW_APPROVED
    state["approved_at"] = now
    state["modified_at"] = now
    state["approved_by"] = "human"
    state["approved_reference"] = build_review_reference_snapshot(data)
    state.pop("approval_invalidated_at", None)
    state.pop("approval_invalidated_reason", None)
    state.pop("invalidated_approved_reference", None)
    state.pop("current_reference", None)
    data["status"] = "perfect"

    if data.get("correction_source") == "gt_assisted" or any(rec.get("correction_source") == "gt_assisted" for rec in chars if isinstance(rec, dict)):
        data["review_source"] = "gt_assist_confirmed"
        for rec in chars:
            if isinstance(rec, dict) and rec.get("correction_source") == "gt_assisted":
                rec["correction_source"] = "gt_assist_confirmed"

    try:
        host._ensure_plate_source_metadata(data, plate_id=pid, modified_by="human")
    except Exception:
        pass

    gold_state = data.get("gold_state")
    if not isinstance(gold_state, dict):
        gold_state = {}
        data["gold_state"] = gold_state
    gold_state["approved"] = True
    gold_state["candidate"] = not bool(gold_state.get("excluded", False))
    try:
        bucket = str(host._get_plate_source_bucket(data) or "").strip()
    except Exception:
        source_info = data.get("source_info")
        bucket = (
            str(source_info.get("bucket", "") or "").strip()
            if isinstance(source_info, dict)
            else ""
        )
    if bucket:
        gold_state["approved_from_bucket"] = bucket

    if refresh:
        _refresh_after_change(
            host, pid, persist=persist,
            message="Tablica została sprawdzona i zatwierdzona do zbioru danych.",
        )

    # Trwały checkpoint AZ jest semantyką decyzji operatora, a nie renderowania UI.
    # Skrót R używa lekkiej ścieżki bez pełnego refreshu, ale nadal musi zapisać AZ.
    if persist:
        _persist_review_az_revision_best_effort(
            host,
            pid,
            data,
            event="review_approved",
        )

    if not quiet:
        try:
            messagebox.showinfo("Tablica zatwierdzona", "Tablica została sprawdzona i może zostać użyta w zbiorze danych.")
        except Exception:
            pass

    return {
        "ok": True,
        "reason": "",
        "plate_id": pid,
        "review_status": REVIEW_APPROVED,
        "status": "perfect",
    }
