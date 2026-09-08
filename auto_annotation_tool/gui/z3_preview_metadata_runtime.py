#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z3 preview metadata, list refresh and undo/redo helpers."""

from __future__ import annotations

import copy
import json
import time
from .z3_metadata_cache import mark_preview_metadata_changed, PreviewMetadataAutosave, path_key
import tkinter as tk

from ..config import logger


def _is_exportable_character_record(self, rec) -> bool:
    if isinstance(rec, dict):
        symbol = rec.get("character", "")
    else:
        symbol = getattr(rec, "character", "")
    if not self._sanitize_preview_char_symbol(symbol):
        return False

    bbox = self._char_record_bbox(rec)
    if not bbox:
        return False
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    except Exception:
        return False
    return bool(x2 > x1 and y2 > y1)

def _derive_preview_status_from_characters(self, chars) -> str:
    if not isinstance(chars, list) or not chars:
        return "needs_fix"
    valid_count = 0
    for rec in chars:
        if self._is_exportable_character_record(rec):
            valid_count += 1
    return "perfect" if valid_count == len(chars) and valid_count > 0 else "needs_fix"

def _preview_has_reference_text_source(self, data: dict | None = None) -> bool:
    source_data = data if isinstance(data, dict) else self._get_preview_active_data(create=False)
    return any(bool(value) for value in self._get_preview_reference_text_values(source_data))

def _get_preview_reference_text_values(self, data: dict | None = None) -> list[str]:
    source_data = data if isinstance(data, dict) else self._get_preview_active_data(create=False)
    if not isinstance(source_data, dict):
        return []
    values = []
    for field_name in ("source_image", "source_name", "filename"):
        prepared = str(source_data.get(field_name, "") or "").strip()
        if prepared:
            values.append(prepared)
    return values

def _normalize_preview_expected_text_values(value) -> list[str]:
    if value is None:
        return []
    source_values = value
    if isinstance(value, str):
        prepared = value.strip()
        if not prepared:
            return []
        if prepared[:1] in ("[", "{"):
            try:
                decoded = json.loads(prepared)
                source_values = decoded
            except Exception:
                source_values = prepared
        else:
            source_values = prepared
    if isinstance(source_values, dict):
        source_values = list(source_values.values())
    elif not isinstance(source_values, (list, tuple, set)):
        source_values = [source_values]

    normalized = []
    seen = set()
    for item in source_values:
        prepared = str(item or "").strip().upper()
        if not prepared or prepared in seen:
            continue
        seen.add(prepared)
        normalized.append(prepared)
    return normalized

def _merge_preview_expected_text_values(*groups) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for item in list(group or []):
            prepared = str(item or "").strip().upper()
            if not prepared or prepared in seen:
                continue
            seen.add(prepared)
            merged.append(prepared)
    return merged

def _get_preview_filename_expected_texts(self, data: dict | None = None) -> list[str]:
    source_data = data if isinstance(data, dict) else self._get_preview_active_data(create=False)
    normalized = []
    seen = set()
    for candidate in self._get_preview_reference_text_values(source_data):
        if not candidate:
            continue
        for item in self._get_true_texts_from_filename(candidate):
            prepared = str(item or "").strip().upper()
            if not prepared or prepared in seen:
                continue
            seen.add(prepared)
            normalized.append(prepared)
    return normalized

def _should_merge_filename_expected_texts(data: dict | None, explicit_values: list[str], filename_values: list[str]) -> bool:
    if not isinstance(data, dict) or not filename_values:
        return False
    explicit_set = {str(item or "").strip().upper() for item in list(explicit_values or []) if str(item or "").strip()}
    filename_set = {str(item or "").strip().upper() for item in list(filename_values or []) if str(item or "").strip()}
    if not filename_set or filename_set.issubset(explicit_set):
        return False

    attrs = data.get("plate_attributes")
    expected_source = str(data.get("source_expected_text_source") or "").strip().lower()
    if not expected_source and isinstance(attrs, dict):
        expected_source = str(attrs.get("source_expected_text_source") or "").strip().lower()
    if expected_source in {"ambiguous_filename_tokens", "filename_order", "filename_order_backfill"}:
        return True

    try:
        source_plate_count = int(data.get("source_plate_count") or 0)
    except Exception:
        source_plate_count = 0
    if source_plate_count > 1:
        return True
    if len(filename_set) > 1:
        return True
    return False

def _get_preview_expected_texts(self, data: dict | None = None) -> list[str]:
    source_data = data if isinstance(data, dict) else self._get_preview_active_data(create=False)
    filename_values = self._get_preview_filename_expected_texts(source_data if isinstance(source_data, dict) else None)
    if isinstance(source_data, dict):
        for field_name in ("source_expected_texts", "expected_texts", "ground_truth_texts"):
            prepared_values = _normalize_preview_expected_text_values(source_data.get(field_name))
            if prepared_values:
                if _should_merge_filename_expected_texts(source_data, prepared_values, filename_values):
                    return _merge_preview_expected_text_values(prepared_values, filename_values)
                return prepared_values
        attrs = source_data.get("plate_attributes")
        if isinstance(attrs, dict):
            for field_name in ("source_expected_texts", "expected_texts", "ground_truth_texts"):
                prepared_values = _normalize_preview_expected_text_values(attrs.get(field_name))
                if prepared_values:
                    if _should_merge_filename_expected_texts(source_data, prepared_values, filename_values):
                        return _merge_preview_expected_text_values(prepared_values, filename_values)
                    return prepared_values
        for field_name in ("source_expected_text", "expected_text", "ground_truth_text"):
            prepared = str(source_data.get(field_name, "") or "").strip().upper()
            if prepared:
                prepared_values = [prepared]
                if _should_merge_filename_expected_texts(source_data, prepared_values, filename_values):
                    return _merge_preview_expected_text_values(prepared_values, filename_values)
                return prepared_values
        if isinstance(attrs, dict):
            for field_name in ("source_expected_text", "expected_text", "ground_truth_text"):
                prepared = str(attrs.get(field_name, "") or "").strip().upper()
                if prepared:
                    prepared_values = [prepared]
                    if _should_merge_filename_expected_texts(source_data, prepared_values, filename_values):
                        return _merge_preview_expected_text_values(prepared_values, filename_values)
                    return prepared_values
    return filename_values

def _resolve_preview_expected_text_for_crop(self, data: dict | None = None, chars=None) -> dict:
    source_data = data if isinstance(data, dict) else self._get_preview_active_data(create=False)
    source_chars = chars
    if source_chars is None and isinstance(source_data, dict):
        source_chars = source_data.get("characters", [])
    if not isinstance(source_chars, list):
        source_chars = []

    expected_texts = self._get_preview_expected_texts(source_data if isinstance(source_data, dict) else None)
    expected_texts = [
        str(text or "").strip().upper()
        for text in (expected_texts or [])
        if str(text or "").strip()
    ]

    try:
        candidate_text = self._characters_to_text(source_chars, data=source_data).strip().upper()
    except Exception:
        candidate_text = ""

    expected_lengths = sorted({len(text) for text in expected_texts if text})
    matched_text = candidate_text if candidate_text and candidate_text in expected_texts else ""
    target_text = ""
    target_length = 0
    count_resolved = False
    resolution = "none"

    if matched_text:
        target_text = matched_text
        target_length = len(matched_text)
        count_resolved = True
        resolution = "exact_reading"
    elif len(expected_texts) == 1:
        target_text = expected_texts[0]
        target_length = len(target_text)
        count_resolved = True
        resolution = "single_expected"
    elif len(expected_lengths) == 1:
        target_length = int(expected_lengths[0])
        count_resolved = True
        resolution = "uniform_length"
    elif expected_lengths:
        resolution = "ambiguous_lengths"

    return {
        "candidate_text": candidate_text,
        "expected_texts": expected_texts,
        "expected_lengths": expected_lengths,
        "matched_text": matched_text,
        "target_text": target_text,
        "target_length": int(target_length or 0),
        "target_lengths": ([int(target_length)] if count_resolved and int(target_length or 0) > 0 else expected_lengths),
        "text_resolved": bool(matched_text),
        "count_resolved": bool(count_resolved),
        "ambiguous": bool(expected_lengths and not count_resolved),
        "resolution": resolution,
    }

def _derive_preview_status_from_data(self, data: dict | None, chars) -> str:
    base_status = self._derive_preview_status_from_characters(chars)
    if base_status != "perfect":
        return base_status
    try:
        if self._preview_layout_separator_conflicts_with_chars(data, chars):
            return "needs_fix"
    except Exception:
        pass

    expected_resolution = self._resolve_preview_expected_text_for_crop(data, chars)
    expected_texts = list(expected_resolution.get("expected_texts", []) or [])
    if not expected_texts:
        if self._preview_has_reference_text_source(data):
            return "needs_fix"
        return base_status

    if bool(expected_resolution.get("text_resolved")):
        return "perfect"
    return "needs_fix"

def _get_preview_live_status(self, data: dict | None = None, chars=None, plate_id: str | None = None) -> str:
    source_data = data if isinstance(data, dict) else self._get_preview_active_data(create=False)
    if not isinstance(source_data, dict):
        return str((data or {}).get("status", "unknown") or "unknown").strip().lower() if isinstance(data, dict) else "unknown"

    probe = dict(source_data)
    resolved_plate_id = str(plate_id or getattr(self, "_preview_active_pid", "") or probe.get("plate_id", "") or "").strip()
    if resolved_plate_id:
        probe["plate_id"] = resolved_plate_id

    source_chars = chars
    if source_chars is None:
        source_chars = probe.get("characters", [])
    return str(self._derive_preview_status_from_data(probe, source_chars) or "unknown").strip().lower()

def _recalculate_preview_statuses_in_metadata(self, metadata: dict | None):
    if not isinstance(metadata, dict):
        return metadata

    for plate_id, raw_data in list(metadata.items()):
        if not isinstance(raw_data, dict):
            continue
        self._ensure_plate_source_metadata(raw_data, plate_id=str(plate_id or ""))
        chars = raw_data.get("characters", None)
        if not isinstance(chars, list):
            chars = []
            raw_data["characters"] = chars
        self._update_preview_plate_layout_metadata(raw_data, chars)
        chars = self._annotate_preview_character_reading_positions(
            self._sort_character_records_by_x(chars, data=raw_data),
            data=raw_data,
        )
        raw_data["characters"] = chars
        if not chars:
            raw_data["status"] = "needs_fix"
            continue

        status_probe = dict(raw_data)
        status_probe["plate_id"] = str(plate_id)
        raw_data["status"] = self._derive_preview_status_from_data(status_probe, chars)

    return metadata

def _persist_preview_metadata(
    self,
    *,
    success_message: str | None = None,
    refresh_list: bool = True,
    sync_access: bool = True,
    mark_current_work: bool = False,
    mark_reason: str = "pz2_manual_ready",
):
    mark_preview_metadata_changed(self)
    writer = getattr(self, "_preview_autosave_writer", None)
    if writer is not None:
        writer.close()
        self._preview_autosave_writer = None
    perf_start = time.perf_counter()
    write_ms = refresh_ms = sync_ms = info_ms = 0.0
    meta_path = self._get_preview_metadata_path()
    if meta_path is None:
        raise RuntimeError("Brak aktywnego preview runu do zapisania.")

    phase_start = time.perf_counter()
    self._atomic_write_json(meta_path, self.preview_metadata)
    write_ms = (time.perf_counter() - phase_start) * 1000.0
    self._loaded_meta_path = meta_path
    try:
        self._loaded_meta_mtime = meta_path.stat().st_mtime
    except Exception:
        self._loaded_meta_mtime = None

    if refresh_list:
        phase_start = time.perf_counter()
        self._refresh_listbox_rows_from_metadata()
        refresh_ms = (time.perf_counter() - phase_start) * 1000.0
    if bool(sync_access):
        phase_start = time.perf_counter()
        self._sync_step3_access_from_preview_state(
            self.preview_metadata,
            mark_current_work=bool(mark_current_work),
            mark_reason=str(mark_reason or "pz2_manual_ready"),
        )
        sync_ms = (time.perf_counter() - phase_start) * 1000.0
    if success_message:
        phase_start = time.perf_counter()
        self._set_preview_box_info(success_message, "success")
        info_ms = (time.perf_counter() - phase_start) * 1000.0
    total_ms = (time.perf_counter() - perf_start) * 1000.0
    if total_ms >= 120.0:
        logger.info(
            "[Z3/PZ2 PERF] persist_preview_metadata total=%.1fms refresh_list=%s sync_access=%s phases=[write=%.1fms, refresh=%.1fms, sync=%.1fms, info=%.1fms]",
            total_ms,
            bool(refresh_list),
            bool(sync_access),
            write_ms,
            refresh_ms,
            sync_ms,
            info_ms,
        )
    try:
        self._log_preview_edit_flow(
            "metadata_saved",
            refresh_list=int(bool(refresh_list)),
            sync_access=int(bool(sync_access)),
            write_ms=round(write_ms, 1),
            sync_ms=round(sync_ms, 1),
        )
    except Exception:
        pass

def _flush_scheduled_preview_metadata_save(self):
    after_id = getattr(self, "_preview_metadata_save_after_id", None)
    if after_id:
        try:
            self.frame.after_cancel(after_id)
        except Exception:
            pass
        self._preview_metadata_save_after_id = None

    try:
        if (after_id or getattr(self, "_preview_pending_save_pids", None)) and self.preview_metadata:
            _start_preview_metadata_autosave(self)
        writer = getattr(self, "_preview_autosave_writer", None)
        if writer is not None:
            mtime = writer.flush()
            _cancel_preview_metadata_write_poll(self)
            if path_key(writer.path) == path_key(self._get_preview_metadata_path()) and mtime is not None:
                self._loaded_meta_path = writer.path
                self._loaded_meta_mtime = mtime
    except Exception as exc:
        logger.error(f"Nie udało się zapisać odłożonych zmian metadata preview: {exc}")

def _cancel_scheduled_preview_metadata_save(self):
    after_id = getattr(self, "_preview_metadata_save_after_id", None)
    if after_id:
        try:
            self.frame.after_cancel(after_id)
        except Exception:
            pass
    self._preview_metadata_save_after_id = None

def _schedule_preview_metadata_save(self, delay_ms: int = 450):
    mark_preview_metadata_changed(self)
    dirty_ids = getattr(self, "_preview_pending_save_pids", None)
    if not isinstance(dirty_ids, set):
        dirty_ids = self._preview_pending_save_pids = set()
    pid = str(getattr(self, "_preview_active_pid", "") or "")
    if pid:
        dirty_ids.add(pid)
    previous_after_id = getattr(self, "_preview_metadata_save_after_id", None)
    if previous_after_id:
        try:
            self.frame.after_cancel(previous_after_id)
        except Exception:
            pass
        self._preview_metadata_save_after_id = None
    self._preview_metadata_save_defer_logged = False
    try:
        self._log_preview_edit_flow("metadata_save_scheduled", delay_ms=max(1, int(delay_ms)))
    except Exception:
        pass

    def _save_later():
        now = time.monotonic()
        try:
            last_edit_interaction = float(getattr(self, "_preview_last_char_edit_interaction_ts", 0.0) or 0.0)
        except Exception:
            last_edit_interaction = 0.0
        recent_char_edit = bool(last_edit_interaction > 0.0 and (now - last_edit_interaction) < 2.2)
        try:
            last_preview_navigation = float(getattr(self, "_preview_last_navigation_interaction_ts", 0.0) or 0.0)
        except Exception:
            last_preview_navigation = 0.0
        recent_preview_navigation = bool(last_preview_navigation > 0.0 and (now - last_preview_navigation) < 1.5)
        hot_char_target = bool(
            getattr(self, "_preview_char_hover_grip", None) is not None
            or getattr(self, "_preview_char_hover_index", None) is not None
        )
        if (
            getattr(self, "_preview_char_drag_state", None) is not None
            or getattr(self, "_preview_char_add_state", None) is not None
            or getattr(self, "_preview_layout_separator_drag_state", None) is not None
            or getattr(self, "_preview_badge_drag_state", None) is not None
            or recent_char_edit
            or recent_preview_navigation
        ):
            if not bool(getattr(self, "_preview_metadata_save_defer_logged", False)):
                self._preview_metadata_save_defer_logged = True
                try:
                    self._log_preview_edit_flow(
                        "metadata_save_deferred",
                        recent_edit=int(bool(recent_char_edit)),
                        recent_nav=int(bool(recent_preview_navigation)),
                        hot_target=int(bool(hot_char_target)),
                        edit_mode=int(bool(getattr(self, "_preview_char_edit_mode", False))),
                    )
                except Exception:
                    pass
            try:
                self._preview_metadata_save_after_id = self.frame.after(650, _save_later)
                return
            except Exception:
                pass
        self._preview_metadata_save_after_id = None
        self._preview_metadata_save_defer_logged = False
        try:
            _start_preview_metadata_autosave(self)
            try:
                self._schedule_preview_info_refresh(delay_ms=900)
            except Exception:
                pass
        except Exception as exc:
            logger.debug(f"Nie udało się zapisać odłożonego układu tablicy: {exc}")

    try:
        self._preview_metadata_save_after_id = self.frame.after(max(1, int(delay_ms)), _save_later)
    except Exception:
        _save_later()

def _cancel_preview_metadata_write_poll(self):
    job = getattr(self, "_preview_metadata_write_poll_after_id", None)
    if job:
        self.frame.after_cancel(job)
    self._preview_metadata_write_poll_after_id = None


def _start_preview_metadata_autosave(self):
    # The directory variable may already point at the next run. Pending edits
    # belong to the dataset actually loaded in memory.
    meta_path = getattr(self, "_loaded_meta_path", None) or self._get_preview_metadata_path()
    if meta_path is None:
        raise RuntimeError("Brak aktywnego preview runu do zapisania.")
    writer = getattr(self, "_preview_autosave_writer", None)
    if writer is None or writer.path != meta_path or writer.metadata is not self.preview_metadata:
        if writer is not None:
            writer.close()
        cache = getattr(self, "_preview_metadata_file_cache", None)
        if (isinstance(cache, tuple) and len(cache) >= 3 and cache[1] is self.preview_metadata
                and cache[0][0] == path_key(meta_path)):
            fragments = cache[2]
        else:
            # Results created by inference/import have no source fragments yet.
            fragments = {pid: json.dumps(data, ensure_ascii=False, separators=(",", ":"))
                         for pid, data in self.preview_metadata.items()}
        writer = self._preview_autosave_writer = PreviewMetadataAutosave(meta_path, self.preview_metadata, fragments)
    dirty_ids = set(getattr(self, "_preview_pending_save_pids", set()) or ())
    pid = str(getattr(self, "_preview_active_pid", "") or "")
    if pid:
        dirty_ids.add(pid)
    future = writer.submit(dirty_ids)
    self._preview_pending_save_pids = set()
    _cancel_preview_metadata_write_poll(self)

    def completed():
        self._preview_metadata_write_poll_after_id = None
        if writer is not getattr(self, "_preview_autosave_writer", None) or future is not writer.future:
            return
        if not future.done():
            self._preview_metadata_write_poll_after_id = self.frame.after(80, completed)
            return
        try:
            mtime = future.result()
            if path_key(meta_path) == path_key(self._get_preview_metadata_path()):
                self._loaded_meta_path = meta_path
                self._loaded_meta_mtime = mtime
            self._log_preview_edit_flow("metadata_saved", background=1, edited_plates=len(dirty_ids))
        except Exception as exc:
            self._preview_pending_save_pids.update(dirty_ids)
            logger.error("Nie udało się zapisać metadata PZ2: %s", exc)
            self._update_preview_edit_status(f"Nie zapisano zmian: {exc}", tone="error")
    self._preview_metadata_write_poll_after_id = self.frame.after(80, completed)


def _schedule_preview_info_refresh(self, delay_ms: int = 180):
    previous_after_id = getattr(self, "_preview_info_refresh_after_id", None)
    if previous_after_id:
        try:
            self.frame.after_cancel(previous_after_id)
        except Exception:
            pass
        self._preview_info_refresh_after_id = None

    def _refresh_later():
        now = time.monotonic()
        try:
            last_edit_interaction = float(getattr(self, "_preview_last_char_edit_interaction_ts", 0.0) or 0.0)
        except Exception:
            last_edit_interaction = 0.0
        recent_char_edit = bool(last_edit_interaction > 0.0 and (now - last_edit_interaction) < 2.2)
        try:
            last_preview_navigation = float(getattr(self, "_preview_last_navigation_interaction_ts", 0.0) or 0.0)
        except Exception:
            last_preview_navigation = 0.0
        recent_preview_navigation = bool(last_preview_navigation > 0.0 and (now - last_preview_navigation) < 1.5)
        hot_char_target = bool(
            getattr(self, "_preview_char_hover_grip", None) is not None
            or getattr(self, "_preview_char_hover_index", None) is not None
        )
        edit_session_recent = bool(last_edit_interaction > 0.0 and (now - last_edit_interaction) < 6.0)
        edit_session_active = bool(
            edit_session_recent
            and (
                getattr(self, "_preview_char_edit_mode", False)
                or getattr(self, "_preview_char_label_mode", False)
                or getattr(self, "_preview_char_label_active_index", None) is not None
                or getattr(self, "_preview_char_selected_index", None) is not None
            )
        )
        if (
            getattr(self, "_preview_char_drag_state", None) is not None
            or getattr(self, "_preview_char_add_state", None) is not None
            or getattr(self, "_preview_layout_separator_drag_state", None) is not None
            or getattr(self, "_preview_badge_drag_state", None) is not None
            or recent_char_edit
            or recent_preview_navigation
            or edit_session_active
        ):
            try:
                self._log_preview_edit_flow(
                    "preview_info_refresh_deferred",
                    recent_edit=int(bool(recent_char_edit)),
                    recent_nav=int(bool(recent_preview_navigation)),
                    hot_target=int(bool(hot_char_target)),
                    edit_mode=int(bool(getattr(self, "_preview_char_edit_mode", False))),
                    edit_session=int(bool(edit_session_active)),
                )
            except Exception:
                pass
            try:
                self._preview_info_refresh_after_id = self.frame.after(750, _refresh_later)
                return
            except Exception:
                pass

        self._preview_info_refresh_after_id = None
        perf_start = time.perf_counter()
        update_ms = sync_ms = 0.0
        try:
            phase_start = time.perf_counter()
            self._update_preview_info_label()
            update_ms = (time.perf_counter() - phase_start) * 1000.0
            phase_start = time.perf_counter()
            self._sync_step3_access_from_preview_state(self.preview_metadata)
            sync_ms = (time.perf_counter() - phase_start) * 1000.0
            total_ms = (time.perf_counter() - perf_start) * 1000.0
            if total_ms >= 120.0:
                logger.info(
                    "[Z3/PZ2 PERF] preview_info_refresh total=%.1fms phases=[update=%.1fms, sync=%.1fms]",
                    total_ms,
                    update_ms,
                    sync_ms,
                )
        except Exception as exc:
            logger.debug(f"Nie udalo sie odswiezyc licznikow PZ2 po edycji: {exc}")

    try:
        self._preview_info_refresh_after_id = self.frame.after(max(1, int(delay_ms)), _refresh_later)
    except Exception:
        _refresh_later()

def _clone_preview_plate_data(self, plate_id: str | None = None):
    pid = str(plate_id or getattr(self, "_preview_active_pid", "") or "").strip()
    if not pid:
        return None
    data = self.preview_metadata.get(pid)
    return copy.deepcopy(data if isinstance(data, dict) else {})

def _get_preview_history_stack(self, kind: str, plate_id: str | None = None, create: bool = False):
    pid = str(plate_id or getattr(self, "_preview_active_pid", "") or "").strip()
    if not pid:
        return None
    store_attr = "_preview_history_undo" if str(kind).lower() == "undo" else "_preview_history_redo"
    store = getattr(self, store_attr, None)
    if not isinstance(store, dict):
        store = {}
        setattr(self, store_attr, store)
    if create:
        return store.setdefault(pid, [])
    return store.get(pid)

def _push_preview_history_snapshot(self, plate_id: str | None = None):
    if bool(getattr(self, "_preview_history_replaying", False)):
        return
    pid = str(plate_id or getattr(self, "_preview_active_pid", "") or "").strip()
    if not pid:
        return
    snapshot = self._clone_preview_plate_data(pid)
    if snapshot is None:
        return

    undo_stack = self._get_preview_history_stack("undo", pid, create=True)
    if isinstance(undo_stack, list) and undo_stack and undo_stack[-1] == snapshot:
        return

    undo_stack.append(snapshot)
    limit = max(8, int(getattr(self, "_preview_history_limit", 30) or 30))
    if len(undo_stack) > limit:
        del undo_stack[:-limit]

    redo_stack = self._get_preview_history_stack("redo", pid, create=True)
    if isinstance(redo_stack, list):
        redo_stack.clear()

def _trim_preview_history_stack(self, stack) -> None:
    if not isinstance(stack, list):
        return
    limit = max(8, int(getattr(self, "_preview_history_limit", 30) or 30))
    if len(stack) > limit:
        del stack[:-limit]

def _refresh_preview_listbox_row(self, plate_id: str | None = None):
    pid = str(plate_id or getattr(self, "_preview_active_pid", "") or "").strip()
    listbox = getattr(self, "plates_listbox", None)
    if not pid or listbox is None:
        return
    pid_map = getattr(self, "_listbox_pid_by_index", [])
    try:
        row_index = pid_map.index(pid)
    except Exception:
        return

    data = self.preview_metadata.get(pid, {})
    label = self._format_plate_listbox_label(pid, data)
    status = str(data.get("status", "unknown")).strip().lower()

    try:
        selected_rows = {int(idx) for idx in listbox.curselection()}
    except Exception:
        selected_rows = set()
    try:
        active_row = int(listbox.index(tk.ACTIVE))
    except Exception:
        active_row = None
    row_selected = row_index in selected_rows

    try:
        listbox.delete(row_index)
        listbox.insert(row_index, label)
        self._apply_plate_listbox_row_style(row_index, status)
        if row_selected:
            listbox.selection_set(row_index)
        if active_row == row_index:
            listbox.activate(row_index)
        if row_selected or active_row == row_index:
            listbox.see(row_index)
    except Exception as exc:
        logger.debug(f"Nie udało się odświeżyć pojedynczego wiersza listy tablic [{pid}]: {exc}")

def _restore_preview_plate_history_snapshot(self, snapshot, *, action_label: str):
    pid = str(getattr(self, "_preview_active_pid", "") or "").strip()
    if not pid:
        return False

    self._preview_history_replaying = True
    try:
        self.preview_metadata[pid] = copy.deepcopy(snapshot if isinstance(snapshot, dict) else {})
        self._preview_char_selected_index = None
        self._preview_char_drag_state = None
        self._preview_char_add_state = None
        self._preview_char_add_click_armed = False
        self._preview_char_hover_index = None
        self._preview_char_hover_label_index = None
        self._preview_char_label_active_index = None
        self._refresh_preview_live_metadata_ui(
            status_message=action_label,
            status_tone="info",
            render_preview=False,
            refresh_row=True,
        )
        if not self._redraw_preview_character_overlays_light():
            self._on_preview_select(None)
        self._persist_preview_metadata(success_message=None, refresh_list=False, sync_access=False)
        try:
            self._schedule_preview_info_refresh(delay_ms=900)
        except Exception:
            pass
        return True
    finally:
        self._preview_history_replaying = False

def _undo_preview_edit(self, event=None):
    pid = str(getattr(self, "_preview_active_pid", "") or "").strip()
    undo_stack = self._get_preview_history_stack("undo", pid, create=False)
    if not pid or not isinstance(undo_stack, list) or not undo_stack:
        self._update_preview_edit_status("Brak zmian do cofnięcia.", tone="warning")
        return "break"

    current_snapshot = self._clone_preview_plate_data(pid)
    redo_stack = self._get_preview_history_stack("redo", pid, create=True)
    if current_snapshot is not None:
        redo_stack.append(current_snapshot)
        self._trim_preview_history_stack(redo_stack)

    target_snapshot = undo_stack.pop()
    self._restore_preview_plate_history_snapshot(target_snapshot, action_label="Cofnięto ostatnią zmianę boxów znaków.")
    return "break"

def _redo_preview_edit(self, event=None):
    pid = str(getattr(self, "_preview_active_pid", "") or "").strip()
    redo_stack = self._get_preview_history_stack("redo", pid, create=False)
    if not pid or not isinstance(redo_stack, list) or not redo_stack:
        self._update_preview_edit_status("Brak zmian do ponowienia.", tone="warning")
        return "break"

    current_snapshot = self._clone_preview_plate_data(pid)
    undo_stack = self._get_preview_history_stack("undo", pid, create=True)
    if current_snapshot is not None:
        undo_stack.append(current_snapshot)
        self._trim_preview_history_stack(undo_stack)

    target_snapshot = redo_stack.pop()
    self._restore_preview_plate_history_snapshot(target_snapshot, action_label="Przywrócono ostatnią cofniętą zmianę boxów znaków.")
    return "break"

def _event_has_control_modifier(event=None) -> bool:
    if event is None:
        return False
    try:
        return bool(int(getattr(event, "state", 0) or 0) & 0x4)
    except Exception:
        return False

def _event_has_shift_modifier(event=None) -> bool:
    if event is None:
        return False
    try:
        return bool(int(getattr(event, "state", 0) or 0) & 0x1)
    except Exception:
        return False

def _pane_has_child(pane, child) -> bool:
    if pane is None or child is None:
        return False
    try:
        return str(child) in {str(item) for item in pane.panes()}
    except Exception:
        return False

def _get_current_preview_list_index(self):
    listbox = getattr(self, "plates_listbox", None)
    if listbox is None:
        return None
    try:
        sel = listbox.curselection()
        if sel:
            idx = int(sel[0])
            if 0 <= idx < len(getattr(self, "_listbox_pid_by_index", [])):
                return idx
    except Exception:
        pass
    try:
        idx = int(listbox.index(tk.ACTIVE))
        if 0 <= idx < len(getattr(self, "_listbox_pid_by_index", [])):
            return idx
    except Exception:
        pass
    return None

def _clear_listbox_selection_fast(listbox) -> None:
    if listbox is None:
        return
    try:
        selected_indices = list(listbox.curselection() or ())
    except Exception:
        selected_indices = []
    for selected_index in selected_indices:
        try:
            listbox.selection_clear(selected_index)
        except Exception:
            pass

def _see_listbox_index_if_needed(listbox, index: int) -> None:
    if listbox is None:
        return
    try:
        target = int(index)
        top = int(listbox.nearest(0))
        bottom = int(listbox.nearest(max(0, int(listbox.winfo_height() or 1) - 1)))
        if top <= target <= bottom:
            return
    except Exception:
        pass
    try:
        listbox.see(int(index))
    except Exception:
        pass

def _handle_preview_list_arrow_nav(self, offset: int):
    try:
        self.plates_listbox.focus_set()
    except Exception:
        pass
    self._select_preview_relative(int(offset))
    return "break"

def _select_preview_relative(self, offset: int):
    listbox = getattr(self, "plates_listbox", None)
    pid_map = getattr(self, "_listbox_pid_by_index", [])
    if listbox is None or not pid_map:
        return False

    current_idx = self._get_current_preview_list_index()
    if current_idx is None:
        current_idx = 0
    target_idx = max(0, min(len(pid_map) - 1, int(current_idx) + int(offset)))
    if target_idx == current_idx and current_idx is not None:
        return False

    try:
        self._preview_last_navigation_interaction_ts = time.monotonic()
        self._suppress_preview_reload_on_list_select = True
        self._preview_fast_select_render = True
        self._clear_listbox_selection_fast(listbox)
        listbox.selection_set(target_idx)
        listbox.activate(target_idx)
        _see_listbox_index_if_needed(listbox, target_idx)
    except Exception:
        return False

    self._preview_char_label_active_index = None
    self._preview_char_hover_label_index = None
    self._preview_char_hover_index = None
    try:
        scheduler = getattr(self, "_schedule_preview_select_render", None)
        if callable(scheduler):
            delay_ms = 1 if bool(getattr(self, "_preview_keyboard_crop_navigation_active", False)) else 18
            scheduler(delay_ms=delay_ms)
        else:
            self._on_preview_select(None)
    except Exception:
        self._on_preview_select(None)
    return True
