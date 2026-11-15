# Optional inline GT inspector for Z2 plate polygons.

from __future__ import annotations

import tkinter as tk
from ..pose_geometry import geometry_warning

from ..campaign_manager import CAMPAIGN
from ..plate_ground_truth import (
    GROUND_TRUTH_SOURCE_MANUAL_Z2,
    PLATE_LAYOUT_GT_ATTR,
    PLATE_LAYOUT_SINGLE_ROW,
    PLATE_LAYOUT_TWO_ROW,
    ensure_plate_layout_gt,
    get_plate_layout_gt,
    normalize_plate_layout_gt,
    set_plate_layout_gt,
    get_plate_ground_truth,
    normalize_plate_ground_truth_text,
    set_plate_ground_truth,
)
from . import z2_gt_pack_runtime
from .web_slim_scrollbar import blend_hex_colors


INLINE_SAVE_DELAY_MS = 2200
INLINE_GATE_REFRESH_DELAY_MS = 180
CARD_WIDTH = 176
CARD_HEIGHT = 40
CARD_GAP = 7
TOGGLE_WIDTH = 42
TOGGLE_HEIGHT = 24


def canvas_viewport_bounds(canvas) -> tuple[float, float, float, float]:
    """Return visible viewport bounds expressed in canvas coordinates."""
    try:
        width = max(1.0, float(canvas.winfo_width() or 1.0))
        height = max(1.0, float(canvas.winfo_height() or 1.0))
    except Exception:
        width = 1.0
        height = 1.0

    try:
        left = float(canvas.canvasx(0.0))
        top = float(canvas.canvasy(0.0))
        right = float(canvas.canvasx(width))
        bottom = float(canvas.canvasy(height))
    except Exception:
        left = 0.0
        top = 0.0
        right = width
        bottom = height

    if right < left:
        left, right = right, left
    if bottom < top:
        top, bottom = bottom, top

    return left, top, right, bottom


def uppercase_registration_input(value) -> str:
    return str(value or "").upper()


def format_plate_info(host, det, ann) -> str:
    try:
        confidence = float(getattr(det, "confidence", 0.0) or 0.0)
    except Exception:
        confidence = 0.0
    if geometry_warning(det):
        return f"DET {confidence:.2f}   ! geometria"
    try:
        fit_score = host._get_plate_detection_fit_score(det, ann)
    except Exception:
        fit_score = None
    if fit_score is None:
        return f"DET {confidence:.2f}   FIT —"
    try:
        return f"DET {confidence:.2f}   FIT {float(fit_score):.2f}"
    except Exception:
        return f"DET {confidence:.2f}   FIT —"


def _layout_gt_label(layout) -> str:
    normalized = normalize_plate_layout_gt(layout)
    return "2R" if normalized == PLATE_LAYOUT_TWO_ROW else "1R"


def _update_record_layout_ui(host, record: dict) -> None:
    if not isinstance(record, dict):
        return
    button = record.get("layout_button")
    layout_var = record.get("layout_var")
    if button is None:
        return

    preview = str(record.get("layout_preview") or "").strip()
    det = record.get("det")
    layout = (
        normalize_plate_layout_gt(preview)
        if preview
        else get_plate_layout_gt(
            getattr(det, "attributes", None)
        )
    )
    label = _layout_gt_label(layout)

    try:
        if layout_var is not None:
            layout_var.set(label)
    except Exception:
        pass

    palette = _palette(host)
    panel_bg = palette.get("panel", "#252526")
    color = (
        palette.get("warning", "#f0a020")
        if layout == PLATE_LAYOUT_TWO_ROW
        else palette.get("success", "#2ecc71")
    )
    fill = blend_hex_colors(panel_bg, color, 0.22)

    try:
        button.configure(
            bg=fill,
            fg=color,
            activebackground=blend_hex_colors(fill, color, 0.16),
            activeforeground=color,
            highlightbackground=color,
            highlightcolor=color,
        )
    except Exception:
        pass


def save_inline_plate_layout_gt_value(
    host,
    ann,
    det,
    requested_layout,
):
    previous_attributes = dict(getattr(det, "attributes", {}) or {})
    det.attributes = dict(previous_attributes)

    normalized = set_plate_layout_gt(
        det.attributes,
        requested_layout,
    )
    if det.attributes == previous_attributes:
        return True, normalized

    try:
        host._mark_preview_image_dirty(ann, refresh_list=False)
    except TypeError:
        try:
            host._mark_preview_image_dirty(ann)
        except Exception:
            pass
    except Exception:
        pass

    save_kwargs = dict(
        interactive=False,
        status_message="",
        refresh_list=False,
        refresh_workflow=False,
        refresh_export_sources=False,
    )
    try:
        saved = bool(host._save_preview_edits(**save_kwargs))
    except TypeError:
        try:
            saved = bool(
                host._save_preview_edits(
                    interactive=False,
                    status_message="",
                )
            )
        except Exception:
            saved = False
    except Exception:
        saved = False

    if not saved:
        det.attributes = previous_attributes
        return False, get_plate_layout_gt(previous_attributes)

    try:
        z2_gt_pack_runtime.sync_plate_layout_after_xml_save(
            host,
            ann,
            det,
            normalized,
            retry_pending=True,
        )
    except Exception:
        # XML remains authoritative locally; GT Pack sync has its own outbox.
        pass

    return True, normalized


def _toggle_record_layout(host, key):
    record = _editor_store(host).get(key)
    if not isinstance(record, dict):
        return "break"

    try:
        if not bool(host._preview_is_editable()):
            return "break"
    except Exception:
        pass

    ann = record.get("ann")
    det = record.get("det")
    if ann is None or det is None:
        return "break"

    current = get_plate_layout_gt(
        getattr(det, "attributes", None)
    )
    target = (
        PLATE_LAYOUT_SINGLE_ROW
        if current == PLATE_LAYOUT_TWO_ROW
        else PLATE_LAYOUT_TWO_ROW
    )

    # Instant visual feedback before the XML write.
    record["layout_preview"] = target
    _update_record_layout_ui(host, record)
    try:
        host.frame.update_idletasks()
    except Exception:
        pass

    ok, normalized = save_inline_plate_layout_gt_value(
        host,
        ann,
        det,
        target,
    )

    record["layout_preview"] = ""
    _update_record_layout_ui(host, record)

    message = (
        f"GT układu tablicy: {_layout_gt_label(normalized)}."
        if ok
        else "Nie udało się zapisać GT układu tablicy."
    )
    try:
        host._update_preview_edit_status(
            message,
            refresh_toolbar=False,
            refresh_debug=False,
        )
    except TypeError:
        try:
            host._update_preview_edit_status(message)
        except Exception:
            pass
    except Exception:
        pass

    return "break"



def count_plate_gt(plate_detections) -> int:
    count = 0
    for det in list(plate_detections or []):
        if get_plate_ground_truth(getattr(det, "attributes", None)):
            count += 1
    return int(count)


def gt_required_for_current_route(host) -> bool:
    from .z2_gt_readiness import gt_required_for_current_route as required
    return required(host, campaign=CAMPAIGN)


def gt_mode_enabled(host) -> bool:
    # Compatibility query only: inline GT is always available.
    return True


def set_gt_mode_enabled(host, enabled: bool) -> bool:
    # Compatibility shim: GT cannot be switched off.
    return True


def toggle_gt_mode(host) -> bool:
    # Compatibility shim: there is no GT ON/OFF mode.
    return True


def _editor_store(host) -> dict:
    store = getattr(host, "_plate_gt_inline_widgets", None)
    if not isinstance(store, dict):
        store = {}
        host._plate_gt_inline_widgets = store
    return store


def suspend_inline_plate_gt_editors(host) -> None:
    # Real Tk widgets are expensive to place/lift/restyle every animation frame.
    if bool(getattr(host, "_plate_gt_inline_interaction_suspended", False)):
        return

    host._plate_gt_inline_interaction_suspended = True
    for record in list(_editor_store(host).values()):
        if not isinstance(record, dict):
            continue
        shell = record.get("shell")
        if shell is None:
            continue
        try:
            shell.place_forget()
        except Exception:
            pass


def _after_store(host) -> dict:
    store = getattr(host, "_plate_gt_inline_save_after", None)
    if not isinstance(store, dict):
        store = {}
        host._plate_gt_inline_save_after = store
    return store


def _offset_store(host) -> dict:
    store = getattr(host, "_plate_gt_inline_offsets", None)
    if not isinstance(store, dict):
        store = {}
        host._plate_gt_inline_offsets = store
    return store


def _cancel_pending_save(host, key) -> None:
    pending = _after_store(host).pop(key, None)
    if not pending:
        return
    try:
        host.frame.after_cancel(pending)
    except Exception:
        pass


def _schedule_inline_gt_gate_refresh(
    host,
    *,
    delay_ms: int = INLINE_GATE_REFRESH_DELAY_MS,
) -> None:
    pending = getattr(host, "_plate_gt_inline_gate_after", None)
    if pending:
        try:
            host.frame.after_cancel(pending)
        except Exception:
            pass
    host._plate_gt_inline_gate_after = None

    def _refresh() -> None:
        host._plate_gt_inline_gate_after = None
        try:
            quiet = host._preview_user_interaction_quiet_remaining_ms(padding_ms=250)
            if quiet > 0 or getattr(host, "_preview_drag_state", None):
                _schedule_inline_gt_gate_refresh(host, delay_ms=max(250, int(quiet)))
                return
        except (AttributeError, TypeError, ValueError):
            pass
        try:
            host._z2_graph_right_panel_render_signature = None
        except Exception:
            pass
        try:
            host._preview_overlay_dock_inline_gate_state = None
            host._preview_overlay_dock_gate_render_key = None
            host._preview_overlay_dock_render_key = None
        except Exception:
            pass
        try:
            host._refresh_step2_action_states(lightweight=True)
        except Exception:
            pass
        try:
            host._place_preview_overlay_dock(force_render=False)
        except Exception:
            pass

    try:
        host._plate_gt_inline_gate_after = host.frame.after(
            max(0, int(delay_ms)),
            _refresh,
        )
    except Exception:
        host._plate_gt_inline_gate_after = None
        _refresh()


def save_inline_plate_gt_value(
    host,
    ann,
    det,
    requested_text,
    *,
    refresh_gate: bool = True,
    lightweight_save: bool = False,
    retry_pack_pending: bool = True,
):
    import time

    save_started_at = time.perf_counter()
    xml_elapsed_ms = 0.0
    pack_elapsed_ms = 0.0
    gate_elapsed_ms = 0.0

    previous_attributes = dict(getattr(det, "attributes", {}) or {})
    det.attributes = dict(previous_attributes)
    ensure_plate_layout_gt(det.attributes)

    normalized = set_plate_ground_truth(
        det.attributes,
        requested_text,
        source=GROUND_TRUTH_SOURCE_MANUAL_Z2,
    )
    changed = det.attributes != previous_attributes

    if changed:
        try:
            host._mark_preview_image_dirty(ann, refresh_list=False)
        except TypeError:
            host._mark_preview_image_dirty(ann)
        except Exception:
            pass

        xml_started_at = time.perf_counter()
        try:
            save_kwargs = dict(
                interactive=False,
                status_message="Zapisano GT.",
            )
            if bool(lightweight_save):
                save_kwargs.update(
                    refresh_list=False,
                    refresh_workflow=False,
                    refresh_export_sources=False,
                )
            saved = bool(host._save_preview_edits(**save_kwargs))
        except Exception:
            saved = False
        xml_elapsed_ms = max(
            0.0,
            (time.perf_counter() - xml_started_at) * 1000.0,
        )

        if not saved:
            det.attributes = previous_attributes
            from .z2_gt_review import refresh_annotation_gt_review
            refresh_annotation_gt_review(host, ann, revoke=False)
            try:
                host._update_preview_edit_status(
                    "Nie udało się automatycznie zapisać GT. "
                    "Przywrócono poprzednią wartość."
                )
            except Exception:
                pass
            return False, get_plate_ground_truth(previous_attributes)

        from .z2_gt_review import refresh_annotation_gt_review
        refresh_annotation_gt_review(host, ann)
        pack_started_at = time.perf_counter()
        try:
            pack_sync = z2_gt_pack_runtime.sync_plate_gt_after_xml_save(
                host,
                ann,
                det,
                normalized,
                retry_pending=bool(retry_pack_pending),
            )
        except Exception as exc:
            pack_sync = {
                "ok": False,
                "enabled": True,
                "queued": True,
                "error": str(exc),
            }
        pack_elapsed_ms = max(
            0.0,
            (time.perf_counter() - pack_started_at) * 1000.0,
        )

        if (
            isinstance(pack_sync, dict)
            and pack_sync.get("enabled")
            and not pack_sync.get("ok")
        ):
            try:
                host._update_preview_edit_status(
                    "GT zapisano w annotations.xml; "
                    "synchronizacja z GT Pack oczekuje na ponowienie."
                )
            except Exception:
                pass

    if refresh_gate:
        gate_started_at = time.perf_counter()
        _schedule_inline_gt_gate_refresh(host)
        gate_elapsed_ms = max(
            0.0,
            (time.perf_counter() - gate_started_at) * 1000.0,
        )

    total_elapsed_ms = max(
        0.0,
        (time.perf_counter() - save_started_at) * 1000.0,
    )
    if changed and total_elapsed_ms >= 80.0:
        try:
            from ..config import logger
            logger.info(
                "[Z2 PERF] inline_gt_save total=%.0fms xml=%.0fms "
                "pack=%.0fms gate=%.0fms lightweight=%s retry_pending=%s",
                total_elapsed_ms,
                xml_elapsed_ms,
                pack_elapsed_ms,
                gate_elapsed_ms,
                int(bool(lightweight_save)),
                int(bool(retry_pack_pending)),
            )
        except Exception:
            pass

    return True, normalized


def queue_inline_plate_gt_value(host, ann, det, requested_text):
    """Accept an edit immediately; persist XML and its GT pack in one worker."""
    from .z2_preview_autosave import queue_gt_sync, start_preview_autosave
    from .z2_gt_review import refresh_annotation_gt_review

    previous = dict(getattr(det, "attributes", {}) or {})
    xml_path = host._get_current_annotation_xml_path()
    if xml_path is None:
        host._update_preview_edit_status("Nie można zapisać GT: brak pliku anotacji.")
        return False, get_plate_ground_truth(previous)
    det.attributes = dict(previous)
    ensure_plate_layout_gt(det.attributes)
    normalized = set_plate_ground_truth(det.attributes, requested_text, source=GROUND_TRUTH_SOURCE_MANUAL_Z2)
    try:
        item = z2_gt_pack_runtime.prepare_plate_gt_sync_item(host, ann, det, normalized)
    except Exception as exc:
        det.attributes = previous
        host._update_preview_edit_status(f"Nie można przygotować zapisu GT: {exc}")
        return False, get_plate_ground_truth(previous)
    host._mark_preview_image_dirty(ann, refresh_list=False, refresh_row=False)
    queue_gt_sync(host, xml_path, item)
    refresh_annotation_gt_review(host, ann)
    if not start_preview_autosave(host, status_message="Zapisano GT."):
        host._schedule_preview_autosave()
    return True, normalized


def _key_for(det) -> str:
    return f"plate:{id(det)}"


def _focused(host, record: dict) -> bool:
    entry = record.get("entry")
    if entry is None:
        return False
    try:
        return host.frame.focus_get() is entry
    except Exception:
        return False


def _palette(host) -> dict:
    try:
        return dict(getattr(getattr(host, "app", None), "palette", {}) or {})
    except Exception:
        return {}


def _apply_record_style(host, record: dict, *, editing: bool | None = None) -> None:
    if not isinstance(record, dict):
        return

    if editing is not None:
        record["editing"] = bool(editing)

    palette = _palette(host)
    style_signature = (
        bool(record.get("editing", False)), bool(record.get("confirmed", False)),
        tuple(sorted(palette.items())),
        get_plate_layout_gt(getattr(record.get("det"), "attributes", None)),
    )
    if record.get("applied_style_signature") == style_signature:
        return
    canvas_bg = palette.get("canvas_bg", palette.get("panel", "#20252b"))
    panel_bg = palette.get("panel", "#252526")
    field_bg = palette.get("field", "#171717")
    muted = palette.get("muted", "#aeb7bf")
    border = palette.get("panel_border", "#555d65")
    accent = palette.get("accent", "#4f8de3")
    edit_color = palette.get("warning", accent)
    confirmed_color = palette.get("info", "#38bdf8")

    neutral_card_bg = blend_hex_colors(panel_bg, canvas_bg, 0.58)
    neutral_top_bg = blend_hex_colors(field_bg, canvas_bg, 0.44)
    neutral_entry_bg = blend_hex_colors(field_bg, neutral_card_bg, 0.18)
    neutral_border = blend_hex_colors(border, canvas_bg, 0.38)

    active = bool(record.get("editing", False))
    confirmed = bool(record.get("confirmed", False))
    if active:
        card_bg = blend_hex_colors(neutral_card_bg, edit_color, 0.24)
        top_bg = blend_hex_colors(neutral_top_bg, edit_color, 0.34)
        entry_bg = blend_hex_colors(neutral_entry_bg, edit_color, 0.18)
        outline = edit_color
        gt_fg = edit_color
        info_fg = palette.get("fg", "#f3f3f3")
        border_width = 2
    elif confirmed:
        card_bg = blend_hex_colors(neutral_card_bg, confirmed_color, 0.22)
        top_bg = blend_hex_colors(neutral_top_bg, confirmed_color, 0.30)
        entry_bg = blend_hex_colors(neutral_entry_bg, confirmed_color, 0.14)
        outline = confirmed_color
        gt_fg = confirmed_color
        info_fg = palette.get("fg", "#f3f3f3")
        border_width = 2
    else:
        card_bg = neutral_card_bg
        top_bg = neutral_top_bg
        entry_bg = neutral_entry_bg
        outline = neutral_border
        gt_fg = muted
        info_fg = muted
        border_width = 1

    try:
        record["shell"].configure(
            bg=card_bg,
            highlightbackground=outline,
            highlightcolor=outline,
            highlightthickness=border_width,
        )
        record["info_row"].configure(bg=top_bg, fg=info_fg)
        record["gt_row"].configure(bg=card_bg)
        record["gt_label"].configure(bg=card_bg, fg=gt_fg)
        record["entry"].configure(
            bg=entry_bg,
            fg=palette.get("fg", "#f3f3f3"),
            insertbackground=palette.get("fg", "#f3f3f3"),
        )
    except Exception:
        pass

    _update_record_layout_ui(host, record)
    record["applied_style_signature"] = style_signature

def _force_uppercase(record: dict) -> str:
    var = record.get("var")
    if var is None:
        return ""
    try:
        raw = str(var.get() or "")
    except Exception:
        return ""

    prepared = uppercase_registration_input(raw)
    if prepared != raw:
        try:
            cursor = int(record["entry"].index(tk.INSERT))
        except Exception:
            cursor = None
        record["uppercase_sync"] = True
        try:
            var.set(prepared)
        finally:
            record["uppercase_sync"] = False
        if cursor is not None:
            try:
                record["entry"].icursor(min(cursor, len(prepared)))
            except Exception:
                pass
    return prepared


def _uppercase_trace(record: dict, *_args) -> None:
    if bool(record.get("uppercase_sync", False)):
        return
    _force_uppercase(record)


def _commit_record(host, key, *, final: bool = False):
    record = _editor_store(host).get(key)
    if not isinstance(record, dict):
        return "break"

    _cancel_pending_save(host, key)
    _force_uppercase(record)

    ann = record.get("ann")
    det = record.get("det")
    var = record.get("var")
    if ann is None or det is None or var is None:
        return "break"

    previous = get_plate_ground_truth(det.attributes)
    requested = normalize_plate_ground_truth_text(var.get() or "")
    if requested == previous:
        # Hiding/repositioning an unchanged card is not an edit. In superzoom
        # every next image used to schedule a full campaign gate recalculation.
        ok, normalized = True, previous
    else:
        ok, normalized = queue_inline_plate_gt_value(host, ann, det, requested)
        if ok:
            record["gate_refresh_pending"] = True

    record["confirmed"] = bool(normalized)

    if ok:
        record["last_saved"] = normalized
        if final:
            record["uppercase_sync"] = True
            try:
                var.set(normalized)
            finally:
                record["uppercase_sync"] = False
    else:
        record["uppercase_sync"] = True
        try:
            var.set(normalized)
        finally:
            record["uppercase_sync"] = False

    if final:
        _apply_record_style(host, record, editing=False)
        if record.pop("gate_refresh_pending", False):
            _schedule_inline_gt_gate_refresh(host)
    return "break"


def _schedule_save(host, key) -> None:
    _cancel_pending_save(host, key)
    try:
        after_id = host.frame.after(
            INLINE_SAVE_DELAY_MS,
            lambda: _commit_record(host, key, final=False),
        )
    except Exception:
        _commit_record(host, key, final=False)
        return
    _after_store(host)[key] = after_id


def flush_inline_plate_gt_editors(host):
    """Commit pending text fields before explicit save/export/shutdown."""
    if getattr(host, "_flushing_inline_gt_edits", False):
        return
    host._flushing_inline_gt_edits = True
    try:
        for key in list(_editor_store(host)):
            _commit_record(host, key, final=True)
    finally:
        host._flushing_inline_gt_edits = False


def _on_focus_in(host, key) -> None:
    record = _editor_store(host).get(key)
    if not isinstance(record, dict):
        return

    ann = record.get("ann")
    plate_idx = record.get("plate_idx")
    if ann is not None and plate_idx is not None:
        try:
            host._set_selected_plate_index_for_ann(ann, int(plate_idx))
        except Exception:
            pass

    _apply_record_style(host, record, editing=True)


def _on_focus_out(host, key) -> None:
    record = _editor_store(host).get(key)
    if not isinstance(record, dict) or bool(record.get("destroying")):
        return

    if bool(record.get("suppress_focus_out_once", False)):
        record["suppress_focus_out_once"] = False
        _apply_record_style(host, record, editing=False)
        return

    if bool(record.get("enter_commit_pending", False)):
        _apply_record_style(host, record, editing=False)
        return

    _commit_record(host, key, final=True)
    _apply_record_style(host, record, editing=False)


def _on_key_press(host, key, _event=None):
    try:
        host._mark_preview_user_interaction(quiet_ms=1400)
    except AttributeError:
        pass
    record = _editor_store(host).get(key)
    if isinstance(record, dict):
        _apply_record_style(host, record, editing=True)
    return None


def _on_key_release(host, key, event=None):
    record = _editor_store(host).get(key)
    if isinstance(record, dict):
        _force_uppercase(record)

    keysym = str(getattr(event, "keysym", "") or "")
    if keysym in {
        "Return", "KP_Enter", "Escape", "Tab",
        "Shift_L", "Shift_R", "Control_L", "Control_R", "Alt_L", "Alt_R",
    }:
        return None
    _schedule_save(host, key)
    return None


def _finish_with_enter(host, key):
    record = _editor_store(host).get(key)
    if not isinstance(record, dict):
        return "break"

    _cancel_pending_save(host, key)
    prepared = _force_uppercase(record)

    # Visual acknowledgement comes first. Persistence is intentionally moved
    # to the next Tk tick so Enter never waits for XML/GT Pack I/O before the
    # user sees that the registration was accepted.
    record["editing"] = False
    record["confirmed"] = bool(str(prepared or "").strip())
    record["enter_commit_pending"] = True
    enter_seq = int(record.get("enter_commit_seq", 0) or 0) + 1
    record["enter_commit_seq"] = enter_seq
    _apply_record_style(host, record, editing=False)

    def _persist_after_visual_ack() -> None:
        current = _editor_store(host).get(key)
        if not isinstance(current, dict):
            return
        if int(current.get("enter_commit_seq", 0) or 0) != enter_seq:
            return

        current["enter_commit_pending"] = False
        _commit_record(host, key, final=True)

        current = _editor_store(host).get(key)
        if isinstance(current, dict):
            current["suppress_focus_out_once"] = True
        try:
            host.preview_canvas.focus_set()
        except Exception:
            pass

    try:
        record["enter_commit_after"] = host.frame.after(
            12,
            _persist_after_visual_ack,
        )
    except Exception:
        record["enter_commit_pending"] = False
        _commit_record(host, key, final=True)
        record["suppress_focus_out_once"] = True
        try:
            host.preview_canvas.focus_set()
        except Exception:
            pass

    return "break"


def _on_escape(host, key):
    record = _editor_store(host).get(key)
    if not isinstance(record, dict):
        return "break"

    _cancel_pending_save(host, key)
    record["uppercase_sync"] = True
    try:
        record["var"].set(
            get_plate_ground_truth(
                getattr(record.get("det"), "attributes", None)
            )
        )
    finally:
        record["uppercase_sync"] = False
    _apply_record_style(host, record, editing=False)

    try:
        host.preview_canvas.focus_set()
    except Exception:
        pass
    return "break"


def _drag_handle_widgets(record: dict) -> list:
    result = []
    for name in ("shell", "info_row", "gt_label"):
        widget = record.get(name)
        if widget is not None:
            result.append(widget)
    return result


def _on_card_drag_start(host, key, event):
    record = _editor_store(host).get(key)
    if not isinstance(record, dict):
        return "break"

    offset = _offset_store(host).get(key, (0.0, 0.0))
    try:
        start_dx, start_dy = float(offset[0]), float(offset[1])
    except Exception:
        start_dx, start_dy = 0.0, 0.0

    record["drag"] = {
        "x_root": float(getattr(event, "x_root", 0.0) or 0.0),
        "y_root": float(getattr(event, "y_root", 0.0) or 0.0),
        "start_dx": start_dx,
        "start_dy": start_dy,
    }
    return "break"


def _on_card_drag_motion(host, key, event):
    record = _editor_store(host).get(key)
    if not isinstance(record, dict):
        return "break"

    drag = record.get("drag")
    if not isinstance(drag, dict):
        return "break"

    dx = float(getattr(event, "x_root", 0.0) or 0.0) - float(
        drag.get("x_root", 0.0)
    )
    dy = float(getattr(event, "y_root", 0.0) or 0.0) - float(
        drag.get("y_root", 0.0)
    )
    _offset_store(host)[key] = (
        float(drag.get("start_dx", 0.0)) + dx,
        float(drag.get("start_dy", 0.0)) + dy,
    )

    try:
        refresh_inline_plate_gt_editors(host)
    except Exception:
        pass
    return "break"


def _on_card_drag_end(host, key, _event=None):
    record = _editor_store(host).get(key)
    if isinstance(record, dict):
        record["drag"] = None
        try:
            _offset_store(host)[key] = (
                float(record.get("card_x", 0.0))
                - float(record.get("base_x", 0.0)),
                float(record.get("card_y", 0.0))
                - float(record.get("base_y", 0.0)),
            )
        except Exception:
            pass
    return "break"


def _create_editor(host, canvas, key, ann, det, plate_idx):
    palette = _palette(host)
    panel_bg = palette.get("panel", "#252526")
    field_bg = palette.get("field", "#171717")
    fg = palette.get("fg", "#f3f3f3")
    muted = palette.get("muted", "#aeb7bf")
    border = palette.get("panel_border", "#555d65")

    layout_value = get_plate_layout_gt(
        getattr(det, "attributes", None)
    )

    shell = tk.Frame(
        canvas,
        bg=panel_bg,
        bd=0,
        highlightthickness=1,
        highlightbackground=border,
        highlightcolor=border,
        cursor="fleur",
    )

    info_var = tk.StringVar(value=format_plate_info(host, det, ann))
    info_row = tk.Label(
        shell,
        textvariable=info_var,
        bg=field_bg,
        fg=muted,
        bd=0,
        anchor="w",
        padx=5,
        pady=0,
        cursor="fleur",
        font=("Segoe UI", 7, "bold"),
    )
    info_row.pack(fill=tk.X, side=tk.TOP)

    gt_row = tk.Frame(shell, bg=panel_bg, bd=0, highlightthickness=0)
    gt_row.pack(fill=tk.BOTH, expand=True, side=tk.TOP)

    gt_label = tk.Label(
        gt_row,
        text="GT",
        bg=panel_bg,
        fg=muted,
        bd=0,
        padx=5,
        pady=0,
        cursor="fleur",
        font=("Segoe UI", 7, "bold"),
    )
    gt_label.pack(side=tk.LEFT, fill=tk.Y)

    layout_var = tk.StringVar(value=_layout_gt_label(layout_value))
    layout_button = tk.Label(
        gt_row,
        textvariable=layout_var,
        bg=panel_bg,
        fg=muted,
        bd=0,
        width=3,
        padx=4,
        pady=0,
        cursor="hand2",
        font=("Segoe UI", 7, "bold"),
        highlightthickness=1,
        highlightbackground=border,
        highlightcolor=border,
    )
    layout_button.pack(
        side=tk.RIGHT,
        fill=tk.Y,
        padx=(2, 2),
        pady=(2, 2),
    )

    var = tk.StringVar(
        value=get_plate_ground_truth(
            getattr(det, "attributes", None)
        )
    )
    entry = tk.Entry(
        gt_row,
        textvariable=var,
        bg=field_bg,
        fg=fg,
        insertbackground=fg,
        selectbackground=palette.get("accent", "#4f8de3"),
        selectforeground=fg,
        relief=tk.FLAT,
        bd=0,
        justify=tk.CENTER,
        font=("Segoe UI", 8, "bold"),
    )
    entry.pack(
        side=tk.LEFT,
        fill=tk.BOTH,
        expand=True,
        padx=(0, 2),
        pady=(2, 2),
    )
    entry._z2_inline_gt_entry = True

    record = {
        "shell": shell,
        "info_row": info_row,
        "info_var": info_var,
        "gt_row": gt_row,
        "gt_label": gt_label,
        "layout_button": layout_button,
        "layout_var": layout_var,
        "layout_preview": "",
        "entry": entry,
        "var": var,
        "ann": ann,
        "det": det,
        "plate_idx": int(plate_idx),
        "destroying": False,
        "editing": False,
        "confirmed": bool(
            get_plate_ground_truth(getattr(det, "attributes", None))
        ),
        "uppercase_sync": False,
        "enter_commit_pending": False,
        "suppress_focus_out_once": False,
        "drag": None,
    }
    _editor_store(host)[key] = record

    try:
        record["trace_id"] = var.trace_add(
            "write",
            lambda *_args, r=record: _uppercase_trace(r),
        )
    except Exception:
        record["trace_id"] = None

    entry.bind("<FocusIn>", lambda _event, k=key: _on_focus_in(host, k), add="+")
    entry.bind("<FocusOut>", lambda _event, k=key: _on_focus_out(host, k), add="+")
    entry.bind("<KeyPress>", lambda event, k=key: _on_key_press(host, k, event), add="+")
    entry.bind("<KeyRelease>", lambda event, k=key: _on_key_release(host, k, event), add="+")
    entry.bind("<Return>", lambda _event, k=key: _finish_with_enter(host, k), add="+")
    entry.bind("<KP_Enter>", lambda _event, k=key: _finish_with_enter(host, k), add="+")
    entry.bind("<Escape>", lambda _event, k=key: _on_escape(host, k), add="+")

    layout_button.bind(
        "<Button-1>",
        lambda _event, k=key: _toggle_record_layout(host, k),
        add="+",
    )

    for widget in _drag_handle_widgets(record):
        widget.bind(
            "<ButtonPress-1>",
            lambda event, k=key: _on_card_drag_start(host, k, event),
            add="+",
        )
        widget.bind(
            "<B1-Motion>",
            lambda event, k=key: _on_card_drag_motion(host, k, event),
            add="+",
        )
        widget.bind(
            "<ButtonRelease-1>",
            lambda event, k=key: _on_card_drag_end(host, k, event),
            add="+",
        )

    _apply_record_style(host, record, editing=False)
    return record


def _destroy_record(host, key, *, flush: bool = True) -> None:
    record = _editor_store(host).get(key)
    if not isinstance(record, dict):
        return

    if flush:
        try:
            _commit_record(host, key, final=True)
        except Exception:
            pass

    _cancel_pending_save(host, key)
    record["destroying"] = True

    trace_id = record.get("trace_id")
    if trace_id:
        try:
            record["var"].trace_remove("write", trace_id)
        except Exception:
            pass

    try:
        record["shell"].destroy()
    except Exception:
        pass
    _editor_store(host).pop(key, None)


def hide_inline_plate_gt_editors(host, *, destroy: bool = False) -> None:
    canvas = getattr(host, "preview_canvas", None)
    if canvas is not None:
        try:
            canvas.delete("preview_gt_tether")
        except Exception:
            pass

    for key, record in list(_editor_store(host).items()):
        if destroy:
            _destroy_record(host, key, flush=True)
            continue
        try:
            record["shell"].place_forget()
        except Exception:
            pass


def _clamp_card_position(canvas_width, canvas_height, x, y):
    max_x = max(2, int(canvas_width) - CARD_WIDTH - 2)
    max_y = max(2, int(canvas_height) - CARD_HEIGHT - 2)
    return (
        max(2, min(int(round(x)), max_x)),
        max(2, min(int(round(y)), max_y)),
    )


def _rects_intersect(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) -> bool:
    return not (
        float(ax2) <= float(bx1)
        or float(ax1) >= float(bx2)
        or float(ay2) <= float(by1)
        or float(ay1) >= float(by2)
    )


def _card_overlaps_plate(card_x, card_y, plate_bbox, margin=5.0) -> bool:
    px1, py1, px2, py2 = [float(v) for v in plate_bbox]
    return _rects_intersect(
        float(card_x),
        float(card_y),
        float(card_x) + CARD_WIDTH,
        float(card_y) + CARD_HEIGHT,
        px1 - float(margin),
        py1 - float(margin),
        px2 + float(margin),
        py2 + float(margin),
    )


def _plate_bbox_meaningfully_visible(
    canvas_width,
    canvas_height,
    plate_bbox,
    *,
    min_visible_ratio: float = 0.10,
) -> bool:
    """Return True when enough of this plate is actually inside the viewport.

    Ratio uses the smaller area of (plate bbox, viewport). That keeps a plate
    visible when it is larger than the viewport at deep zoom, while rejecting
    tiny edge slivers from unrelated/off-screen plates.
    """
    try:
        px1, py1, px2, py2 = [float(v) for v in plate_bbox]
        viewport_w = max(1.0, float(canvas_width))
        viewport_h = max(1.0, float(canvas_height))
    except Exception:
        return False

    if px2 < px1:
        px1, px2 = px2, px1
    if py2 < py1:
        py1, py2 = py2, py1

    plate_w = max(0.0, px2 - px1)
    plate_h = max(0.0, py2 - py1)
    if plate_w <= 0.0 or plate_h <= 0.0:
        return False

    ix1 = max(0.0, px1)
    iy1 = max(0.0, py1)
    ix2 = min(viewport_w, px2)
    iy2 = min(viewport_h, py2)
    intersection_w = max(0.0, ix2 - ix1)
    intersection_h = max(0.0, iy2 - iy1)

    # Do not keep a card alive for a one-pixel strip at the edge.
    if intersection_w < 3.0 or intersection_h < 3.0:
        return False

    intersection_area = intersection_w * intersection_h
    plate_area = plate_w * plate_h
    viewport_area = viewport_w * viewport_h
    reference_area = max(1.0, min(plate_area, viewport_area))

    return bool(
        (intersection_area / reference_area)
        >= float(min_visible_ratio)
    )


def _resolve_non_overlapping_card_position(
    canvas_width,
    canvas_height,
    requested_x,
    requested_y,
    plate_bbox,
):
    requested_x, requested_y = _clamp_card_position(
        canvas_width,
        canvas_height,
        requested_x,
        requested_y,
    )
    if not _card_overlaps_plate(
        requested_x,
        requested_y,
        plate_bbox,
    ):
        return requested_x, requested_y

    px1, py1, px2, py2 = [float(v) for v in plate_bbox]
    margin = 7.0

    raw_candidates = (
        (
            requested_x,
            py1 - CARD_HEIGHT - margin,
        ),
        (
            requested_x,
            py2 + margin,
        ),
        (
            px1 - CARD_WIDTH - margin,
            requested_y,
        ),
        (
            px2 + margin,
            requested_y,
        ),
    )

    safe_candidates = []
    for raw_x, raw_y in raw_candidates:
        card_x, card_y = _clamp_card_position(
            canvas_width,
            canvas_height,
            raw_x,
            raw_y,
        )
        if _card_overlaps_plate(
            card_x,
            card_y,
            plate_bbox,
        ):
            continue
        distance = (
            (float(card_x) - float(requested_x)) ** 2
            + (float(card_y) - float(requested_y)) ** 2
        )
        safe_candidates.append((distance, card_x, card_y))

    if safe_candidates:
        safe_candidates.sort(key=lambda item: item[0])
        _distance, card_x, card_y = safe_candidates[0]
        return int(card_x), int(card_y)

    # Extreme zoom can make the polygon occupy nearly the entire canvas.
    # Search the four canvas corners; if none is safe, do not cover the plate.
    corners = (
        (2, 2),
        (max(2, int(canvas_width) - CARD_WIDTH - 2), 2),
        (2, max(2, int(canvas_height) - CARD_HEIGHT - 2)),
        (
            max(2, int(canvas_width) - CARD_WIDTH - 2),
            max(2, int(canvas_height) - CARD_HEIGHT - 2),
        ),
    )
    for card_x, card_y in corners:
        if not _card_overlaps_plate(
            card_x,
            card_y,
            plate_bbox,
        ):
            return int(card_x), int(card_y)

    fallback_x, fallback_y = _clamp_card_position(
        canvas_width,
        canvas_height,
        requested_x,
        requested_y,
    )
    return int(fallback_x), int(fallback_y)


def _draw_tether(
    host,
    canvas,
    card_x,
    card_y,
    plate_center_x,
    plate_center_y,
):
    palette = _palette(host)
    canvas_bg = palette.get("canvas_bg", palette.get("panel", "#20252b"))
    muted = palette.get("muted", "#aeb7bf")
    line_color = blend_hex_colors(muted, canvas_bg, 0.42)

    x1 = max(card_x, min(plate_center_x, card_x + CARD_WIDTH))
    y1 = max(card_y, min(plate_center_y, card_y + CARD_HEIGHT))

    dx = plate_center_x - x1
    dy = plate_center_y - y1
    mid_x = x1 + (dx * 0.48)
    mid_y = y1 + (dy * 0.48)

    if abs(dx) >= abs(dy):
        mid_y += 8.0 if dy >= 0 else -8.0
    else:
        mid_x += 8.0 if dx >= 0 else -8.0

    try:
        canvas.create_line(
            x1,
            y1,
            mid_x,
            mid_y,
            plate_center_x,
            plate_center_y,
            fill=line_color,
            width=1,
            dash=(3, 3),
            smooth=True,
            splinesteps=10,
            tags=("preview_overlay", "preview_gt_tether"),
        )
        canvas.create_oval(
            plate_center_x - 2,
            plate_center_y - 2,
            plate_center_x + 2,
            plate_center_y + 2,
            outline="",
            fill=line_color,
            tags=("preview_overlay", "preview_gt_tether"),
        )
    except Exception:
        pass


def relocate_inline_plate_gt_editors(
    host,
    *,
    canvas=None,
) -> None:
    """Reposition existing GT cards only; no value/style refresh."""
    canvas = canvas or getattr(host, "preview_canvas", None)
    if canvas is None:
        return

    try:
        canvas.delete("preview_gt_tether")
    except Exception:
        pass

    ann = host._get_preview_annotation()
    if ann is None:
        hide_inline_plate_gt_editors(host, destroy=False)
        return

    try:
        plates = list(host._get_plate_detections(ann) or [])
    except Exception:
        plates = []

    canvas_width = max(1, int(canvas.winfo_width() or 1))
    canvas_height = max(1, int(canvas.winfo_height() or 1))
    viewport_left, viewport_top, _, _ = canvas_viewport_bounds(canvas)

    for plate_idx, det in enumerate(plates):
        key = _key_for(det)
        record = _editor_store(host).get(key)
        if not isinstance(record, dict):
            continue

        polygon = list(host._detection_polygon(det) or [])
        if len(polygon) < 4:
            continue

        record["ann"] = ann
        record["det"] = det
        record["plate_idx"] = int(plate_idx)

        canvas_points = [
            canvas.image_to_canvas_coords(px, py)
            for px, py in polygon
        ]
        xs = [
            float(point[0]) - float(viewport_left)
            for point in canvas_points
        ]
        ys = [
            float(point[1]) - float(viewport_top)
            for point in canvas_points
        ]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        if not _plate_bbox_meaningfully_visible(
            canvas_width,
            canvas_height,
            (min_x, min_y, max_x, max_y),
        ):
            shell = record.get("shell")
            if shell is not None:
                try:
                    shell.place_forget()
                except Exception:
                    pass
            continue
        plate_center_x = (min_x + max_x) / 2.0
        plate_center_y = (min_y + max_y) / 2.0

        base_x = plate_center_x - (CARD_WIDTH / 2.0)
        base_y = min_y - CARD_HEIGHT - CARD_GAP
        if base_y < 2:
            base_y = max_y + CARD_GAP

        offset = _offset_store(host).get(key, (0.0, 0.0))
        try:
            offset_x = float(offset[0])
            offset_y = float(offset[1])
        except Exception:
            offset_x = 0.0
            offset_y = 0.0

        card_x, card_y = _resolve_non_overlapping_card_position(
            canvas_width,
            canvas_height,
            base_x + offset_x,
            base_y + offset_y,
            (min_x, min_y, max_x, max_y),
        )

        record["base_x"] = float(base_x)
        record["base_y"] = float(base_y)
        record["card_x"] = float(card_x)
        record["card_y"] = float(card_y)

        shell = record.get("shell")
        if shell is not None:
            try:
                shell.place(
                    x=int(card_x),
                    y=int(card_y),
                    width=CARD_WIDTH,
                    height=CARD_HEIGHT,
                )
                shell.lift()
            except Exception:
                pass

        _draw_tether(
            host,
            canvas,
            float(card_x) + float(viewport_left),
            float(card_y) + float(viewport_top),
            plate_center_x + float(viewport_left),
            plate_center_y + float(viewport_top),
        )

    try:
        host._plate_gt_inline_interaction_suspended = False
    except Exception:
        pass


def render_inline_plate_gt_editors(
    host,
    canvas,
    plate_detections,
    selected_plate_idx=None,
    *,
    light_overlay: bool = False,
) -> None:
    try:
        canvas.delete("preview_gt_tether")
    except Exception:
        pass

    if light_overlay:
        relocate_inline_plate_gt_editors(host, canvas=canvas)
        return

    host._plate_gt_inline_interaction_suspended = False

    ann = host._get_preview_annotation()
    if ann is None or canvas is None:
        hide_inline_plate_gt_editors(host, destroy=True)
        return

    try:
        editable = bool(host._preview_is_editable())
    except Exception:
        editable = False

    active_keys = set()
    canvas_width = max(1, int(canvas.winfo_width() or 1))
    canvas_height = max(1, int(canvas.winfo_height() or 1))
    (
        viewport_left,
        viewport_top,
        _viewport_right,
        _viewport_bottom,
    ) = canvas_viewport_bounds(canvas)

    for plate_idx, det in enumerate(list(plate_detections or [])):
        polygon = list(host._detection_polygon(det) or [])
        if len(polygon) < 4:
            continue

        key = _key_for(det)
        active_keys.add(key)
        record = _editor_store(host).get(key)
        if not isinstance(record, dict):
            record = _create_editor(host, canvas, key, ann, det, plate_idx)
        else:
            record["ann"] = ann
            record["det"] = det
            record["plate_idx"] = int(plate_idx)

        entry = record["entry"]
        var = record["var"]
        shell = record["shell"]

        try:
            record["info_var"].set(format_plate_info(host, det, ann))
        except Exception:
            pass

        if not _focused(host, record):
            stored_gt = get_plate_ground_truth(
                getattr(det, "attributes", None)
            )
            try:
                if str(var.get() or "") != stored_gt:
                    record["uppercase_sync"] = True
                    try:
                        var.set(stored_gt)
                    finally:
                        record["uppercase_sync"] = False
            except Exception:
                pass

        try:
            entry.configure(state=("normal" if editable else "disabled"))
        except Exception:
            pass

        canvas_points = [
            canvas.image_to_canvas_coords(px, py)
            for px, py in polygon
        ]
        canvas_xs = [float(p[0]) for p in canvas_points]
        canvas_ys = [float(p[1]) for p in canvas_points]

        # tk.place() uses viewport-local widget coordinates, while
        # image_to_canvas_coords() returns scrollregion/canvas coordinates.
        xs = [value - viewport_left for value in canvas_xs]
        ys = [value - viewport_top for value in canvas_ys]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        if not _plate_bbox_meaningfully_visible(
            canvas_width,
            canvas_height,
            (min_x, min_y, max_x, max_y),
        ):
            try:
                shell.place_forget()
            except Exception:
                pass
            continue
        plate_center_x = (min_x + max_x) / 2.0
        plate_center_y = (min_y + max_y) / 2.0

        base_x = plate_center_x - (CARD_WIDTH / 2.0)
        base_y = min_y - CARD_HEIGHT - CARD_GAP
        if base_y < 2:
            base_y = max_y + CARD_GAP

        offset = _offset_store(host).get(key, (0.0, 0.0))
        try:
            offset_x = float(offset[0])
            offset_y = float(offset[1])
        except Exception:
            offset_x = 0.0
            offset_y = 0.0

        resolved_position = _resolve_non_overlapping_card_position(
            canvas_width,
            canvas_height,
            base_x + offset_x,
            base_y + offset_y,
            (min_x, min_y, max_x, max_y),
        )
        if resolved_position is None:
            try:
                shell.place_forget()
            except Exception:
                pass
            continue
        card_x, card_y = resolved_position

        record["base_x"] = float(base_x)
        record["base_y"] = float(base_y)
        record["card_x"] = float(card_x)
        record["card_y"] = float(card_y)

        _apply_record_style(
            host,
            record,
            editing=bool(record.get("editing", False)),
        )

        try:
            shell.place(
                x=card_x,
                y=card_y,
                width=CARD_WIDTH,
                height=CARD_HEIGHT,
            )
            shell.lift()
        except Exception:
            pass

        _draw_tether(
            host,
            canvas,
            card_x + viewport_left,
            card_y + viewport_top,
            plate_center_x + viewport_left,
            plate_center_y + viewport_top,
        )

    for stale_key in set(_editor_store(host)) - active_keys:
        _destroy_record(host, stale_key, flush=True)


def refresh_inline_plate_gt_editors(host) -> None:
    canvas = getattr(host, "preview_canvas", None)
    ann = host._get_preview_annotation()
    if canvas is None or ann is None:
        hide_inline_plate_gt_editors(host, destroy=True)
        return

    plates = list(host._get_plate_detections(ann) or [])
    selected = host._get_selected_plate_index_for_ann(ann)
    render_inline_plate_gt_editors(
        host,
        canvas,
        plates,
        selected,
        light_overlay=bool(
            getattr(host, "_preview_light_overlay_refresh", False)
        ),
    )


def draw_gt_mode_toggle(host, canvas) -> None:
    # Compatibility hook only. Never render a separate GT switch.
    try:
        if canvas is not None:
            canvas.delete("preview_gt_mode_toggle")
    except Exception:
        pass
    host._plate_gt_mode_toggle_bbox = None


def refresh_gt_overlay_after_layout(host) -> None:
    # The combined RAMKA/OK-NOK bar is the single owner of GT/FS icons.
    # Redraw the full preview overlay after fullscreen/layout changes instead
    # of creating a second standalone GT icon.
    try:
        host._refresh_preview_canvas(refresh_chrome=False)
    except Exception:
        pass


def is_gt_mode_toggle_hit(host, canvas_x, canvas_y) -> bool:
    # No GT switch exists on the canvas.
    return False
