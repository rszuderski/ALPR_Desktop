"""Z2 runtime for manual plate ground-truth editing."""

from __future__ import annotations

from ..plate_ground_truth import (
    GROUND_TRUTH_SOURCE_MANUAL_Z2,
    get_plate_ground_truth,
    set_plate_ground_truth,
)


def _selected_plate(host):
    ann = host._get_preview_annotation()
    if ann is None:
        return None, None, None, []

    plates = list(host._get_plate_detections(ann) or [])
    if not plates:
        return ann, None, None, plates

    plate_idx = host._get_selected_plate_index_for_ann(ann)
    if plate_idx is None:
        plate_idx = 0

    try:
        safe_idx = int(plate_idx)
    except Exception:
        safe_idx = 0

    if safe_idx < 0 or safe_idx >= len(plates):
        safe_idx = 0

    return ann, safe_idx, plates[safe_idx], plates


def refresh_plate_gt_editor(host) -> None:
    gt_var = getattr(host, "plate_gt_var", None)
    context_var = getattr(host, "plate_gt_context_var", None)
    entry = getattr(host, "plate_gt_entry", None)
    save_btn = getattr(host, "plate_gt_save_btn", None)
    clear_btn = getattr(host, "plate_gt_clear_btn", None)

    if gt_var is None:
        return

    ann, plate_idx, det, plates = _selected_plate(host)

    try:
        editable = bool(det is not None and host._preview_is_editable())
    except Exception:
        editable = False

    if det is None:
        gt_text = ""
        context_text = "Brak wybranej ramki tablicy."
    else:
        gt_text = get_plate_ground_truth(getattr(det, "attributes", None))
        context_text = (
            f"Tablica {int(plate_idx) + 1}/{len(plates)}"
            + (f" | GT: {gt_text}" if gt_text else " | GT: brak")
        )

    host._plate_gt_editor_syncing = True
    try:
        if str(gt_var.get() or "") != gt_text:
            gt_var.set(gt_text)
        if context_var is not None:
            context_var.set(context_text)
    finally:
        host._plate_gt_editor_syncing = False

    state = "normal" if editable else "disabled"
    try:
        if entry is not None:
            entry.configure(state=state)
    except Exception:
        pass
    try:
        if save_btn is not None:
            save_btn.configure(state=state)
    except Exception:
        pass
    try:
        if clear_btn is not None:
            clear_btn.configure(
                state=("normal" if editable and bool(gt_text) else "disabled")
            )
    except Exception:
        pass


def commit_plate_gt_editor(host, event=None):
    if bool(getattr(host, "_plate_gt_editor_syncing", False)):
        return "break"

    gt_var = getattr(host, "plate_gt_var", None)
    if gt_var is None:
        return "break"

    ann, plate_idx, det, plates = _selected_plate(host)
    if ann is None or det is None or plate_idx is None:
        try:
            host._update_preview_edit_status(
                "Nie można zapisać GT: nie wybrano ramki tablicy."
            )
        except Exception:
            pass
        refresh_plate_gt_editor(host)
        return "break"

    try:
        if not host._preview_is_editable():
            host._update_preview_edit_status(
                "Nie można zmienić GT: bieżący podgląd jest tylko do odczytu."
            )
            refresh_plate_gt_editor(host)
            return "break"
    except Exception:
        pass

    previous_attributes = dict(getattr(det, "attributes", {}) or {})
    requested_text = str(gt_var.get() or "")

    det.attributes = dict(previous_attributes)
    normalized = set_plate_ground_truth(
        det.attributes,
        requested_text,
        source=GROUND_TRUTH_SOURCE_MANUAL_Z2,
    )

    try:
        host._mark_preview_image_dirty(ann, refresh_list=False)
    except TypeError:
        host._mark_preview_image_dirty(ann)

    action = (
        f"Zapisano GT {normalized} dla tablicy {int(plate_idx) + 1}/{len(plates)}."
        if normalized
        else f"Usunięto GT dla tablicy {int(plate_idx) + 1}/{len(plates)}."
    )

    try:
        saved = bool(
            host._save_preview_edits(
                interactive=False,
                status_message=action,
            )
        )
    except Exception:
        saved = False

    if not saved:
        det.attributes = previous_attributes
        try:
            host._update_preview_edit_status(
                "Nie udało się zapisać GT do annotations.xml. Przywrócono poprzednią wartość."
            )
        except Exception:
            pass
        refresh_plate_gt_editor(host)
        return "break"

    refresh_plate_gt_editor(host)
    try:
        host._z2_graph_right_panel_render_signature = None
    except Exception:
        pass
    try:
        host._refresh_step2_action_states(lightweight=False)
    except Exception:
        pass
    try:
        host._refresh_preview_canvas()
    except Exception:
        pass
    return "break"


def clear_plate_gt_editor(host):
    gt_var = getattr(host, "plate_gt_var", None)
    if gt_var is None:
        return
    gt_var.set("")
    commit_plate_gt_editor(host)
