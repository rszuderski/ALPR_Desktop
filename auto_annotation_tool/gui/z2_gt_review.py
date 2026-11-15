"""Small in-memory updates for the Z2 GT review controls."""

from .z2_gt_readiness import annotation_gt_readiness, missing_required_gt


def build_missing_gt_filter(host, parent):
    import tkinter as tk
    from tkinter import ttk

    host.preview_missing_gt_only_var = tk.BooleanVar(master=parent, value=False)
    host.preview_missing_gt_only_check = ttk.Checkbutton(
        parent, text="Brak GT: 0 zdjęć / 0 ramek",
        variable=host.preview_missing_gt_only_var,
        command=lambda: apply_missing_gt_filter(host),
    )
    host.preview_missing_gt_only_check.pack(fill=tk.X, pady=(3, 3))


def missing_gt_filter_active(host):
    variable = getattr(host, "preview_missing_gt_only_var", None)
    return bool(variable is not None and variable.get())


def refresh_missing_gt_count(host):
    widget = getattr(host, "preview_missing_gt_only_check", None)
    if widget is None:
        return
    images = 0
    plates = 0
    for ann in getattr(host, "current_annotations", []) or []:
        missing = annotation_gt_readiness(ann)["missing"]
        images += int(missing > 0)
        plates += missing
    widget.configure(text=f"Brak GT: {images} zdjęć / {plates} ramek")


def apply_missing_gt_filter(host):
    if missing_gt_filter_active(host):
        for name in ("preview_filter_conf_var", "preview_filter_fit_var"):
            variable = getattr(host, name, None)
            if variable is not None:
                variable.set(0.0)
        host._preview_filter_conf_applied = 0.0
        host._preview_filter_fit_applied = 0.0
        callback = getattr(host, "_refresh_preview_filter_bar_state", None)
        if callable(callback):
            callback()
    host._invalidate_preview_list_frozen_order()
    host._refresh_preview_list(preserve_selection=True, render_current=True)
    host._update_preview_edit_status(
        "Lista pokazuje zdjęcia bez numerów tablic. Możesz zatwierdzić geometrię i uzupełnić numery w Z3."
        if missing_gt_filter_active(host) else "Wyłączono filtr brakującego GT."
    )


def revoke_incomplete_approval(host, ann):
    """A cleared GT or a new empty plate must require a fresh OK decision."""
    if not missing_required_gt(host, ann):
        return False
    filename = str(getattr(ann, "filename", "") or "").strip().lower()
    changed = bool(getattr(ann, "_approved_for_training", False))
    ann._approved_for_training = False
    for name in ("_preview_approved_filenames", "_pending_preview_approved_filenames",
                 "_campaign_pending_approved_filenames"):
        values = getattr(host, name, None)
        if isinstance(values, set):
            matches = {v for v in values if str(v).strip().lower() == filename}
            changed = changed or bool(matches)
            values.difference_update(matches)
    if changed:
        host._preview_approval_version = int(getattr(host, "_preview_approval_version", 0) or 0) + 1
        callback = getattr(host, "_schedule_preview_approved_persist", None)
        if callable(callback):
            callback(delay_ms=2600)
    return changed


def refresh_annotation_gt_review(host, ann, *, revoke=True):
    if revoke:
        revoke_incomplete_approval(host, ann)
    # Only metadata and the affected row are refreshed. Keep the edited image
    # visible until the user reapplies the filter; typing must never navigate.
    for name in ("_preview_list_render_state_cache", "_preview_list_summary_cache",
                 "_current_preview_plate_count_cache"):
        setattr(host, name, None)
    for index, current in enumerate(getattr(host, "current_annotations", []) or []):
        if current is ann:
            callback = getattr(host, "_refresh_preview_list_row_for_actual_index", None)
            if callable(callback):
                callback(index, refresh_summary=False, lightweight=True)
            break
    refresh_missing_gt_count(host)
    callback = getattr(host, "_refresh_preview_canvas_light", None)
    if callable(callback):
        callback()
