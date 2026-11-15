"""Selection and group approval actions for the PZ2 plate list."""
import time
import tkinter as tk

from . import z3_review_runtime as review
from .z3_metadata_cache import mark_preview_metadata_changed
from .z3_gt_box_policy import working_characters


def selected_plate_ids(host):
    ids = getattr(host, "_listbox_pid_by_index", []) or []
    return [ids[int(index)] for index in host.plates_listbox.curselection() if 0 <= int(index) < len(ids)]


def capture_selection(host):
    ids = getattr(host, "_listbox_pid_by_index", []) or []
    box = host.plates_listbox
    def pid_at(kind):
        try:
            index = int(box.index(kind))
            return ids[index] if 0 <= index < len(ids) else None
        except (tk.TclError, ValueError):
            return None
    return {"selected": selected_plate_ids(host), "active": pid_at(tk.ACTIVE),
            "anchor": pid_at(tk.ANCHOR), "view": box.yview()[0]}


def restore_selection(host, saved, *, default_first=True):
    ids = getattr(host, "_listbox_pid_by_index", []) or []
    positions = {pid: index for index, pid in enumerate(ids)}
    rows = [positions[pid] for pid in saved.get("selected", []) if pid in positions]
    if not rows and ids and default_first:
        rows = [0]
    box = host.plates_listbox
    host._clear_listbox_selection_fast(box)
    for row in rows:
        box.selection_set(row)
    active = positions.get(saved.get("active"))
    if active not in rows:
        active = rows[0] if rows else None
    if active is not None:
        box.activate(active)
        box.selection_anchor(positions.get(saved.get("anchor"), active))
    box.yview_moveto(saved.get("view", 0))
    return active


def select_all(host, event=None):
    if getattr(host, "_preview_review_batch_running", False):
        return "break"
    host.plates_listbox.selection_set(0, tk.END)
    return "break"


def copy_selected_labels(host):
    box = host.plates_listbox
    labels = [box.get(index) for index in box.curselection()]
    if not labels:
        return None
    text = "\n".join(labels)
    try:
        box.clipboard_clear()
        box.clipboard_append(text)
    except tk.TclError:
        host._update_preview_edit_status("Nie udało się skopiować etykiet do schowka.", tone="warning")
        return None
    host._update_preview_edit_status(f"Skopiowano etykiety tablic: {len(labels)}.", tone="success")
    return text


def clear_automatic_boxes(host, plate_id):
    """Clear working automatic geometry, preserving manual boxes and RAW."""
    data = host.preview_metadata.get(plate_id)
    if not isinstance(data, dict):
        return {"ok": False, "reason": "missing_plate"}
    records = list(working_characters(host, data))
    retained = [rec for index, rec in enumerate(records)
                if host._get_character_box_source_tag(rec, data=data, fallback_index=index) == "manual_box"]
    removed = len(records) - len(retained)
    if not removed:
        return {"ok": True, "unchanged": True, "removed_boxes": 0, "manual_boxes": len(retained)}
    host._push_preview_history_snapshot(plate_id)
    review.mark_review_edit_started(host, data)
    data["characters"] = retained
    data["status"] = "needs_fix"
    data["fusion_strategy"] = "manual_correction"
    for key in ("gt_box_limit", "gt_geometry_recovery", "live_gt_assist", "gt_assist", "_layout_override_chars_backup"):
        data.pop(key, None)
    data["automatic_boxes_cleared"] = {"removed_count": removed, "retained_manual_count": len(retained)}
    return {"ok": True, "removed_boxes": removed, "manual_boxes": len(retained)}


def run_selected_clear_automatic_boxes(host):
    return run_selected_review_action(host, approved=False, action="clear_automatic_boxes")


def change_plate_approval(host, plate_id, approved):
    data = host.preview_metadata.get(plate_id)
    if not isinstance(data, dict):
        return {"ok": False, "reason": "missing_plate"}
    status = review.get_review_state_status(data)
    if not approved:
        if status != review.REVIEW_APPROVED:
            return {"ok": True, "unchanged": True}
        review.mark_review_edit_started(host, data)
        data["status"] = "needs_fix"
        review._persist_review_az_revision_best_effort(
            host, plate_id, data, event="approval_reverted",
        )
        return {"ok": True}
    if (status == review.REVIEW_APPROVED and host._review_approval_is_current(data)
            and host._get_review_quality_status(data) == "perfect"):
        # Repeating the explicit approval also repairs older missing AZ
        # checkpoints without reopening or modifying the local review.
        review._persist_review_az_revision_best_effort(
            host, plate_id, data, event="review_approved",
        )
        return {"ok": True, "unchanged": True}
    if not status and not data.get("characters"):
        return {"ok": False, "reason": "working_annotation_missing"}
    elif status != review.REVIEW_IN_PROGRESS:
        review.mark_review_edit_started(host, data)
    # Batch metadata saves remain coalesced. Each explicit decision still needs
    # its AZ checkpoint, including imported_pending_review -> approved.
    return review.confirm_review_gold(host, plate_id, persist=True, quiet=True, refresh=False)


def run_selected_review_action(host, approved=True, *, action=None):
    if (getattr(host, "_preview_review_batch_running", False)
            or getattr(host, "is_processing", False) or getattr(host, "fast_test_running", False)):
        return None
    ids = selected_plate_ids(host)
    if not ids:
        return None
    metadata = host.preview_metadata
    reset_token = getattr(host, "_project_reset_token", None)
    result = {"total": len(ids), "processed": 0, "changed": [], "unchanged": 0,
              "failed": [], "dirty": [], "approved": bool(approved), "done": False,
              "action": action, "removed_boxes": 0, "manual_boxes": 0}
    host._preview_review_batch_running = True
    host._preview_review_batch_result = result
    host._refresh_detection_review_controls()
    def destroyed(event):
        if event.widget is host.frame:
            job = result.get("after_id")
            if job:
                host.frame.after_cancel(job)
            host._preview_review_batch_running = False
            result.update(done=True, cancelled=True)
    destroy_binding = host.frame.bind("<Destroy>", destroyed, add="+")

    def detach_callback():
        host.frame.unbind("<Destroy>", destroy_binding)

    def step():
        result["after_id"] = None
        if metadata is not host.preview_metadata or reset_token != getattr(host, "_project_reset_token", None):
            result.update(done=True, cancelled=True)
            host._preview_review_batch_running = False
            host._refresh_detection_review_controls()
            detach_callback()
            return
        deadline = time.perf_counter() + .015
        chunk_start = result["processed"]
        while result["processed"] < len(ids):
            pid = ids[result["processed"]]
            try:
                outcome = (clear_automatic_boxes(host, pid) if action == "clear_automatic_boxes"
                           else change_plate_approval(host, pid, approved))
            except Exception as exc:
                outcome = {"ok": False, "reason": "validation_error", "error": str(exc)}
            if outcome.get("unchanged"):
                result["unchanged"] += 1
            elif outcome.get("ok"):
                result["changed"].append(pid)
            else:
                result["failed"].append({"plate_id": pid, "reason": outcome.get("reason", "review_not_perfect")})
            result["removed_boxes"] += outcome.get("removed_boxes", 0)
            result["manual_boxes"] += outcome.get("manual_boxes", 0)
            if not outcome.get("unchanged") and outcome.get("reason") not in {"missing_plate", "missing_raw_detection"}:
                result["dirty"].append(pid)
            host._refresh_preview_listbox_row(pid)
            result["processed"] += 1
            if time.perf_counter() >= deadline or result["processed"] - chunk_start >= 25:
                break
        pending = getattr(host, "_preview_pending_save_pids", None)
        if not isinstance(pending, set):
            pending = host._preview_pending_save_pids = set()
        pending.update(result["dirty"])
        if result["processed"] < len(ids):
            # Coalesce writes while working; leaving the run can still flush
            # completed rows through the existing autosave lifecycle.
            if pending:
                host._schedule_preview_metadata_save(delay_ms=180)
            label = "Usuwanie automatycznych ramek" if action == "clear_automatic_boxes" else "Aktualizacja zatwierdzeń"
            host._update_preview_edit_status(f"{label}: {result['processed']}/{len(ids)}", tone="info")
            result["after_id"] = host.frame.after(1, step)
            return

        host._preview_review_batch_running = False
        if action == "clear_automatic_boxes" and result["changed"]:
            from .z3_detection_runtime import clear_detection_review_snapshot_after_manual_edit
            clear_detection_review_snapshot_after_manual_edit(host)
        # Failed approval can still create a valid, unfinished working review.
        mark_preview_metadata_changed(host)
        if pending:
            host._schedule_preview_metadata_save(delay_ms=30)
        host.preview_box_mode_var.set(host._get_preview_box_mode_label("AUTO"))
        host._refresh_detection_review_controls()
        host._update_preview_info_label()
        host._sync_step3_access_from_preview_state(metadata)
        host._on_preview_select(None)
        verb = "Zatwierdzono" if approved else "Cofnięto zatwierdzenie"
        message = f"{verb}: {len(result['changed'])}/{len(ids)}."
        if action == "clear_automatic_boxes":
            message = (f"Usunięto automatyczne ramki: {result['removed_boxes']}, "
                       f"tablic: {len(result['changed'])}/{len(ids)}. "
                       f"Zachowano ręczne ramki: {result['manual_boxes']}.")
        if result["unchanged"]:
            message += f" Bez zmian: {result['unchanged']}."
        if result["failed"]:
            examples = ", ".join(item["plate_id"] for item in result["failed"][:3])
            message += f" Wymaga poprawy: {len(result['failed'])} ({examples})."
        host._update_preview_edit_status(message, tone="warning" if result["failed"] else "success")
        result["done"] = True
        detach_callback()

    result["after_id"] = host.frame.after(1, step)
    return result


def show_context_menu(host, event):
    if getattr(host, "_preview_review_batch_running", False):
        return "break"
    box = host.plates_listbox
    if not box.size():
        return "break"
    index = int(box.nearest(event.y))
    bounds = box.bbox(index)
    if bounds is None or not bounds[1] <= event.y < bounds[1] + bounds[3]:
        return "break"
    if not box.selection_includes(index):
        host._clear_listbox_selection_fast(box)
        box.selection_set(index)
        box.selection_anchor(index)
    box.activate(index)
    box.focus_set()
    host._schedule_preview_select_render(delay_ms=1)
    old = getattr(host, "_preview_list_context_menu", None)
    if old is not None:
        old.destroy()
    menu = host._preview_list_context_menu = tk.Menu(box, tearoff=False)
    total = len(selected_plate_ids(host))
    busy = bool(getattr(host, "is_processing", False) or getattr(host, "fast_test_running", False))
    state = tk.DISABLED if busy else tk.NORMAL
    menu.add_command(label=f"Zatwierdź zaznaczone ({total})", state=state,
                     command=lambda: run_selected_review_action(host, True))
    menu.add_command(label=f"Cofnij zatwierdzenie ({total})", state=state,
                     command=lambda: run_selected_review_action(host, False))
    menu.add_command(label=f"Usuń automatyczne ramki ({total})", state=state,
                     command=lambda: run_selected_clear_automatic_boxes(host))
    menu.add_separator()
    menu.add_command(label=f"Kopiuj zaznaczone etykiety ({total})",
                     command=lambda: copy_selected_labels(host))
    menu.add_command(label="Zaznacz wszystkie", accelerator="Ctrl+A", command=lambda: select_all(host))
    try:
        menu.tk_popup(event.x_root, event.y_root)
    finally:
        menu.grab_release()
    return "break"
