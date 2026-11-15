"""Prepare/hash E1 manifests off Tk, then approve the saved result on Tk."""
from copy import deepcopy
from queue import SimpleQueue, Empty
from threading import Thread
from tkinter import messagebox

from ..campaign_manager import CAMPAIGN


def approve_ingest_async(host, **options):
    if getattr(host, "_ingest_identity_running", False):
        return None
    from .campaign_step1_ingest import _update_ingest_plan_progress_dialog, _hide_ingest_plan_progress_dialog
    project = CAMPAIGN.get_active_project_name()
    iteration = CAMPAIGN.get_current_iteration_num()
    options = deepcopy(options)
    messages = SimpleQueue()
    host._ingest_identity_running = True
    result = {"done": False, "after_id": None}

    def progress(value, message, **kwargs):
        messages.put(("progress", (value, message, kwargs.get("detail", ""))))

    def worker():
        try:
            path = CAMPAIGN.record_iteration_ingest(
                source_dir=options["source_dir"], selected_source_files=options["selected_source_files"],
                selection_mode=options.get("selection_mode", "manual"),
                selected_source_metadata=options.get("selected_source_metadata"),
                proposal_summary=options.get("proposal_summary"), project_name=project,
                iteration_num=iteration, progress_callback=progress,
            )
            if path is None:
                raise ValueError("Nie udało się zapisać manifestu E1.")
            prepared = CAMPAIGN.load_ingest_manifest(iteration, project)
            if not prepared.get("selected_images"):
                raise ValueError("Manifest E1 nie zawiera dostępnych obrazów.")
            messages.put(("done", prepared))
        except Exception as exc:
            messages.put(("error", str(exc)))

    def finish():
        result["done"] = True
        host._ingest_identity_running = False
        host.frame.unbind("<Destroy>", binding)

    def destroyed(event):
        if event.widget is host.frame:
            if result["after_id"]:
                host.frame.after_cancel(result["after_id"])
            result["done"] = True
            host._ingest_identity_running = False

    binding = host.frame.bind("<Destroy>", destroyed, add="+")

    def poll():
        result["after_id"] = None
        if project != CAMPAIGN.get_active_project_name() or iteration != CAMPAIGN.get_current_iteration_num():
            result["cancelled"] = True
            finish()
            return
        latest = None
        completed = None
        while True:
            try:
                kind, payload = messages.get_nowait()
            except Empty:
                break
            if kind == "progress":
                latest = payload
            else:
                completed = kind, payload
        if latest:
            _update_ingest_plan_progress_dialog(host, *latest)
        if completed:
            finish()
            kind, payload = completed
            if kind == "error":
                _hide_ingest_plan_progress_dialog(host)
                messagebox.showerror("Nie zatwierdzono E1", payload, parent=host.frame.winfo_toplevel())
                result["error"] = payload
                return
            try:
                count = host._approve_current_iteration_package(**options, prepared_manifest=payload)
                summary = payload.get("identity_summary") or {}
                details = (f"Zapisano {count} obrazów. Identyczne pominięte: {summary.get('duplicate_sha256', 0)}. "
                           f"Konflikty nazw: {summary.get('name_collisions', 0)}.")
                _update_ingest_plan_progress_dialog(host, 100, "E1 zatwierdzone.", details)
                host.app.update_status(details, "info")
                result.update(ok=True, count=count, identity_summary=summary)
            except Exception as exc:
                result["error"] = str(exc)
                messagebox.showerror("Nie zatwierdzono E1", str(exc), parent=host.frame.winfo_toplevel())
            finally:
                _hide_ingest_plan_progress_dialog(host, delay_ms=650)
            return
        result["after_id"] = host.frame.after(40, poll)

    result["after_id"] = host.frame.after(40, poll)
    Thread(target=worker, daemon=True, name="campaign-e1-image-identity").start()
    return result
