"""Small controlled-experiment section and asynchronous frozen validation for Z4."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from ..campaign_manager import CAMPAIGN
from .z4_flow_models import TrainingSource, TrainingSourceStats
from ..training.mz_experiment_runtime import (
    MZ_VARIANTS, MZExperimentSelection, prepare_mz_experiment, read_mz_experiment_protocol,
)

_FIELDS = ("dataset_var", "dataset_variant_var", "base_model_var", "base_custom_var", "name_var",
           "epochs_var", "batch_var", "imgsz_var", "lr0_var", "device_var")
_CONTEXT_FIELDS = ("_step4_dataset_mode", "_step4_route_selected", "_step4_train_unlocked",
    "_campaign_training_target", "_last_training_source", "_pending_step4_input_training_source",
    "_step4_dataset_summary_split_counts", "_dataset_variant_choices", "_current_training_dataset_is_pose",
    "_last_training_cockpit_summary", "_step4_creator_source_mode")
_STORAGE_FIELDS = ("history", "trainer")


class _DetachedValue:
    """Route switch sees an empty source without writing normal Tk variables."""
    def __init__(self):
        self.value = ""

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


def is_active(host) -> bool:
    variable = getattr(host, "mz_mode_var", None)
    return bool(variable is not None and variable.get() and getattr(host, "_mz_protocol", None))


def _busy(host) -> bool:
    return bool(getattr(host, "_mz_validation_in_progress", False)
        or getattr(host, "_training_start_in_progress", False)
        or getattr(getattr(host, "trainer", None), "is_training", False))


def _frozen_widgets(host):
    for name in ("dataset_variant_combo", "base_combo", "base_custom_entry", "base_custom_btn",
                 "device_combo", "btn_apply_training_recommendation"):
        widget = getattr(host, name, None)
        if widget is not None:
            yield widget
    for row in getattr(host, "_train_recommendation_cells", []) or []:
        if row.get("editor") is not None:
            yield row["editor"]


def refresh_controls(host) -> None:
    active = is_active(host)
    busy = _busy(host)
    for name in ("mz_mode_check", "mz_load_button", "mz_variant_combo"):
        widget = getattr(host, name, None)
        if widget is not None:
            widget.configure(state="disabled" if busy else ("readonly" if name == "mz_variant_combo" else "normal"))
    if active:
        for widget in _frozen_widgets(host):
            widget.configure(state="disabled")


def _status(host, text):
    variable = getattr(host, "mz_status_var", None)
    if variable is not None:
        variable.set(text)


def _selection_status(selection):
    protocol = selection.protocol
    training = protocol["training"]
    return (f"STRICT • {protocol['experiment_id']} • {selection.variant}\n"
        f"Dataset: {protocol['dataset']['dataset_id']} • Model: {protocol['models'][selection.variant]['architecture']}\n"
        f"Epoki: {training['epochs']} • Batch: {training['batch']} • imgsz: {training['imgsz']}\n"
        f"Optimizer: {training['optimizer']} • Seed: {training['seed']} • Urządzenie: {training['device']}\n"
        "Protokół zweryfikowany. Pola zablokowane przez protokół.")


def frozen_cockpit_summary(host):
    if not is_active(host):
        return None
    protocol = host._mz_protocol
    variant = host._mz_active_variant
    dataset, training = protocol["dataset"], protocol["training"]
    architecture = protocol["models"][variant]["architecture"].replace("yolo", "YOLO", 1) + " Detect"
    environment = protocol.get("environment", {})
    hardware = "CPU" if training["device"] == "cpu" else str(environment.get("gpu") or training["device"])
    memory = float(environment.get("gpu_memory_bytes") or 0) / (1024 ** 3)
    if memory and training["device"] != "cpu":
        hardware += f" | {memory:.1f} GB VRAM"
    counts = {name:dataset[name + "_count"] for name in ("train", "val", "test")}
    return {"status":"Protokół zweryfikowany", "tone":"success",
        "subtitle":f"STRICT • {protocol['experiment_id']} • {variant} • Dataset znaków (YOLO Detect)",
        "cards":[("Dataset", dataset["dataset_id"], "Dataset znaków (YOLO Detect)"),
            ("Próbka", f"{sum(counts.values())} obrazów", f"train {counts['train']} | val {counts['val']} | test {counts['test']}"),
            ("Model", architecture, "Oficjalny checkpoint z protokołu"),
            ("Sprzęt", hardware, f"batch {training['batch']} | imgsz {training['imgsz']} | epoki {training['epochs']}")]}


def frozen_execution_summary_rows(host):
    summary = frozen_cockpit_summary(host)
    if summary is None:
        return None
    training = host._mz_protocol["training"]
    return [("Tor", "Znaki (YOLO Detect)"), ("Eksperyment", host._mz_protocol["experiment_id"]),
        ("Dataset", host._mz_protocol["dataset"]["dataset_id"]),
        ("Model", summary["cards"][2][1]), ("Sprzęt", summary["cards"][3][1]),
        ("Parametry", f"{training['epochs']} epok | batch {training['batch']} | {training['imgsz']}px | LR {training['lr0']}"),
        ("Optimizer / seed", f"{training['optimizer']} / {training['seed']}"), ("Status", summary["status"])]


def validate_gui_context(host, protocol, *, require_target=True):
    if getattr(host, "_step4_fine_tune_parent_run_id", "") or host._resolve_step4_fine_tune_parent_run() is not None:
        raise ValueError("Tryb kontrolowany wymaga oficjalnych wag bazowych. Wyłącz kontekst dotrenowania przed startem.")
    project = CAMPAIGN.get_active_project_name()
    if project and protocol.get("project") and project != protocol["project"]:
        raise ValueError("Otwórz projekt wskazany w protokole: " + protocol["project"])
    if project and protocol.get("iteration") is not None and CAMPAIGN.get_current_iteration_num() != protocol["iteration"]:
        raise ValueError("Aktywna iteracja nie odpowiada protokołowi eksperymentu.")
    locked = host._get_locked_campaign_training_target() if project else None
    if project and locked == "plate":
        raise ValueError("Aktywna kampania ma zablokowany tor tablic (Pose). Protokół wymaga znaków (Detect).")
    if (require_target or project) and host._get_selected_training_target() != "char":
        raise ValueError("Ten eksperyment wymaga toru treningu znaków (Detect).")


def _capture_state(host):
    variables = {name:getattr(host, name).get() for name in (*_FIELDS, "rank_models_dir") if hasattr(host, name)}
    attributes = {name:(hasattr(host, name), deepcopy(getattr(host, name, None))) for name in _CONTEXT_FIELDS}
    storage = {name:(hasattr(host, name), getattr(host, name, None)) for name in _STORAGE_FIELDS}
    widgets = []
    combos = [getattr(host, name, None) for name in ("base_combo", "dataset_variant_combo", "device_combo")]
    for widget in _frozen_widgets(host):
        config = {"state":str(widget.cget("state"))}
        if widget in combos:
            config["values"] = widget.cget("values")
        widgets.append((widget, config))
    return {"variables":variables, "attributes":attributes, "storage":storage, "widgets":widgets,
        "protocol":deepcopy(getattr(host, "_mz_protocol", None)), "variant":getattr(host, "_mz_active_variant", None),
        "mode":is_active(host)}


def _switch_free_route(host, mode):
    saved = {name:getattr(host, name) for name in ("dataset_var", "dataset_variant_var") if hasattr(host, name)}
    try:
        for name in saved:
            setattr(host, name, _DetachedValue())
        host._set_step4_dataset_mode(mode, show_locked_message=False)
        if host._get_selected_training_target() != mode:
            raise ValueError("Nie udało się przełączyć typu datasetu na " + mode + ".")
    finally:
        for name, variable in saved.items():
            setattr(host, name, variable)


def _refresh_context(host):
    for name in ("_refresh_step4_dataset_mode_ui", "_refresh_step4_training_inputs_mode_ui",
                 "_refresh_training_base_model_identity_ui", "_refresh_training_execution_summary",
                 "_refresh_training_start_state"):
        callback = getattr(host, name, None)
        if callable(callback):
            callback()
    callback = getattr(host, "_refresh_training_cockpit", None)
    if callable(callback):
        callback()


def _restore_state(host, snapshot):
    host._mz_applying_protocol = True
    try:
        host.mz_mode_var.set(False)
        for name, (exists, value) in snapshot["storage"].items():
            if exists:
                setattr(host, name, value)
            elif hasattr(host, name):
                delattr(host, name)
        if not CAMPAIGN.get_active_project_name():
            previous_mode = snapshot["attributes"]["_step4_dataset_mode"][1] or "char"
            if host._get_selected_training_target() != previous_mode:
                _switch_free_route(host, previous_mode)
        for name, (exists, value) in snapshot["attributes"].items():
            if exists:
                setattr(host, name, deepcopy(value))
            elif hasattr(host, name):
                delattr(host, name)
        for name, value in snapshot["variables"].items():
            getattr(host, name).set(value)
        for widget, config in snapshot["widgets"]:
            widget.configure(**config)
        host._mz_protocol = deepcopy(snapshot["protocol"])
        host._mz_active_variant = snapshot["variant"]
        host.mz_mode_var.set(snapshot["mode"])
        bind = getattr(host, "_bind_trainer_callbacks", None)
        if callable(bind):
            bind()
    finally:
        host._mz_applying_protocol = False


def _apply_selection(host, selection: MZExperimentSelection) -> None:
    validate_gui_context(host, selection.protocol, require_target=False)
    before = _capture_state(host)
    normal = getattr(host, "_mz_normal_state", None)
    variant_only = before["mode"] and before["protocol"] == selection.protocol
    request = selection.request
    host._mz_applying_protocol = True
    try:
        host._mz_protocol = deepcopy(selection.protocol)
        host._mz_active_variant = selection.variant
        host.mz_variant_var.set(selection.variant)
        host.mz_mode_var.set(True)
        if not variant_only:
            if not CAMPAIGN.get_active_project_name():
                _switch_free_route(host, "char")
            host._step4_dataset_mode = "char"
            host._step4_route_selected = True
            host._step4_train_unlocked = True
            host._current_training_dataset_is_pose = False
            counts = {name:selection.protocol["dataset"][name + "_count"] for name in ("train", "val", "test")}
            counts["total"] = sum(counts.values())
            source = TrainingSource(target="char", kind="yolo_dataset", dataset_dir=request["dataset_path"],
                yaml_path=str(Path(request["dataset_path"]) / "data.yaml"), validated=True,
                stats=TrainingSourceStats.from_mapping(counts), provenance="Protokół eksperymentu", source_stage="Z4/PZ2")
            host._last_training_source = source
            host._pending_step4_input_training_source = source
            host._step4_dataset_summary_split_counts = dict(counts)
            host.dataset_var.set(request["dataset_path"])
            host.dataset_variant_var.set(selection.protocol["dataset"]["dataset_id"])
        label = host._get_custom_base_model_label()
        host.base_custom_var.set(request["base_model"])
        host.base_model_var.set(label)
        for field, key in (("epochs_var", "epochs"), ("batch_var", "batch_size"),
                           ("imgsz_var", "img_size"), ("lr0_var", "lr0"), ("device_var", "device")):
            getattr(host, field).set(request[key])
        host.name_var.set(request["name"])
        _refresh_context(host)
        validate_gui_context(host, selection.protocol)
    except Exception:
        _restore_state(host, before)
        try:
            _refresh_context(host)
        except Exception:
            pass
        raise
    finally:
        host._mz_applying_protocol = False
    if normal is None:
        host._mz_normal_state = before
        host._mz_normal_values = before["variables"]
    _status(host, _selection_status(selection))
    refresh_controls(host)


def disable_mode(host) -> None:
    if _busy(host):
        host.mz_mode_var.set(bool(getattr(host, "_mz_protocol", None)))
        return
    host.mz_mode_var.set(False)
    snapshot = getattr(host, "_mz_normal_state", None)
    if snapshot is not None:
        _restore_state(host, snapshot)
        _refresh_context(host)
    host._mz_normal_state = None
    host._mz_normal_values = None
    _status(host, "Tryb kontrolowany wyłączony. Zwykły trening.")
    host._refresh_training_start_state()


def _run_validation(host, protocol_loader, variant, on_success, *, loading=False, configure_context=False):
    if _busy(host):
        return
    host._mz_validation_in_progress = True
    _status(host, "Sprawdzam protokół, dane, wagi i środowisko…")
    refresh_controls(host)
    host.btn_start_train.configure(state="disabled")

    def finish(selection, error):
        host._mz_validation_in_progress = False
        host._mz_validation_thread = None
        refresh_controls(host)
        if error:
            if is_active(host) and getattr(host, "_mz_active_variant", None):
                host.mz_variant_var.set(host._mz_active_variant)
            if loading and not getattr(host, "_mz_normal_values", None):
                host.mz_mode_var.set(False)
            _status(host, "Protokół nie został zaakceptowany: " + error)
            messagebox.showerror("Eksperyment kontrolowany", error)
            host._refresh_training_start_state()
            return
        try:
            validate_gui_context(host, selection.protocol, require_target=not configure_context)
            on_success(selection)
            _status(host, _selection_status(selection))
        except Exception as exc:
            if is_active(host) and getattr(host, "_mz_active_variant", None):
                host.mz_variant_var.set(host._mz_active_variant)
            if loading and not getattr(host, "_mz_normal_values", None):
                host.mz_mode_var.set(False)
            _status(host, "Start zablokowany: " + str(exc))
            messagebox.showerror("Eksperyment kontrolowany", str(exc))
            host._refresh_training_start_state()

    def worker():
        try:
            selection = prepare_mz_experiment(protocol_loader(), variant)
            error = ""
        except Exception as exc:
            selection, error = None, str(exc)
        host._ui(lambda result=selection, detail=error: finish(result, detail))

    thread = threading.Thread(target=worker, name="Z4MZFrozenPreflight", daemon=True)
    host._mz_validation_thread = thread
    try:
        thread.start()
    except Exception as exc:
        finish(None, str(exc))


def load_protocol(host, path=None):
    if _busy(host):
        return
    path = path or filedialog.askopenfilename(parent=host.frame, title="Wczytaj protokół eksperymentu",
        filetypes=[("Protokół JSON", "*.json")])
    if not path:
        if not getattr(host, "_mz_normal_values", None):
            host.mz_mode_var.set(False)
        return
    variant = host.mz_variant_var.get()
    _run_validation(host, lambda:read_mz_experiment_protocol(path), variant,
        lambda selection:_apply_selection(host, selection), loading=True, configure_context=True)


def toggle_mode(host):
    if host.mz_mode_var.get():
        if getattr(host, "_mz_protocol", None):
            frozen = deepcopy(host._mz_protocol)
            _run_validation(host, lambda:frozen, host.mz_variant_var.get(),
                lambda selection:_apply_selection(host, selection), loading=True, configure_context=True)
        else:
            load_protocol(host)
    else:
        disable_mode(host)


def select_variant(host):
    if not is_active(host):
        return
    frozen = deepcopy(host._mz_protocol)
    _run_validation(host, lambda:frozen, host.mz_variant_var.get(), lambda selection:_apply_selection(host, selection), configure_context=True)


def verify_before_start(host, continuation):
    try:
        validate_gui_context(host, host._mz_protocol)
    except ValueError as exc:
        return messagebox.showerror("Eksperyment kontrolowany", str(exc))
    frozen = deepcopy(host._mz_protocol)
    _run_validation(host, lambda:frozen, host.mz_variant_var.get(), continuation)


def build_section(host, parent):
    host.mz_mode_var = tk.BooleanVar(master=host.frame, value=False)
    host.mz_variant_var = tk.StringVar(master=host.frame, value=MZ_VARIANTS[0])
    host.mz_status_var = tk.StringVar(master=host.frame, value="Tryb kontrolowany wyłączony. Wczytaj frozen protocol JSON.")
    section = ttk.LabelFrame(parent, text=" Eksperyment kontrolowany ", padding=9)
    section.pack(fill=tk.X, pady=(0, getattr(host, "_train_left_section_gap", 16)))
    host.mz_mode_check = ttk.Checkbutton(section, text="Tryb kontrolowany", variable=host.mz_mode_var,
        command=lambda:toggle_mode(host))
    host.mz_mode_check.pack(anchor=tk.W)
    row = ttk.Frame(section)
    row.pack(fill=tk.X, pady=(5, 5))
    host.mz_load_button = ttk.Button(row, text="Wczytaj protokół eksperymentu", command=lambda:load_protocol(host))
    host.mz_load_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
    ttk.Label(row, text="Wariant:").pack(side=tk.LEFT, padx=(8, 4))
    host.mz_variant_combo = ttk.Combobox(row, values=MZ_VARIANTS, textvariable=host.mz_variant_var, state="readonly", width=6)
    host.mz_variant_combo.pack(side=tk.LEFT)
    host.mz_variant_combo.bind("<<ComboboxSelected>>", lambda event:select_variant(host))
    summary = ttk.Label(section, textvariable=host.mz_status_var, justify=tk.LEFT, wraplength=370)
    summary.pack(fill=tk.X)
    host._register_train_left_wrap_target(summary, padding=24, min_wrap=220)
    return section
