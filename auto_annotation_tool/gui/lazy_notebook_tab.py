#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Placeholder for notebook tabs that are built lazily."""

import json
import tkinter as tk
from pathlib import Path
from tkinter import ttk


class _LazyNotebookTab:
    """Lekki placeholder zakładki budowanej dopiero przy pierwszym wejściu."""

    def __init__(self, app, key: str):
        self.app = app
        self.key = str(key or "").strip()
        self.frame = ttk.Frame(app.notebook)
        self.is_processing = False
        self.trainer = None
        self.annotator = None
        self.detector = None
        self.char_detector = None
        self.ocr_engine = None
        self.plate_ocr = None
        self.preview_dir_var = None
        self._preview_fullscreen_active = False
        self._build_placeholder()

    def _build_placeholder(self):
        palette = getattr(self.app, "palette", {})
        bg = palette.get("panel", "#252526")
        fg = palette.get("fg", "#f3f3f3")
        muted = palette.get("muted", "#c7c7c7")
        accent = palette.get("success", "#4ec9b0")

        try:
            self.frame.configure(style="Panel.TFrame")
        except Exception:
            pass

        shell = tk.Frame(
            self.frame,
            bg=bg,
            bd=0,
            highlightthickness=1,
            highlightbackground=palette.get("border", "#3c3c3c"),
        )
        shell.place(relx=0.5, rely=0.5, anchor=tk.CENTER, width=520, height=150)
        self._placeholder_shell = shell

        title = tk.Label(
            shell,
            text=f"{self.app.get_main_tab_label(self.key)}",
            bg=bg,
            fg=fg,
            font=("Segoe UI", 15, "bold"),
            anchor="w",
        )
        title.pack(fill=tk.X, padx=18, pady=(18, 6))
        self._placeholder_title = title

        body = tk.Label(
            shell,
            text=(
                "Zakładka zostanie zbudowana dopiero przy pierwszym wejściu. "
                "Dzięki temu start programu nie musi ładować całego warsztatu naraz."
            ),
            bg=bg,
            fg=muted,
            font=("Segoe UI", 10),
            justify=tk.LEFT,
            anchor="w",
            wraplength=470,
        )
        body.pack(fill=tk.X, padx=18, pady=(0, 10))
        self._placeholder_body = body

        status = tk.Label(
            shell,
            text="Kliknij zakładkę lub przejdź tutaj z wizarda.",
            bg=bg,
            fg=accent,
            font=("Segoe UI", 9),
            anchor="w",
        )
        status.pack(fill=tk.X, padx=18, pady=(0, 16))
        self._placeholder_status = status

    def apply_theme(self):
        palette = getattr(self.app, "palette", {})
        bg = palette.get("panel", "#252526")
        fg = palette.get("fg", "#f3f3f3")
        muted = palette.get("muted", "#c7c7c7")
        accent = palette.get("success", "#4ec9b0")
        border = palette.get("border", "#3c3c3c")
        for widget, options in (
            (getattr(self, "_placeholder_shell", None), {"bg": bg, "highlightbackground": border}),
            (getattr(self, "_placeholder_title", None), {"bg": bg, "fg": fg}),
            (getattr(self, "_placeholder_body", None), {"bg": bg, "fg": muted}),
            (getattr(self, "_placeholder_status", None), {"bg": bg, "fg": accent}),
        ):
            if widget is None:
                continue
            try:
                widget.configure(**options)
            except Exception:
                pass

    def set_loading_state(self):
        try:
            self._placeholder_status.configure(text="Buduję zakładkę... To może chwilę potrwać przy pierwszym wejściu.")
        except Exception:
            pass
        try:
            self._placeholder_body.configure(
                text=(
                    "Pierwsze wejście tworzy pełny interfejs tej zakładki i wczytuje jej lokalny stan. "
                    "Kolejne przełączenia będą już szybkie."
                )
            )
        except Exception:
            pass

    def is_startup_ui_ready(self) -> bool:
        return True

    def release_gpu_resources_for_training(self) -> None:
        return None

    def apply_global_yolo_device_choice(self, *args, **kwargs) -> None:
        return None

    def flush_free_mode_session_state(self) -> None:
        return None

    def clear_campaign_context(self) -> None:
        return None

    def _on_app_close(self, *args, **kwargs) -> None:
        return None

    def capture_free_mode_snapshot_for_project_return(self) -> None:
        return None

    def get_campaign_step2_view_model(self):
        return None

    def get_campaign_step2_wizard_status(self) -> dict:
        return {}

    def get_campaign_step2_source_state(self, *args, **kwargs) -> dict:
        return {}

    def _get_campaign_step2_approval_context(self) -> dict:
        return {}

    def _resolve_safe_annotation_run_dir(self, *args, **kwargs):
        return None

    def _get_run_plate_approved_counts(self, *args, **kwargs) -> tuple[int, int]:
        return 0, 0

    def _get_run_plate_annotation_counts(self, *args, **kwargs) -> tuple[int, int]:
        return 0, 0

    def _build_campaign_char_effective_source(self, *args, **kwargs) -> dict:
        return {}

    def _get_campaign_step3_training_readiness(self, *args, **kwargs) -> dict:
        return {}

    def _has_any_step3_export_outputs(self, *args, **kwargs) -> bool:
        return False

    def _get_gold_export_split_percentages(self, *args, **kwargs) -> tuple[float, float, float]:
        return 80.0, 10.0, 10.0

    @staticmethod
    def _lazy_dataset_split_counts(dataset_path) -> dict:
        counts = {"train": 0, "val": 0, "test": 0, "total": 0}
        try:
            root = Path(dataset_path)
        except Exception:
            return counts
        if root.is_file():
            root = root.parent
        images_root = root / "images"
        if not images_root.exists():
            return counts
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        total = 0
        for split_name in ("train", "val", "test"):
            split_dir = images_root / split_name
            if not split_dir.exists() or not split_dir.is_dir():
                continue
            try:
                split_count = sum(
                    1
                    for item in split_dir.rglob("*")
                    if item.is_file() and item.suffix.lower() in image_exts
                )
            except Exception:
                split_count = 0
            counts[split_name] = int(split_count)
            total += int(split_count)
        counts["total"] = int(total)
        return counts

    @staticmethod
    def _lazy_infer_dataset_target(dataset_path) -> str:
        try:
            text = str(dataset_path or "").lower()
        except Exception:
            text = ""
        if any(token in text for token in ("plate", "plates", "cvat", "pose")):
            return "plate"
        return "char"

    @staticmethod
    def _lazy_load_plate_dataset_manifest(dataset_path) -> dict:
        try:
            root = Path(dataset_path)
            if root.is_file():
                root = root.parent
            manifest_path = root / "dataset_source_manifest.json"
            if not manifest_path.exists():
                return {}
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def _lazy_find_ready_dataset(self, datasets_dir, target: str, *, approved_images: int = 0) -> dict:
        result = {
            "path": "",
            "counts": {"train": 0, "val": 0, "test": 0, "total": 0},
            "stale": False,
        }
        try:
            root = Path(datasets_dir)
        except Exception:
            return result
        if not root.exists():
            return result

        search_roots = []
        for candidate in (root, root / "plates", root / "chars", root / "vehicles"):
            try:
                resolved = candidate.resolve()
            except Exception:
                resolved = candidate
            if candidate.exists() and candidate.is_dir() and resolved not in search_roots:
                search_roots.append(resolved)

        active_project = ""
        current_iteration = 0
        try:
            from ..campaign_manager import CAMPAIGN

            active_project = str(CAMPAIGN.get_active_project_name() or "").strip()
            current_iteration = int(CAMPAIGN.get_current_iteration_num() or 0)
        except Exception:
            pass

        candidates = []
        for search_root in search_roots:
            try:
                for path in Path(search_root).iterdir():
                    if not path.is_dir() or not (path / "data.yaml").exists():
                        continue
                    if self._lazy_infer_dataset_target(path) != target:
                        continue
                    try:
                        stamp = float(path.stat().st_mtime)
                    except Exception:
                        stamp = 0.0
                    candidates.append((path, stamp))
            except Exception:
                continue
        candidates.sort(key=lambda record: record[1], reverse=True)

        stale_path = ""
        stale_counts = {}
        for path, _stamp in candidates:
            counts = self._lazy_dataset_split_counts(path)
            total = int(counts.get("total", 0) or 0)
            if total <= 0:
                continue
            if target != "plate":
                result["path"] = str(path)
                result["counts"] = counts
                return result

            manifest = self._lazy_load_plate_dataset_manifest(path)
            manifest_project = str(manifest.get("project") or "").strip()
            try:
                manifest_iteration = int(manifest.get("iteration", 0) or 0)
            except Exception:
                manifest_iteration = 0
            manifest_images = manifest.get("approved_set_images", None)
            context_ok = (
                bool(active_project)
                and manifest_project == active_project
                and manifest_iteration == current_iteration
            )
            if manifest_images is not None:
                accepted = context_ok and int(manifest_images or 0) == int(approved_images or 0)
            else:
                accepted = context_ok and total >= int(approved_images or 0)
            if accepted or (approved_images > 0 and total == int(approved_images)):
                result["path"] = str(path)
                result["counts"] = counts
                try:
                    from ..campaign_manager import CAMPAIGN

                    CAMPAIGN.set_last_plate_training_source(
                        dataset_path=str(path.resolve()),
                        source_run_path=str(manifest.get("source_run_dir") or ""),
                        source_xml_path=str(manifest.get("source_xml_path") or ""),
                    )
                except Exception:
                    pass
                return result
            if not stale_path:
                stale_path = str(path)
                stale_counts = counts

        if target == "plate" and stale_path and approved_images > 0:
            result["path"] = stale_path
            result["counts"] = stale_counts or result["counts"]
            result["stale"] = True
        return result

    def get_campaign_step4_readiness(self, *args, **kwargs) -> dict:
        if self.key != "training":
            return {"ok": True, "reason": "", "message": ""}
        try:
            from ..campaign_manager import CAMPAIGN
            from ..config import CONFIG
        except Exception:
            return {"ok": True, "reason": "", "message": ""}

        target = str(kwargs.get("iteration_target") or "").strip().lower()
        if not target and args:
            target = str(args[0] or "").strip().lower()
        if target not in {"plate", "char"}:
            target = str(CAMPAIGN.get_iteration_target() or "").strip().lower()
        if target not in {"plate", "char"}:
            target = "char"

        result = {
            "ok": True,
            "reason": "",
            "message": "",
            "iteration_target": target,
            "ready_dataset": "",
            "dataset_hint": "",
            "annotated_images": 0,
            "required_images": 0,
            "required_plates": int(getattr(CONFIG, "CAMPAIGN_MIN_PLATE_ANNOTATIONS", 10) or 10),
            "source_run": "",
            "train_images": 0,
            "val_images": 0,
            "test_images": 0,
            "validation_message": "",
            "project_approved_images": 0,
            "project_approved_plates": 0,
            "project_manual_images": 0,
            "project_auto_accepted_images": 0,
        }

        if not CAMPAIGN.get_active_project_name():
            result.update(ok=False, reason="campaign_inactive", message="Brak aktywnego projektu kampanii.")
            return result
        try:
            if int(CAMPAIGN.get_current_step() or 0) < 4:
                result.update(ok=False, reason="step4_inactive", message="Bramka T07 nie jest jeszcze bieżącą bramką projektu.")
                return result
        except Exception:
            pass

        datasets_dir = CAMPAIGN.get_dir("datasets")
        if datasets_dir is None:
            result.update(
                ok=False,
                reason="missing_datasets_dir",
                message="Projekt nie ma jeszcze poprawnie przygotowanego katalogu datasetów.",
            )
            return result

        if target == "plate":
            try:
                approved_stats = dict(CAMPAIGN.get_plate_approved_set_stats() or {})
            except Exception:
                approved_stats = {}
            approved_images = int(approved_stats.get("images", 0) or 0)
            approved_plates = int(approved_stats.get("plates", 0) or 0)
            result["project_approved_images"] = approved_images
            result["project_approved_plates"] = approved_plates
            result["project_manual_images"] = int(approved_stats.get("manual_images", 0) or 0)
            result["project_auto_accepted_images"] = int(approved_stats.get("auto_accepted_images", 0) or 0)
            result["annotated_images"] = approved_images
            result["source_run"] = "Zatwierdzone tablice projektu"

            if approved_images <= 0:
                result.update(
                    ok=False,
                    reason="missing_plate_annotations",
                    message="Bramka T07 nie ma zatwierdzonych tablic projektu do budowy datasetu YOLO Pose.",
                )
                return result
            min_plates = int(result["required_plates"] or 10)
            if approved_plates < min_plates:
                result.update(
                    ok=False,
                    reason="insufficient_plate_annotations",
                    message=(
                        f"Bramka T07 wymaga minimum {min_plates} zatwierdzonych tablic. "
                        f"Teraz jest {approved_plates}."
                    ),
                )
                return result

            ready = self._lazy_find_ready_dataset(datasets_dir, "plate", approved_images=approved_images)
            if ready.get("path") and not ready.get("stale"):
                counts = dict(ready.get("counts") or {})
                result["ready_dataset"] = str(ready.get("path") or "")
                result["dataset_hint"] = result["ready_dataset"]
                result["train_images"] = int(counts.get("train", 0) or 0)
                result["val_images"] = int(counts.get("val", 0) or 0)
                result["test_images"] = int(counts.get("test", 0) or 0)
                return result
            if ready.get("path") and ready.get("stale"):
                result.update(
                    ok=False,
                    reason="stale_plate_dataset",
                    ready_dataset=str(ready.get("path") or ""),
                    message="Gotowy dataset tablic jest starszy niż aktualny zatwierdzony zbiór projektu.",
                )
                return result
            return result

        try:
            current_iteration = int(CAMPAIGN.get_current_iteration_num() or 0)
            iteration_state = dict(CAMPAIGN.get_iteration_state(iteration_num=current_iteration) or {})
            contracts = iteration_state.get("t06_contracts")
            contracts = dict(contracts) if isinstance(contracts, dict) else {}
            pz3_contract = contracts.get("pz3_char_dataset")
            pz3_contract = dict(pz3_contract) if isinstance(pz3_contract, dict) else {}
            source_iteration = (
                int(pz3_contract.get("source_iteration", 0) or 0)
                or int(pz3_contract.get("created_iteration", 0) or 0)
                or int(pz3_contract.get("iteration", 0) or 0)
            )
            source_path = str(pz3_contract.get("dataset_path") or pz3_contract.get("gold_dataset_path") or "").strip()
            if (
                bool(pz3_contract.get("fulfilled"))
                and source_path
                and (source_iteration <= 0 or source_iteration == current_iteration)
            ):
                root = Path(source_path)
                if root.is_file() and root.name.lower() == "data.yaml":
                    root = root.parent
                if root.exists() and root.is_dir():
                    pairs = (
                        int(pz3_contract.get("source_image_label_pairs", 0) or 0)
                        or int(pz3_contract.get("exportable_plate_count", 0) or 0)
                        or int(pz3_contract.get("perfect_count", 0) or 0)
                    )
                    result.update(
                        ok=True,
                        reason="source_dataset_ready_for_split",
                        ready_dataset="",
                        dataset_hint=str(root),
                        source_dataset=str(root),
                        source_yaml=str(root / "data.yaml"),
                        source_image_label_pairs=int(pairs or 0),
                        validation_message=(
                            "Źródłowy dataset znaków jest gotowy. W Z4/PZ1 utwórz wariant train/val/test."
                        ),
                    )
                    return result
        except Exception:
            pass

        ready = self._lazy_find_ready_dataset(datasets_dir, "char")
        if ready.get("path"):
            counts = dict(ready.get("counts") or {})
            result["ready_dataset"] = str(ready.get("path") or "")
            result["dataset_hint"] = result["ready_dataset"]
            result["train_images"] = int(counts.get("train", 0) or 0)
            result["val_images"] = int(counts.get("val", 0) or 0)
            result["test_images"] = int(counts.get("test", 0) or 0)
        return result

    def get_campaign_step4_finish_state(self, *args, **kwargs) -> dict:
        try:
            from ..campaign_manager import CAMPAIGN

            return dict(CAMPAIGN.get_step4_finish_state() or {})
        except Exception:
            return {}

    def _step4_has_active_operation(self) -> bool:
        return False

    def _get_active_step4_operation_label(self) -> str:
        return ""

    def __getattr__(self, name):
        real_tab = self.app._ensure_tab_loaded(self.key)
        if real_tab is self:
            raise AttributeError(name)
        return getattr(real_tab, name)
