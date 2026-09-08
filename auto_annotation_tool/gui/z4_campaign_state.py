#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z4 campaign readiness and training-source state helpers extracted from tab_training.py."""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import threading
import datetime
import time
import webbrowser
import csv
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from ..campaign_manager import CAMPAIGN
from ..config import (
    CONFIG,
    YOLO_AVAILABLE,
    AVAILABLE_POSE_MODELS,
    AVAILABLE_DETECT_MODELS,
    PIL_AVAILABLE,
    get_torch_module,
    get_yolo_class,
    is_cuda_available,
    logger,
)
from ..icons import IconManager
from ..validators import validate_model_file, format_yolo_model_identity
from ..training import YOLOPoseTrainer, TrainingHistory, TrainingStatus, DatasetCreator, DatasetSplitter
from ..ranking import ModelRanking
from ..utils import cleanup_gpu_memory, safe_load_yaml, get_image_files
from .help_manager import HELP
from .inertial_scroll import InertialScrollController
from .section_header_label import SectionHeaderLabel
from .web_slim_scrollbar import WebSlimScrollbar, blend_hex_colors
from .zoomable_canvas import ZoomableCanvas
from .z4_campaign_flow import (
    build_step4_campaign_navigation_view_model,
    build_step4_dataset_workflow_view_model,
    build_step4_training_inputs_view_model,
    clear_campaign_context,
    complete_campaign_project,
    _current_iteration_pz3_source_contract,
    _current_iteration_step4_dataset_record,
    _resolve_step4_dataset_record_root,
    _step4_dataset_record_counts,
    finish_campaign_step4,
    get_campaign_training_target,
    open_campaign_step4_entry as _flow_open_campaign_step4_entry,
    poll_training_completion,
    restore_step4_campaign_project_state,
    set_campaign_context,
    set_campaign_training_target,
)
from .z4_flow_models import (
    CharYoloDatasetSourceAdapter,
    PlateXmlImagesSourceAdapter,
    TrainingSource,
    TrainingSourceStats,
)
from .z4_free_mode_flow import (
    refresh_free_training_route_cards,
    refresh_free_training_route_ui,
    update_step4_notebook_mode,
)
from .z4_shared_ui import (
    accept_training_input_context,
    clear_step4_guidance,
    guide_step4_builder_action,
    guide_step4_finish_action,
    guide_step4_next_action,
    guide_step4_route_selection,
    mark_step4_dataset_ready,
    open_step4_dataset_stage,
    refresh_step4_campaign_builder_inputs_ui,
    refresh_step4_analysis_tab_visibility,
    refresh_step4_campaign_navigation_ui,
    refresh_step4_dataset_mode_ui,
    refresh_step4_training_inputs_mode_ui,
    set_step4_dataset_mode,
    sync_step4_analysis_nav_buttons,
    step4_dataset_go_back,
    step4_dataset_go_next,
    step4_train_go_back,
)
from . import z4_dataset_sources
from . import z4_training_metrics
from . import z4_dataset_builder
from . import z4_analysis_ranking
from . import z4_training_runtime
from .z4_view_models import (
    Step4CampaignNavigationViewModel,
    Step4DatasetWorkflowViewModel,
    Step4TrainingInputsViewModel,
)

NAV_BUTTON_WIDTH = 18

if PIL_AVAILABLE:
    from PIL import Image, ImageDraw, ImageFont

YOLO = None



def _validate_campaign_finish_run(
    self,
    run_id: str,
    *,
    target: str | None = None,
    history: TrainingHistory | None = None,
) -> dict:
    normalized_target = str(target or self.get_campaign_training_target() or "").strip().lower()
    if normalized_target not in ("char", "plate"):
        return {}

    expected_dataset_path = ""
    if normalized_target == "plate":
        try:
            stored_plate_source = dict(CAMPAIGN.get_last_plate_training_source() or {})
        except Exception:
            stored_plate_source = {}
        expected_dataset_path = str(stored_plate_source.get("dataset_path", "") or "").strip()
    else:
        try:
            readiness = self.get_campaign_step4_readiness(iteration_target=normalized_target) or {}
        except Exception:
            readiness = {}
        expected_dataset_path = str(readiness.get("ready_dataset", "") or "").strip()

    history_source = history or self._get_campaign_history_for_finish_recovery()
    if history_source is None:
        return {}

    try:
        run = history_source.get_run(str(run_id or "").strip())
    except Exception:
        run = None
    if run is None:
        return {}

    status_value = str(getattr(run, "status", "") or "").strip().lower()
    if not self._is_finish_eligible_training_status(status_value):
        return {}

    run_target = ""
    infer_target = getattr(history_source, "_infer_run_target", None)
    try:
        if callable(infer_target):
            run_target = str(infer_target(run) or "").strip().lower()
    except Exception:
        run_target = ""
    if run_target not in ("char", "plate"):
        try:
            run_target = str(
                self._infer_dataset_target(getattr(run, "dataset_path", "")) or ""
            ).strip().lower()
        except Exception:
            run_target = ""
    if run_target != normalized_target:
        return {}

    if expected_dataset_path:
        try:
            expected_dataset_resolved = str(Path(expected_dataset_path).resolve())
        except Exception:
            expected_dataset_resolved = expected_dataset_path
        run_dataset_raw = str(getattr(run, "dataset_path", "") or "").strip()
        try:
            run_dataset_resolved = str(Path(run_dataset_raw).resolve()) if run_dataset_raw else ""
        except Exception:
            run_dataset_resolved = run_dataset_raw
        if run_dataset_resolved and run_dataset_resolved != expected_dataset_resolved:
            return {}

    best_weights = str(getattr(run, "best_weights", "") or "").strip()
    try:
        has_artifacts = bool(best_weights and Path(best_weights).exists())
    except Exception:
        has_artifacts = bool(best_weights)
    if not has_artifacts:
        return {}

    return {
        "ready": True,
        "run_id": str(getattr(run, "id", "") or "").strip(),
        "target": normalized_target,
        "status": status_value,
    }


def _recover_campaign_finish_state_from_history(
    self,
    target: str | None = None,
    *,
    history: TrainingHistory | None = None,
) -> dict:
    if not CAMPAIGN.get_active_project_name():
        return {}

    try:
        current_step = int(CAMPAIGN.get_current_step() or 0)
    except Exception:
        current_step = 0
    if current_step < 4:
        return {}

    normalized_target = str(target or self.get_campaign_training_target() or "").strip().lower()
    if normalized_target not in ("char", "plate"):
        return {}

    expected_dataset_path = ""
    if normalized_target == "plate":
        try:
            stored_plate_source = dict(CAMPAIGN.get_last_plate_training_source() or {})
        except Exception:
            stored_plate_source = {}
        expected_dataset_path = str(stored_plate_source.get("dataset_path", "") or "").strip()
        try:
            expected_dataset_path = str(Path(expected_dataset_path).resolve()) if expected_dataset_path else ""
        except Exception:
            pass
    else:
        try:
            readiness = self.get_campaign_step4_readiness(iteration_target=normalized_target) or {}
        except Exception:
            readiness = {}
        expected_dataset_path = str(readiness.get("ready_dataset", "") or "").strip()
        try:
            expected_dataset_path = str(Path(expected_dataset_path).resolve()) if expected_dataset_path else ""
        except Exception:
            pass

    history_source = history or self._get_campaign_history_for_finish_recovery()
    try:
        all_runs = list(history_source.get_all_runs() or []) if history_source is not None else []
    except Exception:
        all_runs = []
    if not all_runs:
        return {}

    infer_target = getattr(history_source, "_infer_run_target", None)
    for run in all_runs:
        if not self._is_finish_eligible_training_status(getattr(run, "status", "")):
            continue

        run_target = ""
        try:
            if callable(infer_target):
                run_target = str(infer_target(run) or "").strip().lower()
        except Exception:
            run_target = ""
        if run_target not in ("char", "plate"):
            try:
                run_target = str(
                    self._infer_dataset_target(getattr(run, "dataset_path", "")) or ""
                ).strip().lower()
            except Exception:
                run_target = ""
        if run_target != normalized_target:
            continue

        if expected_dataset_path:
            run_dataset_raw = str(getattr(run, "dataset_path", "") or "").strip()
            try:
                run_dataset_path = str(Path(run_dataset_raw).resolve()) if run_dataset_raw else ""
            except Exception:
                run_dataset_path = run_dataset_raw
            if run_dataset_path and run_dataset_path != expected_dataset_path:
                continue

        best_weights = str(getattr(run, "best_weights", "") or "").strip()
        try:
            has_artifacts = bool(best_weights and Path(best_weights).exists())
        except Exception:
            has_artifacts = bool(best_weights)
        if not has_artifacts:
            continue

        return {
            "ready": True,
            "run_id": str(getattr(run, "id", "") or "").strip(),
            "target": normalized_target,
            "status": str(getattr(run, "status", "") or "").strip().lower(),
        }

    return {}


def get_campaign_step4_finish_state(self, *, iteration_target: str | None = None) -> dict:
    if not CAMPAIGN.get_active_project_name():
        return {"ready": False, "run_id": "", "target": "", "iteration": 0}

    target = str(iteration_target or CAMPAIGN.get_iteration_target() or self.get_campaign_training_target() or "").strip().lower()
    if target not in ("char", "plate"):
        target = "char"
    try:
        current_iteration = int(CAMPAIGN.get_current_iteration_num() or 0)
    except Exception:
        current_iteration = 0

    def _declared_training_iteration(record: dict) -> int:
        if not isinstance(record, dict):
            return 0
        for field in ("trained_iteration", "source_iteration", "iteration"):
            try:
                value = int(record.get(field, 0) or 0)
            except Exception:
                value = 0
            if value > 0:
                return value
        return 0

    current_iteration_state = {}
    current_iteration_step4 = {}
    try:
        current_iteration_state = dict(CAMPAIGN.get_iteration_state(iteration_num=current_iteration) or {})
        current_iteration_step4 = dict(current_iteration_state.get("step4_training") or {})
    except Exception:
        current_iteration_state = {}
        current_iteration_step4 = {}
    if current_iteration_step4:
        current_iteration_step4.setdefault("iteration", current_iteration)
        current_iteration_step4.setdefault("trained_iteration", current_iteration)
    if not current_iteration_step4:
        try:
            current_bundle = dict(CAMPAIGN.get_iteration_artifact_bundle(iteration_num=current_iteration) or {})
            bundle_step4 = dict(current_bundle.get("step4_training") or {})
            if _declared_training_iteration(bundle_step4) == current_iteration:
                current_iteration_step4 = bundle_step4
        except Exception:
            current_iteration_step4 = {}

    stored = {}
    try:
        stored = CAMPAIGN.get_step4_finish_state() or {}
    except Exception:
        stored = {}

    history_source = self._get_campaign_history_for_finish_recovery()
    stored_ready = bool(stored.get("ready", False))
    stored_run_id = str(stored.get("run_id", "") or "").strip()
    stored_target = str(stored.get("target", "") or "").strip().lower()
    stored_iteration = int(stored.get("iteration", 0) or 0)
    if stored_target not in ("char", "plate"):
        stored_target = target
    stored_model_path = str(stored.get("model_path", "") or "").strip()
    try:
        stored_model_ready = bool(stored_model_path and Path(stored_model_path).exists() and Path(stored_model_path).is_file())
    except Exception:
        stored_model_ready = bool(stored_model_path)

    if (
        stored_ready
        and stored_run_id
        and bool(stored.get("selection_confirmed", True))
        and stored_iteration == current_iteration
        and (not stored_target or stored_target == target)
        and stored_model_ready
    ):
        return {
            "ready": True,
            "run_id": stored_run_id,
            "target": stored_target or target,
            "iteration": current_iteration,
            "model_path": stored_model_path,
            "selection_confirmed": True,
            "status": "completed",
        }

    bundle_run_id = str(current_iteration_step4.get("run_id", "") or "").strip()
    bundle_target = str(current_iteration_step4.get("target", "") or "").strip().lower()
    bundle_status = str(current_iteration_step4.get("status", "") or "").strip().lower()
    iteration_training_matches = bool(
        bundle_run_id
        and bundle_target == target
        and self._is_finish_eligible_training_status(bundle_status)
    )

    if (
        stored_ready
        and stored_run_id
        and stored_iteration == current_iteration
        and iteration_training_matches
        and bundle_run_id == stored_run_id
    ):
        validated = self._validate_campaign_finish_run(
            stored_run_id,
            target=stored_target,
            history=history_source,
        )
        if validated:
            if stored_target != target and target in ("char", "plate"):
                validated["target"] = target
            validated["iteration"] = current_iteration
            return validated

    # A completed training run is only a candidate. T06 may reopen after restart
    # only when the user explicitly promoted one run/model as the project model.
    recovered = {}

    if stored_ready or stored_run_id:
        try:
            CAMPAIGN.set_step4_finish_state(False)
        except Exception:
            pass

    return {"ready": False, "run_id": "", "target": target, "iteration": current_iteration}


def _build_training_dataset_validation_message(self, dataset_root: Path, msg: str, stats: dict | None = None) -> str:
    stats = stats or {}
    target = self._get_selected_training_target()
    train_images = int(stats.get("train_images", 0) or 0)
    val_images = int(stats.get("val_images", 0) or 0)
    test_images = int(stats.get("test_images", 0) or 0)
    dataset_label = self._format_training_target_label(target)
    dataset_rel = self._format_workspace_relative_path(dataset_root)

    guidance = (
        "Popraw dataset i uruchom trening ponownie."
    )
    if msg == "Brak obrazów w images/val":
        if target == "plate":
            guidance = (
                "Ten dataset ma pusty split walidacyjny `images/val`. "
                "Przebuduj dataset tablic w Z2, aby co najmniej 1 oznaczony obraz trafił do walidacji. "
                "Przy bardzo małych próbkach oznacz przynajmniej 2 obrazy, a praktycznie 3+."
            )
        else:
            guidance = (
                "Ten dataset ma pusty split walidacyjny `images/val`. "
                "Przebuduj lub podziel dataset ponownie tak, aby co najmniej 1 obraz trafił do walidacji."
            )
    elif msg == "Brak obrazów w images/train":
        guidance = (
            "Dataset nie ma zadnych obrazów treningowych. "
            "Przebuduj go tak, aby folder `images/train` zawieral dane do nauki modelu."
        )
    elif msg.startswith("Brak: images/"):
        missing_split = msg.split("Brak:", 1)[-1].strip()
        guidance = (
            f"Dataset nie zawiera wymaganego folderu `{missing_split}`. "
            "Przebuduj dataset, aby YOLO mial komplet wymaganych splitow."
        )

    unlock_note = ""
    if CAMPAIGN.get_active_project_name():
        stage_label = "E4T" if target == "plate" else "E4Z" if target == "char" else "E4T/E4Z"
        unlock_note = (
            f"\n\nDopóki trening w {stage_label} nie wystartuje poprawnie, iteracja nie odblokuje zamknięcia T06."
        )

    return (
        "Dataset nie jest jeszcze gotowy do treningu.\n\n"
        f"Tor: {dataset_label}\n"
        f"Dataset: {dataset_rel}\n"
        f"train={train_images}, val={val_images}, test={test_images}\n"
        f"Walidacja: {msg}\n\n"
        f"{guidance}"
        f"{unlock_note}"
    )


def _get_pose_dataset_size_warning(self, dataset_root: Path | None = None, stats: dict | None = None) -> str:
    try:
        root = Path(dataset_root or str(self.dataset_var.get() or "").strip())
    except Exception:
        return ""

    if not str(root):
        return ""

    yaml_path = root / "data.yaml"
    if not yaml_path.exists():
        return ""

    try:
        cfg = safe_load_yaml(yaml_path)
    except Exception:
        cfg = None

    if not isinstance(cfg, dict) or "kpt_shape" not in cfg:
        return ""

    loaded_stats = dict(stats or {})
    if not loaded_stats:
        is_valid, _validation_msg, validation_stats = self.trainer.validate_dataset(root)
        if not is_valid:
            return ""
        loaded_stats = dict(validation_stats or {})

    train_images = int(loaded_stats.get("train_images", 0) or 0)
    val_images = int(loaded_stats.get("val_images", 0) or 0)
    test_images = int(loaded_stats.get("test_images", 0) or 0)
    total_images = train_images + val_images + test_images

    if total_images <= 0:
        return ""

    if total_images <= 10 or train_images < 8 or val_images < 2:
        size_label = "bardzo mały"
        warning_body = (
            "Przy tak małej próbce model może nauczyć się zgrubnej lokalizacji tablicy, "
            "ale nie geometrii jej rogów. Częstym objawem są małe lub niestabilne wielokąty "
            "pojawiające się w okolicy prawdziwej tablicy."
        )
    elif total_images < 30 or train_images < 20 or val_images < 5:
        size_label = "mały"
        warning_body = (
            "To zwykle wystarcza tylko na bardzo wstępny eksperyment. Geometria rogów tablic "
            "może być nadal niestabilna, dlatego przed oceną modelu warto powiększyć zbiór ręcznych anotacji."
        )
    else:
        return ""

    return (
        f"Ostrzeżenie: dataset YOLO Pose jest {size_label} "
        f"(train={train_images}, val={val_images}, test={test_images}, razem={total_images}).\n"
        f"Przy tak małym zbiorze split 80/10/10 może naturalnie dać np. {train_images}/{val_images}/{test_images}; "
        "to nie jest błąd splitu, tylko skutek małej liczby obrazów.\n"
        f"{warning_body}"
    )


def get_campaign_step4_readiness(self, *, iteration_target: str | None = None) -> dict:
    target = str(iteration_target or CAMPAIGN.get_iteration_target() or "").strip().lower()
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
        result.update(
            ok=False,
            reason="campaign_inactive",
            message="Brak aktywnego projektu kampanii.",
        )
        return result

    datasets_dir = CAMPAIGN.get_dir("datasets")
    if datasets_dir is None:
        logger.error("Brak katalogu datasets dla aktywnego projektu.")
        result.update(
            ok=False,
            reason="missing_datasets_dir",
            message="Projekt nie ma jeszcze poprawnie przygotowanego katalogu datasetow.",
        )
        return result

    if target == "char":
        fast_dataset_record = _current_iteration_step4_dataset_record("char")
        fast_dataset_root = _resolve_step4_dataset_record_root(fast_dataset_record)
        fast_counts = _step4_dataset_record_counts(fast_dataset_record)
        if (
            fast_dataset_root is not None
            and int(fast_counts.get("train", 0) or 0) > 0
            and int(fast_counts.get("val", 0) or 0) > 0
        ):
            try:
                yaml_path = fast_dataset_root / "data.yaml"
                yaml_ok = yaml_path.exists()
            except Exception:
                yaml_ok = False
            if yaml_ok:
                result["ready_dataset"] = str(fast_dataset_root)
                result["dataset_hint"] = str(fast_dataset_root)
                result["train_images"] = int(fast_counts.get("train", 0) or 0)
                result["val_images"] = int(fast_counts.get("val", 0) or 0)
                result["test_images"] = int(fast_counts.get("test", 0) or 0)
                result["validation_message"] = "Wariant treningowy odtworzony z kontraktu Z4/PZ1 bieżącej iteracji."
                source_dataset = str(fast_dataset_record.get("source_dataset", "") or "").strip()
                source_yaml = str(fast_dataset_record.get("source_yaml", "") or "").strip()
                if source_dataset or source_yaml:
                    try:
                        source_root = Path(source_yaml or source_dataset)
                        if source_root.is_file() and source_root.name.lower() == "data.yaml":
                            source_yaml = str(source_root)
                            source_root = source_root.parent
                        elif source_root:
                            source_yaml = source_yaml or str(source_root / "data.yaml")
                        source_dataset = str(source_root.resolve()) if source_root.exists() else str(source_root)
                    except Exception:
                        pass
                    result["source_dataset"] = source_dataset
                    result["source_yaml"] = source_yaml
                return result

        current_step4_record: dict = {}

        def _current_iteration_step4_dataset_path() -> Path | None:
            nonlocal current_step4_record
            try:
                iteration_num = int(CAMPAIGN.get_current_iteration_num() or 0)
            except Exception:
                iteration_num = 0
            if iteration_num <= 0:
                return None

            def _declared_dataset_iteration(record: dict) -> int:
                if not isinstance(record, dict):
                    return 0
                for field in ("dataset_iteration", "created_iteration", "iteration"):
                    try:
                        value = int(record.get(field, 0) or 0)
                    except Exception:
                        value = 0
                    if value > 0:
                        return value
                return 0

            record = {}
            try:
                iteration_state = dict(CAMPAIGN.get_iteration_state(iteration_num=iteration_num) or {})
                record = dict(iteration_state.get("step4_dataset") or {})
            except Exception:
                record = {}
            if not record:
                try:
                    bundle = dict(CAMPAIGN.get_iteration_artifact_bundle(iteration_num=iteration_num) or {})
                    bundle_record = dict(bundle.get("step4_dataset") or {})
                    if _declared_dataset_iteration(bundle_record) == iteration_num:
                        record = bundle_record
                    else:
                        record = {}
                except Exception:
                    record = {}
            if not record:
                return None
            current_step4_record = dict(record)
            record.setdefault("iteration", iteration_num)
            record_target = str(record.get("target", "") or "").strip().lower()
            if record_target and record_target != target:
                return None
            try:
                total_images = int(record.get("total_images", 0) or 0)
            except Exception:
                total_images = 0
            if total_images <= 0:
                try:
                    total_images = (
                        int(record.get("train_images", 0) or 0)
                        + int(record.get("val_images", 0) or 0)
                        + int(record.get("test_images", 0) or 0)
                    )
                except Exception:
                    total_images = 0
            if total_images <= 0:
                return None
            dataset_path = str(record.get("dataset_path", "") or "").strip()
            yaml_path = str(record.get("yaml_path", "") or "").strip()
            candidates = []
            if dataset_path:
                candidates.append(Path(dataset_path))
            if yaml_path:
                candidates.append(Path(yaml_path).parent)
            for candidate in candidates:
                try:
                    if candidate.exists() and candidate.is_dir():
                        return candidate
                except Exception:
                    continue
            return candidates[0] if candidates else None

        ready_dataset = None
        invalid_ready_result = None
        ready_dataset = _current_iteration_step4_dataset_path()

        if ready_dataset is not None:
            ready_dataset_str = str(ready_dataset)
            result["ready_dataset"] = ready_dataset_str
            result["dataset_hint"] = ready_dataset_str
            validation_stats = {}
            try:
                is_valid_dataset, validation_msg, validation_stats = self.trainer.validate_dataset(Path(ready_dataset))
            except Exception as e:
                is_valid_dataset = False
                validation_msg = f"Błąd walidacji datasetu: {e}"
                validation_stats = {}

            validation_stats = dict(validation_stats or {})
            result["train_images"] = int(validation_stats.get("train_images", 0) or 0)
            result["val_images"] = int(validation_stats.get("val_images", 0) or 0)
            result["test_images"] = int(validation_stats.get("test_images", 0) or 0)
            result["validation_message"] = str(validation_msg or "").strip()
            source_dataset = str(current_step4_record.get("source_dataset", "") or "").strip()
            source_yaml = str(current_step4_record.get("source_yaml", "") or "").strip()
            if source_dataset or source_yaml:
                try:
                    source_root = Path(source_yaml or source_dataset)
                    if source_root.is_file() and source_root.name.lower() == "data.yaml":
                        source_yaml = str(source_root)
                        source_root = source_root.parent
                    elif source_root:
                        source_yaml = source_yaml or str(source_root / "data.yaml")
                    source_dataset = str(source_root.resolve()) if source_root.exists() else str(source_root)
                except Exception:
                    pass
                result["source_dataset"] = source_dataset
                result["source_yaml"] = source_yaml

            if not is_valid_dataset:
                invalid_ready_result = {
                    "ok": False,
                    "reason": "invalid_char_dataset",
                    "message": (
                        "E4Z w torze znaków nie ma jeszcze gotowego wariantu treningowego train/val.\n\n"
                        f"{self._build_training_dataset_validation_message(Path(ready_dataset), validation_msg, validation_stats)}\n\n"
                        "Jeżeli to jest źródłowy dataset z PZ3, przejdź do Z4/PZ1 i utwórz wariant datasetu ze splitem."
                    ),
                }
            else:
                return result

        pz3_source = _current_iteration_pz3_source_contract()
        pz3_source_path = str(pz3_source.get("dataset_path") or "").strip()
        if pz3_source_path:
            source_pairs = (
                int(pz3_source.get("source_image_label_pairs", 0) or 0)
                or int(pz3_source.get("exportable_plate_count", 0) or 0)
                or int(pz3_source.get("perfect_count", 0) or 0)
            )
            result.update(
                ok=True,
                reason="source_dataset_ready_for_split",
                ready_dataset="",
                dataset_hint=pz3_source_path,
                source_dataset=pz3_source_path,
                source_yaml=str(Path(pz3_source_path) / "data.yaml"),
                train_images=0,
                val_images=0,
                test_images=0,
                source_image_label_pairs=int(source_pairs or 0),
                validation_message=(
                    "Źródłowy dataset YOLO Detect znaków jest wskazany przez PZ3 bieżącej iteracji. "
                    "W Z4/PZ1 utwórz wariant treningowy train/val/test."
                ),
            )
            return result

        latest_source = None
        latest_source_info = {}
        try:
            source_candidates = []
            for path in self._find_dataset_source_candidates(Path(datasets_dir)):
                try:
                    inferred = self._infer_dataset_target(str(path))
                except Exception:
                    inferred = "char"
                if inferred != "char":
                    continue
                info = self._validate_char_yolo_split_source(
                    path,
                    create_missing_yaml=False,
                    overwrite_incompatible_yaml=False,
                    resolve_nested_dataset=True,
                )
                if not bool(info.get("ok")):
                    continue
                src = info.get("src") or path
                try:
                    stamp = float(Path(src).stat().st_mtime)
                except Exception:
                    stamp = float(Path(path).stat().st_mtime)
                source_candidates.append((Path(src), dict(info), stamp))
            if source_candidates:
                latest_source, latest_source_info, _stamp = max(source_candidates, key=lambda rec: rec[2])
        except Exception:
            latest_source = None
            latest_source_info = {}

        if latest_source is not None:
            source_stats = dict(latest_source_info.get("stats") or {})
            dataset_hint = str(latest_source)
            result.update(
                ok=True,
                reason="source_dataset_ready_for_split",
                ready_dataset="",
                dataset_hint=dataset_hint,
                source_dataset=dataset_hint,
                train_images=0,
                val_images=0,
                test_images=0,
                source_image_label_pairs=int(source_stats.get("total", 0) or 0),
                validation_message=(
                    "Źródłowy dataset YOLO Detect znaków jest gotowy. "
                    "W Z4/PZ1 utwórz wariant treningowy train/val/test."
                ),
            )
            return result

        if invalid_ready_result is not None:
            result.update(invalid_ready_result)
            return result

    if target != "plate":
        dataset_hint = self._format_workspace_relative_path(Path(datasets_dir))
        result.update(
            ok=False,
            reason="missing_char_dataset",
            dataset_hint=dataset_hint,
            message=(
                "E4Z w torze znaków pozostaje zablokowane, bo Z3 nie przygotowało jeszcze poprawnego datasetu treningowego.\n\n"
                f"Katalog datasetow projektu: {dataset_hint}\n\n"
                "Wróć do Z3, wyeksportuj albo podziel dataset znaków i upewnij się, ze zawiera niepuste obrazy w train i val."
            ),
        )
        return result

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
    result["source_run"] = "ApprovedSet projektu"

    if approved_images <= 0:
        result.update(
            ok=False,
            reason="missing_plate_annotations",
            message=(
                "E4T w torze tablic pozostaje zablokowane, bo zbiór zatwierdzonych tablic projektu "
                "jest jeszcze pusty.\n\n"
                "Najpierw przygotuj i zatwierdź ręcznie pierwszy zestaw zdjęć w Z2. "
                "Dopiero wtedy projekt będzie miał własne źródło do budowy datasetu YOLO Pose."
            ),
        )
        return result

    min_plate_approval_plates = int(getattr(CONFIG, "CAMPAIGN_MIN_PLATE_ANNOTATIONS", 10) or 10)
    if approved_plates < min_plate_approval_plates:
        manual_images = int(result.get("project_manual_images", 0) or 0)
        auto_images = int(result.get("project_auto_accepted_images", 0) or 0)
        missing_plates = max(0, int(min_plate_approval_plates) - int(approved_plates or 0))
        result.update(
            ok=False,
            reason="insufficient_plate_annotations",
            message=(
                f"E4T w torze tablic wymaga co najmniej {min_plate_approval_plates} zatwierdzonych tablic.\n\n"
                f"Zatwierdzony zbiór projektu ma teraz {approved_images} obraz(y) i {approved_plates} tablic(e).\n"
                f"Brakuje jeszcze: {missing_plates} tablic.\n"
                f"W tym: ręczne {manual_images}, zaakceptowane po autoanotacji {auto_images}.\n"
                "Wróć do Z2, dodaj brakujące oznaczenia albo zaakceptuj kolejne obrazy i dopiero wtedy przejdź dalej."
            ),
        )
        return result

    plate_dataset_info = self._resolve_campaign_plate_ready_dataset(Path(datasets_dir))
    ready_dataset = plate_dataset_info.get("path")
    ready_counts = dict(plate_dataset_info.get("counts") or {})
    if ready_dataset is not None and not bool(plate_dataset_info.get("stale")):
        result["ready_dataset"] = str(ready_dataset)
        result["dataset_hint"] = str(ready_dataset)
        result["train_images"] = int(ready_counts.get("train", 0) or 0)
        result["val_images"] = int(ready_counts.get("val", 0) or 0)
        result["test_images"] = int(ready_counts.get("test", 0) or 0)
        return result

    if ready_dataset is not None and bool(plate_dataset_info.get("stale")):
        stale_total = int(ready_counts.get("total", 0) or 0)
        result.update(
            ok=False,
            reason="stale_plate_dataset",
            message=(
                "Z4 w torze tablic widzi nowszy ApprovedSet projektu niż ostatnio przygotowany dataset treningowy.\n\n"
                f"Aktualny ApprovedSet: {approved_images} obraz(y), {approved_plates} tablic(e).\n"
                f"Ostatni gotowy dataset: {stale_total} obraz(y).\n\n"
                "Wróć do PZ1 i przebuduj dataset tablic, aby trening korzystał z aktualnego zbioru projektu."
            ),
        )
        return result

    return result


def open_campaign_step4_entry(
    self,
    *,
    iteration_target: str | None = None,
    preferred_subtab: str | None = None,
) -> dict:
    return _flow_open_campaign_step4_entry(
        self,
        iteration_target=iteration_target,
        preferred_subtab=preferred_subtab,
    )


def _restore_step4_campaign_project_state(self):
    restore_step4_campaign_project_state(self)


def _plate_dataset_source_manifest_path(dataset_dir: Path) -> Path:
    return Path(dataset_dir) / "dataset_source_manifest.json"


def _load_plate_dataset_source_manifest(self, dataset_dir: Path) -> dict:
    manifest_path = self._plate_dataset_source_manifest_path(dataset_dir)
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_plate_dataset_source_manifest(
    self,
    dataset_dir: Path,
    *,
    source_kind: str,
    source_run_dir: Path | None = None,
    source_xml_path: Path | None = None,
    source_images_dir: Path | None = None,
) -> None:
    dataset_dir = Path(dataset_dir)
    payload = {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "project": str(CAMPAIGN.get_active_project_name() or "").strip(),
        "iteration": int(CAMPAIGN.get_current_iteration_num() or 0) if CAMPAIGN.get_active_project_name() else 0,
        "dataset_dir": str(dataset_dir.resolve()),
        "source_kind": str(source_kind or "").strip(),
        "source_run_dir": "",
        "source_run_name": "",
        "source_xml_path": "",
        "source_images_dir": "",
    }

    try:
        approved_stats = dict(CAMPAIGN.get_plate_approved_set_stats() or {})
    except Exception:
        approved_stats = {}
    payload["approved_set_images"] = int(approved_stats.get("images", 0) or 0)
    payload["approved_set_plates"] = int(approved_stats.get("plates", 0) or 0)

    if source_run_dir is not None:
        try:
            source_run_dir = Path(source_run_dir)
            payload["source_run_dir"] = str(source_run_dir.resolve())
            payload["source_run_name"] = source_run_dir.name
        except Exception:
            payload["source_run_dir"] = str(source_run_dir)
            payload["source_run_name"] = str(Path(source_run_dir).name)

    if source_xml_path is not None:
        try:
            payload["source_xml_path"] = str(Path(source_xml_path).resolve())
        except Exception:
            payload["source_xml_path"] = str(source_xml_path)

    if source_images_dir is not None:
        try:
            payload["source_images_dir"] = str(Path(source_images_dir).resolve())
        except Exception:
            payload["source_images_dir"] = str(source_images_dir)

    self._plate_dataset_source_manifest_path(dataset_dir).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _resolve_plate_training_source_from_dataset(self, dataset_path: Path | None) -> dict:
    result = {
        "dataset_path": "",
        "source_run_path": "",
        "source_xml_path": "",
    }
    if dataset_path is None:
        return result

    try:
        dataset_path = Path(dataset_path)
    except Exception:
        return result

    try:
        if not dataset_path.exists():
            return result
    except Exception:
        return result

    result["dataset_path"] = str(dataset_path.resolve()) if dataset_path.exists() else str(dataset_path)

    if dataset_path.is_file():
        dataset_path = dataset_path.parent

    manifest = self._load_plate_dataset_source_manifest(dataset_path)
    source_run_path = str(manifest.get("source_run_dir") or "").strip()
    source_xml_path = str(manifest.get("source_xml_path") or "").strip()

    if source_run_path and Path(source_run_path).exists():
        result["source_run_path"] = str(Path(source_run_path).resolve())
    if source_xml_path and Path(source_xml_path).exists():
        result["source_xml_path"] = str(Path(source_xml_path).resolve())

    if result["source_run_path"] or result["source_xml_path"]:
        return result

    run_name = str(manifest.get("source_run_name") or "").strip()
    if not run_name:
        match = re.match(r"^Plates_Z2_(.+)_\d{8}_\d{6}$", dataset_path.name)
        if match:
            run_name = str(match.group(1) or "").strip()

    if not run_name:
        return result

    auto_dir = CAMPAIGN.get_dir("auto_ann")
    if auto_dir is None:
        return result

    candidates = []
    try:
        for candidate in Path(auto_dir).rglob(run_name):
            if (
                candidate.is_dir()
                and candidate.name == run_name
                and (candidate / "annotations.xml").exists()
            ):
                candidates.append(candidate)
    except Exception:
        candidates = []

    if not candidates:
        return result

    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    best = candidates[0]
    result["source_run_path"] = str(best.resolve())
    result["source_xml_path"] = str((best / "annotations.xml").resolve())
    return result


def _remember_campaign_plate_training_source(self, dataset_path: Path | None) -> None:
    if not CAMPAIGN.get_active_project_name():
        return
    if str(self.get_campaign_training_target() or "").strip().lower() != "plate":
        return

    source_info = self._resolve_plate_training_source_from_dataset(dataset_path)
    CAMPAIGN.set_last_plate_training_source(
        dataset_path=source_info.get("dataset_path", ""),
        source_run_path=source_info.get("source_run_path", ""),
        source_xml_path=source_info.get("source_xml_path", ""),
    )


def _remember_campaign_training_run_in_registry(
    self,
    *,
    run_id: str | None = None,
    status: str | None = None,
    target: str | None = None,
) -> None:
    if not CAMPAIGN.get_active_project_name():
        return

    normalized_target = str(target or self.get_campaign_training_target() or "").strip().lower()
    if normalized_target not in ("char", "plate"):
        normalized_target = "char"

    resolved_run_id = str(run_id or self.current_run_id or "").strip()
    run = None
    if resolved_run_id:
        try:
            run = self.history.get_run(resolved_run_id)
        except Exception:
            run = None

    status_value = str(
        status
        or (getattr(run, "status", "") if run is not None else "")
        or ""
    ).strip().lower()

    payload = {
        "run_id": resolved_run_id,
        "target": normalized_target,
        "status": status_value,
        "iteration": int(CAMPAIGN.get_current_iteration_num() or 0),
        "trained_iteration": int(CAMPAIGN.get_current_iteration_num() or 0),
        "dataset_path": str(getattr(run, "dataset_path", "") or "").strip() if run is not None else "",
        "output_dir": str(getattr(run, "output_dir", "") or "").strip() if run is not None else "",
        "best_weights": str(getattr(run, "best_weights", "") or "").strip() if run is not None else "",
        "last_weights": str(getattr(run, "last_weights", "") or "").strip() if run is not None else "",
        "current_epoch": int(getattr(run, "current_epoch", 0) or 0) if run is not None else 0,
        "epochs": int(getattr(run, "epochs", 0) or 0) if run is not None else 0,
        "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }

    try:
        CAMPAIGN.upsert_iteration_state(
            iteration_num=CAMPAIGN.get_current_iteration_num(),
            updates={"step4_training": payload},
        )
    except Exception as e:
        logger.debug(f"Nie udało się zapisać runu treningowego do iteracyjnego rejestru: {e}")
    try:
        CAMPAIGN.upsert_iteration_artifact_bundle(
            iteration_num=CAMPAIGN.get_current_iteration_num(),
            updates={"step4_training": payload},
        )
    except Exception:
        pass
