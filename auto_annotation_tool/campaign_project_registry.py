"""Project registry and defaults for ``CampaignManager``.

This module keeps the basic project lifecycle separate from the larger campaign
iteration and artifact logic.  The methods are bound back to CampaignManager so
existing call sites keep using the same public/private method names.
"""

from __future__ import annotations

import json
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from .config import logger
from . import project_attachment


def _iter_project_workspace_dirs(root: Path) -> list[Path]:
    auto_ann_root = root / "2_auto_annotations"
    datasets_root = root / "4_training_datasets"
    runs_root = root / "5_training_runs"
    models_root = root / "6_models"
    rankings_root = root / "7_rankings"
    presets_root = root / "8_presets"
    staging_root = root / "_staging"

    return [
        root / "1_raw_images",
        auto_ann_root,
        root / "3_cropped_characters",
        datasets_root,
        runs_root,
        models_root,
        rankings_root,
        presets_root,
        presets_root / "ocr",
        presets_root / "detection_pipeline",
        presets_root / "augmentation",
        presets_root / "augmentation" / "plate",
        presets_root / "augmentation" / "char",
        presets_root / "training",
        presets_root / "training" / "plate",
        presets_root / "training" / "char",
        presets_root / "ranking_scenarios",
        # Legacy fallback kept for existing OCR presets.
        root / "8_ocr_presets",
        staging_root,
        staging_root / "auto_annotations",
        staging_root / "plate_manual_stage",
        root / "_campaign_state",
        root / "_campaign_state" / "ingest",
    ]


def _ensure_project_workspace_tree(self, root: Path) -> None:
    for path in self._iter_project_workspace_dirs(root):
        path.mkdir(parents=True, exist_ok=True)


def _get_project_default_fields(self) -> Dict[str, Any]:
    return {
        "master_pool_dir": "",
        "ingest_batch_size": 200,
        "project_start_mode": "",
        "project_start_scope_plate_run": "",
        "project_start_scope_plate_model": "",
        "project_start_scope_char_model": "",
        "project_start_plate_source_run": "",
        "project_start_plate_source_xml": "",
        "project_start_plate_source_input": "",
        "project_start_plate_source_mode": "",
        "project_start_plate_source_iteration": 0,
        "e1_resource_contract_baseline_iteration": 0,
        "e1_resource_contract_baseline": {},
        "e1_resource_contract_last_rollback_at": "",
        "e1_resource_contract_last_rollback_from_path": "",
        "e1_resource_contract_last_rollback_to_path": "",
        "t02_at_review_committed_iteration": 0,
        "t02_at_review_committed_at": "",
        "t02_at_review_committed_run": "",
        "t02_at_review_committed_images": 0,
        "t02_at_review_committed_plates": 0,
        "step1_source_manual_clear_iteration": 0,
        "step1_restored_image_source_dir": "",
        "step1_status": "pending",
        "iteration_target": "",
        "iteration_path": "",
        "graph_selected_edge_key": "",
        "last_iteration_target": "",
        "last_plate_manual_source_run": "",
        "last_plate_manual_source_xml": "",
        "last_plate_manual_source_input": "",
        "last_plate_training_dataset": "",
        "last_plate_training_source_run": "",
        "last_plate_training_source_xml": "",
        "step4_finish_ready": False,
        "step4_last_run_id": "",
        "step4_last_target": "",
        "step4_last_iteration": 0,
        "step4_without_training_ready": False,
        "step4_without_training_target": "",
        "step4_without_training_iteration": 0,
        "step3_extract_entry_mode": "",
        "step3_extract_workflow_step": "entry",
        "step3_extract_annotation_run_dir": "",
        "step3_extract_xml_path": "",
        "step3_extract_images_dir": "",
        "step3_preview_dir": "",
        "step3_detection_yolo_model": "",
        "project_status": "active",
        "project_paused_at": "",
        "project_completed_at": "",
    }


def _ensure_project_defaults(self, project_data: Dict[str, Any]) -> Dict[str, Any]:
    if "step1_status" not in project_data:
        try:
            inferred_step = int(project_data.get("current_step", 1) or 1)
        except Exception:
            inferred_step = 1
        project_data["step1_status"] = "approved" if inferred_step >= 2 else "pending"
    for key, value in self._get_project_default_fields().items():
        project_data.setdefault(key, value)
    return project_data


def _get_default_project_template(self, name: str) -> Dict[str, Any]:
    clean_name = re.sub(r"[^A-Za-z0-9_\-]", "_", name)
    proj_id = uuid.uuid4().hex[:6].upper()
    data = {
        "folder_name": f"{clean_name}_{proj_id}",
        "created_at": datetime.now().isoformat(),
        "current_iteration": 1,
        "current_step": 1,
        "project_start_mode": "",
        "project_start_scope_plate_run": "",
        "project_start_scope_plate_model": "",
        "project_start_scope_char_model": "",
        "project_start_plate_source_run": "",
        "project_start_plate_source_xml": "",
        "project_start_plate_source_input": "",
        "project_start_plate_source_mode": "",
        "project_start_plate_source_iteration": 0,
        "e1_resource_contract_baseline_iteration": 0,
        "e1_resource_contract_baseline": {},
        "e1_resource_contract_last_rollback_at": "",
        "e1_resource_contract_last_rollback_from_path": "",
        "e1_resource_contract_last_rollback_to_path": "",
        "t02_at_review_committed_iteration": 0,
        "t02_at_review_committed_at": "",
        "t02_at_review_committed_run": "",
        "t02_at_review_committed_images": 0,
        "t02_at_review_committed_plates": 0,
        "step1_status": "pending",
        "step2_status": "pending",
        "step2_staging_run": "",
            "iteration_target": "",
            "iteration_path": "",
            "graph_selected_edge_key": "",
            "last_iteration_target": "",
        "last_plate_manual_source_run": "",
        "last_plate_manual_source_xml": "",
        "last_plate_manual_source_input": "",
        "last_plate_training_dataset": "",
        "last_plate_training_source_run": "",
        "last_plate_training_source_xml": "",
        "step4_finish_ready": False,
        "step4_last_run_id": "",
        "step4_last_target": "",
        "step4_last_iteration": 0,
        "step4_without_training_ready": False,
        "step4_without_training_target": "",
        "step4_without_training_iteration": 0,
        "best_vehicle_model": "",
        "best_plate_model": "",
        "best_char_model": "",
        "step3_status": "pending",
        "step3_substep": 1,
        "step3_stage1_done": False,
        "step3_stage2_done": False,
        "step3_extract_entry_mode": "",
        "step3_extract_workflow_step": "entry",
        "step3_extract_annotation_run_dir": "",
        "step3_extract_xml_path": "",
        "step3_extract_images_dir": "",
        "step3_preview_dir": "",
        "step3_detection_yolo_model": "",
        "project_status": "active",
        "project_paused_at": "",
        "project_completed_at": "",
    }
    return self._ensure_project_defaults(data)


def _load_state(self) -> Dict[str, Any]:
    if self.state_file.exists():
        try:
            with open(self.state_file, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
                if "projects" in data and "active_project" in data:
                    for project_name, project_data in list(data.get("projects", {}).items()):
                        if isinstance(project_data, dict):
                            data["projects"][project_name] = self._ensure_project_defaults(project_data)
                    data["active_project"] = ""
                    self._registered_project_names = set(data.get("projects", {}))
                    return data
        except Exception as e:
            logger.error(f"Błąd czytania rejestru kampanii: {e}")

    self._registered_project_names = set()
    return {
        "active_project": "",
        "projects": {},
    }


def save_state(self):
    try:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        # Preserve projects attached by another instance since this one loaded.
        disk = project_attachment._read_json(self.state_file, {})
        known = getattr(self, "_registered_project_names", set(self.state.get("projects", {})))
        for name, data in disk.get("projects", {}).items():
            if name not in known and name not in self.state["projects"]:
                self.state["projects"][name] = data
        project_attachment._atomic_bytes(self.state_file, project_attachment._json_bytes(self.state))
        self._registered_project_names = set(self.state["projects"])
    except Exception as e:
        logger.error(f"Błąd zapisu rejestru kampanii: {e}")
        return False
    for name in self.state.get("projects", {}):
        try:
            project_attachment.save_project_descriptor(self, name)
        except Exception as exc:
            logger.warning(f"Nie udało się zapisać przenośnego opisu projektu {name}: {exc}")
    return True


def list_attachable_projects(self):
    return project_attachment.list_attachable_projects(self)


def inspect_existing_project(self, folder):
    return project_attachment.inspect_existing_project(self, Path(folder))


def attach_existing_project(self, plan, *, name=None):
    return project_attachment.attach_existing_project(self, plan, name=name)


def _resolve_project_name(self, name: str = None) -> str:
    project_name = str(name or self.get_active_project_name() or "").strip()
    if not project_name:
        return ""
    if project_name not in self.state.get("projects", {}):
        return ""
    return project_name


def _read_json_file(self, path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_json_file(self, path: Path, payload: Dict[str, Any]) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Nie udało się zapisać pliku {path}: {e}")
        return False


def get_all_projects(self) -> List[str]:
    return list(self.state["projects"].keys())


def get_active_project_name(self) -> str:
    act = self.state.get("active_project", "")
    if not act:
        return ""
    if act not in self.state.get("projects", {}):
        return ""
    return act


def set_active_project(self, name: str):
    if name in self.state["projects"]:
        self.state["active_project"] = name
        self.save_state()


def clear_active_project(self):
    self.state["active_project"] = ""
    self.save_state()


def create_project(self, name: str) -> bool:
    name = name.strip()
    if not name:
        return False
    if name in self.state["projects"]:
        return False

    self.state["projects"][name] = self._get_default_project_template(name)
    self.state["active_project"] = name
    self.save_state()

    root = self.get_project_root_dir(name)
    self._ensure_project_workspace_tree(root)
    project_attachment.save_project_descriptor(self, name)
    try:
        self.append_project_history_event(
            "project",
            "Utworzono projekt",
            status="ok",
            project_name=name,
            details={"folder": str(root)},
        )
    except Exception:
        pass
    return True


def delete_project(self, name: str) -> bool:
    if name not in self.state["projects"]:
        return False

    root = self.get_project_root_dir(name)
    try:
        if root.exists():
            shutil.rmtree(root)
    except Exception as e:
        logger.error(f"Nie można usunąć projektu {root}: {e}")

    del self.state["projects"][name]

    if self.state.get("active_project") == name:
        self.state["active_project"] = ""

    self.save_state()
    return True


_INSTANCE_METHODS = (
    "_ensure_project_workspace_tree",
    "_get_project_default_fields",
    "_ensure_project_defaults",
    "_get_default_project_template",
    "_load_state",
    "save_state",
    "_resolve_project_name",
    "_read_json_file",
    "_write_json_file",
    "get_all_projects",
    "get_active_project_name",
    "set_active_project",
    "clear_active_project",
    "create_project",
    "delete_project",
    "list_attachable_projects",
    "inspect_existing_project",
    "attach_existing_project",
)


_STATIC_METHODS = (
    "_iter_project_workspace_dirs",
)


def bind_campaign_project_registry_methods(manager_cls):
    for method_name in _INSTANCE_METHODS:
        setattr(manager_cls, method_name, globals()[method_name])
    for method_name in _STATIC_METHODS:
        setattr(manager_cls, method_name, staticmethod(globals()[method_name]))
    return manager_cls
