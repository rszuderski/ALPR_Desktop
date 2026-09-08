"""Stage and route state methods for ``CampaignManager``.

This module stores campaign step/status setters and getters.  Methods are bound
back to CampaignManager so callers keep using the same API while the central
manager file stays focused on orchestration.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from .campaign_iteration_paths import (
    default_iteration_path_for_target,
    iteration_path_target,
    normalize_iteration_path,
)

_E1_RESOURCE_CONTRACT_BASELINE_KEYS = (
    "master_pool_dir",
    "master_pool_selected_iteration",
    "project_start_mode",
    "project_start_scope_plate_run",
    "project_start_scope_plate_model",
    "project_start_scope_char_model",
    "project_start_plate_source_run",
    "project_start_plate_source_xml",
    "project_start_plate_source_input",
    "project_start_plate_source_mode",
    "project_start_plate_source_iteration",
    "step1_source_manual_clear_iteration",
    "step1_restored_image_source_dir",
)


def _e1_resource_contract_snapshot(project_data: Dict[str, Any]) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {}
    for key in _E1_RESOURCE_CONTRACT_BASELINE_KEYS:
        value = project_data.get(key, "")
        if key in {"master_pool_selected_iteration", "project_start_plate_source_iteration", "step1_source_manual_clear_iteration"}:
            try:
                value = int(value or 0)
            except Exception:
                value = 0
        else:
            value = str(value or "").strip()
        snapshot[key] = value
    return snapshot


def _invalidate_e1_resource_contract_caches(self) -> None:
    for cache_name in (
        "_latest_ingest_plan_summary_cache",
        "_latest_ingest_plan_summary_runtime_cache",
        "_iteration_image_count_cache",
        "_iteration_image_source_dir_cache",
    ):
        try:
            cache = getattr(self, cache_name, None)
            if cache is not None:
                cache.clear()
        except Exception:
            pass
    try:
        invalidator = getattr(self, "invalidate_step3_char_source_state_cache", None)
        if callable(invalidator):
            invalidator()
    except Exception:
        pass


def _safe_positive_int(value: Any, default: int = 0) -> int:
    if value is None:
        return int(default or 0)
    if isinstance(value, str) and not value.strip():
        return int(default or 0)
    try:
        return max(0, int(value or 0))
    except Exception:
        return int(default or 0)


def _safe_path_token(value: Any) -> str:
    raw_value = str(value or "").strip()
    if not raw_value:
        return ""
    try:
        return str(Path(raw_value).expanduser().resolve()).strip().lower()
    except Exception:
        return raw_value.lower()


def _same_artifact_path(left: Any, right: Any) -> bool:
    left_token = _safe_path_token(left)
    right_token = _safe_path_token(right)
    if not left_token or not right_token:
        return False
    if left_token == right_token:
        return True
    try:
        return Path(left_token).name.lower() == Path(right_token).name.lower()
    except Exception:
        return False


def _entry_matches_iteration(entry: Dict[str, Any], iteration: int) -> bool:
    for key in ("approved_iteration", "first_approved_iteration", "iteration"):
        try:
            if int(entry.get(key, 0) or 0) == int(iteration or 0):
                return True
        except Exception:
            continue
    return False


def _entry_plate_count(entry: Dict[str, Any]) -> int:
    for key in ("plate_count", "valid_plate_count", "plates_count"):
        count = _safe_positive_int(entry.get(key), -1)
        if count >= 0:
            return count
    plates = entry.get("plates")
    if isinstance(plates, list):
        return len(plates)
    return 0


def _infer_t02_at_review_commit_from_approved_set(
    self,
    project_name: str,
    project_data: Dict[str, Any],
    current_iteration: int,
) -> Dict[str, Any]:
    source_run = str(project_data.get("project_start_plate_source_run", "") or "").strip()
    source_xml = str(project_data.get("project_start_plate_source_xml", "") or "").strip()
    if not source_run and not source_xml:
        return {"committed": False}

    try:
        entries = list(self.list_plate_approved_entries(project_name) or [])
    except Exception:
        entries = []
    if not entries:
        return {"committed": False}

    matched_entries: list[Dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if not _entry_matches_iteration(entry, current_iteration):
            continue
        entry_run = str(entry.get("approved_from_run", "") or "").strip()
        entry_xml = str(entry.get("approved_from_xml", "") or "").strip()
        if (
            (source_run and _same_artifact_path(entry_run, source_run))
            or (source_xml and _same_artifact_path(entry_xml, source_xml))
        ):
            matched_entries.append(entry)

    if not matched_entries:
        return {"committed": False}

    approved_images = len(matched_entries)
    approved_plates = sum(_entry_plate_count(entry) for entry in matched_entries)
    approved_plates = int(approved_plates or approved_images)
    committed_at = ""
    for entry in matched_entries:
        committed_at = str(entry.get("approved_at", "") or entry.get("first_approved_at", "") or "").strip()
        if committed_at:
            break
    return {
        "committed": True,
        "inferred": True,
        "iteration": int(current_iteration or 0),
        "current_iteration": int(current_iteration or 0),
        "committed_at": committed_at,
        "run_dir": source_run,
        "approved_images": int(approved_images or 0),
        "approved_plates": int(approved_plates or 0),
    }


def _get_active_data(self) -> Dict[str, Any]:
    act = self.state.get("active_project", "")
    return self.state["projects"].get(act, {})

def get_safe_project_folder_name(self) -> str:
    data = self._get_active_data()
    return data.get("folder_name", "UNNAMED_PROJECT")

def get_project_created_at(self, name: str = None) -> str:
    project_name = (name or self.get_active_project_name() or "").strip()
    if not project_name:
        return ""

    project_data = self.state.get("projects", {}).get(project_name, {})
    return str(project_data.get("created_at", "") or "").strip()

def get_current_iteration_num(self) -> int:
    return self._get_active_data().get("current_iteration", 1)
    
def get_current_step(self) -> int:
    return self._get_active_data().get("current_step", 1)

def set_current_step(self, step: int):
    act = self.state.get("active_project", "")
    if not act or act not in self.state.get("projects", {}): return
    self.state["projects"][act]["current_step"] = step
    self.save_state()

def approve_step1(self):
    """Oznacza krok 1 jako zatwierdzony."""
    act = self.get_active_project_name()
    if not act:
        return
    self.state["projects"][act]["step1_status"] = "approved"
    self.save_state()

def reset_step1(self):
    """Resetuje stan Kroku 1 dla nowej iteracji."""
    act = self.get_active_project_name()
    if not act:
        return
    self.state["projects"][act]["step1_status"] = "pending"
    self.save_state()

def get_step1_status(self) -> str:
    act = self.get_active_project_name()
    if not act:
        return "pending"
    return self.state["projects"][act].get("step1_status", "pending")

@staticmethod
def _normalize_project_start_mode(mode: str | None) -> str:
    value = str(mode or "").strip().lower()
    if not value:
        return ""
    if value in {"assets", "import", "resource", "resources", "mam_zasoby"}:
        return "assets"
    return "fresh"

def set_project_start_mode(self, mode: str | None, project_name: str = None) -> bool:
    project_name = self._resolve_project_name(project_name)
    if not project_name:
        return False

    self.state["projects"][project_name]["project_start_mode"] = self._normalize_project_start_mode(mode)
    self.save_state()
    return True

def get_project_start_mode(self, project_name: str = None) -> str:
    project_name = self._resolve_project_name(project_name)
    if not project_name:
        return ""

    raw_value = self.state["projects"][project_name].get("project_start_mode", "")
    return self._normalize_project_start_mode(raw_value)

@staticmethod
def _normalize_project_start_asset_scope(scope: str | None) -> str:
    value = str(scope or "").strip().lower()
    if value in {"project", "freemode", "na"}:
        return value
    return ""

@staticmethod
def _project_start_asset_scope_state_key(row_key: str | None) -> str:
    normalized = str(row_key or "").strip().lower()
    mapping = {
        "plate_run": "project_start_scope_plate_run",
        "plate_model": "project_start_scope_plate_model",
        "char_model": "project_start_scope_char_model",
    }
    return str(mapping.get(normalized) or "").strip()

def set_project_start_asset_scope(self, row_key: str, scope: str | None, project_name: str = None) -> bool:
    project_name = self._resolve_project_name(project_name)
    if not project_name:
        return False

    state_key = self._project_start_asset_scope_state_key(row_key)
    if not state_key:
        return False

    try:
        self.ensure_e1_resource_contract_baseline(project_name=project_name)
    except Exception:
        pass
    normalized = self._normalize_project_start_asset_scope(scope)
    if str(self.state["projects"][project_name].get(state_key, "") or "").strip().lower() == normalized:
        return True
    self.state["projects"][project_name][state_key] = normalized
    self.save_state()
    return True

def get_project_start_asset_scope(self, row_key: str, project_name: str = None) -> str:
    project_name = self._resolve_project_name(project_name)
    if not project_name:
        return ""

    state_key = self._project_start_asset_scope_state_key(row_key)
    if not state_key:
        return ""

    raw_value = self.state["projects"][project_name].get(state_key, "")
    return self._normalize_project_start_asset_scope(raw_value)

def set_project_start_plate_source(
    self,
    source_run_path: str = "",
    source_xml_path: str = "",
    source_input_path: str = "",
    source_mode: str = "",
    project_name: str = None,
) -> bool:
    project_name = self._resolve_project_name(project_name)
    if not project_name:
        return False

    project_data = self.state["projects"][project_name]
    source_run = str(source_run_path or "").strip()
    source_xml = str(source_xml_path or "").strip()
    source_input = str(source_input_path or "").strip()
    source_mode_value = str(source_mode or "").strip().lower()
    if source_mode_value not in {"approved", "draft"}:
        source_mode_value = ""
    has_source = bool(source_run or source_xml or source_input)
    try:
        current_iteration = int(project_data.get("current_iteration", 1) or 1)
    except Exception:
        current_iteration = 1

    try:
        self.ensure_e1_resource_contract_baseline(project_name=project_name)
    except Exception:
        pass
    project_data["project_start_plate_source_run"] = source_run
    project_data["project_start_plate_source_xml"] = source_xml
    project_data["project_start_plate_source_input"] = source_input
    project_data["project_start_plate_source_mode"] = source_mode_value if has_source else ""
    project_data["project_start_plate_source_iteration"] = int(current_iteration if has_source else 0)
    self.save_state()
    return True

def clear_project_start_plate_source(self, project_name: str = None) -> bool:
    return self.set_project_start_plate_source("", "", "", project_name=project_name)

def get_project_start_plate_source(self, project_name: str = None) -> Dict[str, str]:
    project_name = self._resolve_project_name(project_name)
    if not project_name:
        return {
            "source_run_path": "",
            "source_xml_path": "",
            "source_input_path": "",
            "source_mode": "",
            "source_iteration": "",
        }

    project_data = self.state["projects"].get(project_name, {})
    try:
        source_iteration = int(project_data.get("project_start_plate_source_iteration", 0) or 0)
    except Exception:
        source_iteration = 0
    try:
        current_iteration = int(project_data.get("current_iteration", 1) or 1)
    except Exception:
        current_iteration = 1

    if source_iteration and source_iteration != current_iteration:
        return {
            "source_run_path": "",
            "source_xml_path": "",
            "source_input_path": "",
            "source_mode": "",
            "source_iteration": "",
        }

    return {
        "source_run_path": str(project_data.get("project_start_plate_source_run", "") or "").strip(),
        "source_xml_path": str(project_data.get("project_start_plate_source_xml", "") or "").strip(),
        "source_input_path": str(project_data.get("project_start_plate_source_input", "") or "").strip(),
        "source_mode": str(project_data.get("project_start_plate_source_mode", "") or "").strip().lower(),
        "source_iteration": str(source_iteration or ""),
    }

@staticmethod
def _normalize_iteration_target(target: str | None) -> str:
    value = str(target or "").strip().lower()
    if value in {"plate", "plates", "tablica", "tablice", "pose"}:
        return "plate"
    if value in {"char", "chars", "character", "characters", "znak", "znaki"}:
        return "char"
    return ""

def set_iteration_target(self, target: str | None):
    act = self.get_active_project_name()
    if not act:
        return

    normalized_target = self._normalize_iteration_target(target)
    project_data = self.state["projects"][act]
    project_data["iteration_target"] = normalized_target
    current_path = normalize_iteration_path(project_data.get("iteration_path", ""))
    if not normalized_target:
        project_data["iteration_path"] = ""
    elif current_path and iteration_path_target(current_path) != normalized_target:
        project_data["iteration_path"] = ""
    self.save_state()

def get_iteration_target(self) -> str:
    act = self.get_active_project_name()
    if not act:
        return ""
    return self._normalize_iteration_target(self.state["projects"][act].get("iteration_target", ""))

def clear_iteration_target(self):
    self.set_iteration_target("")

@staticmethod
def _normalize_iteration_path(path: str | None) -> str:
    return normalize_iteration_path(path)

def set_iteration_path(self, path: str | None):
    act = self.get_active_project_name()
    if not act:
        return
    normalized_path = normalize_iteration_path(path)
    project_data = self.state["projects"][act]
    try:
        t01_state = dict(self.get_t01_entry_commit_state(act) or {})
        t01_committed = bool(t01_state.get("committed"))
    except Exception:
        t01_state = {}
        t01_committed = False
    if t01_committed:
        locked_path = normalize_iteration_path(t01_state.get("path") or project_data.get("iteration_path", ""))
        if locked_path not in {"plate_training", "char_from_images"}:
            current_target = self._normalize_iteration_target(project_data.get("iteration_target", ""))
            locked_path = "plate_training" if current_target == "plate" else "char_from_images"
        if normalized_path != locked_path:
            project_data["iteration_path"] = locked_path
            target = iteration_path_target(locked_path)
            if target:
                project_data["iteration_target"] = target
            self.save_state()
            return
    if normalized_path != "char_from_ready_plates":
        try:
            t02_committed = bool(self.get_t02_at_review_commit_state(act).get("committed"))
        except Exception:
            t02_committed = False
        if t02_committed:
            project_data["iteration_path"] = "char_from_ready_plates"
            project_data["iteration_target"] = "char"
            self.save_state()
            return
    project_data["iteration_path"] = normalized_path
    target = iteration_path_target(normalized_path)
    if target:
        project_data["iteration_target"] = target
    elif not normalized_path:
        project_data["iteration_target"] = ""
    self.save_state()

def get_iteration_path(self) -> str:
    act = self.get_active_project_name()
    if not act:
        return ""
    project_data = self.state["projects"][act]
    normalized_path = normalize_iteration_path(project_data.get("iteration_path", ""))
    target = self._normalize_iteration_target(project_data.get("iteration_target", ""))
    if normalized_path and (not target or iteration_path_target(normalized_path) == target):
        return normalized_path
    return default_iteration_path_for_target(target)

def get_explicit_iteration_path(self) -> str:
    act = self.get_active_project_name()
    if not act:
        return ""
    project_data = self.state["projects"][act]
    normalized_path = normalize_iteration_path(project_data.get("iteration_path", ""))
    target = self._normalize_iteration_target(project_data.get("iteration_target", ""))
    if normalized_path and (not target or iteration_path_target(normalized_path) == target):
        return normalized_path
    return ""

def clear_iteration_path(self):
    self.set_iteration_path("")

def set_graph_selected_edge_key(self, edge_key: str | None):
    act = self.get_active_project_name()
    if not act:
        return
    self.state["projects"][act]["graph_selected_edge_key"] = str(edge_key or "").strip()
    self.save_state()

def get_graph_selected_edge_key(self) -> str:
    act = self.get_active_project_name()
    if not act:
        return ""
    return str(self.state["projects"][act].get("graph_selected_edge_key", "") or "").strip()

def clear_graph_selected_edge_key(self):
    self.set_graph_selected_edge_key("")


def ensure_e1_resource_contract_baseline(
    self,
    *,
    force: bool = False,
    project_name: str = None,
) -> bool:
    project_name = self._resolve_project_name(project_name)
    if not project_name:
        return False

    project_data = self.state["projects"][project_name]
    try:
        current_iteration = int(project_data.get("current_iteration", 1) or 1)
    except Exception:
        current_iteration = 1
    try:
        baseline_iteration = int(project_data.get("e1_resource_contract_baseline_iteration", 0) or 0)
    except Exception:
        baseline_iteration = 0
    baseline = project_data.get("e1_resource_contract_baseline")
    if (
        force
        or baseline_iteration != current_iteration
        or not isinstance(baseline, dict)
        or not baseline
    ):
        project_data["e1_resource_contract_baseline_iteration"] = int(current_iteration)
        project_data["e1_resource_contract_baseline"] = _e1_resource_contract_snapshot(project_data)
        project_data["e1_resource_contract_last_rollback_at"] = ""
        project_data["e1_resource_contract_last_rollback_from_path"] = ""
        project_data["e1_resource_contract_last_rollback_to_path"] = ""
        self.save_state()
        return True
    return True


def get_e1_resource_contract_state(self, project_name: str = None) -> Dict[str, Any]:
    project_name = self._resolve_project_name(project_name)
    if not project_name:
        return {
            "baseline_ready": False,
            "has_draft": False,
            "changed_keys": [],
            "current_iteration": 0,
            "baseline_iteration": 0,
        }

    project_data = self.state["projects"].get(project_name, {})
    try:
        current_iteration = int(project_data.get("current_iteration", 1) or 1)
    except Exception:
        current_iteration = 1
    try:
        baseline_iteration = int(project_data.get("e1_resource_contract_baseline_iteration", 0) or 0)
    except Exception:
        baseline_iteration = 0
    baseline = project_data.get("e1_resource_contract_baseline")
    baseline_ready = bool(isinstance(baseline, dict) and baseline and baseline_iteration == current_iteration)
    current = _e1_resource_contract_snapshot(project_data)
    changed_keys: list[str] = []
    if baseline_ready:
        for key in _E1_RESOURCE_CONTRACT_BASELINE_KEYS:
            if current.get(key, "") != baseline.get(key, ""):
                changed_keys.append(key)
    return {
        "baseline_ready": baseline_ready,
        "has_draft": bool(changed_keys),
        "changed_keys": changed_keys,
        "current_iteration": int(current_iteration or 0),
        "baseline_iteration": int(baseline_iteration or 0),
        "current": current,
        "baseline": dict(baseline or {}) if isinstance(baseline, dict) else {},
    }


def has_e1_resource_contract_draft_current_iteration(self, project_name: str = None) -> bool:
    return bool(self.get_e1_resource_contract_state(project_name).get("has_draft"))


def restore_e1_resource_contract_baseline(
    self,
    *,
    from_path: str = "",
    to_path: str = "",
    project_name: str = None,
) -> bool:
    project_name = self._resolve_project_name(project_name)
    if not project_name:
        return False

    state = self.get_e1_resource_contract_state(project_name)
    if not bool(state.get("baseline_ready")):
        return False

    project_data = self.state["projects"][project_name]
    baseline = dict(state.get("baseline") or {})
    for key in _E1_RESOURCE_CONTRACT_BASELINE_KEYS:
        if key in baseline:
            project_data[key] = baseline.get(key)
        else:
            project_data[key] = 0 if key.endswith("_iteration") else ""
    project_data["e1_resource_contract_last_rollback_at"] = datetime.now().isoformat(timespec="seconds")
    project_data["e1_resource_contract_last_rollback_from_path"] = normalize_iteration_path(from_path)
    project_data["e1_resource_contract_last_rollback_to_path"] = normalize_iteration_path(to_path)
    self.save_state()
    _invalidate_e1_resource_contract_caches(self)
    return True


def mark_t02_at_review_committed(
    self,
    *,
    run_dir: str | None = "",
    approved_images: int = 0,
    approved_plates: int = 0,
    project_name: str = None,
) -> bool:
    project_name = self._resolve_project_name(project_name)
    if not project_name:
        return False

    project_data = self.state["projects"][project_name]
    try:
        current_iteration = int(project_data.get("current_iteration", 1) or 1)
    except Exception:
        current_iteration = 1
    project_data["t02_at_review_committed_iteration"] = int(current_iteration)
    project_data["t02_at_review_committed_at"] = datetime.now().isoformat(timespec="seconds")
    project_data["t02_at_review_committed_run"] = str(run_dir or "").strip()
    try:
        project_data["t02_at_review_committed_images"] = max(0, int(approved_images or 0))
    except Exception:
        project_data["t02_at_review_committed_images"] = 0
    try:
        project_data["t02_at_review_committed_plates"] = max(0, int(approved_plates or 0))
    except Exception:
        project_data["t02_at_review_committed_plates"] = 0
    self.save_state()
    return True


def get_t02_at_review_commit_state(self, project_name: str = None) -> Dict[str, Any]:
    project_name = self._resolve_project_name(project_name)
    if not project_name:
        return {"committed": False}

    project_data = self.state["projects"].get(project_name, {})
    try:
        current_iteration = int(project_data.get("current_iteration", 1) or 1)
    except Exception:
        current_iteration = 1
    try:
        committed_iteration = int(project_data.get("t02_at_review_committed_iteration", 0) or 0)
    except Exception:
        committed_iteration = 0
    committed = bool(committed_iteration > 0 and committed_iteration == current_iteration)
    try:
        approved_images = max(0, int(project_data.get("t02_at_review_committed_images", 0) or 0))
    except Exception:
        approved_images = 0
    try:
        approved_plates = max(0, int(project_data.get("t02_at_review_committed_plates", 0) or 0))
    except Exception:
        approved_plates = 0
    inferred_state: Dict[str, Any] = {}
    if not committed:
        inferred_state = _infer_t02_at_review_commit_from_approved_set(
            self,
            project_name,
            project_data,
            current_iteration,
        )
        if bool(inferred_state.get("committed")):
            try:
                project_data["t02_at_review_committed_iteration"] = int(current_iteration)
                project_data["t02_at_review_committed_at"] = str(inferred_state.get("committed_at", "") or "").strip()
                project_data["t02_at_review_committed_run"] = str(inferred_state.get("run_dir", "") or "").strip()
                project_data["t02_at_review_committed_images"] = _safe_positive_int(inferred_state.get("approved_images"))
                project_data["t02_at_review_committed_plates"] = _safe_positive_int(inferred_state.get("approved_plates"))
                self.save_state()
            except Exception:
                pass
            return inferred_state
    return {
        "committed": committed,
        "inferred": False,
        "iteration": int(committed_iteration or 0),
        "current_iteration": int(current_iteration or 0),
        "committed_at": str(project_data.get("t02_at_review_committed_at", "") or "").strip(),
        "run_dir": str(project_data.get("t02_at_review_committed_run", "") or "").strip(),
        "approved_images": approved_images,
        "approved_plates": approved_plates,
    }


def is_t02_at_review_committed_current_iteration(self, project_name: str = None) -> bool:
    return bool(self.get_t02_at_review_commit_state(project_name).get("committed"))


def get_t01_entry_commit_state(self, project_name: str = None) -> Dict[str, Any]:
    project_name = self._resolve_project_name(project_name)
    if not project_name:
        return {"committed": False}

    project_data = self.state["projects"].get(project_name, {})
    current_path = normalize_iteration_path(project_data.get("iteration_path", ""))
    if current_path not in {"plate_training", "char_from_images"}:
        return {"committed": False}

    try:
        current_iteration = int(project_data.get("current_iteration", 1) or 1)
    except Exception:
        current_iteration = 1
    try:
        current_step = int(project_data.get("current_step", 1) or 1)
    except Exception:
        current_step = 1
    step1_status = str(project_data.get("step1_status", "pending") or "pending").strip().lower()
    committed = bool(step1_status == "approved" or current_step > 1)
    reason = ""
    if committed:
        reason = "step1_approved" if step1_status == "approved" else "current_step"
    return {
        "committed": committed,
        "path": current_path,
        "target": iteration_path_target(current_path),
        "iteration": int(current_iteration or 0),
        "current_step": int(current_step or 0),
        "step1_status": step1_status,
        "reason": reason,
    }


def is_t01_entry_committed_current_iteration(self, project_name: str = None) -> bool:
    return bool(self.get_t01_entry_commit_state(project_name).get("committed"))


def get_last_iteration_target(self) -> str:
    act = self.get_active_project_name()
    if not act:
        return ""
    return self._normalize_iteration_target(self.state["projects"][act].get("last_iteration_target", ""))

def get_project_status(self, project_name: str = None) -> str:
    project_name = self._resolve_project_name(project_name)
    if not project_name:
        return "active"

    status = str(self.state["projects"][project_name].get("project_status", "active") or "").strip().lower()
    if status == "completed":
        return "completed"
    if status == "paused":
        return "paused"
    return "active"

def is_project_completed(self, project_name: str = None) -> bool:
    return self.get_project_status(project_name) == "completed"

def is_project_paused(self, project_name: str = None) -> bool:
    return self.get_project_status(project_name) == "paused"

def get_project_paused_at(self, project_name: str = None) -> str:
    project_name = self._resolve_project_name(project_name)
    if not project_name:
        return ""
    return str(self.state["projects"][project_name].get("project_paused_at", "") or "").strip()

def get_project_completed_at(self, project_name: str = None) -> str:
    project_name = self._resolve_project_name(project_name)
    if not project_name:
        return ""
    return str(self.state["projects"][project_name].get("project_completed_at", "") or "").strip()

def pause_project(self, project_name: str = None) -> bool:
    project_name = self._resolve_project_name(project_name)
    if not project_name:
        return False

    project_data = self.state["projects"][project_name]
    project_data["project_status"] = "paused"
    project_data["project_paused_at"] = datetime.now().isoformat()
    project_data["project_completed_at"] = ""
    self.save_state()
    return True

def complete_project(self, project_name: str = None) -> bool:
    project_name = self._resolve_project_name(project_name)
    if not project_name:
        return False

    project_data = self.state["projects"][project_name]
    if int(project_data.get("current_step", 1) or 1) < 5:
        project_data["current_step"] = 5
    project_data["project_status"] = "completed"
    project_data["project_paused_at"] = ""
    project_data["project_completed_at"] = datetime.now().isoformat()
    self.save_state()
    return True

def reopen_project(self, project_name: str = None) -> bool:
    project_name = self._resolve_project_name(project_name)
    if not project_name:
        return False

    project_data = self.state["projects"][project_name]
    project_data["project_status"] = "active"
    project_data["project_paused_at"] = ""
    project_data["project_completed_at"] = ""
    self.save_state()
    return True

def set_step2_generated(self, staging_run_path: str):
    """Zapisuje informację, że krok 2 został wykonany, ale niezatwierdzony."""
    act = self.get_active_project_name()
    if not act:
        return
    self.state["projects"][act]["step2_status"] = "generated"
    self.state["projects"][act]["step2_staging_run"] = str(staging_run_path)
    self.save_state()

def approve_step2(self):
    """Oznacza krok 2 jako zatwierdzony."""
    act = self.get_active_project_name()
    if not act:
        return
    self.state["projects"][act]["step2_status"] = "approved"
    self.save_state()

def reset_step2(self):
    """Resetuje stan Kroku 2 (Autoanotacja) do oczekiwania na nowe zatwierdzenie."""
    act = self.get_active_project_name()
    if not act:
        return
    self.state["projects"][act]["step2_status"] = "pending"
    self.state["projects"][act]["step2_staging_run"] = ""
    self.save_state()

def get_step2_status(self) -> str:
    act = self.get_active_project_name()
    if not act:
        return "pending"
    return self.state["projects"][act].get("step2_status", "pending")

def get_step2_staging_run(self) -> str:
    act = self.get_active_project_name()
    if not act:
        return ""
    return self.state["projects"][act].get("step2_staging_run", "")

def set_step3_needs_rework(self):
    """Oznacza krok 3 jako wymagający poprawy."""
    act = self.get_active_project_name()
    if not act:
        return
    self.state["projects"][act]["step3_status"] = "needs_rework"
    self.save_state()

def set_step3_ready(self):
    """Oznacza krok 3 jako gotowy do zatwierdzenia w wizardzie."""
    act = self.get_active_project_name()
    if not act:
        return
    self.state["projects"][act]["step3_status"] = "ready"
    self.save_state()

def approve_step3(self):
    """Oznacza krok 3 jako zakończony powodzeniem."""
    act = self.get_active_project_name()
    if not act:
        return
    self.state["projects"][act]["step3_status"] = "approved"
    self.save_state()

def set_step3_pending(self):
    """Przywraca krok 3 do stanu w toku bez resetu zapisanej pracy."""
    act = self.get_active_project_name()
    if not act:
        return
    self.state["projects"][act]["step3_status"] = "pending"
    self.save_state()

def reset_step3(self):
    """Resetuje stan kroku 3."""
    act = self.get_active_project_name()
    if not act:
        return

    self.state["projects"][act]["step3_status"] = "pending"
    self.state["projects"][act]["step3_substep"] = 1
    self.state["projects"][act]["step3_stage1_done"] = False
    self.state["projects"][act]["step3_stage2_done"] = False
    self.state["projects"][act]["step3_extract_entry_mode"] = ""
    self.state["projects"][act]["step3_extract_workflow_step"] = "entry"
    self.state["projects"][act]["step3_extract_annotation_run_dir"] = ""
    self.state["projects"][act]["step3_extract_xml_path"] = ""
    self.state["projects"][act]["step3_extract_images_dir"] = ""
    self.state["projects"][act]["step3_preview_dir"] = ""
    self.save_state()

def get_step3_status(self) -> str:
    act = self.get_active_project_name()
    if not act:
        return "pending"
    return self.state["projects"][act].get("step3_status", "pending")

def get_step4_status(self) -> str:
    act = self.get_active_project_name()
    if not act:
        return "pending"

    project_data = self.state["projects"].get(act, {})
    try:
        current_step = int(project_data.get("current_step", 1) or 1)
    except Exception:
        current_step = 1
    project_status = str(project_data.get("project_status", "active") or "active").strip().lower()
    if project_status == "completed" or current_step > 4:
        return "approved"

    finish_ready = False
    try:
        finish_state = dict(self.get_step4_finish_state() or {})
        finish_ready = bool(finish_state.get("ready"))
    except Exception:
        finish_ready = bool(project_data.get("step4_finish_ready", False))

    without_training_ready = False
    try:
        no_training_state = dict(self.get_step4_without_training_decision() or {})
        without_training_ready = bool(no_training_state.get("ready"))
    except Exception:
        without_training_ready = bool(project_data.get("step4_without_training_ready", False))

    if finish_ready or without_training_ready:
        return "ready"
    return "pending"

def get_step3_substep(self) -> int:
    act = self.get_active_project_name()
    if not act:
        return 1
    return int(self.state["projects"][act].get("step3_substep", 1))


def set_step3_substep(self, value: int):
    act = self.get_active_project_name()
    if not act:
        return

    value = int(value)
    if value < 1:
        value = 1
    if value > 3:
        value = 3

    self.state["projects"][act]["step3_substep"] = value
    self.save_state()


def is_step3_stage1_done(self) -> bool:
    act = self.get_active_project_name()
    if not act:
        return False
    return bool(self.state["projects"][act].get("step3_stage1_done", False))


def set_step3_stage1_done(self, done: bool):
    act = self.get_active_project_name()
    if not act:
        return

    self.state["projects"][act]["step3_stage1_done"] = bool(done)
    self.save_state()


def is_step3_stage2_done(self) -> bool:
    act = self.get_active_project_name()
    if not act:
        return False
    return bool(self.state["projects"][act].get("step3_stage2_done", False))


def set_step3_stage2_done(self, done: bool):
    act = self.get_active_project_name()
    if not act:
        return

    self.state["projects"][act]["step3_stage2_done"] = bool(done)
    self.save_state()


def reset_step3_progress(self):
    act = self.get_active_project_name()
    if not act:
        return

    self.state["projects"][act]["step3_substep"] = 1
    self.state["projects"][act]["step3_stage1_done"] = False
    self.state["projects"][act]["step3_stage2_done"] = False
    self.state["projects"][act]["step3_extract_entry_mode"] = ""
    self.state["projects"][act]["step3_extract_workflow_step"] = "entry"
    self.state["projects"][act]["step3_extract_annotation_run_dir"] = ""
    self.state["projects"][act]["step3_extract_xml_path"] = ""
    self.state["projects"][act]["step3_extract_images_dir"] = ""
    self.state["projects"][act]["step3_preview_dir"] = ""
    self.save_state()

def get_step3_extract_state(self) -> Dict[str, str]:
    act = self.get_active_project_name()
    if not act:
        return {
            "entry_mode": "",
            "workflow_step": "entry",
            "annotation_run_dir": "",
            "xml_path": "",
            "images_dir": "",
        }

    project_data = self.state["projects"][act]
    return {
        "entry_mode": str(project_data.get("step3_extract_entry_mode", "") or "").strip(),
        "workflow_step": str(project_data.get("step3_extract_workflow_step", "entry") or "entry").strip(),
        "annotation_run_dir": str(project_data.get("step3_extract_annotation_run_dir", "") or "").strip(),
        "xml_path": str(project_data.get("step3_extract_xml_path", "") or "").strip(),
        "images_dir": str(project_data.get("step3_extract_images_dir", "") or "").strip(),
    }

def get_step3_preview_dir(self) -> str:
    act = self.get_active_project_name()
    if not act:
        return ""
    value = str(self.state["projects"][act].get("step3_preview_dir", "") or "").strip()
    if value:
        return value

    try:
        if self.state_file.exists():
            loaded = json.loads(self.state_file.read_text(encoding="utf-8"))
            project_data = dict((loaded.get("projects") or {}).get(act) or {})
            fallback_value = str(project_data.get("step3_preview_dir", "") or "").strip()
            if fallback_value:
                try:
                    self.state["projects"][act]["step3_preview_dir"] = fallback_value
                except Exception:
                    pass
                return fallback_value
    except Exception:
        pass

    return ""

def set_step3_preview_dir(self, preview_dir: str | None) -> None:
    act = self.get_active_project_name()
    if not act:
        return
    self.state["projects"][act]["step3_preview_dir"] = str(preview_dir or "").strip()
    self.save_state()

def get_step3_detection_yolo_model(self) -> str:
    act = self.get_active_project_name()
    if not act:
        return ""
    return str(self.state["projects"][act].get("step3_detection_yolo_model", "") or "").strip()

def set_step3_detection_yolo_model(self, model_path: str | None) -> None:
    act = self.get_active_project_name()
    if not act:
        return
    self.state["projects"][act]["step3_detection_yolo_model"] = str(model_path or "").strip()
    self.save_state()

def set_step3_extract_state(
    self,
    *,
    entry_mode: str | None = None,
    workflow_step: str | None = None,
    annotation_run_dir: str | None = None,
    xml_path: str | None = None,
    images_dir: str | None = None,
) -> None:
    act = self.get_active_project_name()
    if not act:
        return

    project_data = self.state["projects"][act]

    if entry_mode is not None:
        project_data["step3_extract_entry_mode"] = str(entry_mode or "").strip()
    if workflow_step is not None:
        project_data["step3_extract_workflow_step"] = str(workflow_step or "entry").strip() or "entry"
    if annotation_run_dir is not None:
        project_data["step3_extract_annotation_run_dir"] = str(annotation_run_dir or "").strip()
    if xml_path is not None:
        project_data["step3_extract_xml_path"] = str(xml_path or "").strip()
    if images_dir is not None:
        project_data["step3_extract_images_dir"] = str(images_dir or "").strip()

    self.save_state()


_INSTANCE_METHODS = ('_get_active_data', 'get_safe_project_folder_name', 'get_project_created_at', 'get_current_iteration_num', 'get_current_step', 'set_current_step', 'approve_step1', 'reset_step1', 'get_step1_status', 'set_project_start_mode', 'get_project_start_mode', 'set_project_start_asset_scope', 'get_project_start_asset_scope', 'set_project_start_plate_source', 'clear_project_start_plate_source', 'get_project_start_plate_source', 'set_iteration_target', 'get_iteration_target', 'clear_iteration_target', 'set_iteration_path', 'get_iteration_path', 'get_explicit_iteration_path', 'clear_iteration_path', 'set_graph_selected_edge_key', 'get_graph_selected_edge_key', 'clear_graph_selected_edge_key', 'ensure_e1_resource_contract_baseline', 'get_e1_resource_contract_state', 'has_e1_resource_contract_draft_current_iteration', 'restore_e1_resource_contract_baseline', 'mark_t02_at_review_committed', 'get_t02_at_review_commit_state', 'is_t02_at_review_committed_current_iteration', 'get_t01_entry_commit_state', 'is_t01_entry_committed_current_iteration', 'get_last_iteration_target', 'get_project_status', 'is_project_completed', 'is_project_paused', 'get_project_paused_at', 'get_project_completed_at', 'pause_project', 'complete_project', 'reopen_project', 'set_step2_generated', 'approve_step2', 'reset_step2', 'get_step2_status', 'get_step2_staging_run', 'set_step3_needs_rework', 'set_step3_ready', 'approve_step3', 'set_step3_pending', 'reset_step3', 'get_step3_status', 'get_step4_status', 'get_step3_substep', 'set_step3_substep', 'is_step3_stage1_done', 'set_step3_stage1_done', 'is_step3_stage2_done', 'set_step3_stage2_done', 'reset_step3_progress', 'get_step3_extract_state', 'get_step3_preview_dir', 'set_step3_preview_dir', 'get_step3_detection_yolo_model', 'set_step3_detection_yolo_model', 'set_step3_extract_state')


_STATIC_METHODS = ('_normalize_project_start_mode', '_normalize_project_start_asset_scope', '_project_start_asset_scope_state_key', '_normalize_iteration_target', '_normalize_iteration_path')


def bind_campaign_stage_state_methods(manager_cls):
    for method_name in _INSTANCE_METHODS:
        setattr(manager_cls, method_name, globals()[method_name])
    for method_name in _STATIC_METHODS:
        setattr(manager_cls, method_name, staticmethod(globals()[method_name]))
    return manager_cls
